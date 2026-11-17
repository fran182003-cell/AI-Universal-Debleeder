import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
import zmq
import queue
import threading
import scipy.signal as sig
import scipy.io.wavfile as wav
import sys
import time

class ZyliaNetworkClientGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Zylia Cliente: Dúplex y Remuestreo en Vivo")
        self.root.geometry("500x650")
        
        # --- Configuración Zylia y Red ---
        self.FS_HW = 48000
        self.FS_SERVER = 44100
        self.BLOCK_HW = 4800   # 100ms exactos a 48kHz
        self.HOP_SERVER = 4096 # Lo que espera la red neuronal
        
        self.ZYLIA_CHANNELS_ASIO = 20
        self.ZYLIA_CHANNELS_REAL = 19
        self.NUM_SOURCES = 7
        
        self.PORT_SEND = 5555
        self.PORT_RECV = 7777
        
        self.source_names = [f"Posición {i*50}º" for i in range(self.NUM_SOURCES)]
        
        # --- Buffers y Estado ---
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        self.in_stream = None
        self.out_stream = None
        self.send_thread = None
        self.recv_thread = None
        
        self.context = zmq.Context()
        self.socket_send = None
        self.socket_recv = None
        
        self.output_buffers = {i: [] for i in range(self.NUM_SOURCES)}
        self.final_audio = {}
        self.current_monitor_idx = 0
        
        self.setup_ui()

    def setup_ui(self):
        # 1. Configuración de Red
        frame_red = ttk.LabelFrame(self.root, text="Conexión SSH / Red")
        frame_red.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(frame_red, text="IP del servidor de inferencia:").grid(row=0, column=0, padx=5, pady=5)
        self.ip_var = tk.StringVar(value="localhost")
        ttk.Entry(frame_red, textvariable=self.ip_var, width=20).grid(row=0, column=1, padx=5, pady=5)
        
        # 2. Hardware de Audio
        ttk.Label(self.root, text="Micrófono (ASIO Zylia):", font=("Arial", 10)).pack(pady=(10, 0))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(self.root, textvariable=self.device_var, state="readonly", width=50)
        
        devices = sd.query_devices()
        input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] >= self.ZYLIA_CHANNELS_REAL]
        if not input_devices: input_devices = ["No se detectan micrófonos de 19+ canales"]
        
        self.device_combo['values'] = input_devices
        self.device_combo.current(0)
        self.device_combo.pack(pady=5)

        # 3. Monitorización
        ttk.Label(self.root, text="🎛️ Monitorizar EN DIRECTO:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        self.monitor_combo = ttk.Combobox(self.root, state="readonly", width=30)
        self.monitor_combo['values'] = [f"Canal {i}: {self.source_names[i]}" for i in range(self.NUM_SOURCES)]
        self.monitor_combo.current(0)
        self.monitor_combo.pack(pady=5)
        self.monitor_combo.bind("<<ComboboxSelected>>", self.change_monitor_channel)

        # 4. Controles
        self.lbl_status = ttk.Label(self.root, text="Estado: Desconectado", font=("Arial", 12, "bold"))
        self.lbl_status.pack(pady=15)

        self.btn_start = ttk.Button(self.root, text="🔴 Iniciar Transmisión Dúplex", command=self.start_recording)
        self.btn_start.pack(pady=5, fill=tk.X, padx=50)

        self.btn_stop = ttk.Button(self.root, text="⏹ Detener y Guardar en PC", command=self.stop_recording, state=tk.DISABLED)
        self.btn_stop.pack(pady=5, fill=tk.X, padx=50)

        # 5. Reproducción Local
        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(self.root, text="Archivos Locales (Post-Grabación):", font=("Arial", 10, "bold")).pack()

        self.play_buttons = []
        for i in range(self.NUM_SOURCES):
            btn = ttk.Button(
                self.root, text=f"▶ Escuchar: {self.source_names[i]}", 
                command=lambda idx=i: self.play_channel(idx), state=tk.DISABLED
            )
            btn.pack(pady=2, fill=tk.X, padx=50)
            self.play_buttons.append(btn)

    def change_monitor_channel(self, event):
        self.current_monitor_idx = self.monitor_combo.current()

    def audio_input_callback(self, indata, frames, time_info, status):
        if status: print(f"[!] Audio Status: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def network_send_task(self):
        buffer_44k = np.zeros((0, self.ZYLIA_CHANNELS_REAL), dtype=np.float32)
        
        while self.is_recording:
            try:
                # 1. Obtener 4800 muestras a 48kHz
                chunk_48k = self.audio_queue.get(timeout=0.5)
                
                # 2. Descartar el canal 20 vacío
                chunk_48k = chunk_48k[:, :self.ZYLIA_CHANNELS_REAL]

                # 3. Remuestreo a 44100Hz
                chunk_44k = sig.resample_poly(chunk_48k, up=441, down=480, axis=0).astype(np.float32)
                buffer_44k = np.concatenate((buffer_44k, chunk_44k), axis=0)

                # 4. Empaquetar y enviar bloques de 4096 exactos
                while buffer_44k.shape[0] >= self.HOP_SERVER:
                    chunk_to_send = buffer_44k[:self.HOP_SERVER, :]
                    buffer_44k = buffer_44k[self.HOP_SERVER:, :]
                    self.socket_send.send_pyobj(chunk_to_send)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[!] Error de envío: {e}")
                break

    def network_recv_task(self):
        while self.is_recording:
            try:
                # El servidor responde con el tensor [7, 4096]
                processed_chunk = self.socket_recv.recv_pyobj(flags=zmq.NOBLOCK)
                
                # 1. Guardar en buffers locales para el .wav final
                for i in range(self.NUM_SOURCES):
                    self.output_buffers[i].append(processed_chunk[i, :])
                
                # 2. Reproducción en vivo del canal seleccionado
                if self.out_stream and self.out_stream.active:
                    audio_directo = processed_chunk[self.current_monitor_idx, :]
                    audio_directo = audio_directo.reshape(-1, 1) # Forma que espera sounddevice
                    self.out_stream.write(audio_directo)
                    
            except zmq.Again:
                time.sleep(0.01) # No hay datos aún, esperamos sin bloquear
            except Exception as e:
                print(f"[!] Error de recepción: {e}")
                break

    def start_recording(self):
        device_str = self.device_var.get()
        if "No se detectan" in device_str:
            messagebox.showerror("Error", "Debes conectar el Zylia y configurar ASIO.")
            return

        ip_server = self.ip_var.get()
        device_id = int(device_str.split(":")[0])
        self.current_monitor_idx = self.monitor_combo.current()

        # Limpieza previa
        self.output_buffers = {i: [] for i in range(self.NUM_SOURCES)}
        self.final_audio = {}
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # Iniciar Red
        self.socket_send = self.context.socket(zmq.PUSH)
        self.socket_send.connect(f"tcp://{ip_server}:{self.PORT_SEND}")
        
        self.socket_recv = self.context.socket(zmq.PULL)
        self.socket_recv.connect(f"tcp://{ip_server}:{self.PORT_RECV}")

        self.is_recording = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        for btn in self.play_buttons: btn.config(state=tk.DISABLED)
        self.lbl_status.config(text="Estado: 🔴 Conectado a Servidor y Grabando...", foreground="red")

        # Iniciar Streams de Audio Hardware
        jitter_buffer_seconds = (self.HOP_SERVER * 4) / self.FS_SERVER
        
        self.out_stream = sd.OutputStream(
            samplerate=self.FS_SERVER, channels=1, blocksize=self.HOP_SERVER,
            dtype='float32', latency=jitter_buffer_seconds
        )
        self.out_stream.start()

        self.in_stream = sd.InputStream(
            device=device_id, samplerate=self.FS_HW, channels=self.ZYLIA_CHANNELS_ASIO,
            blocksize=self.BLOCK_HW, dtype='float32', callback=self.audio_input_callback
        )
        self.in_stream.start()

        # Lanzar Hilos Asíncronos
        self.send_thread = threading.Thread(target=self.network_send_task, daemon=True)
        self.recv_thread = threading.Thread(target=self.network_recv_task, daemon=True)
        self.send_thread.start()
        self.recv_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.lbl_status.config(text="Estado: ⏳ Cerrando red y guardando WAVs...", foreground="orange")
        self.root.update()

        # Apagar hardware
        if self.in_stream:
            self.in_stream.stop()
            self.in_stream.close()
        if self.out_stream:
            self.out_stream.stop()
            self.out_stream.close()

        # Enviar orden de apagado al servidor
        if self.socket_send:
            self.socket_send.send_pyobj("FIN")
            
        # Esperar a que terminen los hilos
        if self.send_thread: self.send_thread.join()
        if self.recv_thread: self.recv_thread.join()
        
        # Cerrar sockets
        if self.socket_send: self.socket_send.close()
        if self.socket_recv: self.socket_recv.close()

        # GUARDADO FÍSICO EN DISCO LOCAL
        archivos_guardados = 0
        for i in range(self.NUM_SOURCES):
            if self.output_buffers[i]:
                audio_completo = np.concatenate(self.output_buffers[i], axis=-1)
                self.final_audio[i] = audio_completo
                
                nombre_archivo = f"local_hdemucs_canal_{i}_pos_{i*50}deg.wav"
                wav.write(nombre_archivo, self.FS_SERVER, audio_completo)
                archivos_guardados += 1
                
        self.lbl_status.config(text=f"Estado: ✅ Listo. {archivos_guardados} archivos WAV locales guardados.", foreground="green")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        for btn in self.play_buttons: btn.config(state=tk.NORMAL)

    def play_channel(self, idx):
        if idx in self.final_audio:
            audio_data = self.final_audio[idx]
            sd.play(audio_data, samplerate=self.FS_SERVER)
        else:
            messagebox.showwarning("Aviso", "No hay audio procesado para este canal.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ZyliaNetworkClientGUI(root)
    
    def on_closing():
        if app.is_recording: app.stop_recording()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()