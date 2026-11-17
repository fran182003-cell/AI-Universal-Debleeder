import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import threading
import queue
import numpy as np
import torch as th
import yaml
import time
import gc
import scipy.signal as sig
import SH  # Tu librería de Beamformer

# Parche PyTorch
_original_load = th.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
th.load = _patched_load

from demucs.apply import apply_model
from demucs.hdemucs import HDemucs
from LightingModel2 import LightningModel

# =========================================================================
# FAST BEAMFORMER OPTIMIZADO (STATELESS)
# =========================================================================
class FastBeamformer:
    def __init__(self, sampling_points, ldirs_fijas, fs, order=3, r=49e-3, hop_len=4096, winhop=1024):
        self.fs = fs
        self.hop_len = hop_len
        self.winhophalf = winhop // 2
        self.win_len = hop_len + self.winhophalf # 4608

        Y_q = SH.SHMatrix(sampling_points, order=order)      
        self.Y_q_pinv = np.linalg.pinv(Y_q)                  
        self.Y_l = SH.SHMatrix(ldirs_fijas, order=order)     

        f = np.fft.rfftfreq(self.win_len, d=1/fs)
        b = SH.b_function(k=2*np.pi*f/343, r=r, ra=r, order=order)
        W, crossover_indices = SH.AmpModMatrix(f, b)
        D = SH.MaxDirectivityBeamformerCoeffMatrix(f, crossover_indices)
        self.WD = W @ D 

        hann = sig.windows.hann(winhop, sym=False).astype(np.float32)
        self.window = np.concatenate([
            hann[:self.winhophalf],
            np.ones(self.hop_len - self.winhophalf, dtype=np.float32),
            hann[-self.winhophalf:]
        ])

    def process(self, x: np.ndarray) -> np.ndarray:
        x_win = x * self.window
        x_fft = np.fft.rfft(x_win, n=self.win_len, axis=-1)
        p_nm = np.einsum('NQ,Qf->Nf', self.Y_q_pinv, x_fft, optimize=True)
        y_fft = np.einsum('vN,Nf,Nf->vf', self.Y_l, p_nm, self.WD, optimize=True)
        y_out = np.fft.irfft(y_fft, n=self.win_len, axis=-1)
        return y_out 

# =========================================================================
# CLASE EN TIEMPO REAL CON OLA
# =========================================================================
class RealTimeModel:
    def __init__(self, model: th.nn.Module, segment: int = 45056, hop: int = 4096, device: str = 'cuda' if th.cuda.is_available() else 'cpu'):
        self.model = model
        self.model.to(device)
        self.model.eval()

        self.fs = self.model.samplerate
        self._first_pass = True
        channels = self.model.audio_channels
        nsources = len(self.model.sources)

        self.device = device
        self.hop = hop
        self.winhop = round(self.model.nfft / 4)
        self.winhophalf = round(self.winhop / 2)

        hann = th.hann_window(self.winhop, periodic=True, device=self.device)
        hann = th.cat([hann[:self.winhophalf],
                       th.ones(hop - self.winhophalf, device=self.device),
                       hann[-self.winhophalf:]])
        
        self.buffer = th.zeros(1, channels, segment, device=self.device)
        self.in_ola_buffer = th.zeros(1, channels, self.winhophalf, device=self.device)

        with th.inference_mode():
            dummy_in = th.zeros(1, channels, segment, device=device)
            dummy_out = apply_model(self.model, dummy_in, shifts=0, split=False, device=device)
        
        self.out_dim = dummy_out.dim()
        
        if self.out_dim == 3:
            self.win = hann.unsqueeze(0).unsqueeze(1).expand(1, nsources, -1)
            self.overlapadd = th.zeros(1, nsources, self.winhophalf, device=self.device)
        else:
            self.win = hann.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(1, nsources, channels, -1)
            self.overlapadd = th.zeros(1, nsources, channels, self.winhophalf, device=self.device)

    def separate(self, frame_win: th.Tensor):
        valid_hop = frame_win[:, :, :self.hop].clone()
        valid_hop[:, :, :self.winhophalf] += self.in_ola_buffer
        self.in_ola_buffer = frame_win[:, :, -self.winhophalf:].clone()

        self.buffer[:, :, :-self.hop] = self.buffer[:, :, self.hop:]
        self.buffer[:, :, -self.hop:] = valid_hop
        
        output = apply_model(self.model, self.buffer, shifts=0, split=False, device=self.device)
        self._first_pass = False 

        if self.out_dim == 3:
            output[:, :, -self.hop - self.winhophalf:] *= self.win
            output[:, :, -self.hop - self.winhophalf:-self.hop] += self.overlapadd
            self.overlapadd = output[:, :, -self.winhophalf:]
            return output[:, :, -self.hop - self.winhophalf:-self.winhophalf]
        else:
            output[:, :, :, -self.hop - self.winhophalf:] *= self.win
            output[:, :, :, -self.hop - self.winhophalf:-self.hop] += self.overlapadd
            self.overlapadd = output[:, :, :, -self.winhophalf:]
            return output[:, :, :, -self.hop - self.winhophalf:-self.winhophalf]

# =========================================================================
# INTERFAZ GRÁFICA ZYLIA DÚPLEX
# =========================================================================
class RealTimeSeparationGUI:
    def __init__(self, root, conf_path, ckpt_path):
        self.root = root
        self.root.title("Zylia Live Dúplex: Monitorización en Vivo")
        self.root.geometry("480x650")
        
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'
        th.backends.cudnn.deterministic = False
        th.backends.cudnn.benchmark = True 
        
        self.context_sec = 1.0
        self.hop = 4096
        
        self.zylia_channels = 19 
        self.num_sources = 7 
        self.source_names = [f"Posición {i*50}º" for i in range(self.num_sources)]
        
        mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
        mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
        self.sampling_points = np.column_stack((mic_el, mic_az))
        
        elevacion = np.pi / 2
        paso_rad = np.deg2rad(50)
        self.fixed_dirs = np.array([[elevacion, i * paso_rad] for i in range(self.num_sources)])

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_simulation = False
        self.in_stream = None
        self.out_stream = None
        self.inference_thread = None
        self.dummy_thread = None
        
        self.current_monitor_idx = 0 
        
        self.load_model(conf_path, ckpt_path)
        
        # Buffer de entrada para el beamformer (4608 muestras)
        self.beamformer_win_len = self.hop + self.model_rt.winhophalf
        self.mic_buffer = np.zeros((self.zylia_channels, self.beamformer_win_len), dtype=np.float32)

        self.output_buffers = {i: [] for i in range(self.num_sources)}
        self.final_audio = {} 

        self.setup_ui()

    def load_model(self, conf_path, ckpt_path):
        import yaml
        print("Cargando y optimizando modelo...")
        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)

        self.fs = conf['hdemucs_conf']['samplerate']
        
        # 1. LEER DIMENSIONES REALES DEL ENTRENAMIENTO
        sources_entrenamiento = conf['hdemucs_conf']['sources']
        audio_channels_entrenamiento = conf['hdemucs_conf']['audio_channels']
        
        confdemucs = {
            'sources': sources_entrenamiento, 
            'audio_channels': audio_channels_entrenamiento,
            'cac': conf['hdemucs_conf']['cac'],
            'samplerate': self.fs,
            'channels': conf['hdemucs_conf']['channels'],
            'segment': conf['dataset']['chunk_duration']
        }

        # 2. CARGAR EL MODELO BASE Y LOS PESOS
        modelo_base = HDemucs(**confdemucs)
        lightning_model = LightningModel.load_from_checkpoint(ckpt_path, model=modelo_base, strict=False)
        
        context_samples = int(self.context_sec * self.fs)
        self.model_rt = RealTimeModel(
            model=lightning_model.model, 
            segment=context_samples, 
            hop=self.hop, 
            device=self.device
        )
        
        # 3. INICIALIZAR EL BEAMFORMER (Apuntando a las 7 direcciones fijas)
        # Asegúrate de tener self.sampling_points y self.fixed_dirs definidos en el __init__
        self.beamformer = FastBeamformer(
            self.sampling_points, self.fixed_dirs,
            self.fs, order=3, r=49e-3, hop_len=self.hop, winhop=self.model_rt.winhop
        )
        
        print("Realizando Warm-Up de la GPU...")
        with torch.inference_mode():
            # El dummy chunk ahora usa los canales exactos que espera el modelo (7 canales)
            dummy_chunk = torch.zeros(1, audio_channels_entrenamiento, self.hop + self.model_rt.winhophalf, device=self.device)
            for _ in range(5):
                with torch.autocast(device_type=self.device if self.device != 'cpu' else 'cpu'):
                    _ = self.model_rt.separate(dummy_chunk)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        self.model_rt.overlapadd = torch.zeros_like(self.model_rt.overlapadd)
        self.model_rt.in_ola_buffer = torch.zeros_like(self.model_rt.in_ola_buffer)
        self.model_rt.buffer = torch.zeros_like(self.model_rt.buffer)
        self.model_rt._first_pass = True
        print("Modelo cargado y listo.")

    def setup_ui(self):
        ttk.Label(self.root, text="Dispositivo de Entrada (Zylia):", font=("Arial", 10)).pack(pady=(10, 0))
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(self.root, textvariable=self.device_var, state="readonly", width=50)
        
        devices = sd.query_devices()
        input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] >= self.zylia_channels]
        
        # MODO SIMULACIÓN SI NO HAY ZYLIA
        if not input_devices: 
            input_devices = ["SIMULACIÓN: Generador interno (Sin Zylia)"]
        else:
            input_devices.append("SIMULACIÓN: Generador interno (Sin Zylia)")
            
        self.device_combo['values'] = input_devices
        self.device_combo.current(0)
        self.device_combo.pack(pady=5)

        ttk.Label(self.root, text="🎛️ Canal a monitorizar EN DIRECTO:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        self.monitor_var = tk.StringVar()
        self.monitor_combo = ttk.Combobox(self.root, textvariable=self.monitor_var, state="readonly", width=30)
        self.monitor_combo['values'] = [f"Canal {i}: {self.source_names[i]}" for i in range(self.num_sources)]
        self.monitor_combo.current(0)
        self.monitor_combo.pack(pady=5)
        self.monitor_combo.bind("<<ComboboxSelected>>", self.change_monitor_channel)

        self.lbl_status = ttk.Label(self.root, text="Estado: Listo", font=("Arial", 12, "bold"))
        self.lbl_status.pack(pady=15)

        self.btn_start = ttk.Button(self.root, text="🔴 Iniciar Captura y Escucha", command=self.start_recording)
        self.btn_start.pack(pady=5, fill=tk.X, padx=50)

        self.btn_stop = ttk.Button(self.root, text="⏹ Detener", command=self.stop_recording, state=tk.DISABLED)
        self.btn_stop.pack(pady=5, fill=tk.X, padx=50)

        ttk.Separator(self.root, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(self.root, text="Reproducir Salidas Completas (Post-Grabación):", font=("Arial", 10, "bold")).pack()

        self.play_buttons = []
        for i in range(self.num_sources):
            btn = ttk.Button(
                self.root, 
                text=f"▶ Escuchar Completo: {self.source_names[i]}", 
                command=lambda idx=i: self.play_channel(idx),
                state=tk.DISABLED
            )
            btn.pack(pady=3, fill=tk.X, padx=50)
            self.play_buttons.append(btn)

    def change_monitor_channel(self, event):
        self.current_monitor_idx = self.monitor_combo.current()

    def audio_input_callback(self, indata, frames, time, status):
        if status: print(f"Input status: {status}")
        self.audio_queue.put(indata.copy())

    def dummy_audio_generator(self):
        """ Simula la entrada de un micrófono generando bloques de silencio (o ruido) """
        sleep_time = self.hop / self.fs
        while self.is_recording:
            # Generamos silencio (puedes cambiarlo por ruido np.random.randn si quieres probar señal)
            dummy_chunk = np.zeros((self.hop, self.zylia_channels), dtype=np.float32)
            self.audio_queue.put(dummy_chunk)
            time.sleep(sleep_time)

    def process_queue(self):
        self.output_buffers = {i: [] for i in range(self.num_sources)}
        prefill_blocks = 4  
        prefill_done = False
        
        gc.disable()
        
        with th.inference_mode():
            while self.is_recording or not self.audio_queue.empty():
                
                if not prefill_done and self.is_recording:
                    if self.audio_queue.qsize() < prefill_blocks:
                        time.sleep(0.005)
                        continue
                    else:
                        prefill_done = True

                try:
                    chunk_np = self.audio_queue.get(timeout=0.5) 
                except queue.Empty:
                    continue

                if chunk_np.shape[0] != self.hop:
                    if chunk_np.shape[0] < self.hop:
                        pad = np.zeros((self.hop - chunk_np.shape[0], self.zylia_channels))
                        chunk_np = np.concatenate((chunk_np, pad), axis=0)
                    else:
                        chunk_np = chunk_np[:self.hop, :]
                
                chunk_t = chunk_np.T 
                
                # --- NUEVA GESTIÓN DE BUFFER (4608 Muestras) ---
                self.mic_buffer[:, :-self.hop] = self.mic_buffer[:, self.hop:]
                self.mic_buffer[:, -self.hop:] = chunk_t

                # Entran 4608, salen 4608
                chunk_in_np = self.beamformer.process(self.mic_buffer) 
                chunk_th = th.from_numpy(chunk_in_np).float().unsqueeze(0).to(self.device)

                with th.autocast(device_type=self.device if self.device != 'cpu' else 'cpu'):
                    out = self.model_rt.separate(chunk_th) # Retorna 4096
                
                out_cpu = out[0].cpu().numpy() 
                
                for i in range(self.num_sources):
                    self.output_buffers[i].append(out_cpu[i, :])

                audio_para_altavoz = out_cpu[self.current_monitor_idx, :]
                audio_para_altavoz = np.expand_dims(audio_para_altavoz, axis=1) 
                
                if self.out_stream and self.is_recording and not self.is_simulation:
                    self.out_stream.write(audio_para_altavoz.astype(np.float32))

        gc.enable()

    def start_recording(self):
        device_str = self.device_var.get()
        self.current_monitor_idx = self.monitor_combo.current()

        self.is_recording = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        for btn in self.play_buttons: btn.config(state=tk.DISABLED)
        self.lbl_status.config(text="Estado: 🔴 Grabando y Escuchando...", foreground="red")

        jitter_buffer_seconds = (self.hop * 4) / self.fs
        
        if "SIMULACIÓN" in device_str:
            self.is_simulation = True
            print("Iniciando en MODO SIMULACIÓN. No se usará audio de entrada real ni salida en vivo.")
            self.dummy_thread = threading.Thread(target=self.dummy_audio_generator)
            self.dummy_thread.start()
        else:
            self.is_simulation = False
            device_id = int(device_str.split(":")[0])
            
            # Solo abrimos los streams físicos si hay hardware real
            self.out_stream = sd.OutputStream(
                samplerate=self.fs,
                channels=1, 
                blocksize=self.hop,
                dtype='float32',
                latency=jitter_buffer_seconds
            )
            self.out_stream.start()

            self.in_stream = sd.InputStream(
                device=device_id,
                samplerate=self.fs, 
                channels=self.zylia_channels, 
                blocksize=self.hop,
                latency=jitter_buffer_seconds,
                callback=self.audio_input_callback
            )
            self.in_stream.start()

        self.inference_thread = threading.Thread(target=self.process_queue)
        self.inference_thread.start()

    def stop_recording(self):
        self.is_recording = False
        
        if self.in_stream:
            self.in_stream.stop()
            self.in_stream.close()
            
        if self.dummy_thread:
            self.dummy_thread.join()
        
        self.lbl_status.config(text="Estado: Procesando restos...", foreground="orange")
        self.root.update()
        
        if self.inference_thread:
            self.inference_thread.join()

        if self.out_stream:
            self.out_stream.stop()
            self.out_stream.close()

        for i in range(self.num_sources):
            if self.output_buffers[i]:
                self.final_audio[i] = np.concatenate(self.output_buffers[i], axis=-1)
        
        self.lbl_status.config(text="Estado: Listo. Audio guardado.", foreground="green")
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        for btn in self.play_buttons: btn.config(state=tk.NORMAL)

    def play_channel(self, idx):
        if idx in self.final_audio:
            audio_data = self.final_audio[idx]
            # Usamos el dispositivo de salida por defecto de tu SO
            sd.play(audio_data, samplerate=self.fs)
        else:
            messagebox.showwarning("Aviso", "No hay audio guardado.")

if __name__ == "__main__":
    CONF_PATH = DEFAULT_CONF
    CKPT_PATH = os.environ.get("CKPT_PATH") or sys.exit("Define la variable de entorno CKPT_PATH con la ruta al checkpoint .ckpt")
    
    root = tk.Tk()
    app = RealTimeSeparationGUI(root, CONF_PATH, CKPT_PATH)
    
    def on_closing():
        if app.is_recording: app.stop_recording()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()