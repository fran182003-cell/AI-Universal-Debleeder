import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import zmq
import numpy as np
import torch as th
import time
import os
import yaml
import scipy.signal as sig
import sys

import SH

# Parche PyTorch
_original_load = th.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
th.load = _patched_load

from demucs.apply import apply_model
from demucs.hdemucs import HDemucs
from LightingModel2 import LightningModel

class FastBeamformer:
    def __init__(self, sampling_points, ldirs_fijas, fs, order=3, r=49e-3, hop_len=4096, winhop=1024):
        self.fs = fs
        self.hop_len = hop_len
        self.winhophalf = winhop // 2
        self.win_len = hop_len + self.winhophalf 

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
        x_win = x_win - np.mean(x_win, axis=-1, keepdims=True)
        x_fft = np.fft.rfft(x_win, n=self.win_len, axis=-1)
        x_fft[:, :6] = 0.0 + 0.0j
        p_nm = np.einsum('NQ,Qf->Nf', self.Y_q_pinv, x_fft, optimize=True)
        y_fft = np.einsum('vN,Nf,Nf->vf', self.Y_l, p_nm, self.WD, optimize=True)
        return np.fft.irfft(y_fft, n=self.win_len, axis=-1)

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
        hann = th.cat([hann[:self.winhophalf], th.ones(hop - self.winhophalf, device=self.device), hann[-self.winhophalf:]])
        
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
        if frame_win.shape[-1] != self.hop + self.winhophalf:
            raise ValueError(f"El frame de entrada debe ser de {self.hop + self.winhophalf} muestras.")

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

def main():
    CONF_PATH = DEFAULT_CONF
    CKPT_PATH = os.environ.get("CKPT_PATH") or sys.exit("Define la variable de entorno CKPT_PATH con la ruta al checkpoint .ckpt")  # modelo inicial (version_12)
    CONTEXT_SEC = 1.0
    HOP = 4096
    C = 7
    PORT_IN = int(os.getenv("AUDIO_INPUT_PORT", "5555"))
    # Por defecto solo escucha en local: el cliente llega mediante un túnel SSH.
    # Usa AUDIO_INPUT_HOST=0.0.0.0 solo en una red de confianza (los mensajes usan pickle).
    AUDIO_INPUT_HOST = os.getenv("AUDIO_INPUT_HOST", "127.0.0.1")
    PORT_OUT = int(os.getenv("WEB_STREAM_PORT", "7777"))
    WEB_STREAM_HOST = os.getenv("WEB_STREAM_HOST", "127.0.0.1")
    DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

    context = zmq.Context()

    socket_in = context.socket(zmq.PULL)
    socket_in.setsockopt(zmq.LINGER, 0)
    socket_in.bind(f"tcp://{AUDIO_INPUT_HOST}:{PORT_IN}")

    # El servicio web mantiene el PULL enlazado. Este proceso se conecta a él,
    # de modo que el servicio puede permanecer activo entre trabajos de GPU.
    socket_out = context.socket(zmq.PUSH)
    socket_out.setsockopt(zmq.SNDHWM, 8)
    socket_out.setsockopt(zmq.LINGER, 0)
    socket_out.connect(f"tcp://{WEB_STREAM_HOST}:{PORT_OUT}")

    print(
        f"[*] Servidor escuchando audio en {PORT_IN} y enviando a la web "
        f"en {WEB_STREAM_HOST}:{PORT_OUT} (HW: {DEVICE.upper()})"
    )

    with open(CONF_PATH, 'r') as f:
        conf = yaml.safe_load(f)
    sample_rate = conf['hdemucs_conf']['samplerate']
    
    confdemucs = {
        'sources': [str(i) for i in range(C)],
        'audio_channels': C,
        'cac': conf['hdemucs_conf']['cac'],
        'samplerate': sample_rate,
        'channels': conf['hdemucs_conf']['channels'],
        'segment': conf['dataset']['chunk_duration']
    }
    
    modelo_base = HDemucs(**confdemucs)
    lightning_model = LightningModel.load_from_checkpoint(CKPT_PATH, model=modelo_base, strict=False)
    modelo_puro = lightning_model.model

    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))

    elevacion_base = 1.5707963267949  
    direcciones_activas = [[elevacion_base, np.deg2rad(i*50)] for i in range(3)]
    todas_direcciones = list(direcciones_activas)
    faltantes = C - len(direcciones_activas)
    
    if faltantes > 0:
        paso_rad = np.deg2rad(50)
        current_az = direcciones_activas[-1][1] 
        for _ in range(faltantes):
            current_az = (current_az + paso_rad) % (2 * np.pi)
            todas_direcciones.append([elevacion_base, current_az])

    direcciones_fijas = np.array(todas_direcciones)
    winhop = int(modelo_puro.nfft / 4)
    beamformer = FastBeamformer(sampling_points, direcciones_fijas, sample_rate, order=3, r=49e-3, hop_len=HOP, winhop=winhop)
    
    context_samples = int(CONTEXT_SEC * sample_rate)
    modelo_rt = RealTimeModel(model=modelo_puro, segment=context_samples, hop=HOP, device=DEVICE)

    bloques_procesados = 0
    mic_buffer = np.zeros((19, beamformer.win_len), dtype=np.float32)
    print("\n[*] Esperando trama de audio del Cliente...")

    try:
        while True:
            chunk_raw = socket_in.recv_pyobj() 
            
            if isinstance(chunk_raw, str) and chunk_raw == "FIN":
                print("\n[!] Fin de transmisión del cliente. Reseteando buffers y esperando nuevo audio...")
                mic_buffer.fill(0)  # Esto es NumPy, se mantiene igual
                
                # Sustitución de tensores PyTorch sin in-place update
                modelo_rt.overlapadd = th.zeros_like(modelo_rt.overlapadd)
                modelo_rt.in_ola_buffer = th.zeros_like(modelo_rt.in_ola_buffer)
                modelo_rt.buffer = th.zeros_like(modelo_rt.buffer)
                
                modelo_rt._first_pass = True
                bloques_procesados = 0
                continue

            chunk_raw = chunk_raw.T  
            mic_buffer[:, :-HOP] = mic_buffer[:, HOP:]
            mic_buffer[:, -HOP:] = chunk_raw

            chunk_in_np = beamformer.process(mic_buffer)
            
            chunk_in = th.from_numpy(chunk_in_np).float().unsqueeze(0).to(DEVICE)
            with th.inference_mode():
                with th.autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu'):
                    chunk_out = modelo_rt.separate(chunk_in)
            
            if chunk_out.dim() == 3:
                chunk_out_np = chunk_out.squeeze(0).cpu().numpy()
            else:
                chunk_out_np = chunk_out.cpu().numpy()
                
            try:
                socket_out.send_pyobj(chunk_out_np, flags=zmq.NOBLOCK)
            except zmq.Again:
                # La inferencia no debe bloquearse si el servicio web se reinicia.
                if bloques_procesados % 50 == 0:
                    print("[!] Servicio web no disponible; bloque de salida descartado.")

            bloques_procesados += 1

            if bloques_procesados % 10 == 0:
                print(f" -> Procesados {bloques_procesados} bloques de audio...", end='\r')

    except KeyboardInterrupt:
        print("\n[!] Recepción detenida manualmente.")
    finally:
        socket_in.close()
        socket_out.close()
        context.term()
        print("[+] Servidor finalizado.")

if __name__ == "__main__":
    main()