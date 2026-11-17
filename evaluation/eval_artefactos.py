import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import argparse
import typing as tp
import torch as th
import yaml
import fast_bss_eval
import numpy as np
from tqdm import tqdm
import torchaudio
import SH  # Beamformer
import time
import gc  # Para controlar el recolector de basura de Python
import scipy.signal as sig # Necesario para generar la ventana
import os

# Parche PyTorch
_original_load = th.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
th.load = _patched_load

from demucs.apply import apply_model
from demucs.hdemucs import HDemucs

# --- IMPORTACIONES ACTUALIZADAS A TU VERSIÓN 2 ---
from LightingModel2 import LightningModel

# =========================================================================
# FAST BEAMFORMER OPTIMIZADO (STATELESS)
# =========================================================================
class FastBeamformer:
    """
    Versión optimizada SIN ESTADO INTERNO.
    Aplica una ventana idéntica a la de salida de HDemucs:
    (Fade-in de 512) + (Plano de 3584) + (Fade-out de 512) = 4608 muestras.
    El OLA lo gestiona el consumidor posterior (RealTimeModel).
    """
    def __init__(self, sampling_points, ldirs_fijas, fs, order=3, r=49e-3, hop_len=4096, winhop=1024):
        self.fs = fs
        self.hop_len = hop_len
        self.winhophalf = winhop // 2
        # El bloque debe ser el salto (4096) + la cola de solapamiento (512)
        self.win_len = hop_len + self.winhophalf # 4608

        # ---- Todo esto es constante: se calcula UNA sola vez ----
        Y_q = SH.SHMatrix(sampling_points, order=order)      
        self.Y_q_pinv = np.linalg.pinv(Y_q)                  
        self.Y_l = SH.SHMatrix(ldirs_fijas, order=order)     

        f = np.fft.rfftfreq(self.win_len, d=1/fs)
        b = SH.b_function(k=2*np.pi*f/343, r=r, ra=r, order=order)
        W, crossover_indices = SH.AmpModMatrix(f, b)
        D = SH.MaxDirectivityBeamformerCoeffMatrix(f, crossover_indices)
        
        self.WD = W @ D 

        # --- VENTANA IDÉNTICA A HDEMUCS (Flat-top con bordes de Hann) ---
        hann = sig.windows.hann(winhop, sym=False).astype(np.float32)
        self.window = np.concatenate([
            hann[:self.winhophalf],
            np.ones(self.hop_len - self.winhophalf, dtype=np.float32),
            hann[-self.winhophalf:]
        ])

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        x: (Q, win_len) señales de micrófono de entrada (4608 muestras).
        return: (v, win_len) señales beamformadas listas para OLA.
        """
        # 1. Aplicar ventana
        x_win = x * self.window

        # 2. FFT
        x_fft = np.fft.rfft(x_win, n=self.win_len, axis=-1)
        
        # 3. Procesamiento SH (Beamforming)
        p_nm = np.einsum('NQ,Qf->Nf', self.Y_q_pinv, x_fft, optimize=True)
        y_fft = np.einsum('vN,Nf,Nf->vf', self.Y_l, p_nm, self.WD, optimize=True)
        
        # 4. IFFT 
        y_out = np.fft.irfft(y_fft, n=self.win_len, axis=-1)
        
        return y_out # Devuelve el bloque completo (4608)

# =========================================================================
# CLASE EN TIEMPO REAL (Soporte dinámico 3D/4D y OLA Nativo de Entrada y Salida)
# =========================================================================
class RealTimeModel:
    def __init__(self, model: th.nn.Module,
                 segment: int = 45056,
                 hop: int = 4096,
                 device: str = 'cuda' if th.cuda.is_available() else 'cpu'):
        
        self.model = model
        self.model.to(device)
        self.model.eval()

        fs = self.model.samplerate
        self.fs = fs
        self._first_pass = True
        channels = self.model.audio_channels
        nsources = len(self.model.sources)

        self.device = device
        self.hop = hop
        segmentsamp = segment
        self.winhop = round(self.model.nfft / 4)
        self.winhophalf = round(self.winhop / 2)

        # Hanning window base para salida
        hann = th.hann_window(self.winhop, periodic=True, device=self.device)
        hann = th.cat([hann[:self.winhophalf],
                       th.ones(hop - self.winhophalf, device=self.device),
                       hann[-self.winhophalf:]])
        
        # Buffer principal de Demucs
        self.buffer = th.zeros(1, channels, segmentsamp, device=self.device)

        # --- NUEVO: Buffer para el OLA de ENTRADA (cola de 512 muestras) ---
        self.in_ola_buffer = th.zeros(1, channels, self.winhophalf, device=self.device)

        with th.inference_mode():
            dummy_in = th.zeros(1, channels, segmentsamp, device=device)
            dummy_out = apply_model(self.model, dummy_in, shifts=0, split=False, device=device)
        
        self.out_dim = dummy_out.dim()
        
        if self.out_dim == 3:
            self.win = hann.unsqueeze(0).unsqueeze(1).expand(1, nsources, -1)
            self.overlapadd = th.zeros(1, nsources, self.winhophalf, device=self.device)
        else:
            self.win = hann.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(1, nsources, channels, -1)
            self.overlapadd = th.zeros(1, nsources, channels, self.winhophalf, device=self.device)

    def separate(self, frame_win: th.Tensor):
        """
        frame_win: Tensor de tamaño (1, canales, 4608) proveniente del Beamformer.
        """
        if frame_win.shape[-1] != self.hop + self.winhophalf:
            raise ValueError(f"El frame de entrada debe ser de {self.hop + self.winhophalf} muestras.")

        # --- 1. OVERLAP-ADD DE ENTRADA ---
        # Extraemos las 4096 muestras válidas (hop)
        valid_hop = frame_win[:, :, :self.hop].clone()
        
        # Sumamos la cola (fade-out) del bloque anterior a la cabeza (fade-in) del bloque actual
        valid_hop[:, :, :self.winhophalf] += self.in_ola_buffer
        
        # Guardamos la cola del bloque actual (fade-out) para el futuro
        self.in_ola_buffer = frame_win[:, :, -self.winhophalf:].clone()

        # --- 2. ALIMENTAR HDEMUCS ---
        # Desplazamos el buffer de contexto y metemos el salto limpio y reconstruido
        self.buffer[:, :, :-self.hop] = self.buffer[:, :, self.hop:]
        self.buffer[:, :, -self.hop:] = valid_hop
        
        if self._first_pass:
            samples_in = self.buffer.shape[-1]
            secs_in = samples_in / self.fs
            print(f"\n[DEBUG RT] 📥 TENSOR DE ENTRADA AL MODELO:")
            print(f"           - Shape de bloque tras OLA de entrada: {valid_hop.shape}")
            print(f"           - Shape buffer total: {self.buffer.shape}")
            print(f"           - Contexto aplicado: {samples_in} muestras ({secs_in:.2f} s)")

        output = apply_model(self.model, self.buffer, shifts=0, split=False, device=self.device)
        
        if self._first_pass:
            samples_out = output.shape[-1]
            secs_out = samples_out / self.fs
            print(f"[DEBUG RT] 📤 TENSOR DE SALIDA DEL MODELO:")
            print(f"           - Shape: {output.shape}")
            print(f"           - Generado: {samples_out} muestras ({secs_out:.2f} s)\n")
            self._first_pass = False 

        # --- 3. OVERLAP-ADD DE SALIDA (Propio de Demucs) ---
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
# LÓGICA DE EVALUACIÓN
# =========================================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Evaluación SDR con RealTimeModel (V2)")
    parser.add_argument("--pt_file", type=str, required=True, help="Ruta al archivo .pt precargado")
    parser.add_argument("--conf", type=str, default=DEFAULT_CONF)
    parser.add_argument("--ckpt", type=str, required=True, help="Ruta al checkpoint .ckpt")
    parser.add_argument("-c", "--context", type=float, default=1.0, help="Ventana temporal en segundos")
    parser.add_argument("--hop", type=int, default=4096, help="Tamaño de salto (muestras)")
    parser.add_argument("-d", "--device", default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--cache", type=str, default="mezcla_precargada.pt", help="Ruta para guardar/cargar la mezcla")
    return parser

def main():
    def set_seed(seed=42):
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        th.backends.cudnn.deterministic = False 
        th.backends.cudnn.benchmark = True 

    set_seed(42)
    args = get_parser().parse_args()
    device = th.device(args.device)

    with open(args.conf, 'r') as f:
        conf = yaml.safe_load(f)

    sample_rate = conf['hdemucs_conf']['samplerate']
    
    C = 7
    confdemucs = {
        'sources': [str(i) for i in range(C)],
        'audio_channels': C,
        'cac': conf['hdemucs_conf']['cac'],
        'samplerate': sample_rate,
        'channels': conf['hdemucs_conf']['channels'],
        'segment': conf['dataset']['chunk_duration']
    }

    modelo_base = HDemucs(**confdemucs)
    lightning_model = LightningModel.load_from_checkpoint(args.ckpt, model=modelo_base, strict=False)
    modelo_puro = lightning_model.model

    idx_vocals = confdemucs['sources'].index('vocals') if 'vocals' in confdemucs['sources'] else 0

    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))

    print(f"\n[*] Cargando mezcla y referencias directas desde disco: {args.pt_file}")
    if not os.path.exists(args.pt_file):
        raise FileNotFoundError(f"No se encontró el archivo {args.pt_file}.")
        
    datos_cache = th.load(args.pt_file)
    mezcla_cruda_np = datos_cache['mezcla_cruda_np']
    referencias = datos_cache['referencias']
    if referencias.dim() == 2:
        referencias = referencias.unsqueeze(0)
    referencias = referencias.to(device)
    active_dirs = datos_cache['active_dirs']
    nsamples = mezcla_cruda_np.shape[-1]
    
    todas_direcciones = []
    for ldir in active_dirs:
        todas_direcciones.append(ldir)

    faltantes = C - len(active_dirs)
    if faltantes > 0:
        elevacion_base = np.pi / 2
        paso_rad = np.deg2rad(50)
        usados_az = [d[1] for d in active_dirs]
        current_az = usados_az[-1] if len(usados_az) > 0 else 0.0
        libres_az = []
        
        while len(libres_az) < faltantes:
            current_az = (current_az + paso_rad) % (2 * np.pi)
            is_used = any(np.isclose(current_az, u, atol=1e-2) for u in usados_az)
            is_in_libres = any(np.isclose(current_az, l, atol=1e-2) for l in libres_az)
            if not is_used and not is_in_libres:
                libres_az.append(current_az)
                
        for az_extra in libres_az:
            todas_direcciones.append([elevacion_base, az_extra])

    context_samples = int(args.context * sample_rate)
    modelo_rt = RealTimeModel(model=modelo_puro, segment=context_samples, hop=args.hop, device=args.device)

    # Calculamos la ventana exacta basada en el HDemucs puro
    hop_len = args.hop
    winhop = int(modelo_puro.nfft / 4)
    beamformer_win_len = hop_len + (winhop // 2)

    beamformer = FastBeamformer(
        sampling_points, np.array(todas_direcciones),
        sample_rate, order=3, r=49e-3, hop_len=hop_len, winhop=winhop
    )

    print("\n=> Warm-up de cuDNN/autocast...")
    with th.inference_mode():
        dummy_chunk = th.zeros(1, C, beamformer_win_len, device=device)
        for _ in range(5):
            with th.autocast(device_type=device.type):
                _ = modelo_rt.separate(dummy_chunk)
                
    th.cuda.synchronize(device)
    print("=> Warm-up completo.\n")

    modelo_rt.overlapadd = th.zeros_like(modelo_rt.overlapadd)
    modelo_rt.in_ola_buffer = th.zeros_like(modelo_rt.in_ola_buffer)
    modelo_rt.buffer = th.zeros_like(modelo_rt.buffer)
    modelo_rt._first_pass = True

    buffer_salida = []
    latencias_ms = []
    index = 0

    print(f"\n=> Procesando {nsamples/sample_rate:.1f}s | Contexto: {args.context}s | Hop: {args.hop}")
    print("=> ATENCIÓN: OLA de entrada alineado con ventana de HDemucs (Latencia reducida a 11.6ms)")

    gc.disable()

    with th.inference_mode():
        with tqdm(total=nsamples/sample_rate, unit='s', desc="Inferencia RT") as pbar:
            # Avanzamos de hop en hop (4096)
            while nsamples > index + hop_len:
                
                th.cuda.synchronize(device)
                t_inicio = time.perf_counter()

                # Extraemos el bloque asimétrico (4608 muestras)
                end_idx = min(index + beamformer_win_len, nsamples)
                chunk_crudo_np = mezcla_cruda_np[:, index:end_idx].numpy()

                # Padding de seguridad al final
                if chunk_crudo_np.shape[1] < beamformer_win_len:
                    pad = np.zeros((chunk_crudo_np.shape[0], beamformer_win_len - chunk_crudo_np.shape[1]), dtype=np.float32)
                    chunk_crudo_np = np.concatenate([chunk_crudo_np, pad], axis=-1)

                # Beamformer puro (Entran 4608, salen 4608 ya separadas)
                chunk_in_np = beamformer.process(chunk_crudo_np)

                chunk_in = th.from_numpy(chunk_in_np).float().unsqueeze(0).to(device)

                with th.autocast(device_type=device.type):
                    chunk_out = modelo_rt.separate(chunk_in)
                
                th.cuda.synchronize(device)
                t_fin = time.perf_counter()

                latencia_hardware_ms = (t_fin - t_inicio) * 1000
                latencias_ms.append(latencia_hardware_ms)

                buffer_salida.append(chunk_out.cpu())
                
                index += hop_len
                pbar.update(hop_len / sample_rate)

    gc.enable()

    # --- SIMULACIÓN Y EVALUACIÓN ---
    latencias_np = np.array(latencias_ms)
    presupuesto_ms = (args.hop / sample_rate) * 1000
    
    print("\n" + "="*50)
    print(" REPORTE DE LATENCIA Y JITTER BUFFER")
    print("="*50)
    print(f" Media de procesamiento        : {latencias_np.mean():.2f} ms")
    print(f" Máximo pico registrado (P100) : {latencias_np.max():.2f} ms")

    audio_reconstruido = th.cat(buffer_salida, dim=-1).to(device)

    # El retraso del algoritmo se ha reducido dramáticamente. 
    # HDemucs introduce un retraso algorítmico interno idéntico a su tamaño de overlap de salida.
    offset_alineacion = winhop // 2
    len_eval = min(audio_reconstruido.shape[-1] - offset_alineacion, referencias.shape[-1])

    est_vocals = audio_reconstruido[:, idx_vocals, offset_alineacion : offset_alineacion + len_eval].unsqueeze(1)
    ref_vocals = referencias[:, idx_vocals, :len_eval].unsqueeze(1)

    start_eval_time = 5.0
    end_eval_time = 55.0

    start_sample = int(start_eval_time * sample_rate)
    end_sample = int(end_eval_time * sample_rate)

    est_vocals_valido = est_vocals[:, :, start_sample:end_sample]
    ref_vocals_valido = ref_vocals[:, :, start_sample:end_sample]

    audio_guardar = est_vocals.squeeze(0).cpu()
    
    nombre_salida = f"estimacion_voces_fft_ola_input_ctx_{args.context}s.wav"
    torchaudio.save(nombre_salida, audio_guardar, sample_rate)
    print(f"\n[+] Audio estimado guardado en: {nombre_salida}")

    try:
        sdr, _, _ = fast_bss_eval.bss_eval_sources(ref_vocals_valido, est_vocals_valido, compute_permutation=False)
        sdr_val = round(sdr.mean().item(), 3)
        print("\n" + "="*50)
        print(f" MÉTRICA FINAL SDR (Contexto: {args.context}s) ->  {sdr_val} dB")
        print("="*50 + "\n")
    except Exception as e:
        print(f"Error SDR: {e}")

if __name__ == "__main__":
    main()