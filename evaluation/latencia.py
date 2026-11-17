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

# Parche PyTorch
_original_load = th.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
th.load = _patched_load

from demucs.apply import apply_model
from demucs.hdemucs import HDemucs

# --- IMPORTACIONES ACTUALIZADAS A TU VERSIÓN 2 ---
from MyDataset_2 import AutoSourceDataset
from LightingModel2 import LightningModel

# =========================================================================
# FAST BEAMFORMER OPTIMIZADO (FFT / IFFT)
# =========================================================================
class FastBeamformer:
    """
    Versión optimizada de SH.MaxDirectivityBeamformer:
    - Precalcula todo lo que NO depende de la señal de entrada (una sola vez).
    - Procesa TODAS las direcciones de mirada en una sola pasada de FFT/IFFT.
    """
    def __init__(self, sampling_points, ldirs_fijas, fs, order=3, r=49e-3, win_len=2048):
        self.fs = fs
        self.win_len = win_len

        # ---- Todo esto es constante: se calcula UNA sola vez ----
        Y_q = SH.SHMatrix(sampling_points, order=order)      # (Q, N) complex
        self.Y_q_pinv = np.linalg.pinv(Y_q)                  # (N, Q) complex
        self.Y_l = SH.SHMatrix(ldirs_fijas, order=order)     # (v, N) complex

        # Frecuencias de la FFT (dependen solo de win_len y fs)
        f = np.fft.rfftfreq(win_len, d=1/fs)
        b = SH.b_function(k=2*np.pi*f/343, r=r, ra=r, order=order)
        W, crossover_indices = SH.AmpModMatrix(f, b)
        D = SH.MaxDirectivityBeamformerCoeffMatrix(f, crossover_indices)
        
        # Multiplicación base: W @ D da (N, F)
        self.WD = W @ D 

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        x: (19, L) señales de micrófono en el bloque actual.
        return: (7, L) señales beamformadas para TODAS las direcciones fijas.
        """
        # 1. Una sola FFT real sobre el eje temporal
        x_fft = np.fft.rfft(x, n=self.win_len, axis=-1)
        
        # 2. Codificación a Armónicos Esféricos
        p_nm = self.Y_q_pinv @ x_fft
        
        # 3. Filtrado radial/directividad
        p_nm_filtrado = p_nm * self.WD
        
        # 4. Decodificación a las 7 direcciones
        y_fft = self.Y_l @ p_nm_filtrado
        
        # 5. IFFT para volver al dominio del tiempo
        y = np.fft.irfft(y_fft, n=self.win_len, axis=-1)
        return y


# =========================================================================
# CLASE EN TIEMPO REAL
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
        self.winhophalf = round(self.winhop / 2) # Exportado como self para acceder fuera

        # Hanning window base
        hann = th.hann_window(self.winhop, periodic=True, device=self.device)
        hann = th.cat([hann[:self.winhophalf],
                       th.ones(hop - self.winhophalf, device=self.device),
                       hann[-self.winhophalf:]])
        
        # Buffer de entrada
        self.buffer = th.zeros(1, channels, segmentsamp, device=self.device)

        # Detección dinámica de dimensiones (3D vs 4D)
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

    def separate(self, frame: th.Tensor):
        if frame.shape != (1, self.model.audio_channels, self.hop):
            raise ValueError(f"Frame must have size 1x{self.model.audio_channels}x{self.hop}")

        self.buffer[:, :, :-self.hop] = self.buffer[:, :, self.hop:]
        self.buffer[:, :, -self.hop:] = frame
        
        if self._first_pass:
            samples_in = self.buffer.shape[-1]
            secs_in = samples_in / self.fs
            print(f"\n[DEBUG RT] 📥 TENSOR DE ENTRADA AL MODELO:")
            print(f"           - Shape: {self.buffer.shape}")
            print(f"           - Contexto aplicado: {samples_in} muestras ({secs_in:.2f} s)")

        output = apply_model(self.model, self.buffer, shifts=0, split=False, device=self.device)
        
        if self._first_pass:
            samples_out = output.shape[-1]
            secs_out = samples_out / self.fs
            print(f"[DEBUG RT] 📤 TENSOR DE SALIDA DEL MODELO:")
            print(f"           - Shape: {output.shape}")
            print(f"           - Generado: {samples_out} muestras ({secs_out:.2f} s)\n")
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
# LÓGICA DE EVALUACIÓN
# =========================================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Evaluación SDR con RealTimeModel (V2)")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV de 60s")
    parser.add_argument("--conf", type=str, default=DEFAULT_CONF)
    parser.add_argument("--ckpt", type=str, required=True, help="Ruta al checkpoint .ckpt")
    parser.add_argument("-c", "--context", type=float, default=1.0, help="Ventana temporal en segundos")
    parser.add_argument("--hop", type=int, default=4096, help="Tamaño de salto (muestras)")
    parser.add_argument("-d", "--device", default="cuda" if th.cuda.is_available() else "cpu")
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

    dataset_eval = AutoSourceDataset(
        csv_mixtures=args.csv,
        rir_root_dir=conf['dataset']['rir_root_dir'],
        chunk_duration=60.0,
        sample_rate=sample_rate,
        sampling_points=sampling_points,
    )

    mezcla_cruda_np, referencias, active_dirs = dataset_eval[0]
    referencias = referencias.unsqueeze(0).to(device)
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
    
    # ==========================================
    # CREACIÓN DEL BUFFER OLA (BEAMFORMER)
    # ==========================================
    zylia_channels = 19
    beamformer_win_len = args.hop + modelo_rt.winhophalf
    mic_buffer = np.zeros((zylia_channels, beamformer_win_len), dtype=np.float32)

    # Instanciamos el FastBeamformer optimizado con el win_len exacto
    beamformer = FastBeamformer(
        sampling_points, np.array(todas_direcciones),
        sample_rate, order=3, r=49e-3, win_len=beamformer_win_len
    )

    # ==========================================
    # WARM-UP COMPLETO (Beamformer + Demucs)
    # ==========================================
    print("\n=> Warm-up del Pipeline (Beamformer + cuDNN/autocast)...")
    with th.inference_mode():
        dummy_mic = np.zeros((zylia_channels, beamformer_win_len), dtype=np.float32)
        dummy_in = th.zeros(1, C, args.hop, device=device)
        for _ in range(5):
            _ = beamformer.process(dummy_mic) 
            with th.autocast(device_type=device.type):
                _ = modelo_rt.separate(dummy_in)
                
    th.cuda.synchronize(device)
    print("=> Warm-up completo.\n")

    modelo_rt.overlapadd = th.zeros_like(modelo_rt.overlapadd)
    modelo_rt.buffer = th.zeros_like(modelo_rt.buffer)
    modelo_rt._first_pass = True

    buffer_salida = []
    latencias_ms = []
    index = 0

    print(f"\n=> Procesando {nsamples/sample_rate:.1f}s | Contexto: {args.context}s | Hop: {args.hop}")

    gc.disable()

    with th.inference_mode(): 
        with tqdm(total=nsamples/sample_rate, unit='s', desc="Inferencia RT") as pbar:
            # Procesamos de bloque en bloque imitando el directo (hop a hop)
            while nsamples >= index + args.hop:
                
                # 1. Chunk crudo desde el dataset (simula audio llegando de tarjeta de sonido)
                chunk_raw = mezcla_cruda_np[:, index : index + args.hop].numpy()

                th.cuda.synchronize(device)
                t_inicio = time.perf_counter()

                # 2. OLA del Beamformer
                mic_buffer[:, :-args.hop] = mic_buffer[:, args.hop:]
                mic_buffer[:, -args.hop:] = chunk_raw

                # 3. Beamformer con FFT/IFFT 
                y_beam_multi = beamformer.process(mic_buffer) 
                
                # 4. Extraer el bloque resultante válido
                chunk_in_np = y_beam_multi[:, -args.hop:]      

                # 5. Envío a la GPU e inferencia
                chunk_in = th.from_numpy(chunk_in_np).float().unsqueeze(0).to(device)
                
                with th.autocast(device_type=device.type):
                    chunk_out = modelo_rt.separate(chunk_in)

                th.cuda.synchronize(device)
                t_fin = time.perf_counter()

                latencia_hardware_ms = (t_fin - t_inicio) * 1000
                latencias_ms.append(latencia_hardware_ms)

                buffer_salida.append(chunk_out.cpu())
                index += args.hop
                pbar.update(args.hop / sample_rate)

    gc.enable()

    # ==========================================
    # SIMULACIÓN Y REPORTE DE JITTER BUFFER
    # ==========================================
    latencias_np = np.array(latencias_ms)
    presupuesto_ms = (args.hop / sample_rate) * 1000
    
    def simular_jitter_buffer(latencias, presupuesto, jitter_blocks):
        t_out_prev = 0
        dropouts = 0
        for i, lat in enumerate(latencias):
            t_in = i * presupuesto
            t_start_gpu = max(t_in, t_out_prev)
            t_out = t_start_gpu + lat
            t_out_prev = t_out
            
            t_deadline = t_in + presupuesto + (jitter_blocks * presupuesto)
            
            if t_out > t_deadline:
                dropouts += 1
        return dropouts

    def calcular_jitter_minimo(latencias, presupuesto):
        t_out_prev = 0
        max_retraso_ms = 0.0
        for i, lat in enumerate(latencias):
            t_in = i * presupuesto
            t_start_gpu = max(t_in, t_out_prev)
            t_out = t_start_gpu + lat
            t_out_prev = t_out

            deadline_base = t_in + presupuesto 
            retraso = t_out - deadline_base
            if retraso > max_retraso_ms:
                max_retraso_ms = retraso

        jitter_minimo = int(np.ceil(max_retraso_ms / presupuesto)) if max_retraso_ms > 0 else 0
        return jitter_minimo, max_retraso_ms

    dropouts_0 = simular_jitter_buffer(latencias_np, presupuesto_ms, 0)
    dropouts_1 = simular_jitter_buffer(latencias_np, presupuesto_ms, 1)
    dropouts_2 = simular_jitter_buffer(latencias_np, presupuesto_ms, 2)

    jitter_minimo, max_retraso_ms = calcular_jitter_minimo(latencias_np, presupuesto_ms)
    dropouts_minimo = simular_jitter_buffer(latencias_np, presupuesto_ms, jitter_minimo)

    print("\n" + "="*50)
    print(" REPORTE DE LATENCIA Y JITTER BUFFER")
    print("="*50)
    print(f" Presupuesto base por bloque   : {presupuesto_ms:.2f} ms")
    print(f" Media de procesamiento        : {latencias_np.mean():.2f} ms")
    print(f" Máximo pico registrado (P100) : {latencias_np.max():.2f} ms")
    print("-" * 50)
    print(" SIMULACIÓN DE CORTES DE AUDIO EN DIRECTO:")
    print(f" -> Sin Jitter Buffer (0 extra) : {dropouts_0} cortes ({100*dropouts_0/len(latencias_np):.1f}%)")
    print(f" -> Jitter Buffer de 1 bloque   : {dropouts_1} cortes ({100*dropouts_1/len(latencias_np):.1f}%) [Añade {presupuesto_ms:.1f}ms de latencia]")
    print(f" -> Jitter Buffer de 2 bloques  : {dropouts_2} cortes ({100*dropouts_2/len(latencias_np):.1f}%) [Añade {presupuesto_ms*2:.1f}ms de latencia]")
    print("-" * 50)
    print(f" Peor retraso registrado        : {max_retraso_ms:.2f} ms")
    print(f" Jitter Buffer MÍNIMO para 0 cortes : {jitter_minimo} bloques "
          f"[Añade {presupuesto_ms*jitter_minimo:.1f}ms de latencia total]")
    print(f"   -> Verificación: con {jitter_minimo} bloques quedan {dropouts_minimo} cortes "
          f"({100*dropouts_minimo/len(latencias_np):.1f}%)")
    print("="*50 + "\n")

    audio_reconstruido = th.cat(buffer_salida, dim=-1).to(device)

    # ==========================================
    # CORRECCIÓN DE ALINEACIÓN PARA CÁLCULO SDR
    # ==========================================
    offset_alineacion = args.hop 
    len_eval = min(audio_reconstruido.shape[-1], referencias.shape[-1] - offset_alineacion)

    est_vocals = audio_reconstruido[:, idx_vocals, :len_eval].unsqueeze(1)
    ref_vocals = referencias[:, idx_vocals, offset_alineacion : offset_alineacion + len_eval].unsqueeze(1)

    start_eval_time = 5.0
    end_eval_time = 55.0

    start_sample = int(start_eval_time * sample_rate)
    end_sample = int(end_eval_time * sample_rate)

    est_vocals_valido = est_vocals[:, :, start_sample:end_sample]
    ref_vocals_valido = ref_vocals[:, :, start_sample:end_sample]

    audio_guardar = est_vocals.squeeze(0).cpu()
    
    nombre_salida = f"estimacion_voces_ctx_{args.context}s.wav"
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