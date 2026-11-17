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
import scipy.signal as sig # Necesario para stft e istft

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
# FAST BEAMFORMER OPTIMIZADO
# =========================================================================
class FastBeamformer:
    """
    Versión optimizada de SH.MaxDirectivityBeamformer:
    - Precalcula todo lo que NO depende de la señal de entrada (una sola vez).
    - Procesa TODAS las direcciones de mirada en una sola pasada de STFT/ISTFT,
      en vez de repetir el cálculo completo por cada dirección.
    """
    def __init__(self, sampling_points, ldirs_fijas, fs, order=3, r=49e-3, win_len=2048):
        self.fs = fs
        self.win_len = win_len

        # ---- Todo esto es constante: se calcula UNA sola vez ----
        Y_q = SH.SHMatrix(sampling_points, order=order)      # (Q, N) complex
        self.Y_q_pinv = np.linalg.pinv(Y_q)                  # (N, Q) complex
        self.Y_l = SH.SHMatrix(ldirs_fijas, order=order)     # (v, N) complex

        # Frecuencias del STFT (dependen solo de win_len y fs, no de la señal)
        f = np.fft.rfftfreq(win_len, d=1/fs)
        b = SH.b_function(k=2*np.pi*f/343, r=r, ra=r, order=order)
        W, crossover_indices = SH.AmpModMatrix(f, b)
        D = SH.MaxDirectivityBeamformerCoeffMatrix(f, crossover_indices)
        
        # En tu SH.py, D es (F, F) diagonal, y W es (N, F).
        # Por tanto, la multiplicación W @ D da (N, F) y es 100% equivalente a tu código.
        self.WD = W @ D 

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        x: (Q, L) señales de micrófono, numpy real.
        return: (v, L) señales beamformadas para TODAS las direcciones fijas.
        """
        # 1. Una sola STFT para los 19 micrófonos a la vez
        f, t, x_stft = sig.stft(x, fs=self.fs, nperseg=self.win_len)
        
        # 2. Codificación a Armónicos Esféricos
        p_nm = np.einsum('NQ,Qft->Nft', self.Y_q_pinv, x_stft, optimize=True)
        
        # 3. Filtrado radial, directividad y decodificación a 7 canales (en un paso)
        y_stft = np.einsum('vN,Nft,Nf->vft', self.Y_l, p_nm, self.WD, optimize=True)
        
        # 4. Una sola ISTFT para los 7 canales resultantes
        _, y = sig.istft(y_stft, fs=self.fs, nperseg=self.win_len)
        return y   # (v, L)

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
        winhophalf = round(self.winhop / 2)

        # Hanning window base
        hann = th.hann_window(self.winhop, periodic=True, device=self.device)
        hann = th.cat([hann[:winhophalf],
                       th.ones(hop - winhophalf, device=self.device),
                       hann[-winhophalf:]])
        
        # Buffer de entrada
        self.buffer = th.zeros(1, channels, segmentsamp, device=self.device)

        # Detección dinámica de dimensiones (3D vs 4D)
        with th.inference_mode():
            dummy_in = th.zeros(1, channels, segmentsamp, device=device)
            dummy_out = apply_model(self.model, dummy_in, shifts=0, split=False, device=device)
        
        self.out_dim = dummy_out.dim()
        
        if self.out_dim == 3:
            self.win = hann.unsqueeze(0).unsqueeze(1).expand(1, nsources, -1)
            self.overlapadd = th.zeros(1, nsources, winhophalf, device=self.device)
        else:
            self.win = hann.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(1, nsources, channels, -1)
            self.overlapadd = th.zeros(1, nsources, channels, winhophalf, device=self.device)

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

        winhophalf = round(self.winhop / 2)

        if self.out_dim == 3:
            output[:, :, -self.hop - winhophalf:] *= self.win
            output[:, :, -self.hop - winhophalf:-self.hop] += self.overlapadd
            self.overlapadd = output[:, :, -winhophalf:]
            return output[:, :, -self.hop - winhophalf:-winhophalf]
        else:
            output[:, :, :, -self.hop - winhophalf:] *= self.win
            output[:, :, :, -self.hop - winhophalf:-self.hop] += self.overlapadd
            self.overlapadd = output[:, :, :, -winhophalf:]
            return output[:, :, :, -self.hop - winhophalf:-winhophalf]


# =========================================================================
# LÓGICA DE EVALUACIÓN
# =========================================================================
def get_parser():
    parser = argparse.ArgumentParser(description="Evaluación SDR con RealTimeModel (V2)")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV de 60s")
    parser.add_argument("--conf", type=str, default=DEFAULT_CONF)
    parser.add_argument("--ckpt", type=str, required=True, help="Ruta al checkpoint .ckpt")
    parser.add_argument("-c", "--context", type=float, default=2.0, help="Ventana temporal en segundos")
    parser.add_argument("--hop", type=int, default=4096, help="Tamaño de salto (muestras)")
    parser.add_argument("-d", "--device", default="cuda" if th.cuda.is_available() else "cpu")
    return parser

def main():
    def set_seed(seed=42):
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        th.backends.cudnn.deterministic = False # Desactivado para permitir Benchmark
        # OPTIMIZACIÓN 3: Habilitar CUDNN Benchmark
        # Obliga a la gráfica a buscar el algoritmo más rápido para tensores de tamaño fijo (como el nuestro)
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
    
    # Instanciamos el FastBeamformer optimizado
    beamformer = FastBeamformer(
        sampling_points, np.array(todas_direcciones),
        sample_rate, order=3, r=49e-3, win_len=2048
    )

    # ==========================================
    # WARM-UP: fuerza a cuDNN a buscar el algoritmo óptimo
    # ANTES de entrar al bucle cronometrado (evita el pico inicial)
    # ==========================================
    print("\n=> Warm-up de cuDNN/autocast...")
    with th.inference_mode():
        dummy_chunk = th.zeros(1, C, args.hop, device=device)
        for _ in range(5):
            # Corrección de la sintaxis deprecada
            with th.autocast(device_type=device.type):
                _ = modelo_rt.separate(dummy_chunk)
                
    th.cuda.synchronize(device)
    print("=> Warm-up completo.\n")

    # Reseteamos el estado que el warm-up haya podido dejar en el overlap-add.
    # Usamos zeros_like (crea tensores NUEVOS) en vez de .zero_() (in-place),
    # porque overlapadd/buffer quedaron como vistas de tensores de inferencia
    # tras el warm-up, y las operaciones in-place sobre esas vistas no están permitidas.
    modelo_rt.overlapadd = th.zeros_like(modelo_rt.overlapadd)
    modelo_rt.buffer = th.zeros_like(modelo_rt.buffer)
    modelo_rt._first_pass = True

    buffer_salida = []
    latencias_ms = []
    index = 0

    print(f"\n=> Procesando {nsamples/sample_rate:.1f}s | Contexto: {args.context}s | Hop: {args.hop}")

    # ==========================================
    # OPTIMIZACIÓN 2: APAGAR RECOLECTOR DE BASURA
    # ==========================================
    gc.disable()

    with th.inference_mode(): 
        with tqdm(total=nsamples/sample_rate, unit='s', desc="Inferencia RT") as pbar:
            while nsamples >= index + (args.hop * 3):
                
                th.cuda.synchronize(device)
                t_inicio = time.perf_counter()

                chunk_triple_np = mezcla_cruda_np[:, index : index + (args.hop * 3)].numpy()

                # Todo el proceso del beamformer se resuelve ahora en una sola pasada rápida
                y_beam_multi = beamformer.process(chunk_triple_np)          # (7, L)
                chunk_in_np = y_beam_multi[:, args.hop : args.hop * 2]      # (7, hop)

                chunk_in = th.from_numpy(chunk_in_np).float().unsqueeze(0).to(device)
                
                # ==========================================
                # OPTIMIZACIÓN 1: PRECISIÓN MIXTA (FP16)
                # ==========================================
                with th.autocast(device_type=device.type):
                    chunk_out = modelo_rt.separate(chunk_in)

                th.cuda.synchronize(device)
                t_fin = time.perf_counter()

                latencia_hardware_ms = (t_fin - t_inicio) * 1000
                latencias_ms.append(latencia_hardware_ms)

                buffer_salida.append(chunk_out.cpu())
                index += args.hop
                pbar.update(args.hop / sample_rate)

    # Volvemos a encender el recolector al acabar el bucle crítico
    gc.enable()

    # ==========================================
    # SIMULACIÓN Y REPORTE DE JITTER BUFFER
    # ==========================================
    latencias_np = np.array(latencias_ms)
    presupuesto_ms = (args.hop / sample_rate) * 1000
    
    def simular_jitter_buffer(latencias, presupuesto, jitter_blocks):
        """ Simula la línea de tiempo de un sistema en vivo para ver si hay cortes """
        t_out_prev = 0
        dropouts = 0
        for i, lat in enumerate(latencias):
            t_in = i * presupuesto
            # La GPU empieza a calcular en cuanto el bloque llega, o en cuanto termina el bloque anterior
            t_start_gpu = max(t_in, t_out_prev)
            t_out = t_start_gpu + lat
            t_out_prev = t_out
            
            # Fecha límite para reproducir el bloque sin que haya cortes (Drop-outs)
            t_deadline = t_in + presupuesto + (jitter_blocks * presupuesto)
            
            if t_out > t_deadline:
                dropouts += 1
        return dropouts

    def calcular_jitter_minimo(latencias, presupuesto):
        """
        Calcula, para cada bloque, cuánto tiempo extra (en unidades de 'presupuesto')
        necesitaría el buffer para que ese bloque nunca llegue tarde.
        Devuelve el jitter_blocks mínimo que lleva los cortes a 0.
        """
        t_out_prev = 0
        max_retraso_ms = 0.0
        for i, lat in enumerate(latencias):
            t_in = i * presupuesto
            t_start_gpu = max(t_in, t_out_prev)
            t_out = t_start_gpu + lat
            t_out_prev = t_out

            deadline_base = t_in + presupuesto  # deadline con jitter_blocks=0
            retraso = t_out - deadline_base
            if retraso > max_retraso_ms:
                max_retraso_ms = retraso

        # Cuántos bloques completos de presupuesto hacen falta para cubrir el peor retraso
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