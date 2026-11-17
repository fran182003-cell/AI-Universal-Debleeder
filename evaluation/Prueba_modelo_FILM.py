import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from torchmetrics.audio import SignalDistortionRatio
from LightingModel3_RoPE import LightningModel, PerfectedFiLMHDemucs
from MyDataset_3_RoPE import AutoSourceDataset

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load


def tiene_mismo_tipo(instrument_names):
    nombres_validos = [n for n in instrument_names if n != '']
    return len(nombres_validos) != len(set(nombres_validos))

def test_permutacion_espacial(checkpoint_path, conf_path, n_batches=100):
    
    with open(conf_path, 'r') as f:
        conf = yaml.safe_load(f)

    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 
              2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 
              2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 
              1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 
              1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reemplaza la parte de carga del modelo por esto:

# Reconstruir confdemucs igual que en train.py
    C = 7
    confdemucs = {
        'sources': [str(i) for i in range(C)],
        'audio_channels': C,
        'cac': conf['hdemucs_conf']['cac'],
        'samplerate': conf['hdemucs_conf']['samplerate'],
        'channels': conf['hdemucs_conf']['channels'],
        'segment': conf['dataset']['chunk_duration']
    }

    # Instanciar el modelo igual que en train.py
    model = PerfectedFiLMHDemucs(confdemucs=confdemucs, sh_dim=32)

    # Cargar el checkpoint pasando el modelo ya instanciado
    lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        model=model,
        optimizer=None,
        conf=conf
    )
    lightning_model.eval()
    lightning_model.to(device)
    
    sdr_metric = SignalDistortionRatio().to(device)

    val_dataset = AutoSourceDataset(
        csv_mixtures=conf['dataset']['val_estandar_csv'],
        rir_root_dir=conf['dataset']['rir_root_dir'],
        chunk_duration=conf['dataset']['chunk_duration'],
        sample_rate=conf['hdemucs_conf']['samplerate'],
        sampling_points=sampling_points,
    )
    # batch_size=1 para clasificación por muestra limpia
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    sdrs_mismo_tipo_normal    = []
    sdrs_mismo_tipo_permutado = []
    sdrs_mismo_tipo_aleatorio = []
    sdrs_distinto_tipo_normal    = []
    sdrs_distinto_tipo_permutado = []
    sdrs_distinto_tipo_aleatorio = []

    with torch.no_grad():
        for batch_idx, (mix, refs, sh_features, instrument_names) in enumerate(val_loader):
            if batch_idx >= n_batches:
                break

            mix         = mix.to(device)
            refs        = refs.to(device)
            sh_features = sh_features.to(device)

            # Clasificar la muestra
            # instrument_names es lista de 7 elementos, cada uno tensor/string
            nombres = [instrument_names[j][0] if isinstance(instrument_names[j], (list, tuple)) 
                      else instrument_names[j] for j in range(len(instrument_names))]
            es_mismo_tipo = tiene_mismo_tipo(nombres)

            # 1. Forward normal
            out_normal = lightning_model(mix, sh_features)
            
            # Calcular SDR evitando fuentes silenciosas
            rms = torch.sqrt(torch.mean(refs ** 2, dim=-1))
            mask = rms > 1e-4
            if mask.any():
                sdr_n = sdr_metric(out_normal[mask], refs[mask]).item()
            else:
                continue

            # 2. Forward permutado
            perm_idx = torch.randperm(7)
            sh_permutado = sh_features[:, perm_idx, :]
            out_permutado = lightning_model(mix, sh_permutado)
            sdr_p = sdr_metric(out_permutado[mask], refs[mask]).item()

            # 3. Forward aleatorio (destrucción total)
            sh_aleatorio = torch.randn_like(sh_features)
            out_aleatorio = lightning_model(mix, sh_aleatorio)
            sdr_a = sdr_metric(out_aleatorio[mask], refs[mask]).item()

            # Clasificar y guardar
            if es_mismo_tipo:
                sdrs_mismo_tipo_normal.append(sdr_n)
                sdrs_mismo_tipo_permutado.append(sdr_p)
                sdrs_mismo_tipo_aleatorio.append(sdr_a)
            else:
                sdrs_distinto_tipo_normal.append(sdr_n)
                sdrs_distinto_tipo_permutado.append(sdr_p)
                sdrs_distinto_tipo_aleatorio.append(sdr_a)

            if (batch_idx + 1) % 20 == 0:
                print(f"Procesadas {batch_idx + 1}/{n_batches} muestras...")

    # Resultados
    print("\n" + "="*60)
    print("RESULTADOS DEL EXPERIMENTO DE PERMUTACIÓN")
    print("="*60)

    if sdrs_mismo_tipo_normal:
        n  = np.mean(sdrs_mismo_tipo_normal)
        p  = np.mean(sdrs_mismo_tipo_permutado)
        a  = np.mean(sdrs_mismo_tipo_aleatorio)
        print(f"MISMO TIPO ({len(sdrs_mismo_tipo_normal)} muestras)")
        print(f"  SDR normal:    {n:.3f} dB")
        print(f"  SDR permutado: {p:.3f} dB  (degradación: {n-p:.3f} dB)")
        print(f"  SDR aleatorio: {a:.3f} dB  (degradación: {n-a:.3f} dB)")

    print("-"*60)

    if sdrs_distinto_tipo_normal:
        n  = np.mean(sdrs_distinto_tipo_normal)
        p  = np.mean(sdrs_distinto_tipo_permutado)
        a  = np.mean(sdrs_distinto_tipo_aleatorio)
        print(f"DISTINTO TIPO ({len(sdrs_distinto_tipo_normal)} muestras)")
        print(f"  SDR normal:    {n:.3f} dB")
        print(f"  SDR permutado: {p:.3f} dB  (degradación: {n-p:.3f} dB)")
        print(f"  SDR aleatorio: {a:.3f} dB  (degradación: {n-a:.3f} dB)")

    print("="*60)
    print("INTERPRETACIÓN:")
    
    if sdrs_mismo_tipo_normal and sdrs_distinto_tipo_normal:
        deg_mismo    = np.mean(sdrs_mismo_tipo_normal) - np.mean(sdrs_mismo_tipo_permutado)
        deg_distinto = np.mean(sdrs_distinto_tipo_normal) - np.mean(sdrs_distinto_tipo_permutado)
        
        if deg_mismo > deg_distinto * 1.5:
            print("→ FiLM usa los SH principalmente para resolver ambigüedades espectrales")
            print("  La degradación es mayor en mezclas de mismo tipo, como se esperaba")
        elif deg_mismo > 0.5:
            print("→ FiLM usa los SH de forma significativa en ambas categorías")
        else:
            print("→ FiLM aún no usa los SH de forma concluyente en esta época")

if __name__ == '__main__':
    test_permutacion_espacial(
        checkpoint_path=os.environ.get("CKPT_PATH") or sys.exit("Define la variable de entorno CKPT_PATH con la ruta al checkpoint .ckpt"),  # modelo con FiLM (version_7 en la memoria)
        conf_path=DEFAULT_CONF,
        n_batches=200
    )