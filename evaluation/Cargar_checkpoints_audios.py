import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import os
import yaml
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm

# Importaciones de tus módulos del proyecto
from MyDataset_3_permut import AutoSourceDataset
from lightingModel_3_permut import LightningModel
from demucs.hdemucs import HDemucs

# Parche de seguridad para la carga de PyTorch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

def main():
    # 1. Configuración de precisión y dispositivo
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizando dispositivo: {device}")

    # 2. Cargar archivo de configuración
    with open(DEFAULT_CONF, 'r') as f:
        conf = yaml.safe_load(f)

    sample_rate = conf['hdemucs_conf']['samplerate']
    ruta_checkpoint = os.environ.get("CKPT_PATH") or sys.exit("Define la variable de entorno CKPT_PATH con la ruta al checkpoint .ckpt")  # modelo con permutación (version_28 en la memoria)

    # 3. Puntos de muestreo de la esfera del array Zylia
    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))

    # 4. Inicializar Datasets de Validación (Estándar, Poco Reverb, Reverb)
    print("\n[+] Inicializando los datasets de validación...")
    val_datasets = {
        "val_estandar": AutoSourceDataset(
            csv_mixtures=conf['dataset']['val_estandar_csv'],
            rir_root_dir=conf['dataset']['rir_root_dir'],
            chunk_duration=conf['dataset']['chunk_duration'],
            sample_rate=sample_rate,
            sampling_points=sampling_points,
        ),
        "val_poco_reverb": AutoSourceDataset(
            csv_mixtures=conf['dataset']['val_poco_reverb_csv'],
            rir_root_dir=conf['dataset']['rir_root_dir'],
            chunk_duration=conf['dataset']['chunk_duration'],
            sample_rate=sample_rate,
            sampling_points=sampling_points,
        ),
        "val_reverb": AutoSourceDataset(
            csv_mixtures=conf['dataset']['val_reverb_csv'],
            rir_root_dir=conf['dataset']['rir_root_dir'],
            chunk_duration=conf['dataset']['chunk_duration'],
            sample_rate=sample_rate,
            sampling_points=sampling_points,
        )
    }

    # 5. Instanciar y cargar pesos en el modelo HDemucs
    C = 7
    confdemucs = {
        'sources': [str(i) for i in range(C)],
        'audio_channels': C, 
        'cac': conf['hdemucs_conf']['cac'],
        'samplerate': sample_rate,
        'channels': conf['hdemucs_conf']['channels'],
        'segment': conf['dataset']['chunk_duration']
    }
    
    print("[+] Cargando arquitectura HDemucs y cargando checkpoint...")
    model_base = HDemucs(**confdemucs)
    
    # Cargamos el LightningModel con el modelo base
    lightning_model = LightningModel.load_from_checkpoint(
        ruta_checkpoint, 
        model=model_base, 
        optimizer=None, 
        conf=conf
    )
    lightning_model.to(device)
    lightning_model.eval()  # Desactivar Dropout, BatchNorm, etc.

    # 6. Directorio base para guardar los archivos de audio resultados
    output_base_dir = "./resultados_audio"
    os.makedirs(output_base_dir, exist_ok=True)

    # 7. Bucle principal de extracción (primeros 10 de cada set)
    num_ejemplos = 10

    with torch.no_grad(): # Desactivar tracking de gradientes para ahorrar memoria/tiempo
        for set_name, dataset in val_datasets.items():
            print(f"\n" + "="*60)
            print(f" PROCESANDO SET DE VALIDACIÓN: {set_name.upper()}")
            print("="*60)

            # Iterar sobre las primeras 10 muestras
            for idx in tqdm(range(min(num_ejemplos, len(dataset))), desc=f"Procesando {set_name}"):
                
                # Obtener los elementos desde el dataset
                mix, refs, sh_features, instrument_names, mix_raw = dataset[idx]
                
                # Preparar dimensiones para el modelo: añadir dimensión de batch (1, C, T)
                mix_batched = mix.unsqueeze(0).to(device)
                sh_features_batched = sh_features.unsqueeze(0).to(device)

                # Pasar por el modelo para obtener la estimación
                estimated_sources = lightning_model(mix_batched, sh_features_batched)
                
                # Quitar dimensión de batch (C, T) y pasar a CPU como numpy
                estimated_sources = estimated_sources.squeeze(0).cpu().numpy()
                mix_np = mix.numpy()
                refs_np = refs.numpy()
                mix_raw_np = mix_raw.T
                # Crear directorio de guardado para la mezcla específica
                mix_output_dir = os.path.join(output_base_dir, set_name, f"mezcla_{idx:02d}")
                os.makedirs(mix_output_dir, exist_ok=True)

                try:
                # Si mix_raw_np tiene forma (19, Tiempo), tomamos el canal 0
                    if mix_raw_np.ndim == 2:
                        if mix_raw_np.shape[0] < mix_raw_np.shape[1]:
                            mic_0_to_save = mix_raw_np[0, :]  # Forma (Tiempo,)
                        else:
                            mic_0_to_save = mix_raw_np[:, 0]  # Forma (Tiempo,)
                    else:
                        mic_0_to_save = mix_raw_np

                    ruta_mic0_mono = os.path.join(mix_output_dir, "mezcla_original_mic0_mono.wav")
                    
                    # Al ser un archivo de 1 solo canal, VS Code lo reproducirá instantáneamente
                    sf.write(ruta_mic0_mono, mic_0_to_save, sample_rate)
                    print(f" -> [Opcional] Guardado micrófono 0 (Mono) para VS Code en: {ruta_mic0_mono}")

                except Exception as e:
                    print(f"⚠️ No se pudo guardar la previsualización Mono: {e}")

                # Guardar canales del Beamformer de la Mezcla Original (Input del modelo)
                for ch in range(C):
                    sf.write(
                        os.path.join(mix_output_dir, f"mix_beamformed_ch{ch}.wav"), 
                        mix_np[ch], 
                        sample_rate
                    )

                # Guardar los Targets (Referencias limpias) y las Estimaciones correspondientes
                for ch in range(C):
                    instrument_label = instrument_names[ch] if instrument_names[ch] != '' else f"vacio_ch{ch}"
                    
                    # 1. Guardar referencia
                    sf.write(
                        os.path.join(mix_output_dir, f"target_{ch:02d}_{instrument_label}.wav"), 
                        refs_np[ch], 
                        sample_rate
                    )
                    
                    # 2. Guardar predicción estimada
                    sf.write(
                        os.path.join(mix_output_dir, f"estimado_{ch:02d}_{instrument_label}.wav"), 
                        estimated_sources[ch], 
                        sample_rate
                    )

    print(f"\n[✔] Proceso completado exitosamente. Audios exportados a: {os.path.abspath(output_base_dir)}")

if __name__ == '__main__':
    main()