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
import torchaudio
import numpy as np
from torch.utils.data import DataLoader

# Importamos tus clases reales
from LightingModel2 import LightningModel
from MyDataset_2 import AutoSourceDataset

def main():
    # ==========================================
    # 1. CARGAR CONFIGURACIÓN Y PARÁMETROS
    # ==========================================
    config_file = DEFAULT_CONF
    print(f"Leyendo configuración desde: {config_file}")
    
    with open(config_file, 'r') as f:
        conf = yaml.safe_load(f)
        
    sample_rate = conf['dataset']['sample_rate']          # 44100
    chunk_duration = conf['dataset']['chunk_duration']    # 5.0
    rir_dir = conf['dataset']['rir_root_dir']
    
    # Parámetros del array (extraídos de tu main de entrenamiento)
    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))
    
    # Rutas de archivos
    checkpoint_path = os.environ.get("CKPT_PATH") or sys.exit("Define la variable de entorno CKPT_PATH con la ruta al checkpoint .ckpt")  # modelo inicial (version_12 en la memoria)
    test_csv = os.environ.get("TEST_CSV", "./data/info/Pruebas_mismo_instrumentos.csv")
    
    out_dir = './resultados_audios_test_2'
    num_simulations_to_save = 1
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================
    # 2. CARGAR DATASET Y MODELO
    # ==========================================
    print(f"Instanciando dataset con {test_csv}...")
    dataset = AutoSourceDataset(
        csv_mixtures=test_csv,
        rir_root_dir=rir_dir,
        chunk_duration=chunk_duration,
        sample_rate=sample_rate,
        sampling_points=sampling_points
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Cargando modelo...")
    # Cargamos el modelo Lightning que envuelve a HDemucs
    model = LightningModel.load_from_checkpoint(checkpoint_path,weights_only=False)
    model.eval() 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Usando dispositivo: {device}")

   # ==========================================
    # 3. BUCLE DE INFERENCIA
    # ==========================================
    print("Iniciando separación de fuentes...")
    with torch.no_grad(): 
        for i, (mix, targets, instrument_names) in enumerate(dataloader):
            if i >= num_simulations_to_save:
                break
                
            print(f" -> Procesando simulación {i+1}...")
            
            # mix: [1, 7, tiempo]
            mix = mix.to(device) 
            
            estimations = model(mix) 
            
            # Pasamos todo a CPU
            mix_cpu = mix.squeeze(0).cpu()           # [7, tiempo]
            targets_cpu = targets.squeeze(0).cpu()   # [7, tiempo]
            estimations_cpu = estimations.squeeze(0).cpu() # [7, 1, tiempo] o [7, tiempo]
            
            sim_dir = os.path.join(out_dir, f'simulacion_{i+1:03d}')
            os.makedirs(sim_dir, exist_ok=True)
            
            # Recorremos ABSOLUTAMENTE TODOS los canales (los 7 ángulos)
            for c_idx in range(mix_cpu.shape[0]):
                
                # Sacamos el nombre o ponemos 'vacio'
                class_name = instrument_names[c_idx][0] if isinstance(instrument_names[c_idx], (list, tuple)) else instrument_names[c_idx]
                etiqueta = class_name if class_name != '' else 'vacio'
                
                # 1. GUARDAR BEAMFORMER (Mezcla en este ángulo)
                bf_audio = mix_cpu[c_idx:c_idx+1, :]
                torchaudio.save(
                    os.path.join(sim_dir, f'1_mezcla_ch{c_idx}_{etiqueta}.wav'), 
                    bf_audio, 
                    sample_rate
                )
                
                # 2. GUARDAR GROUND TRUTH (Será silencio si está 'vacio')
                gt_audio = targets_cpu[c_idx:c_idx+1, :] 
                torchaudio.save(
                    os.path.join(sim_dir, f'2_gt_ch{c_idx}_{etiqueta}.wav'), 
                    gt_audio, 
                    sample_rate
                )
                
                # 3. GUARDAR ESTIMACIÓN DEL MODELO
                est_audio = estimations_cpu[c_idx]
                if est_audio.dim() > 1 and est_audio.shape[0] == 1:
                    pass # Ya es [1, tiempo]
                elif est_audio.dim() == 1:
                    est_audio = est_audio.unsqueeze(0)
                else:
                    est_audio = est_audio[0:1, :]

                torchaudio.save(
                    os.path.join(sim_dir, f'3_est_ch{c_idx}_{etiqueta}.wav'), 
                    est_audio, 
                    sample_rate
                )

    print(f"\nProceso finalizado. Resultados en: {out_dir}")

if __name__ == '__main__':
    main()