import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import pandas as pd
import random
import os
import soundfile as sf
import numpy as np

random.seed(42)

df_sources = pd.read_csv('./info/master_sources_info.csv')
available_classes = df_sources['class_label'].unique().tolist()

# =================================================================
# FUNCIÓN: Comprobador de Energía ultrarrápido (Modificado para 60s)
# =================================================================
def check_audio_energy(filepath, start_sec, chunk_length=60.0, threshold=1e-3):
    real_filepath = filepath
    if real_filepath.startswith('./'):
        real_filepath = os.path.join(os.environ.get("AUDIO_ROOT", "./data/BBDD"), real_filepath[2:])
        
    try:
        info = sf.info(real_filepath)
        sr = info.samplerate
        start_frame = int(start_sec * sr)
        frames_to_read = int(chunk_length * sr)
        
        if start_frame + frames_to_read > info.frames:
            return False
            
        audio, _ = sf.read(real_filepath, frames=frames_to_read, start=start_frame, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        energia = np.mean(np.abs(audio))
        return energia > threshold
    except Exception as e:
        return False

# =================================================================
# FUNCIÓN PRINCIPAL
# =================================================================
def generate_test_mixture(rir_csv_path, out_csv_path, chunk_length=60.0):
    df_rirs = pd.read_csv(rir_csv_path)
    
    # --- LA NUEVA LÓGICA DE BÚSQUEDA DE SALA ---
    num_fuentes_deseadas = 3
    
    # 1. Contamos cuántas posiciones (RIRs) tiene cada sala
    conteo_por_sala = df_rirs['room_id'].value_counts()
    
    # 2. Nos quedamos solo con las salas que tienen 3 o más posiciones
    salas_validas = conteo_por_sala[conteo_por_sala >= num_fuentes_deseadas].index.tolist()
    
    if not salas_validas:
        raise ValueError(f"No hay ninguna sala en {rir_csv_path} que tenga {num_fuentes_deseadas} o más posiciones.")
    
    # 3. Elegimos una sala aleatoria de las válidas
    sala_elegida = random.choice(salas_validas)
    rirs_sala = df_rirs[df_rirs['room_id'] == sala_elegida]
    
    # 4. Seleccionamos aleatoriamente exactamente 3 posiciones de esa sala
    rirs_seleccionados = rirs_sala.sample(n=num_fuentes_deseadas)
    
    clases_asignadas = ['vocals'] * num_fuentes_deseadas
    
    mixtures_data = []
    print(f"Generando mezcla en la sala '{sala_elegida}'...")
    
    for (idx, rir_row), chosen_class in zip(rirs_seleccionados.iterrows(), clases_asignadas):
        rir_filepath = rir_row['filepath']
        
        fuentes_validas = df_sources[
            (df_sources['class_label'] == chosen_class) & 
            (df_sources['duration_sec'] >= chunk_length)
        ]
        
        if fuentes_validas.empty:
            print(f"[Aviso] No hay audios de 60s para {chosen_class}. Usando uno más corto.")
            fuentes_validas = df_sources[df_sources['class_label'] == chosen_class]
            
        while True: 
            src_row = fuentes_validas.sample(1).iloc[0]
            src_filepath = src_row['filepath']
            duration = src_row['duration_sec']
            
            if duration >= chunk_length:
                min_start = duration * 0.10
                max_start = duration - chunk_length
                
                if max_start < min_start:
                    min_start = 0
                    
                intentos = 0
                tiempo_valido = False
                
                while intentos < 20 and not tiempo_valido:
                    candidato = round(random.uniform(min_start, max_start), 3)
                    if check_audio_energy(src_filepath, candidato, chunk_length):
                        start_second = candidato
                        tiempo_valido = True
                    intentos += 1
                
                if tiempo_valido:
                    break
            else:
                start_second = 0.0
                break
                
        mixtures_data.append({
            'mixture_id': 1,
            'room_id': sala_elegida,
            'rir_id': rir_row['rir_id'],
            'rir_filepath': rir_filepath,
            'rir_azimuth': rir_row['az'],
            'rir_elevation': rir_row['el'],
            'source_id': src_row['source_id'],
            'source_class': chosen_class, 
            'source_filepath': src_filepath,
            'start_second': start_second
        })
            
    df_mix = pd.DataFrame(mixtures_data)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_mix.to_csv(out_csv_path, index=False)
    
    print(f"\n[OK] Mezcla guardada en: {out_csv_path}")
    print(f" -> Total fuentes generadas: {len(df_mix)}")
    print(f" -> Acimuts reales seleccionados al azar: {df_mix['rir_azimuth'].tolist()}")

# EJECUCIÓN
if __name__ == '__main__':
    generate_test_mixture(
        rir_csv_path='./info_2/val_rir_info.csv', 
        out_csv_path='./info_2/latency_test_mixture_60s.csv',
        chunk_length=60.0
    )