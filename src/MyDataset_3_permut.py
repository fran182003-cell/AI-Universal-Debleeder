import torch
import soundfile as sf
import pandas as pd
import numpy as np
import scipy.io
import scipy.signal as sig
import os
import h5py

import SH # Tu librería del beamformer

# Carpeta raíz de las bases de datos de audio (rutas relativas de los CSV)
AUDIO_ROOT = os.environ.get("AUDIO_ROOT", "./data/BBDD")

class AutoSourceDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_mixtures, 
                 rir_root_dir=None,
                 chunk_duration=5.0,
                 sample_rate=44100,
                 sampling_points=None,
                 order=3,
                 r=49e-3,
                 win_len=2048):
        
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_len = int(self.sample_rate * self.chunk_duration)
        self.rir_root_dir = rir_root_dir
        
        self.df = pd.read_csv(csv_mixtures)
        self.mixture_ids = self.df['mixture_id'].unique()
        
        self.sampling_points = sampling_points
        self.order = order
        self.r = r
        self.win_len = win_len

    def __len__(self):
        return len(self.mixture_ids)

    def __getitem__(self, idx):
        mix_id_actual = self.mixture_ids[idx]
        df_mix = self.df[self.df['mixture_id'] == mix_id_actual]
     
        convolved_sources_raw = []
        active_dirs = []
        instrument_names = []

        # =================================================================
        # FASE 1: LEER AUDIOS, CONVOLUCIONAR CON RIR Y GUARDAR
        # =================================================================
        for i, (_, row) in enumerate(df_mix.iterrows()):
            ruta_audio = row['source_filepath']
            if ruta_audio.startswith('./'):
                ruta_audio = os.path.join(AUDIO_ROOT, ruta_audio[2:])
            
            ruta_rir = os.path.join(self.rir_root_dir, row['rir_filepath']) if self.rir_root_dir else row['rir_filepath']
                   
            try:
                info = sf.info(ruta_audio)
                fs_file = info.samplerate
                
                start_sample = int(row['start_second'] * fs_file)
                stop_sample = start_sample + int(self.chunk_duration * fs_file)
                
                s_raw, _ = sf.read(ruta_audio, 
                                   start=start_sample, 
                                   stop=stop_sample, 
                                   dtype='float32')
                
                if s_raw.ndim > 1: s_raw = s_raw[:, 0]

                if fs_file != self.sample_rate:
                    import librosa 
                    s = librosa.resample(s_raw, orig_sr=fs_file, target_sr=self.sample_rate)
                else:
                    s = s_raw

                if len(s) < self.chunk_len: 
                    s = np.pad(s, (0, self.chunk_len - len(s)), 'constant')
                else:
                    s = s[:self.chunk_len]

            except Exception as e:
                print(f"Error procesando {ruta_audio}: {e}")
                s = np.zeros(self.chunk_len, dtype='float32')
          
            try:
                with h5py.File(ruta_rir, 'r') as f:
                    rir_actual = np.array(f['sim_rir']).T
            except:
                mat = scipy.io.loadmat(ruta_rir)
                rir_actual = mat['sim_rir']
           
            x_convolved = sig.fftconvolve(s[:, None], rir_actual.T, axes=0)[:len(s), :]
            
            convolved_sources_raw.append(x_convolved)
            active_dirs.append([row['rir_elevation'], row['rir_azimuth']])
            instrument_names.append(row['source_class'])

        # =================================================================
        # FASE 2: SUMAR PARA OBTENER LA MEZCLA Y NORMALIZAR
        # =================================================================
        x_array_mezcla = sum(convolved_sources_raw)
        
        max_amp = np.max(np.abs(x_array_mezcla))
        if max_amp > 1.0:
            factor_norm = max_amp / 0.99
            x_array_mezcla = x_array_mezcla / factor_norm
            convolved_sources_norm = [src / factor_norm for src in convolved_sources_raw]
        else:
            convolved_sources_norm = convolved_sources_raw

        # =================================================================
        # FASE 3: BEAMFORMING DE FUENTES AISLADAS (REFERENCIAS TARGET)
        # =================================================================
        fuentes_beamformed = []
        for x_conv, ldir in zip(convolved_sources_norm, active_dirs):
            ldirs_np = np.array([ldir])
            y_beam = SH.MaxDirectivityBeamformer(
                x_conv.T, self.sample_rate, ldirs_np, self.sampling_points,
                order=self.order, r=self.r, win_len=self.win_len
            )
            fuentes_beamformed.append(y_beam.flatten())

        # =================================================================
        # FASE 4: BEAMFORMING DE LA MEZCLA Y RECOPILACIÓN DE DIRECCIONES
        # =================================================================
        entradas_demucs = []
        all_ldirs = [] # NUEVO: Guardamos todas las direcciones exactas de los 7 canales
        target_channels = 7
        
        # 4.1 Apuntar a fuentes activas
        for ldir in active_dirs:
            ldirs_np = np.array([ldir])
            all_ldirs.append(ldir) # Guardar dirección
            y_entrada_ruidosa = SH.MaxDirectivityBeamformer(
                x_array_mezcla.T, self.sample_rate, ldirs_np, self.sampling_points,
                order=self.order, r=self.r, win_len=self.win_len
            )
            entradas_demucs.append(y_entrada_ruidosa.flatten())

        # 4.2 Rellenar canales restantes
        if len(entradas_demucs) < target_channels:
            faltantes = target_channels - len(entradas_demucs)
            elevacion_base = np.pi / 2
            paso_rad = np.deg2rad(50) # Los 50 grados en radianes
            
            # Extraemos los azimuths que ya están ocupados (en radianes)
            usados_az = [d[1] for d in active_dirs]
            
            # Tomamos el azimuth de la última fuente como punto de partida
            if len(usados_az) > 0:
                current_az = usados_az[-1]
            else:
                current_az = 0.0 # Por seguridad, si una mezcla viniera sin fuentes
                
            libres_az = []
            
            # Buscamos 'faltantes' posiciones sumando 50º iterativamente
            while len(libres_az) < faltantes:
                # Sumamos 50º y aplicamos módulo 2*PI para dar la vuelta al círculo
                current_az = (current_az + paso_rad) % (2 * np.pi)
                
                # Comprobamos si este ángulo ya está siendo usado por una fuente real 
                # o si ya lo hemos añadido a la lista de libres (usando una pequeña tolerancia)
                is_used = any(np.isclose(current_az, u, atol=1e-2) for u in usados_az)
                is_in_libres = any(np.isclose(current_az, l, atol=1e-2) for l in libres_az)
                
                if not is_used and not is_in_libres:
                    libres_az.append(current_az)
            
            # Generamos los beamformers para estos canales extra encontrados
            for az_extra in libres_az:
                dir_extra = [elevacion_base, az_extra]
                all_ldirs.append(dir_extra) # Guardamos la dirección exacta para los SH
                
                ldirs_extra = np.array([dir_extra])
                y_extra = SH.MaxDirectivityBeamformer(
                    x_array_mezcla.T, self.sample_rate, ldirs_extra, self.sampling_points,
                    order=self.order, r=self.r, win_len=self.win_len
                )
                entradas_demucs.append(y_extra.flatten())

        # =================================================================
        # FASE 5: EMPAQUETADO Y GENERACIÓN DE EMBEDDINGS SH
        # =================================================================
        mix_np = np.stack(entradas_demucs, axis=0) 
        refs_np = np.stack(fuentes_beamformed, axis=0)
        
        # --- NUEVO: CÁLCULO DE LOS SH-EMBEDDINGS ---
        all_ldirs_np = np.array(all_ldirs) # Forma (7, 2)
        # Obtenemos matriz Y de tamaño (7, 16) complejos (para order=3)
        Y_complex = SH.SHMatrix(all_ldirs_np, order=self.order) 
        # Concatenamos la parte Real e Imaginaria -> Tamaño final: (7, 32)
        sh_features_np = np.concatenate([np.real(Y_complex), np.imag(Y_complex)], axis=-1)
        # -------------------------------------------

        if refs_np.shape[0] < target_channels:
            pad_val = target_channels - refs_np.shape[0]
            refs_np = np.pad(refs_np, ((0, pad_val), (0, 0)))

        if len(instrument_names) < target_channels:
            instrument_names += [''] * (target_channels - len(instrument_names))
            
        return torch.from_numpy(mix_np).float(), \
               torch.from_numpy(refs_np).float(), \
               torch.from_numpy(sh_features_np).float(), \
               instrument_names,x_array_mezcla