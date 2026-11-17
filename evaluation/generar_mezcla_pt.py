import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import argparse
import torch as th
import yaml
import numpy as np
from MyDataset_2 import AutoSourceDataset

def get_parser():
    parser = argparse.ArgumentParser(description="Generador de archivo .pt con la mezcla espacializada")
    parser.add_argument("--csv", type=str, required=True, help="Ruta al CSV de 60s original")
    parser.add_argument("--conf", type=str, default=DEFAULT_CONF, help="Ruta al archivo de configuración")
    parser.add_argument("--out", type=str, default="mezcla_precargada.pt", help="Nombre del archivo .pt de salida")
    return parser

def main():
    args = get_parser().parse_args()

    print(f"[*] Cargando configuración desde: {args.conf}")
    with open(args.conf, 'r') as f:
        conf = yaml.safe_load(f)

    sample_rate = conf['hdemucs_conf']['samplerate']
    rir_root_dir = conf['dataset']['rir_root_dir']

    # Coordenadas de los 19 micrófonos del Zylia
    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))

    print(f"[*] Instanciando AutoSourceDataset con el CSV: {args.csv}")
    dataset = AutoSourceDataset(
        csv_mixtures=args.csv,
        rir_root_dir=rir_root_dir,
        chunk_duration=60.0,
        sample_rate=sample_rate,
        sampling_points=sampling_points,
    )

    print("[*] Generando la mezcla espacial (esto puede tardar unos segundos)...")
    # Obtenemos el primer elemento del dataset (que abarca los 60s completos)
    mezcla_cruda_np, referencias, active_dirs = dataset[0]

    print(f"[*] ¡Mezcla generada! Dimensiones:")
    print(f"    - Mezcla cruda: {mezcla_cruda_np.shape}")
    print(f"    - Referencias : {referencias.shape}")
    print(f"    - Direcciones activas: {len(active_dirs)}")

    # Empaquetamos todo en un diccionario
    datos_a_guardar = {
        'mezcla_cruda_np': mezcla_cruda_np,
        'referencias': referencias,
        'active_dirs': active_dirs
    }

    print(f"[*] Guardando datos en disco: {args.out} ...")
    th.save(datos_a_guardar, args.out)
    print(f"[+] Proceso completado con éxito. Ya puedes usar {args.out} en tu script de evaluación de latencia.")

if __name__ == "__main__":
    main()