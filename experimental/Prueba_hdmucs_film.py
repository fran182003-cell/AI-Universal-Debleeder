import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import torch

def analizar_pesos_film(ruta_ckpt, nombre_descriptivo):
    # Cargamos el checkpoint directamente a la CPU para no gastar VRAM
    checkpoint = torch.load(ruta_ckpt, map_location='cpu', weights_only=False)
    state_dict = checkpoint['state_dict']
    
    # Buscamos los tensores de nuestras capas FiLM
    try:
        pesos_t = state_dict['model.proj_t.weight']
        pesos_f = state_dict['model.proj_f.weight']
    except KeyError:
        print(f"Error: No se encontraron los pesos de FiLM en {nombre_descriptivo}.")
        return

    print(f"=== {nombre_descriptivo} ===")
    
    # Para saber si han aprendido, miramos la Desviación Estándar (std) y el valor máximo absoluto.
    # En la inicialización, la std debería ser ~0.01 y el MaxAbs muy bajito.
    print(f"Rama TIEMPO (proj_t):")
    print(f"  - Desv. Estándar: {pesos_t.std().item():.5f}")
    print(f"  - Max Absoluto:   {pesos_t.abs().max().item():.5f}")
    print(f"  - Media:          {pesos_t.mean().item():.5f}")
    
    print(f"Rama FRECUENCIA (proj_f):")
    print(f"  - Desv. Estándar: {pesos_f.std().item():.5f}")
    print(f"  - Max Absoluto:   {pesos_f.abs().max().item():.5f}\n")


# SUSTITUYE ESTAS RUTAS por los archivos reales que tengas en lightning_logs/

ckpt_avanzado = os.environ.get("CKPT_PATH") or sys.exit("Define la variable de entorno CKPT_PATH con la ruta al checkpoint .ckpt")

print("Comprobando el aprendizaje de los Armónicos Esféricos...\n")

analizar_pesos_film(ckpt_avanzado, "ÉPOCA AVANZADA")