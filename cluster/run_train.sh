#!/bin/bash
# Entrenamiento en un clúster con Slurm. Lanzar desde la raíz del repositorio:
#   sbatch -J train -o ./logs/log_%j.out -e ./logs/log_%j.err --gres=gpu:1 cluster/run_train.sh base
# El argumento elige el modelo: base | spatial | film

MODEL="${1:-base}"
srun python -u train.py --model "$MODEL"
