"""
Entrenamiento de los modelos del TFG con un único script.

Durante el TFG cada modelo se entrenaba con una copia del mismo script principal,
cambiando los imports del dataset y del LightningModule. Este archivo reúne esas
variantes sin cambiar el procedimiento: mismo dataset, configuración, optimizador
Adam de 8 bits, logger de TensorBoard, checkpoints y recorte de gradiente.

Modelos (--model):
  base     Modelo inicial: HDemucs sobre las 7 salidas del beamformer (LightingModel2).
  spatial  HDemucs con el dataset actualizado; la permutación de fuentes se activa con
           train.permute_sources en la configuración (Lighting_model_sinfilm).
  film     HDemucs condicionado con FiLM a partir de los coeficientes en armónicos
           esféricos de las 7 direcciones (LightingModel3_RoPE.PerfectedFiLMHDemucs).
           Las capas FiLM usan un learning rate 20 veces mayor y sin weight decay.

Uso (desde la raíz del repositorio):
  python train.py --model base
  python train.py --model film --config configs/conf.yaml
"""
import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch
import yaml
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from demucs.hdemucs import HDemucs

from MyDataset_3_RoPE import AutoSourceDataset

NUM_SOURCES = 7  # 7 direcciones del beamformer = 7 canales de entrada y 7 fuentes de salida

# Posiciones de las 19 cápsulas del micrófono esférico Zylia ZM-1 (radianes)
MIC_AZ = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326,
          3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
MIC_EL = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315,
          1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
SAMPLING_POINTS = np.column_stack((MIC_EL, MIC_AZ))


class WithoutSH(torch.utils.data.Dataset):
    """Adapta el dataset al modelo inicial, que no recibe los coeficientes SH."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mix, refs, _sh_features, instrument_names = self.dataset[idx]
        return mix, refs, instrument_names


def build_datasets(conf):
    d = conf['dataset']
    common = dict(rir_root_dir=d['rir_root_dir'], chunk_duration=d['chunk_duration'],
                  sampling_points=SAMPLING_POINTS)
    train = AutoSourceDataset(csv_mixtures=d['train_csv'], sample_rate=d['sample_rate'], **common)
    val = [AutoSourceDataset(csv_mixtures=d[key], sample_rate=conf['hdemucs_conf']['samplerate'], **common)
           for key in ('val_estandar_csv', 'val_poco_reverb_csv', 'val_reverb_csv')]
    return train, val


def demucs_config(conf):
    return {
        'sources': [str(i) for i in range(NUM_SOURCES)],
        'audio_channels': NUM_SOURCES,
        'cac': conf['hdemucs_conf']['cac'],
        'samplerate': conf['hdemucs_conf']['samplerate'],
        'channels': conf['hdemucs_conf']['channels'],
        'segment': conf['dataset']['chunk_duration'],
    }


def make_optimizer(param_groups, opt_kwargs, kind):
    if kind == 'adam8bit':
        import bitsandbytes as bnb  # Adam de 8 bits para ahorrar memoria de GPU
        return bnb.optim.Adam8bit(param_groups, **opt_kwargs)
    return torch.optim.Adam(param_groups, **opt_kwargs)


def build_model(name, conf, optimizer_kind):
    """Devuelve el LightningModule, la métrica que se vigila y el nombre del experimento."""
    confdemucs = demucs_config(conf)
    tr = conf['train']
    suffix = (f"_loss={tr['loss_function']}_channels={confdemucs['channels']}"
              f"_perm={tr['permute_sources']}_cac={conf['hdemucs_conf']['cac']}")

    if name == 'base':
        from LightingModel2 import LightningModel
        model = HDemucs(**confdemucs)
        optimizer = make_optimizer(model.parameters(), conf['optimizer_conf'], optimizer_kind)
        return (LightningModel(model=model, optimizer=optimizer, conf=conf),
                'val_estandar_sdr', tr['exp_name'] + suffix)

    if name == 'spatial':
        from Lighting_model_sinfilm import LightningModel
        model = HDemucs(**confdemucs)
        optimizer = make_optimizer(model.parameters(), conf['optimizer_conf'], optimizer_kind)
        return (LightningModel(model=model, optimizer=optimizer, conf=conf),
                'val_estandar_sdr_FINAL', tr['exp_name'] + suffix + '_SH_DSM')

    if name == 'film':
        from LightingModel3_RoPE import LightningModel, PerfectedFiLMHDemucs
        model = PerfectedFiLMHDemucs(confdemucs=confdemucs, sh_dim=32)
        film_params, base_params = [], []
        for pname, param in model.named_parameters():
            (film_params if ('proj_t' in pname or 'proj_f' in pname) else base_params).append(param)
        base_lr = conf['optimizer_conf']['lr']
        opt_kwargs = {k: v for k, v in conf['optimizer_conf'].items() if k != 'lr'}
        groups = [
            {"params": base_params, "lr": base_lr, "weight_decay": conf['optimizer_conf'].get('weight_decay')},
            # Learning rate x20 y sin weight decay para que las capas FiLM no se apaguen
            {"params": film_params, "lr": base_lr * 20.0, "weight_decay": 0.0},
        ]
        optimizer = make_optimizer(groups, opt_kwargs, optimizer_kind)
        return (LightningModel(model=model, optimizer=optimizer, conf=conf),
                'val_estandar_sdr_FINAL', tr['exp_name'])

    raise ValueError(f"Modelo desconocido: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=['base', 'spatial', 'film'], required=True)
    parser.add_argument('--config', default=str(REPO_ROOT / 'configs' / 'conf.yaml'))
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--optimizer', choices=['adam8bit', 'adam'], default='adam8bit',
                        help="adam8bit (bitsandbytes, el usado en el TFG) o adam estándar")
    parser.add_argument('--fast-dev-run', action='store_true', help="Ejecuta un único paso para comprobar el montaje")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)

    train_ds, val_ds = build_datasets(conf)
    if args.model == 'base':
        train_ds, val_ds = WithoutSH(train_ds), [WithoutSH(v) for v in val_ds]
    print(f"\nTrain set length: {len(train_ds)}")
    for i, v in enumerate(val_ds):
        print(f" - Validation set {i} length: {len(v)}")

    loader_kwargs = dict(batch_size=conf['dataset']['batch_size'], pin_memory=True,
                         num_workers=conf['dataset']['num_workers'])
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loaders = [torch.utils.data.DataLoader(v, shuffle=False, **loader_kwargs) for v in val_ds]

    lightning_model, monitor, exp_name = build_model(args.model, conf, args.optimizer)
    print(f"Running experiment {exp_name}")

    callbacks = [
        ModelCheckpoint(monitor=monitor, save_top_k=1, mode='max', filename='best_{epoch:02d}'),
        ModelCheckpoint(every_n_epochs=10, save_top_k=-1, filename='checkpoint_{epoch:02d}'),
    ]
    trainer = pl.Trainer(
        max_epochs=conf['train']['num_epochs'],
        enable_checkpointing=True,
        log_every_n_steps=1,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=TensorBoardLogger('lightning_logs/', name=exp_name),
        callbacks=callbacks,
        gradient_clip_val=5.0,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(lightning_model, train_loader, val_loaders)


if __name__ == '__main__':
    main()
