import os
import sys
import pathlib

# Rutas del repositorio: los módulos comunes están en src/ y la configuración en configs/
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
DEFAULT_CONF = os.environ.get("CONF_PATH", str(REPO_ROOT / "configs" / "conf.yaml"))

import torch
import lightning as pl
import yaml
import pathlib
from MyDataset_3_RoPE import AutoSourceDataset
from LightingModel3_RoPE import LightningModel
from Lighting_model_htdmucs import PerfectedFiLMHTDemucs
from lightning.pytorch.loggers import TensorBoardLogger
import bitsandbytes as bnb
import numpy as np 
from lightning.pytorch.callbacks import ModelCheckpoint

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

def main():
    torch.set_float32_matmul_precision('medium')
    
    with open(DEFAULT_CONF,'r') as f:
        conf = yaml.safe_load(f)
        
    mic_az = [0, 0.0306, 2.096, 4.1898, 4.8512, 5.6267, 0.6612, 1.4362, 2.7555, 3.5326, 3.8028, 4.5778, 5.8971, 0.391, 1.7075, 2.4851, 3.111, 5.2376, 1.0482]
    mic_el = [0, 0.7308, 0.7307, 0.7299, 1.2319, 1.2316, 1.2319, 1.2317, 1.2316, 1.2315, 1.9097, 1.9099, 1.91, 1.9101, 1.9098, 1.9099, 2.4108, 2.4109, 2.4117]
    sampling_points = np.column_stack((mic_el, mic_az))

    train_dataset = AutoSourceDataset(
        csv_mixtures=conf['dataset']['train_csv'], 
        rir_root_dir=conf['dataset']['rir_root_dir'],
        chunk_duration=conf['dataset']['chunk_duration'],
        sample_rate=conf['dataset']['sample_rate'],
        sampling_points=sampling_points
    )
    print(f'\nTrain set length: {len(train_dataset)}')
    
    val_datasets = [
        AutoSourceDataset(
            csv_mixtures=conf['dataset']['val_estandar_csv'],
            rir_root_dir = conf['dataset']['rir_root_dir'],
            chunk_duration = conf['dataset']['chunk_duration'],
            sample_rate = conf['hdemucs_conf']['samplerate'],
            sampling_points = sampling_points,
        ),
        AutoSourceDataset(
            csv_mixtures=conf['dataset']['val_poco_reverb_csv'],
            rir_root_dir = conf['dataset']['rir_root_dir'],
            chunk_duration = conf['dataset']['chunk_duration'],
            sample_rate = conf['hdemucs_conf']['samplerate'],
            sampling_points = sampling_points,
        ),
        AutoSourceDataset(
            csv_mixtures=conf['dataset']['val_reverb_csv'],
            rir_root_dir = conf['dataset']['rir_root_dir'],
            chunk_duration = conf['dataset']['chunk_duration'],
            sample_rate = conf['hdemucs_conf']['samplerate'],
            sampling_points = sampling_points,
        )
    ]

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = conf['dataset']['batch_size'],
        shuffle = True,
        pin_memory = True,
        num_workers= conf['dataset']['num_workers']
    )

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset,
        batch_size = conf['dataset']['batch_size'],
        shuffle = False,
        pin_memory = True,
        num_workers = conf['dataset']['num_workers']
    ) for val_dataset in val_datasets]


    C = 7
    confdemucs = {
        'sources': [str(i) for i in range(C)],
        'audio_channels': C, 
        'samplerate': conf['hdemucs_conf']['samplerate'],
        'segment': conf['dataset']['chunk_duration'],
        # Puedes mantener 'channels' si lo tienes en tu YAML (por defecto en HTDemucs suele ser 48)
        'channels': conf['hdemucs_conf'].get('channels', 48),
        't_layers': 5,       
        'depth': 4,  # en el original ponía '4, 8' (edición a medias); se deja 4
    }
    
    # -------------------------------------------------------------
    # INSTANCIAMOS EL MODELO RIGUROSO FiLM (AHORA HTDEMUCS)
    # -------------------------------------------------------------
    print("Instanciando HTDemucs condicionado con FiLM antes del Cross-Transformer...")
    model = PerfectedFiLMHTDemucs(confdemucs=confdemucs, sh_dim=32)

    # -------------------------------------------------------------
    # 2. CONFIGURAMOS EL OPTIMIZADOR GLOBAL
    # -------------------------------------------------------------
    # El optimizador Adam8bit ahora registrará tanto los parámetros 
    # de HDemucs como los de nuestras nuevas capas nn.Linear
    # -------------------------------------------------------------
    # 2. CONFIGURAMOS EL OPTIMIZADOR GLOBAL (PLAN B: LR Diferencial)
    # -------------------------------------------------------------
    print("Configurando optimizador con Tasas de Aprendizaje Diferenciales...")
    
    film_params = []
    base_hdemucs_params = []
    
    for name, param in model.named_parameters():
        if 'proj_t' in name or 'proj_f' in name:
            film_params.append(param)
        else:
            base_hdemucs_params.append(param)
            
    # Obtenemos el Learning Rate base de tu archivo YAML
    base_lr = conf['optimizer_conf'].get('lr')
    
    # ¡LA MAGIA AQUÍ!: Le damos a FiLM una tasa x20 veces mayor
    film_lr = base_lr * 20.0 
    
    # Copiamos los argumentos extra del optimizador, pero quitamos el 'lr' general
    opt_kwargs = {k: v for k, v in conf['optimizer_conf'].items() if k != 'lr'}
    
    optimizer_grouped_parameters = [
        {"params": base_hdemucs_params, "lr": base_lr ,"weight_decay": conf['optimizer_conf'].get('weight_decay')},
        {"params": film_params, "lr": film_lr,"weight_decay": 0.0}  # sin decay en FiLM, no queremos frenarlo
    ]
    
    optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, **opt_kwargs)
    
    
    
    exp_name = conf['train']['exp_name']
    tb_logger = TensorBoardLogger('lightning_logs/', name=exp_name)

    best_checkpoint_callback = ModelCheckpoint(
        monitor = 'val_estandar_sdr_FINAL',
        save_top_k = 1,
        mode = 'max',
        filename = 'best_{epoch:02d}'
    )

    every_5_epochs_checkpoint_callback = ModelCheckpoint(
        every_n_epochs = 10,
        save_top_k = -1,
        filename = 'checkpoint_{epoch:02d}'
    )
    
    trainer = pl.Trainer(
        max_epochs = conf['train']['num_epochs'],
        enable_checkpointing = True,
        log_every_n_steps = 1,
        accelerator = "gpu",
        devices = 1,
                     
        logger = tb_logger,
        callbacks = [best_checkpoint_callback, every_5_epochs_checkpoint_callback],
        gradient_clip_val = 5.0,
       
    )

    lightning_model = LightningModel(
        model=model,         # Tu instancia de PerfectedFiLMHTDemucs
        optimizer=optimizer, # Tu optimizador bnb.optim.Adam8bit con LR diferencial
        conf=conf            # El diccionario YAML cargado
    )
    
    # 3. Llamamos a fit SIN ckpt_path
    trainer.fit(lightning_model, train_loader, val_loaders)

if __name__ == '__main__':
    main()