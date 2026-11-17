import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.audio import SignalDistortionRatio
import bitsandbytes as bnb
from demucs.hdemucs import HDemucs  # O HTDemucs si usas la v4
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.io
import scipy.signal as sig
import h5py
import auraloss

# Tus librerías personalizadas
import SH 
from LossFunction import perceptual_loss
from MyDataset_3_RoPE import AutoSourceDataset

# ==============================================================================
# MODELO LIGHTNING BASELINE (SIN FiLM)
# ==============================================================================
class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, conf):
        super(LightningModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.optimizer = optimizer
        self.conf = conf
        
        self.shuller = perceptual_loss
        self.sdr = SignalDistortionRatio()
        sr = conf['hdemucs_conf']['samplerate']
        self.auralossnew = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=sr,
            perceptual_weighting=True,
        )

    def _compute_masked_sdr(self, preds, target):
        rms_target = torch.sqrt(torch.mean(target ** 2, dim=-1))
        valid_mask = (rms_target > 1e-4) 
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        valid_preds = preds[valid_mask]
        valid_target = target[valid_mask]
        
        try:
            return self.sdr(valid_preds, valid_target)
        except torch.linalg.LinAlgError:
            return self.sdr(valid_preds, valid_target + 1e-6)
            
    def forward(self, batched_mixtures):
        # Corrección de dimensiones si es necesario
        if batched_mixtures.shape[1] > batched_mixtures.shape[2]:
            mix_in = batched_mixtures.permute(0, 2, 1)
        else:
            mix_in = batched_mixtures
            
        # Llamamos al modelo base (SOLO procesa el audio de entrada)
        estimated_sources = self.model(mix_in)

        # Ajuste de dimensiones de salida
        if estimated_sources.dim() == 4 and estimated_sources.shape[1] == 1:
            estimated_sources = estimated_sources.squeeze(1)
            
        return estimated_sources
        
    def training_step(self, batch, batch_idx):
        # El dataloader sigue devolviendo sh_features, pero simplemente las ignoramos
        mix, refs, sh_features, instrument_names = batch
        permute_sources = self.conf.get('train').get('permute_sources')
        name_loss = self.conf.get('train').get('loss_function')

        # Lógica de permutación
        if permute_sources:
            B, S = mix.shape[0:2]
            mix_list, refs_list = [], []
            for i in range(B):
                perm = torch.randperm(S)
                mix_list.append(mix[i, perm])
                refs_list.append(refs[i, perm])
                
            mix = torch.stack(mix_list)
            refs = torch.stack(refs_list)

        # Pasamos SOLO el audio al forward
        estimated_sources = self(mix)

        # Cálculo de la pérdida
        loss = 0.0
        if name_loss == 'l1':
            loss = F.l1_loss(estimated_sources, refs, reduction='none').mean()
            self.log('train_L1_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        elif name_loss == 'auraloss':
            estimated_sources = estimated_sources.contiguous()
            refs = refs.contiguous()
            loss = self.auralossnew(estimated_sources, refs)
            self.log('train_AuralossNEW_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        elif name_loss == 'perceptual':
            sr = self.conf['hdemucs_conf']['samplerate']
            loss = self.shuller(estimated_sources, refs, sr)
            self.log('train_Perceptual_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        elif name_loss == 'sdr': 
            loss = -self._compute_masked_sdr(estimated_sources, refs)
            self.log('train_sdr_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        train_sdr = self._compute_masked_sdr(estimated_sources, refs)
        self.log('train_sdr', train_sdr, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Eliminados los logs de film_weight_norm que estaban aquí
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        nombres_validacion = ['val_estandar', 'val_poco_reverberantes', 'val_reverberantes']
        base_name = nombres_validacion[dataloader_idx] if dataloader_idx < len(nombres_validacion) else f'val_dataset_{dataloader_idx}'

        mix, refs, sh_features, _ = batch
        
        # Pasamos SOLO el audio
        estimated_sources = self(mix)
        
        val_sdr = self._compute_masked_sdr(estimated_sources, refs)
            
        self.log(f'{base_name}_sdr_FINAL', val_sdr, prog_bar=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=mix.shape[0])

        return val_sdr

    def configure_optimizers(self):
        if self.optimizer:
            return self.optimizer
        else:
            return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)