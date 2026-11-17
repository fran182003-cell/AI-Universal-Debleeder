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
from demucs.hdemucs import HDemucs
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
# ENVOLTORIO FiLM: INYECCIÓN RIGUROSA EN EL BOTTLENECK
# ==============================================================================
class PerfectedFiLMHDemucs(nn.Module):
    """
    HDemucs condicionado con FiLM. Usa un forward probing para detectar
    los canales latentes de forma segura. Aplana los SH para proveer
    un contexto espacial global al bottleneck mezclado de la CNN.
    """
    def __init__(self, confdemucs, sh_dim=32):
        super().__init__()
        self.hdemucs = HDemucs(**confdemucs)
        self.num_sources = confdemucs['audio_channels']  # 7
        self.sh_total_dim = self.num_sources * sh_dim    # 224

        self.t_layer = self.hdemucs.tencoder[-1]
        self.f_layer = self.hdemucs.encoder[-1]

        # 1. Obtenemos las dimensiones EXACTAS por separado
        self.C_t, self.C_f = self._probe_bottleneck_channels(confdemucs)
        print(f"[PerfectedFiLMHDemucs] Canales detectados -> Tiempo: {self.C_t}, Frecuencia: {self.C_f}")

        # 2. Proyecciones Lineales INDEPENDIENTES para cada rama
        self.proj_t = nn.Linear(self.sh_total_dim, self.C_t * 2, bias=False)
        self.proj_f = nn.Linear(self.sh_total_dim, self.C_f * 2, bias=False)

        nn.init.normal_(self.proj_t.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.proj_f.weight, mean=0.0, std=0.01)

        self._sh_for_hook = None

        self._hook_handles = [
            self.t_layer.register_forward_hook(self._make_hook('t')),
            self.f_layer.register_forward_hook(self._make_hook('f')),
        ]

    def _probe_bottleneck_channels(self, confdemucs):
        captured = {}
        def _capture_t(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured['C_t'] = t.shape[1]
        def _capture_f(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            captured['C_f'] = t.shape[1]

        h1 = self.t_layer.register_forward_hook(_capture_t)
        h2 = self.f_layer.register_forward_hook(_capture_f)

        C = confdemucs['audio_channels']
        sr = confdemucs['samplerate']
        seg = confdemucs.get('segment', 5.0)
        
        # PARCHE CUDA: Hacemos el probe en la GPU si está disponible 
        # para que no choque con los tensores internos de HDemucs
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.hdemucs.to(device)
        dummy = torch.zeros(1, C, int(sr * seg), device=device)
        
        with torch.no_grad():
            try:
                self.hdemucs(dummy)
            except Exception as e:
                # Es normal que falle si la red requiere datos reales para el decodificador,
                # pero nos da igual porque ya hemos capturado los canales en el encoder.
                pass

        h1.remove()
        h2.remove()
        
        # Devolvemos a CPU para que Lightning tome el control
        self.hdemucs.to('cpu')

        C_t = captured.get('C_t', None)
        C_f = captured.get('C_f', None)
        
        if C_t is None or C_f is None:
            raise RuntimeError(
                f"El probe no capturó los canales del bottleneck. "
                f"C_t={C_t}, C_f={C_f}. Revisa la estructura interna de HDemucs."
            )
        
        return C_t, C_f

    def _make_hook(self, branch: str):
        # Asignamos la proyección y el tamaño correctos según la rama
        proj = self.proj_t if branch == 't' else self.proj_f
        channels = self.C_t if branch == 't' else self.C_f

        def hook(module, inp, out):
            if self._sh_for_hook is None:
                return 

            sh_flat = self._sh_for_hook 
            is_tuple = isinstance(out, (tuple, list))
            x = out[0] if is_tuple else out

            params = proj(sh_flat)  # (B, 2 * channels)
            gamma = params[:, :channels] + 1.0 
            beta  = params[:, channels:]       

            if x.dim() == 3: # Tiempo: (B, C, T)
                gamma = gamma.unsqueeze(-1)
                beta  = beta.unsqueeze(-1)
            elif x.dim() == 4: # Frecuencia: (B, C, F, T)
                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta  = beta.unsqueeze(-1).unsqueeze(-1)

            x_out = (x * gamma) + beta

            return (x_out,) + out[1:] if is_tuple else x_out

        return hook

    def forward(self, mix, sh_features):
        B = mix.shape[0]
        
        assert sh_features.shape == (B, self.num_sources, 32), \
            f"Shape inesperada de sh_features: {sh_features.shape}, esperada ({B}, {self.num_sources}, 32)"
        
        if mix.shape[1] > mix.shape[2]:
            mix = mix.permute(0, 2, 1)

        self._sh_for_hook = sh_features.view(B, -1)
        out = self.hdemucs(mix)
        self._sh_for_hook = None
        return out

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()

# ==============================================================================
# MODELO LIGHTNING UNIFICADO Y LIMPIO
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
            
    def forward(self, batched_mixtures, sh_features):
        # Corrección de dimensiones si es necesario
        if batched_mixtures.shape[1] > batched_mixtures.shape[2]:
            mix_in = batched_mixtures.permute(0, 2, 1)
        else:
            mix_in = batched_mixtures
            
        # Llamamos al modelo FiLM (procesa audio y features simultáneamente)
        estimated_sources = self.model(mix_in, sh_features)

        # Ajuste de dimensiones de salida
        if estimated_sources.dim() == 4 and estimated_sources.shape[1] == 1:
            estimated_sources = estimated_sources.squeeze(1)
            
        return estimated_sources
        
    def training_step(self, batch, batch_idx):
        mix, refs, sh_features, instrument_names = batch
        permute_sources = self.conf.get('train').get('permute_sources')
        name_loss = self.conf.get('train').get('loss_function')

        # Lógica de permutación (se desactivará si en el YAML pones false)
        if permute_sources:
            B, S = mix.shape[0:2]
            mix_list, refs_list, sh_list = [], [], []
            for i in range(B):
                perm = torch.randperm(S)
                mix_list.append(mix[i, perm])
                refs_list.append(refs[i, perm])
                sh_list.append(sh_features[i, perm])
                
            mix = torch.stack(mix_list)
            refs = torch.stack(refs_list)
            sh_features = torch.stack(sh_list)

        # Pasamos audio y coordenadas
        estimated_sources = self(mix, sh_features)

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
        self.log('film_weight_norm_t', self.model.proj_t.weight.norm(), prog_bar=False, on_epoch=True)
        self.log('film_weight_norm_f', self.model.proj_f.weight.norm(), prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        nombres_validacion = ['val_estandar', 'val_poco_reverberantes', 'val_reverberantes']
        base_name = nombres_validacion[dataloader_idx] if dataloader_idx < len(nombres_validacion) else f'val_dataset_{dataloader_idx}'

        mix, refs, sh_features, _ = batch
        estimated_sources = self(mix, sh_features)
        
        val_sdr = self._compute_masked_sdr(estimated_sources, refs)
            
        self.log(f'{base_name}_sdr_FINAL', val_sdr, prog_bar=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=mix.shape[0])

        return val_sdr

    def configure_optimizers(self):
        # Usar el optimizador global (Adam8bit) para entrenar todo a la vez
        if self.optimizer:
            return self.optimizer
        else:
            return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def on_before_optimizer_step(self, optimizer):
        # Tu log original para TensorBoard
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

     
            


# ==============================================================================
# MAIN BUCLE DE ENTRENAMIENTO
# ==============================================================================
