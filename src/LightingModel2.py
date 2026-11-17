import torch
import lightning as pl
from torchmetrics.audio import SignalDistortionRatio
import torch.nn.functional as F
from LossFunction import perceptual_loss
from lightning.pytorch.utilities import grad_norm
import auraloss

class LightningModel(pl.LightningModule):
    def __init__(self, model, optimizer, conf):
        super(LightningModel, self).__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = optimizer
        self.conf = conf

        self.shuller = perceptual_loss
        self.sdr = SignalDistortionRatio()
        self.auralossnew = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel",
            n_bins=128,
            sample_rate=model.samplerate,
            perceptual_weighting=True,
        )

    def _compute_masked_sdr(self, preds, target):
        # 1. CAMBIO CLAVE: Usar RMS en lugar de suma absoluta
        # Calculamos la raíz cuadrada de la media de los cuadrados a lo largo del tiempo (dim=-1)
        rms_target = torch.sqrt(torch.mean(target ** 2, dim=-1))
        
        # 2. UMBRAL REALISTA: 1e-4 (aprox -80 dB). Ignora ruido de fondo y colas de reverb.
        valid_mask = (rms_target > 1e-4) 
        
        # Si en todo el batch no hay ni un solo canal con audio útil, devolvemos 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
        valid_preds = preds[valid_mask]
        valid_target = target[valid_mask]
        
        # 3. Calculamos el SDR normal (ya no necesitas sumar epsilon si usas una buena métrica)
        # Ojo: asegúrate de que self.sdr esté configurado preferiblemente como SI-SDR
        try:
            return self.sdr(valid_preds, valid_target)
        except torch.linalg.LinAlgError:
            # Red de seguridad matemática
            return self.sdr(valid_preds, valid_target + 1e-6)
            
    def forward(self, batched_mixtures):
        if batched_mixtures.shape[1] > batched_mixtures.shape[2]:
            batched_mixtures = batched_mixtures.permute(0, 2, 1)
            
        estimated_sources = self.model(batched_mixtures)
        
        # Ajuste para que la salida de HDemucs encaje con tus referencias 3D
        if estimated_sources.dim() == 4 and estimated_sources.shape[1] == 1:
            estimated_sources = estimated_sources.squeeze(1)
        
        return estimated_sources

    def training_step(self, batch, batch_idx):
        mix, refs, instrument_names = batch

        permute_sources = self.conf.get('train').get('permute_sources')
        name_loss = self.conf.get('train').get('loss_function')

        if permute_sources:
            B, S = mix.shape[0:2]
            for i in range(B):
                perm = torch.randperm(S)
                mix[i] = mix[i, perm]
                refs[i] = refs[i, perm]

        estimated_sources = self(mix)
        assert not torch.isnan(estimated_sources).any()

        # Pérdidas originales sobre el tensor COMPLETO (incluyendo los ceros del padding)
        if name_loss == 'l1':
            loss = F.l1_loss(estimated_sources, refs, reduction='none').mean()
            self.log('train_L1_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if name_loss == 'auraloss':
            estimated_sources = estimated_sources.contiguous()
            refs = refs.contiguous()
            loss = self.auralossnew(estimated_sources, refs)
            self.log('train_AuralossNEW_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if name_loss == 'perceptual':
            loss = self.shuller(estimated_sources, refs, self.model.samplerate)
            self.log('train_Perceptual_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if name_loss == 'sdr': 
            # CAMBIO 1: Usar la función enmascarada para la función de pérdida SDR
            loss = -self._compute_masked_sdr(estimated_sources, refs)
            self.log('train_sdr_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

        # CAMBIO 2: Usar la función enmascarada para registrar la métrica SDR de entrenamiento
        train_sdr = self._compute_masked_sdr(estimated_sources, refs)
        self.log('train_sdr', train_sdr, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        nombres_validacion = [
            'val_estandar',
            'val_poco_reverberantes',
            'val_reverberantes'
        ]
        
        if dataloader_idx < len(nombres_validacion):
            base_name = nombres_validacion[dataloader_idx]
        else:
            base_name = f'val_dataset_extra_{dataloader_idx}'

        log_label_sdr = f'{base_name}_sdr'

        mix, refs, _ = batch
        assert not torch.isnan(mix).any()

        

        estimated_sources = self(mix)

        # CAMBIO 3: Usar la función enmascarada para registrar la métrica SDR de validación
        val_sdr = self._compute_masked_sdr(estimated_sources, refs)
            
        # Añadido add_dataloader_idx=False para limpieza en Tensorboard
        self.log(log_label_sdr, val_sdr, prog_bar=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=mix.shape[0])

        return val_sdr

    def configure_optimizers(self):
        if self.optimizer:
            return self.optimizer
        else:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)