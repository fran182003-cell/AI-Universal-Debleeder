from demucs.htdemucs import HTDemucs
import torch
import torch.nn as nn

# ==============================================================================
# ENVOLTORIO FiLM: INYECCIÓN RIGUROSA EN EL BOTTLENECK DEL TRANSFORMER
# ==============================================================================
class PerfectedFiLMHTDemucs(nn.Module):
    """
    HTDemucs condicionado con FiLM. Usa un forward probing para detectar
    los canales latentes de forma segura. Inyecta el contexto espacial (SH) 
    justo a la salida de los encoders, antes de entrar al Cross-Transformer.
    """
    def __init__(self, confdemucs, sh_dim=32):
        super().__init__()
        
        # 1. Instanciamos el modelo HTDemucs en lugar de HDemucs
        self.htdemucs = HTDemucs(**confdemucs)
        self.num_sources = confdemucs['audio_channels']  # 7
        self.sh_total_dim = self.num_sources * sh_dim    # 224

        # En HTDemucs, tencoder y encoder son ModuleLists.
        # Su capa [-1] es la ideal para inyectar antes del CrossTransformer.
        self.t_layer = self.htdemucs.tencoder[-1]
        self.f_layer = self.htdemucs.encoder[-1]

        # 2. Obtenemos las dimensiones EXACTAS por separado
        self.C_t, self.C_f = self._probe_bottleneck_channels(confdemucs)
        print(f"[PerfectedFiLMHTDemucs] Canales detectados -> Tiempo: {self.C_t}, Frecuencia: {self.C_f}")

        # 3. Proyecciones Lineales INDEPENDIENTES para cada rama
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
        sr = confdemucs.get('samplerate', 44100)
        seg = confdemucs.get('segment', 5.0)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.htdemucs.to(device)
        
        # Tensor dummy para activar el forward pass
        dummy = torch.zeros(1, C, int(sr * seg), device=device)
        
        with torch.no_grad():
            try:
                self.htdemucs(dummy)
            except Exception as e:
                # El transformer fallará por la falta de estructura real, pero no importa, 
                # los encoders ya habrán guardado las dimensiones.
                pass

        h1.remove()
        h2.remove()
        
        self.htdemucs.to('cpu')

        C_t = captured.get('C_t', None)
        C_f = captured.get('C_f', None)
        
        if C_t is None or C_f is None:
            raise RuntimeError(
                f"El probe no capturó los canales. "
                f"C_t={C_t}, C_f={C_f}. Revisa la arquitectura HTDemucs."
            )
        
        return C_t, C_f

    def _make_hook(self, branch: str):
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
        # Pasamos por HTDemucs
        out = self.htdemucs(mix)
        self._sh_for_hook = None
        return out

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()