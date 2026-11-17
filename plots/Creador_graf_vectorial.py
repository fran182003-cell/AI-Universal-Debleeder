import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. CARGA DE DATOS Y CONFIGURACIÓN
# ==========================================
df_def = pd.read_csv('detalles_BSS_EVAL_SDR_SIR_SAR_conservatorio_def.csv')
df_permut = pd.read_csv('detalles_BSS_EVAL_SDR_SIR_SAR_conservatorio_permut.csv')
df_film = pd.read_csv('detalles_BSS_EVAL_SDR_SIR_SAR_conservatorio_film.csv')
df_permut_false = pd.read_csv('detalles_BSS_EVAL_SDR_SIR_SAR_conservatorio_permut_false.csv')

metrics = ['SDR [dB]', 'SIR [dB]', 'SAR [dB]']

# Asegúrate de que los nombres coinciden con lo que vimos antes
col_metrics_bf = ['SDR_Beamformer (dB)', 'SIR_Beamformer (dB)', 'SAR_Beamformer (dB)'] 
col_metrics_net = ['SDR_HDemucs (dB)', 'SIR_HDemucs (dB)', 'SAR_HDemucs (dB)'] 

methods = ['Beamformer', 'Modelo Inicial', 'Modelo Permut', 'Modelo Film', 'Modelo Permut False']
data_sources = [df_def, df_def, df_permut, df_film, df_permut_false]

# ==========================================
# 2. CÁLCULO DE ESTADÍSTICAS (LIMPIANDO INF Y NAN)
# ==========================================
num_methods = len(methods)
num_metrics = len(metrics)

p50 = np.zeros((num_methods, num_metrics)) 
p25 = np.zeros((num_methods, num_metrics)) 
p75 = np.zeros((num_methods, num_metrics)) 

for i, (method, df) in enumerate(zip(methods, data_sources)):
    for j in range(num_metrics):
        if method == 'Beamformer':
            col_name = col_metrics_bf[j]
        else:
            col_name = col_metrics_net[j]
            
        # MAGIA AQUÍ: Reemplazamos los 'inf' y '-inf' por NaN, y luego tiramos todos los NaN.
        data = df[col_name].replace([np.inf, -np.inf], np.nan).dropna().values
        
        # Calculamos solo si quedaron datos válidos tras la limpieza
        if len(data) > 0:
            p50[i, j] = np.mean(data)
            p25[i, j] = np.percentile(data, 25)
            p75[i, j] = np.percentile(data, 75)
        else:
            # Si por algún motivo una columna entera era 'inf', la dejamos a 0
            p50[i, j] = p25[i, j] = p75[i, j] = 0

# ==========================================
# 3. CREACIÓN DE LA GRÁFICA
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(num_metrics) 
groupwidth = min(0.8, num_methods / (num_methods + 1.5))
bar_width = groupwidth / num_methods

# Solución al Warning de deprecación
colors = plt.colormaps['tab10'].colors

for i in range(num_methods):
    offset = (i - num_methods / 2 + 0.5) * bar_width
    x_pos = x + offset
    
    # Calculamos el error. Usamos np.maximum(0, ...) por si algún redondeo infinitesimal de Python diera negativo
    lower_error = np.maximum(0, p50[i, :] - p25[i, :])
    upper_error = np.maximum(0, p75[i, :] - p50[i, :])
    yerr = [lower_error, upper_error]
    
    ax.bar(x_pos, p50[i, :], bar_width, label=methods[i], color=colors[i], zorder=3)
    ax.errorbar(x_pos, p50[i, :], yerr=yerr, fmt='none', ecolor='k', capsize=0, zorder=4)

# ==========================================
# 4. ESTILOS FINALES Y EXPORTACIÓN
# ==========================================
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=15)
ax.set_ylim([-4, 20])

ax.grid(True, linestyle='-', linewidth=0.5, zorder=0)
ax.set_axisbelow(True) 
ax.tick_params(axis='y', labelsize=15)

ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig('res_percent.eps', format='eps')
plt.show()