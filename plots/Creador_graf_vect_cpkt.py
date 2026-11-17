import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ==========================================
# 1. CARGA Y FILTRADO DE DATOS
# ==========================================
# Cargar el archivo CSV
df = pd.read_csv('Spheres_HDEMUCS_loss=l1_channels=48_perm=True_cac=False_SH_DSM_version_28.csv')

# Filtrar los datos para quedarnos solo hasta el paso 10.000
df_filtered = df[df['Step'] <= 10000]

# ==========================================
# 2. CONFIGURACIÓN DE ESTILO PARA PAPER
# ==========================================
# Usar fuente serif (estilo clásico de artículos académicos y LaTeX)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# Crear la figura con una proporción clásica (ej. 8x5)
fig, ax = plt.subplots(figsize=(8, 5))

# Si mantienes df_filtered = df[df['Step'] >= 20000]
# Normaliza los pasos para que empiecen en 0
pasos_normalizados = df_filtered['Step'] - df_filtered['Step'].min()

# Y al dibujar la gráfica usas la variable normalizada
ax.plot(pasos_normalizados, df_filtered['Value'], 
        color='#003366', linewidth=1.5, label='Valor de la Métrica')

# Y luego limitas el eje X
ax.set_xlim(0, 10000)

# ==========================================
# 4. FORMATO DE EJES Y LIMPIEZA VISUAL
# ==========================================
# Etiquetas de los ejes (¡Cámbialas según sea Loss o SDR!)
ax.set_xlabel('Paso de Entrenamiento (Step)')
ax.set_ylabel('SDR (dB)') 

ax.set_title('Evolución del SDR en el conjunto de validación reverberante para el modelo con permutación', fontsize=14, pad=15)
# Añadir un grid muy sutil solo en el eje Y para guiar la vista sin molestar
ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

# Formatear el eje X para que muestre "2k, 4k..." en lugar de "2000, 4000"
formatter = ticker.EngFormatter()
ax.xaxis.set_major_formatter(formatter)

# Eliminar los bordes superior y derecho (Estilo APA / IEEE)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Si quieres poner una leyenda, la colocamos sin caja exterior (frameon=False)
# ax.legend(loc='best', frameon=False)

# Ajustar los márgenes automáticamente
plt.tight_layout()

# ==========================================
# 5. EXPORTACIÓN VECTORIAL
# ==========================================
# Guardar en EPS (ideal para compilar en LaTeX)
plt.savefig('evolucion_10k_perm_muy.eps', format='eps', bbox_inches='tight')



print("Gráficas vectoriales generadas con éxito: 'evolucion_10k_ini.eps' y 'evolucion_10k.pdf'")
plt.show()