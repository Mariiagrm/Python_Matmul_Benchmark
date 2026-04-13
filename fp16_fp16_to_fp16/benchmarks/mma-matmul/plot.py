import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar datos
try:
    df = pd.read_csv("benchmark_mma-matmul.csv")
except FileNotFoundError:
    print("❌ Archivo no encontrado. Asegúrate de que 'benchmark_mma-matmul.csv' existe.")
    exit()

# Convertir Kernel a string para que Seaborn lo trate como categoría (colores discretos)
df['Kernel'] = df['Kernel'].astype(str)

# Configurar el estilo visual (estética profesional y limpia)
sns.set_theme(style="whitegrid", context="talk")
palette = sns.color_palette("tab20", n_colors=df['Kernel'].nunique())

# Crear la figura con 3 subgráficos (1 fila, 3 columnas)
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Análisis de Rendimiento: Kernels MMA vs cuBLAS (RTX 4090)', fontsize=20, fontweight='bold', y=1.05)

# ==========================================
# Gráfica 1: TFLOPS vs Tamaño (Matrices Cuadradas)
# ==========================================
df_square = df[df['Type'] == 'Square'].copy()

sns.lineplot(
    data=df_square, x='M', y='TFLOPS', hue='Kernel', marker='o', 
    linewidth=2.5, markersize=8, palette=palette, ax=axes[0]
)
axes[0].set_title('Escalado en Matrices Cuadradas (M=N=K)', fontsize=16)
axes[0].set_xlabel('Dimensión de la Matriz (M)')
axes[0].set_ylabel('Rendimiento (TFLOPS)')
axes[0].set_xscale('log', base=2) # Escala logarítmica porque los tamaños saltan x2
axes[0].set_xticks(df_square['M'].unique())
axes[0].set_xticklabels(df_square['M'].unique(), rotation=45)

# ==========================================
# Gráfica 2: TFLOPS vs Tamaño (K Fijo = 8192)
# ==========================================
df_fixed = df[df['Type'] == 'Fixed_K'].copy()

sns.lineplot(
    data=df_fixed, x='M', y='TFLOPS', hue='Kernel', marker='s', 
    linewidth=2.5, markersize=8, palette=palette, ax=axes[1]
)
axes[1].set_title('Escalado con K Fijo (K=8192)', fontsize=16)
axes[1].set_xlabel('Dimensión (M=N)')
axes[1].set_ylabel('Rendimiento (TFLOPS)')
axes[1].set_xscale('log', base=2)
axes[1].set_xticks(df_fixed['M'].unique())
axes[1].set_xticklabels(df_fixed['M'].unique(), rotation=45)

# ==========================================
# Gráfica 3: TFLOPS Máximos Alcanzados por Kernel
# ==========================================
# Agrupamos por Kernel para encontrar su pico absoluto de TFLOPS
peak_tflops = df.groupby('Kernel')['TFLOPS'].max().reset_index()
peak_tflops = peak_tflops.sort_values(by='TFLOPS', ascending=False)

sns.barplot(
    data=peak_tflops, x='Kernel', y='TFLOPS', 
    palette="viridis", ax=axes[2], hue='Kernel', legend=False
)
axes[2].set_title('Pico Máximo de Rendimiento por Kernel', fontsize=16)
axes[2].set_xlabel('ID del Kernel')
axes[2].set_ylabel('TFLOPS Máximos')

# Añadir el valor de TFLOPS encima de cada barra
for p in axes[2].patches:
    axes[2].annotate(f'{p.get_height():.0f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=12, fontweight='bold', color='black', xytext=(0, 5), textcoords='offset points')

# Ajustar las leyendas
axes[0].legend(title='Kernel ID', fontsize=10, title_fontsize=12, loc='upper left')
axes[1].get_legend().remove() # Quitamos la leyenda repetida del centro

# Guardar y mostrar
plt.tight_layout()
plt.savefig('analisis_rendimiento_kernels.png', dpi=300, bbox_inches='tight')
print("✅ Gráficas generadas y guardadas como 'analisis_rendimiento_kernels.png'")
plt.show()