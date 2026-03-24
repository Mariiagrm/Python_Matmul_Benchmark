import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------------
# 1. CONFIGURACIÓN
# ---------------------------------------------------------
archivos_csv = [
    "rtx4090_benchmark_jit_specific.csv",
    "rtx4090_pytorch_eager_specific.csv",
    "rtx4090_torch_aoti_benchmark.csv",
    "rtx4090_benchmark_cuda.csv"
]

col_tflops = "TFLOPS"

# ---------------------------------------------------------
# 2. PROCESAMIENTO DE DATOS
# ---------------------------------------------------------
dataframes_list = []

print("📂 Leyendo archivos...")

for archivo in archivos_csv:
    if os.path.exists(archivo):
        try:
            df = pd.read_csv(archivo)
            df.columns = df.columns.str.strip()
            
            required_cols = ['Type', 'M', 'N', 'K', col_tflops]
            if not all(col in df.columns for col in required_cols):
                continue

            df['Type'] = df['Type'].astype(str).str.strip()
            df['Config'] = df.apply(lambda row: f"{int(row['M'])}x{int(row['N'])}x{int(row['K'])}", axis=1)
            df['Size_Sort'] = df['M'] * df['N'] * df['K']
            
            nombre_origen = archivo.replace(".csv", "").replace("rtx4090_", "")
            df['Origen'] = nombre_origen
            
            dataframes_list.append(df)
            
        except Exception as e:
            print(f"❌ Error leyendo {archivo}: {e}")
    else:
        print(f"⚠️  ARCHIVO NO ENCONTRADO: '{archivo}' no existe en esta carpeta.")

if not dataframes_list:
    print("❌ No se cargaron datos. Revisa los archivos.")
    exit()

df_final = pd.concat(dataframes_list)

# --- ¡LA SOLUCIÓN! TRADUCIR/UNIFICAR LAS ETIQUETAS ---
df_final['Type'] = df_final['Type'].replace({
    'Cuadrada': 'Square',
    'K_Fijo': 'Fixed_K'
})

print("\n--- RESUMEN DE DATOS CARGADOS ---")
print("Orígenes encontrados:", df_final['Origen'].unique())
print("Tipos unificados:", df_final['Type'].unique())
print("---------------------------------\n")

# ---------------------------------------------------------
# 3. DEFINICIÓN DE FUNCIÓN DE GRAFICADO
# ---------------------------------------------------------
def generar_grafica(df_input, titulo, nombre_archivo):
    if df_input.empty:
        print(f"⚠️  No hay datos para generar: {titulo}")
        return

    df_sorted = df_input.sort_values(by=['Size_Sort', 'Origen'])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))

    chart = sns.barplot(
        data=df_sorted,
        x='Config',
        y=col_tflops,
        hue='Origen',
        palette='viridis',
        edgecolor='black',
        errorbar=None
    )

    plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Configuración (M x N x K)', fontsize=12, fontweight='bold')
    plt.ylabel('TFLOPS (Más alto es mejor)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Benchmark / Modo', loc='upper left', bbox_to_anchor=(1, 1))
    
    for container in chart.containers:
        chart.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=300)
    print(f"✅ Gráfica guardada: {nombre_archivo}")
    plt.show() 
    plt.close() 

# ---------------------------------------------------------
# 4. GENERACIÓN DE GRÁFICAS (SEPARADAS POR TYPE)
# ---------------------------------------------------------

# --- GRÁFICA 1: Fixed_K ---
print("\n📊 Generando Gráfica 1: Fixed_K...")
df_fixed = df_final[df_final['Type'] == 'Fixed_K'].copy()
generar_grafica(df_fixed, 
                "Rendimiento con Dimensión K Fija (8192)", 
                "benchmark_fixed_k.png")

# --- GRÁFICA 2: Square ---
print("\n📊 Generando Gráfica 2: Square...")
df_square = df_final[df_final['Type'] == 'Square'].copy()
generar_grafica(df_square, 
                "Rendimiento con Matrices Cuadradas (Escalado Completo)", 
                "benchmark_square.png")