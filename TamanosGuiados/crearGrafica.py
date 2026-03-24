import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuración inicial
archivos_csv = [
    "rtx4090_benchmark_jit_specific.csv",
    "rtx4090_pytorch_eager_specific.csv",
    "rtx4090_torch_aoti_benchmark.csv",
    "rtx4090_benchmark_cuda.csv" 
]

col_tflops = "TFLOPS"

# Diccionario para nombres legibles
nombres_metodos = {
    "rtx4090_benchmark_jit_specific.csv": "Torch JIT",
    "rtx4090_pytorch_eager_specific.csv": "PyTorch Eager",
    "rtx4090_torch_aoti_benchmark.csv": "Torch AOTI",
    "rtx4090_benchmark_cuda.csv": "CUDA Nativo"
}

dataframes = []

# Leer y procesar cada archivo
for archivo in archivos_csv:
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        
        # Crear la etiqueta de configuración (MxNxK)
        df['Config_M_N_K'] = df['M'].astype(str) + "x" + df['N'].astype(str) + "x" + df['K'].astype(str)
        
        # Extraer las columnas necesarias, INCLUYENDO 'Type'
        df_resumen = df[['Type', 'Config_M_N_K', col_tflops]].copy()
        df_resumen['Metodo'] = nombres_metodos[archivo]
        
        dataframes.append(df_resumen)
    else:
        print(f"⚠️ Advertencia: El archivo '{archivo}' no se encontró.")

if not dataframes:
    print("❌ No se pudo cargar ningún archivo. Verifica las rutas.")
else:
    # Unir todos los dataframes
    df_total = pd.concat(dataframes, ignore_index=True)

    # Identificar los modos disponibles en la columna 'Type' (ej. 'Cuadrada', 'K_Fijo')
    # Si tus nombres exactos son diferentes, el script se adaptará automáticamente.
    modos = df_total['Type'].unique()
    
    print(f"Modos detectados en los datos: {modos}\n")

    # Generar un gráfico por cada Modo
    for modo in modos:
        print(f"=== Procesando gráfico para el modo: {modo} ===")
        
        # Filtrar solo los datos de este modo
        df_modo = df_total[df_total['Type'] == modo]
        
        # Crear la tabla pivote para este modo específico
        df_pivot = df_modo.pivot_table(
            index='Config_M_N_K', 
            columns='Metodo', 
            values=col_tflops, 
            aggfunc='mean'
        )

        # Crear el gráfico
        plt.figure(figsize=(12, 6))
        
        # Generar las barras
        df_pivot.plot(kind='bar', figsize=(12, 6), width=0.8, colormap='viridis')
        
        plt.title(f'Rendimiento (TFLOPS) - RTX 4090 - Modo: {modo}', fontsize=16, pad=15)
        plt.xlabel('Tamaño de Matrices (M x N x K)', fontsize=12)
        plt.ylabel('Rendimiento (TFLOPS)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Método de Ejecución', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Guardar el gráfico con un nombre dinámico según el modo
        nombre_archivo_salida = f"comparativa_tflops_{modo.lower()}.png"
        plt.savefig(nombre_archivo_salida, dpi=300)
        print(f"✅ Gráfico guardado como '{nombre_archivo_salida}'.\n")
        
        # Mostrar el gráfico en pantalla
        plt.show()