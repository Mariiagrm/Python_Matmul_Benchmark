import pandas as pd
import os
import glob
from pathlib import Path

# Obtener la ruta absoluta del directorio donde está este script
script_dir = Path(__file__).parent.resolve()

# Cambiar el directorio de trabajo actual a la carpeta del script
os.chdir(script_dir)

# 1. Obtener la lista de todos los archivos CSV en la carpeta actual
archivos_csv = [
    "../results/benchmark_mma-matmul.csv",
    "../results/rtx4090_benchmark_jit.csv",
"../results/rtx4090_pytorch_eager.csv",
"../results/rtx4090_torch_aoti_benchmark.csv",
"../results/rtx4090_benchmark_cuda.csv"]


if not archivos_csv:
    print("No se encontraron archivos CSV en el directorio.")
else:
    # 2. Leer cada archivo y almacenarlos en una lista de DataFrames
    lista_dfs = []
    for archivo in archivos_csv:
        try:
            df = pd.read_csv(archivo)
            # Opción A: Nombre del archivo sin extensión (ej: 'benchmark_fp16')
            nombre_limpio = os.path.splitext(os.path.basename(archivo))[0]
            
            df["Mode"] = nombre_limpio
            lista_dfs.append(df)
            print(f"Leído correctamente: {archivo}")
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")

    if lista_dfs:
        # 3. Concatenar todos los DataFrames en uno solo
        df_final = pd.concat(lista_dfs, ignore_index=True)

        # 4. Verificar si existe la columna TFLOPS y ordenar
        # Usamos case=False por si acaso está escrito en minúsculas
        columna_tflops = next((col for col in df_final.columns if col.upper() == 'TFLOPS'), None)

        if columna_tflops:
            # Convertir a numérico por si acaso hay datos en formato string
            df_final[columna_tflops] = pd.to_numeric(df_final[columna_tflops], errors='coerce')
            # Ordenar de mayor a menor
            df_final = df_final.sort_values(by=columna_tflops, ascending=False)
            print(f"Datos ordenados por {columna_tflops} de mayor a menor.")
        else:
            print("Aviso: No se encontró la columna 'TFLOPS'. El archivo se unirá sin ordenar.")

        # 5. Guardar el resultado en benchmarkCompleto.csv
        df_final.to_csv("../results/benchmarkCompleto.csv", index=False)
        print("Archivo 'benchmarkCompleto.csv' generado con éxito.")