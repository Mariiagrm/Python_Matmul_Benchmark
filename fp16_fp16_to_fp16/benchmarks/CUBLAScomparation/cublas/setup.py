from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Asegúrate de que los nombres de archivo coincidan exactamente con los que tienes
sources = [
    'binding.cpp', # Asumo que el código PYBIND11 que me diste está aquí
    'cublas/hgemm_cublas.cu',
    'hgemm_cublaslt_heuristic.cu',
    'hgemm_cublaslt_auto_tuning.cu',
    # 'cutlass_wrapper.cu' # Descomenta si tienes el archivo de cutlass
]

# Filtrar archivos que no existan para evitar errores de compilación
valid_sources = [s for s in sources if os.path.exists(s)]

setup(
    name='cublas_benchmark_ext',
    ext_modules=[
        CUDAExtension(
            name='cublas_benchmark_ext',
            sources=valid_sources,
            libraries=['cublas', 'cublasLt'], # Enlazamos librerías de NVIDIA
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3', 
                    '-gencode=arch=compute_89,code=sm_89', # Arquitectura Ada Lovelace (4090)
                    '--ptxas-options=-v',
                    '-lineinfo'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)