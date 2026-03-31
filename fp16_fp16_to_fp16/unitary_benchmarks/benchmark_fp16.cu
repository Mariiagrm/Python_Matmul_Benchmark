#include <iostream>
#include <vector>
#include <cstdlib> // Para std::atoi
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

const int N_ITER = 100; // Número de iteraciones para promediar
const int N_WARMUP = 10; // Iteraciones de calentamiento

void checkCuda(cudaError_t result, const char* func) {
    if (result != cudaSuccess) {
        std::cerr << "Error de CUDA en " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

void checkCublas(cublasStatus_t result, const char* func) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Error de cuBLAS en " << func << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // 1. Parsear los argumentos de la terminal
    // ./benchmark_fp16 

    int M = 16384;
    int N = 16384;
    int K = 16384;

    if (argc == 2) {
        // Un solo argumento = Matriz Cuadrada
        M = N = K = std::atoi(argv[1]);
    } else if (argc == 4) {
        // Tres argumentos = Dimensiones M, N, K
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::cerr << "Uso incorrecto." << std::endl;
        std::cerr << "Para matrices cuadradas: " << argv[0] << " [TAMANIO]" << std::endl;
        std::cerr << "Para matrices MxNxK:     " << argv[0] << " [M] [N] [K]" << std::endl;
        return 1;
    }

    std::cout << "--- Benchmark RTX 4090 (FP16 / Tensor Cores) ---" << std::endl;
    std::cout << "Dimensiones (M x N x K): " << M << " x " << N << " x " << K << std::endl;

    // 2. Inicializar cuBLAS
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");

    // 3. Cálculos de tamaño de memoria seguros para matrices enormes (usando size_t)
    size_t size_A = (size_t)M * K * sizeof(__half);
    size_t size_B = (size_t)K * N * sizeof(__half);
    size_t size_C = (size_t)M * N * sizeof(__half);

    __half *d_A, *d_B, *d_C;

    checkCuda(cudaMalloc(&d_A, size_A), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_B, size_B), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_C, size_C), "cudaMalloc C");

    checkCuda(cudaMemset(d_A, 0x3C00, size_A), "Init A"); // 1.0 en FP16
    checkCuda(cudaMemset(d_B, 0x3C00, size_B), "Init B");
    checkCuda(cudaMemset(d_C, 0x0000, size_C), "Init C");

    const __half alpha = 1.0f;
    const __half beta = 0.0f;

    // 4. Calentamiento (Warm-up)
    std::cout << "Calentando GPU..." << std::endl;
    for (int i = 0; i < N_WARMUP; ++i) {
        checkCublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                M, N, K,
                                &alpha,
                                d_A, M,
                                d_B, K,
                                &beta,
                                d_C, M), "cublasHgemm Warmup");
    }
    cudaDeviceSynchronize();

    // 5. Benchmark Loop
    std::cout << "Ejecutando " << N_ITER << " iteraciones..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < N_ITER; ++i) {
        checkCublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                M, N, K,
                                &alpha,
                                d_A, M,
                                d_B, K,
                                &beta,
                                d_C, M), "cublasHgemm Loop"); 
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 6. Cálculos de Rendimiento
    double seconds = milliseconds / 1000.0;
    double avg_seconds = seconds / N_ITER;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops / avg_seconds) / 1e12;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Tiempo total: " << seconds << " s" << std::endl;
    std::cout << "Tiempo promedio por iteración: " << (avg_seconds * 1000.0) << " ms" << std::endl;
    std::cout << "Rendimiento estimado: " << tflops << " TFLOPS (FP16)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Limpieza
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}