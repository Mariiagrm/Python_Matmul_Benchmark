#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

const int N_ITER   = 100;
const int N_WARMUP = 10;

/* ./benchmark_fp16 */

// Workspace de 32 MB para cuBLAS: habilita algoritmos split-K y otros
// que requieren buffer auxiliar y suelen ser mas rapidos en shapes grandes.
constexpr size_t CUBLAS_WORKSPACE_BYTES = 32ULL * 1024ULL * 1024ULL;

void checkCuda(cudaError_t r, const char* f) {
    if (r != cudaSuccess) {
        std::cerr << "CUDA error en " << f << ": " << cudaGetErrorString(r) << "\n";
        std::exit(1);
    }
}
void checkCublas(cublasStatus_t r, const char* f) {
    if (r != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error en " << f << " (code " << (int)r << ")\n";
        std::exit(1);
    }
}

// Rellena un buffer __half con un valor escalar (cudaMemset no sirve porque
// trabaja byte a byte; 0x3C00 se truncaria a 0x00).
__global__ void fill_half(__half* p, __half v, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
static void launch_fill_half(__half* p, float v, size_t n) {
    int block = 256;
    size_t grid = (n + block - 1) / block;
    if (grid > 2147483647ULL) grid = 2147483647ULL;
    fill_half<<<(int)grid, block>>>(p, __float2half(v), n);
    checkCuda(cudaGetLastError(), "fill_half launch");
}

int main(int argc, char* argv[]) {
    int M = 16384, N = 16384, K = 16384;
    if (argc == 2) {
        M = N = K = std::atoi(argv[1]);
    } else if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::cerr << "Uso: " << argv[0] << " [N]   o   " << argv[0] << " [M] [N] [K]\n";
        return 1;
    }

    std::cout << "--- Benchmark RTX 4090 (FP16 / FP16-acc / Tensor Cores) ---\n"
              << "Dimensiones (M x N x K): " << M << " x " << N << " x " << K << "\n";

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // Workspace explicito (cuBLAS por defecto solo reserva ~4 MB)
    void* d_workspace = nullptr;
    checkCuda(cudaMalloc(&d_workspace, CUBLAS_WORKSPACE_BYTES), "cudaMalloc workspace");
    checkCublas(cublasSetWorkspace(handle, d_workspace, CUBLAS_WORKSPACE_BYTES),
                "cublasSetWorkspace");

    // Habilitar Tensor Cores (en cuBLAS modernos esto es el default, pero lo
    // hacemos explicito por claridad)
    checkCublas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");

    size_t size_A = (size_t)M * K * sizeof(__half);
    size_t size_B = (size_t)K * N * sizeof(__half);
    size_t size_C = (size_t)M * N * sizeof(__half);

    __half *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, size_A), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_B, size_B), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_C, size_C), "cudaMalloc C");

    // Inicializacion correcta a 1.0 / 1.0 / 0.0 via kernel propio
    launch_fill_half(d_A, 1.0f, (size_t)M * K);
    launch_fill_half(d_B, 1.0f, (size_t)K * N);
    launch_fill_half(d_C, 0.0f, (size_t)M * N);
    checkCuda(cudaDeviceSynchronize(), "fill sync");

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    // --- LA CLAVE: cublasGemmEx con CUBLAS_COMPUTE_16F ---
    // Esto fuerza acumulacion en FP16 (techo 330 TFLOPS), no FP32 (techo 165).
    // El algoritmo CUBLAS_GEMM_DEFAULT_TENSOR_OP deja a cuBLAS elegir el mejor
    // kernel de Tensor Cores via heuristica.
    auto run_gemm = [&]() {
        return cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_16F, M,
            d_B, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_16F, M,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };

    // Warmup: la heuristica de cuBLAS afina la eleccion de algoritmo con las
    // primeras llamadas; sin warmup la primera iteracion mide mas tiempo.
    std::cout << "Warmup (" << N_WARMUP << " iters)...\n";
    for (int i = 0; i < N_WARMUP; ++i) checkCublas(run_gemm(), "GemmEx warmup");
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    std::cout << "Midiendo (" << N_ITER << " iters)...\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < N_ITER; ++i) checkCublas(run_gemm(), "GemmEx loop");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, start, stop);

    double avg_s   = (ms_total / 1000.0) / N_ITER;
    double flops   = 2.0 * (double)M * (double)N * (double)K;
    double tflops  = (flops / avg_s) / 1e12;

    // Peak teorico ajustado al lock de clock (asumimos 2500/2520 MHz)
    constexpr double PEAK_4090_BOOST = 330.0;  // FP16/FP16-acc @ 2520 MHz
    double pct_peak = 100.0 * tflops / PEAK_4090_BOOST;

    std::cout << "------------------------------------------------\n"
              << "Tiempo total:                  " << (ms_total / 1000.0) << " s\n"
              << "Tiempo promedio por iteración: " << (avg_s * 1000.0) << " ms\n"
              << "Rendimiento estimado:          " << tflops << " TFLOPS (FP16/FP16-acc)\n"
              << "% del peak teorico (330):      " << pct_peak << " %\n"
              << "------------------------------------------------\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_workspace);
    cublasDestroy(handle);
    return 0;
}
