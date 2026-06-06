#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

const int N_ITER   = 100;
const int N_WARMUP = 10;

// Workspace de 32 MB para cuBLAS
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

// Inicializa un buffer float con valor escalar (cudaMemset rellena bytes,
// no floats; para float != 0 hay que usar un kernel).
__global__ void fill_float(float* p, float v, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
static void launch_fill_float(float* p, float v, size_t n) {
    int block = 256;
    size_t grid = (n + block - 1) / block;
    if (grid > 2147483647ULL) grid = 2147483647ULL;
    fill_float<<<(int)grid, block>>>(p, v, n);
    checkCuda(cudaGetLastError(), "fill_float launch");
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

    std::cout << "--- Benchmark RTX 4090 (TF32 / Tensor Cores) ---\n"
              << "Dimensiones (M x N x K): " << M << " x " << N << " x " << K << "\n";

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    void* d_workspace = nullptr;
    checkCuda(cudaMalloc(&d_workspace, CUBLAS_WORKSPACE_BYTES), "cudaMalloc workspace");
    checkCublas(cublasSetWorkspace(handle, d_workspace, CUBLAS_WORKSPACE_BYTES),
                "cublasSetWorkspace");

    // Habilitar Tensor Cores con TF32
    checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH),
                "cublasSetMathMode");

    // TF32 trabaja con buffers FP32 reales; los multiplicadores se redondean
    // internamente al formato TF32 (mantisa 10 bits) dentro de los Tensor Cores.
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCuda(cudaMalloc(&d_A, size_A), "cudaMalloc A");
    checkCuda(cudaMalloc(&d_B, size_B), "cudaMalloc B");
    checkCuda(cudaMalloc(&d_C, size_C), "cudaMalloc C");

    launch_fill_float(d_A, 1.0f, (size_t)M * K);
    launch_fill_float(d_B, 1.0f, (size_t)K * N);
    launch_fill_float(d_C, 0.0f, (size_t)M * N);
    checkCuda(cudaDeviceSynchronize(), "fill sync");

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cublasGemmEx con compute type TF32 (CUBLAS_COMPUTE_32F_FAST_TF32):
    // entradas/salidas en FP32, multiplicacion interna en TF32 sobre Tensor Cores.
    auto run_gemm = [&]() {
        return cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, CUDA_R_32F, M,
            d_B, CUDA_R_32F, K,
            &beta,
            d_C, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };

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

    // Peak teorico RTX 4090: TF32 = 82.6 TFLOPS dense (mitad que FP16 con FP32-acc)
    constexpr double PEAK_4090_TF32 = 82.6;
    double pct_peak = 100.0 * tflops / PEAK_4090_TF32;

    std::cout << "------------------------------------------------\n"
              << "Tiempo total:                  " << (ms_total / 1000.0) << " s\n"
              << "Tiempo promedio por iteración: " << (avg_s * 1000.0) << " ms\n"
              << "Rendimiento estimado:          " << tflops << " TFLOPS (TF32)\n"
              << "% del peak teorico (82.6):     " << pct_peak << " %\n"
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
