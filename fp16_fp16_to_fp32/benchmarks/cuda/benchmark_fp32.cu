#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Helper to check CUDA errors
void checkCuda(cudaError_t result, const char* func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << func << ": " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// Helper to check cuBLAS errors
void checkCublas(cublasStatus_t result, const char* func) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error at " << func << " (Code: " << result << ")" << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    int M = 16384, N = 16384, K = 16384;
    if (argc == 2) M = N = K = std::atoi(argv[1]);
    
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    // Enable Tensor Cores
    checkCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "MathMode");

    // 1. Calculate total needed bytes first
    size_t total_bytes = ((size_t)M * K + (size_t)K * N + (size_t)M * N) * sizeof(__half);
    std::cout << "Attempting to allocate: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;

    __half *d_all;
    __half *d_A, *d_B, *d_C;
    // Attempt a single large allocation
    cudaError_t err = cudaMalloc(&d_all, total_bytes);

    if (err != cudaSuccess) {
        std::cerr << "OOM! GPU cannot provide a single block of " << total_bytes << " bytes." << std::endl;
        exit(1);
    }

    // Sub-divide the block
    d_A = d_all;
    d_B = d_all + ((size_t)M * K);
    d_C = d_all + ((size_t)M * K) + ((size_t)K * N);

    // Important: alpha and beta are FLOAT for FP32 accumulation
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Warmup
    for (int i = 0; i < 10; ++i) {
        checkCublas(cublasGemmEx(handle, 
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                M, N, K,
                                &alpha,           // Alpha as float*
                                d_A, CUDA_R_16F, M,
                                d_B, CUDA_R_16F, K,
                                &beta,            // Beta as float*
                                d_C, CUDA_R_16F, M,
                                CUBLAS_COMPUTE_32F, // <--- FP32 Accumulator
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP), "Warmup");
    }
    cudaDeviceSynchronize();

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                     &alpha, d_A, CUDA_R_16F, M,
                     d_B, CUDA_R_16F, K,
                     &beta, d_C, CUDA_R_16F, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double avg_s = (ms / 1000.0) / iterations;
    double tflops = (2.0 * M * N * K) / (avg_s * 1e12);

    std::cout << "Average seconds: " << avg_s << std::endl;
    std::cout << "Performance (FP16 Inputs, FP32 Accumulate): " << tflops << " TFLOPS" << std::endl;

    cudaFree(d_all);
    cublasDestroy(handle);
    return 0;
}