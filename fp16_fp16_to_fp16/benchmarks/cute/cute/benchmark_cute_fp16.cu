#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <functional>
#include <limits>
#include <unordered_map>
#include <fstream>
#include <sstream>

#include <cuda_fp16.h>
#include <cublas_v2.h>

// CUTLASS Includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"

#include "autotune_cache.h"

// -------------------------------------------------------------------------
// MACROS & UTILS
// -------------------------------------------------------------------------
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
}


// -------------------------------------------------------------------------
// CUTLASS TEMPLATES & CONFIGURATIONS
// -------------------------------------------------------------------------
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t; 

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, 
    ElementAccumulator, ElementAccumulator
>;

/*
// Config 0: Small Tile, 4 Stages
using Gemm_128x128x64_4 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4
>;

// Config 1: Massive Tile, 3 Stages
using Gemm_256x128x64_3 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3
>;

// Config 2: Tall Tile, 4 Stages
using Gemm_128x256x64_4 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4
>; */

// -------------------------------------------------------------------------
// CUTLASS TEMPLATES & CONFIGURATIONS (RTX 4090 / 100KB SMEM SAFE)
// -------------------------------------------------------------------------
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = cutlass::half_t; 

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, 
    ElementAccumulator, ElementAccumulator
>;

// Config 0: Standard Tile, 3 Stages (~96 KB SMEM - Pushing the Ada limits!)
using Gemm_128x128x64_3 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3
>;

// Config 1: Wide Tile, thinner K, 3 Stages (~72 KB SMEM)
using Gemm_256x128x32_3 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3
>;

// Config 2: Safe Fallback, 4 Stages (~64 KB SMEM)
using Gemm_128x128x32_4 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4
>;

// -------------------------------------------------------------------------
// BENCHMARK RUNNERS
// -------------------------------------------------------------------------
float benchmark_cublas(cublasHandle_t handle, int m, int n, int k, __half* d_A, __half* d_B, __half* d_C, int iterations) {
    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return ms / iterations;
}

template <typename GemmConfig>
float test_config(int m, int n, int k, cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C, int iterations) {
    GemmConfig gemm_op;
    cutlass::half_t alpha = cutlass::half_t(1.0f);
    cutlass::half_t beta  = cutlass::half_t(0.0f);
    
    typename GemmConfig::Arguments args({m, n, k}, {A, m}, {B, k}, {C, m}, {C, m}, {alpha, beta});
    
    int smem_size = int(sizeof(typename GemmConfig::GemmKernel::SharedStorage));
    if (smem_size > (48 * 1024)) {
        cudaFuncSetAttribute(cutlass::Kernel<typename GemmConfig::GemmKernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }

    if (gemm_op.can_implement(args) != cutlass::Status::kSuccess) return -1.0f; 

    size_t workspace_size = GemmConfig::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

    if (gemm_op.initialize(args, workspace) != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -1.0f;
    }

    // Warmup
    if (gemm_op() != cutlass::Status::kSuccess) {
        if (workspace) cudaFree(workspace);
        return -1.0f;
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        gemm_op();
    }
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    if (workspace) cudaFree(workspace);

    return ms / iterations; 
}


// -------------------------------------------------------------------------
// MAIN EXECUTION
// -------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int M = 16384;
    int N = 16384;
    int K = 16384;

    if (argc == 2) {
        M = N = K = std::atoi(argv[1]);
    } else if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::cerr << "Uso incorrecto.\n";
        return 1;
    }

    std::cout << "--- Benchmark RTX 4090 CUTE (FP16 / Tensor Cores) ---\n";
    std::cout << "Dimensiones (M x N x K): " << M << " x " << N << " x " << K << "\n\n";

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    size_t size_A = (size_t)M * K * sizeof(__half);
    size_t size_B = (size_t)K * N * sizeof(__half);
    size_t size_C = (size_t)M * N * sizeof(__half);

    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    CHECK_CUDA(cudaMalloc(&d_B, size_B));
    CHECK_CUDA(cudaMalloc(&d_C, size_C));

    CHECK_CUDA(cudaMemset(d_A, 0, size_A));
    CHECK_CUDA(cudaMemset(d_B, 0, size_B));
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    int iters = 100; 
    
    // 1. Run cuBLAS Baseline
    std::cout << "Running cuBLAS baseline...\n";
    float ms_cublas = benchmark_cublas(handle, M, N, K, d_A, d_B, d_C, iters);

    // 2. Initialize Cache & Check Status
    AutotuneCache cache("./cutlass_heuristics.csv");
    
    if (cache.did_file_exist()) {
        std::cout << "[INFO] Cache file loaded successfully. (" << cache.get_size() << " saved heuristics)\n";
    } else {
        std::cout << "[INFO] No existing cache file found. A new one will be created.\n";
    }

    auto cut_A = reinterpret_cast<cutlass::half_t*>(d_A);
    auto cut_B = reinterpret_cast<cutlass::half_t*>(d_B);
    auto cut_C = reinterpret_cast<cutlass::half_t*>(d_C);


  
    // 3. Define Search Space
    std::vector<std::function<float()>> search_space = {
        [&]() { return test_config<Gemm_128x128x64_3>(M, N, K, cut_A, cut_B, cut_C, iters); },
        [&]() { return test_config<Gemm_256x128x32_3>(M, N, K, cut_A, cut_B, cut_C, iters); },
        [&]() { return test_config<Gemm_128x128x32_4>(M, N, K, cut_A, cut_B, cut_C, iters); }
    };

    int best_idx = cache.get_best_kernel(M, N, K);
    float ms_cute = -1.0f;

    // 4. Autotune Logic
    if (best_idx != -1) {
        std::cout << "[CACHE HIT] Skipping autotuning. Using Kernel Index: " << best_idx << "\n";
        ms_cute = search_space[best_idx](); 
    } else {
        std::cout << "[CACHE MISS] Autotuning " << search_space.size() << " configurations...\n";
        float best_time = 999999.0f;
        
        for (int i = 0; i < search_space.size(); ++i) {
            float time = search_space[i]();
            if (time > 0 && time < best_time) {
                best_time = time;
                best_idx = i;
            }
        }

        // --- PROPER OOM / FAILURE HANDLING ---
        if (best_idx == -1) {
            std::cerr << "\n[FATAL ERROR] All CUTLASS configurations failed!\n";
            std::cerr << "This usually means the matrix is too large (VRAM OOM) or the shapes are unsupported.\n";
            
            // Clean up memory before exiting so we don't leave zombie data on the GPU
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cublasDestroy(handle);
            
            // Force the program to terminate immediately
            return 1; 
        } else {
            std::cout << "[AUTOTUNE COMPLETE] Best Kernel Index: " << best_idx 
                      << " | Time: " << best_time << " ms\n";
            cache.save_heuristic(M, N, K, best_idx); 
            ms_cute = best_time;
        }
    }

    // 5. Calculate and Print TFLOPS
    std::cout << "\n===============================================================================\n";
    std::cout << std::left << std::setw(12) << "Type" 
              << std::setw(8) << "M" << std::setw(8) << "N" << std::setw(8) << "K" 
              << std::setw(18) << "cuBLAS (TFLOPS)" << "CuTe (TFLOPS)\n";
    std::cout << "===============================================================================\n";

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops_cublas = (flops / (ms_cublas / 1000.0)) / 1e12;
    
    if (ms_cute < 0.0f) {
        std::cout << std::left << std::setw(12) << "Custom" 
                  << std::setw(8) << M << std::setw(8) << N << std::setw(8) << K 
                  << std::setw(18) << std::fixed << std::setprecision(2) << tflops_cublas 
                  << "ERROR\n";
    } else {
        double tflops_cute = (flops / (ms_cute / 1000.0)) / 1e12;
        std::cout << std::left << std::setw(12) << "Custom" 
                  << std::setw(8) << M << std::setw(8) << N << std::setw(8) << K 
                  << std::setw(18) << std::fixed << std::setprecision(2) << tflops_cublas 
                  << std::fixed << std::setprecision(2) << tflops_cute << "\n";
    }
    std::cout << "===============================================================================\n";

    // 6. Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}