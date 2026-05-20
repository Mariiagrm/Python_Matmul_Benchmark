// =====================================================================
//  hgemm_wmma_ada.cu
//  HGEMM (FP16 in / FP32 acc / FP32 out) optimizado para Ada Lovelace
//  (RTX 4090, sm_89), basado en la API WMMA (nvcuda::wmma).
//
//  Diferencia clave respecto a sgemm_ada.cu:
//      * sgemm_ada.cu  ->  256 hilos x sub-tile 8x8 sobre CUDA Cores FP32.
//      * hgemm_wmma_ada.cu  ->  8 warps x fragmentos 16x16x16 sobre
//                              Tensor Cores de 4a generacion. El sub-tile
//                              ya no es propiedad de un hilo sino del
//                              warp entero, distribuido internamente por
//                              el hardware.
//
//  Jerarquia de tiles (estandar CUTLASS, simplificado):
//      ThreadblockShape  = 128 x 128 x  32    (BM x BN x BK)
//      WarpShape         =  64 x  32 x  32    (WM x WN x BK)
//      InstructionShape  =  16 x  16 x  16    (WMMA fragment)
//
//  Disposicion de warps: 2 (en M) x 4 (en N) = 8 warps por bloque
//      => 256 hilos por bloque. Cada warp computa
//         FRAGS_M x FRAGS_N = 4 x 2 = 8 fragmentos de 16x16,
//         encadenando FRAGS_K = 2 fragmentos en la dimension K por
//         iteracion externa.
//
//  Optimizaciones aplicadas (espejo del kernel FP32):
//      * Doble buffer + cp.async + __pipeline_wait_prior(1).
//      * Cargas vectorizadas float4 (8 halfs por float4).
//      * Padding de 8 halfs (=16 B = 1 banco) en cada fila de shared
//        memory para reducir conflictos de banco.
//      * --launch_bounds__(256, 2) para garantizar >= 2 bloques/SM.
//
//  Compilacion:
//      nvcc -O3 -arch=sm_89 -use_fast_math -std=c++17 \
//           -lineinfo --ptxas-options=-v \
//           -lcublas hgemm_wmma_ada.cu -o hgemm_wmma_ada
//
//  Convencion: A (MxK) y B (KxN) en FP16, row-major.  C (MxN) en FP32.
//  Para cuBLAS se aplica el truco row-major -> column-major calculando
//  B*A en lugar de A*B.
// =====================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

using namespace nvcuda;

// --------------------------- Parametros del kernel ---------------------------
// Threadblock tile
#define BM 128
#define BN 128
#define BK  32

// Warp tile
#define WM 64
#define WN 32

// WMMA fragment shape (instrucciones mma.sync de 4a gen, FP16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Layout de warps dentro del bloque
#define WARPS_M    (BM / WM)               // 2
#define WARPS_N    (BN / WN)               // 4
#define NUM_WARPS  (WARPS_M * WARPS_N)     // 8
#define NUM_THREADS (NUM_WARPS * 32)       // 256

// Fragmentos por warp
#define FRAGS_M    (WM   / WMMA_M)         // 4
#define FRAGS_N    (WN   / WMMA_N)         // 2
#define FRAGS_K    (BK   / WMMA_K)         // 2

// Padding antibancos (8 halfs = 16 bytes = 1 banco de 32x4B)
// Mantiene stride multiplo de 8 halfs, condicion requerida por WMMA
// para load_matrix_sync con tipo half.
#define A_PAD 8
#define B_PAD 8
#define A_STRIDE (BK + A_PAD)              // 40 halfs
#define B_STRIDE (BN + B_PAD)              // 136 halfs

// --------------------------- Macros de error --------------------------------
#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t _err = (stmt);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d -> %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_err));                                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(stmt)                                                     \
    do {                                                                       \
        cublasStatus_t _err = (stmt);                                          \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d -> %d\n", __FILE__, __LINE__,  \
                    (int)_err);                                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ============================================================================
//                              KERNEL PRINCIPAL
// ============================================================================
//
//  Cada bloque (bx, by) calcula el tile:
//      C[ by*BM : by*BM+BM ,  bx*BN : bx*BN+BN ]
//
//  Dentro del bloque, el warp (warp_row, warp_col) calcula la subregion:
//      C[ by*BM + warp_row*WM : ... + WM ,
//         bx*BN + warp_col*WN : ... + WN ]
//
//  El warp mantiene en registros FRAGS_M x FRAGS_N acumuladores (8 frags
//  de 16x16 FP32, repartidos entre los 32 hilos por el hardware).
//
// ============================================================================
__global__ __launch_bounds__(NUM_THREADS, 2)
void hgemm_wmma_kernel(const half*  __restrict__ A,
                       const half*  __restrict__ B,
                       float*       __restrict__ C,
                       int M, int N, int K)
{
    const int bx       = blockIdx.x;
    const int by       = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;                 // tid / 32   in [0, 8)
    const int warp_row = warp_id / WARPS_N;        // in [0, 2)
    const int warp_col = warp_id % WARPS_N;        // in [0, 4)

    // ---------------- Memoria compartida con doble buffer -------------------
    //  Tamano total:
    //      As: 2 * 128 *  40 * 2 B =  20.0 KB
    //      Bs: 2 *  32 * 136 * 2 B =  17.0 KB
    //      total                  ~  37 KB  (<< 100 KB disponibles)
    // -----------------------------------------------------------------------
    __shared__ half As[2][BM][A_STRIDE];
    __shared__ half Bs[2][BK][B_STRIDE];

    // ---------------- Fragmentos WMMA --------------------------------------
    //  matrix_a y matrix_b en row-major (coincide con el layout de As y Bs).
    //  accumulator en FP32 (CUBLAS_COMPUTE_32F equivalente).
    // -----------------------------------------------------------------------
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> a_frag[FRAGS_M];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                   half, wmma::row_major> b_frag[FRAGS_N];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                   float> c_frag[FRAGS_M][FRAGS_N];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    // ---------------- Indices de carga global -> shared ---------------------
    //
    //  A (128 x 32 halfs = 4096 halfs = 512 float4) con 256 hilos
    //      => 2 float4 / hilo.  Cada float4 = 8 halfs consecutivos en K.
    //      Una fila de A tiene 32 halfs = 4 float4.
    //      Pattern: row_step1 = tid/4   col_byte = (tid%4)*8
    //               row_step2 = row_step1 + 64
    //
    //  B ( 32 x 128 halfs = 4096 halfs = 512 float4) con 256 hilos
    //      => 2 float4 / hilo.  Cada float4 = 8 halfs consecutivos en N.
    //      Una fila de B tiene 128 halfs = 16 float4.
    //      Pattern: row_step1 = tid/16  col_byte = (tid%16)*8
    //               row_step2 = row_step1 + 16
    // -----------------------------------------------------------------------
    const int loadA_col = (tid & 3) << 3;   // (tid % 4) * 8
    const int loadA_row = tid >> 2;         //  tid / 4
    const int loadB_col = (tid & 15) << 3;  // (tid % 16) * 8
    const int loadB_row = tid >> 4;         //  tid / 16

    const half* A_block = A + by * BM * K;
    const half* B_block = B + bx * BN;

    // ---------------- Helper: emite cp.async para un tile -------------------
    auto issue_async_load = [&](int buf, int kStart) {
        // ---- A ----
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int row = loadA_row + s * 64;
            const half* src = &A_block[row * K + kStart + loadA_col];
            __pipeline_memcpy_async(
                &As[buf][row][loadA_col], src, sizeof(float4));
        }
        // ---- B ----
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int row = loadB_row + s * 16;
            const half* src = &B_block[(kStart + row) * N + loadB_col];
            __pipeline_memcpy_async(
                &Bs[buf][row][loadB_col], src, sizeof(float4));
        }
    };

    // ---------------- Helper: bucle de WMMA sobre un buffer dado -----------
    auto compute_buffer = [&](int buf) {
        #pragma unroll
        for (int kf = 0; kf < FRAGS_K; ++kf) {
            // Cargar fragmentos de A propios de este warp (FRAGS_M)
            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                const int row = warp_row * WM + i * WMMA_M;
                wmma::load_matrix_sync(
                    a_frag[i],
                    &As[buf][row][kf * WMMA_K],
                    A_STRIDE);
            }
            // Cargar fragmentos de B propios de este warp (FRAGS_N)
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                const int col = warp_col * WN + j * WMMA_N;
                wmma::load_matrix_sync(
                    b_frag[j],
                    &Bs[buf][kf * WMMA_K][col],
                    B_STRIDE);
            }
            // Producto exterior 4x2 de fragmentos -> 8 MMA por kf
            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    wmma::mma_sync(c_frag[i][j],
                                   a_frag[i], b_frag[j],
                                   c_frag[i][j]);
                }
            }
        }
    };

    // =================== PROLOGO: precargar buffer 0 ========================
    const int numTiles = K / BK;             // se asume K % BK == 0

    issue_async_load(0, 0);
    __pipeline_commit();

    // =================== BUCLE PRINCIPAL =====================================
    for (int t = 0; t < numTiles - 1; ++t) {
        const int curBuf  = t & 1;
        const int nextBuf = curBuf ^ 1;

        issue_async_load(nextBuf, (t + 1) * BK);
        __pipeline_commit();

        // Esperar al tile actual; queda 1 commit pendiente (el next).
        __pipeline_wait_prior(1);
        __syncthreads();

        compute_buffer(curBuf);
        __syncthreads();
    }

    // =================== EPILOGO: procesar el ultimo tile ===================
    {
        const int lastBuf = (numTiles - 1) & 1;
        __pipeline_wait_prior(0);
        __syncthreads();
        compute_buffer(lastBuf);
    }

    // =================== ESCRITURA DE C (FP32, store_matrix_sync) ===========
    //  Cada warp escribe FRAGS_M x FRAGS_N fragmentos de 16x16 FP32.
    //  store_matrix_sync se encarga internamente de la distribucion
    //  inversa del fragmento a la matriz global; se requiere stride en
    //  elementos (no en bytes) y mem_row_major.
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            const int row = by * BM + warp_row * WM + i * WMMA_M;
            const int col = bx * BN + warp_col * WN + j * WMMA_N;
            wmma::store_matrix_sync(
                &C[row * N + col],
                c_frag[i][j],
                N,
                wmma::mem_row_major);
        }
    }
}

// ============================================================================
//                            UTILIDADES DEL HOST
// ============================================================================

static void launch_hgemm(const half* dA, const half* dB, float* dC,
                         int M, int N, int K)
{
    dim3 block(NUM_THREADS, 1, 1);                  // 256 hilos en 1D
    dim3 grid (N / BN, M / BM, 1);                  // (N/128, M/128, 1)
    hgemm_wmma_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

// cuBLAS row-major <-> column-major: calcular B*A en column-major equivale
// a A*B en row-major.  Operandos FP16, salida FP32, computo FP32 sobre TC.
static void cublas_hgemm_rm(cublasHandle_t handle,
                            const half* dA, const half* dB, float* dC,
                            int M, int N, int K)
{
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        dB, CUDA_R_16F, N,
        dA, CUDA_R_16F, K,
        &beta,
        dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void fill_random_fp16(half* p, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        // Rango [-1, 1] para mantener acotada la magnitud del producto
        // y limitar el error acumulado de FP16.
        float v = (float)((rand() % 2001) - 1000) / 1000.0f;
        p[i] = __float2half(v);
    }
}

// Error relativo medio y maximo.  Mas robusto que |diff| absoluto para FP16.
static void analyze_error(const float* a, const float* b, size_t n,
                          float& max_rel, float& mean_rel, float& max_abs)
{
    double sum_rel = 0.0;
    max_rel = 0.0f;
    max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        float denom = std::fabs(b[i]) + 1e-6f;
        float r = d / denom;
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
        sum_rel += r;
    }
    mean_rel = (float)(sum_rel / n);
}

// ============================================================================
//                                   MAIN
// ============================================================================
int main(int argc, char** argv)
{
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    if (M % BM || N % BN || K % BK) {
        fprintf(stderr,
                "Las dimensiones deben ser multiplos de (BM=%d, BN=%d, BK=%d)\n",
                BM, BN, BK);
        return EXIT_FAILURE;
    }

    printf("HGEMM (FP16 -> FP32)  M=%d N=%d K=%d\n", M, N, K);
    printf("  Threadblock: %dx%dx%d   Warp: %dx%d   MMA: %dx%dx%d\n",
           BM, BN, BK, WM, WN, WMMA_M, WMMA_N, WMMA_K);
    printf("  Warps/bloque: %d   Hilos/bloque: %d\n", NUM_WARPS, NUM_THREADS);

    const size_t sA = (size_t)M * K;
    const size_t sB = (size_t)K * N;
    const size_t sC = (size_t)M * N;

    // ---- Host ----
    half*  hA   = (half*) std::malloc(sA * sizeof(half));
    half*  hB   = (half*) std::malloc(sB * sizeof(half));
    float* hC   = (float*)std::malloc(sC * sizeof(float));
    float* hRef = (float*)std::malloc(sC * sizeof(float));
    fill_random_fp16(hA, sA, 1);
    fill_random_fp16(hB, sB, 2);

    // ---- Device ----
    half  *dA, *dB;
    float *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA,   sA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dB,   sB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dC,   sC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, sC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sA * sizeof(half),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sB * sizeof(half),  cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // ---- Warm-up ----
    launch_hgemm(dA, dB, dC, M, N, K);
    cublas_hgemm_rm(handle, dA, dB, dRef, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Validacion numerica ----
    CUDA_CHECK(cudaMemcpy(hC,   dC,   sC * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef, dRef, sC * sizeof(float), cudaMemcpyDeviceToHost));
    float max_rel, mean_rel, max_abs;
    analyze_error(hC, hRef, sC, max_rel, mean_rel, max_abs);
    printf("Error frente a cuBLAS:  max|diff|=%.3e  rel_max=%.3e  rel_mean=%.3e  %s\n",
           max_abs, max_rel, mean_rel,
           (mean_rel < 1e-2f) ? "[OK]" : "[FALLO]");

    // ---- Benchmark ----
    const int iters = 50;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch_hgemm(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_k;
    CUDA_CHECK(cudaEventElapsedTime(&ms_k, e0, e1));
    ms_k /= iters;

    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i)
        cublas_hgemm_rm(handle, dA, dB, dRef, M, N, K);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_c;
    CUDA_CHECK(cudaEventElapsedTime(&ms_c, e0, e1));
    ms_c /= iters;

    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double tK = flops / (ms_k * 1.0e9);   // TFLOP/s
    const double tC = flops / (ms_c * 1.0e9);
    printf("Kernel WMMA propio: %7.3f ms   %7.2f TFLOP/s\n", ms_k, tK);
    printf("cuBLAS (TC)       : %7.3f ms   %7.2f TFLOP/s\n", ms_c, tC);
    printf("Ratio propio / cuBLAS: %.1f %%\n", 100.0 * tK / tC);

    // ---- Limpieza ----
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    std::free(hA); std::free(hB); std::free(hC); std::free(hRef);
    return 0;
}
