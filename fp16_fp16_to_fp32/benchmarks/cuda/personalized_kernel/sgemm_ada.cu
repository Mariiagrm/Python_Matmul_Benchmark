// =====================================================================
//  sgemm_ada.cu
//  SGEMM optimizado para la arquitectura Ada Lovelace (RTX 4090, sm_89)
//
//    - Threadblock tile de 128 x 128 elementos de C.
//    - 256 hilos por bloque organizados como una rejilla logica 16 x 16.
//    - Cada hilo computa un sub-tile de 8 x 8 (register tiling => 64
//      acumuladores en registros, mas operandos: ~80 reg/hilo).
//    - Doble buffer en memoria compartida (pipeline asincrono) usando
//      cp.async a traves de __pipeline_memcpy_async.
//    - Cargas vectorizadas float4 (16 bytes / hilo / transaccion).
//    - Padding en memoria compartida para mitigar bank conflicts.
//    - Escrituras vectorizadas float4 a memoria global para C.
//    - Validacion numerica frente a cuBLAS (cublasSgemm) y benchmark
//      relativo en GFLOPs.
//
//  Compilacion (RTX 4090):
//      nvcc -O3 -arch=sm_89 -use_fast_math \
//           -lineinfo --ptxas-options=-v \
//           -lcublas sgemm_ada.cu -o sgemm_ada
//
//  Convencion: A (MxK), B (KxN), C (MxN), todas en formato row-major.
//  Para coincidir con cuBLAS (column-major), se invoca como
//      cublasSgemm(... N, M, K, ..., B, N, A, K, ..., C, N)
//  que es el truco habitual para comparar con row-major.
// =====================================================================

#include <cuda_runtime.h>
#include <cuda_pipeline.h>     // __pipeline_memcpy_async, __pipeline_commit, ...
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// --------------------------- Parametros del kernel ---------------------------
#define BM 128            // Tile del bloque en M (filas de C)
#define BN 128            // Tile del bloque en N (cols  de C)
#define BK   8            // Profundidad K cargada por iteracion externa
#define TM   8            // Sub-tile por hilo en M
#define TN   8            // Sub-tile por hilo en N

// 16 x 16 = 256 hilos por bloque  (8 warps)
#define BLOCK_DIM_X  (BN / TN)        // 16
#define BLOCK_DIM_Y  (BM / TM)        // 16
#define NUM_THREADS  (BLOCK_DIM_X * BLOCK_DIM_Y)  // 256

// Padding de la fila en Bs para mitigar bank conflicts (32 banks * 4 B)
#define BN_PAD       4
#define BN_STRIDE    (BN + BN_PAD)    // 132

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
//  Layout de memoria compartida:
//      As[2][BK][BM]              -> A transpuesta al cargarla; permite que
//                                    a_reg[i] = As[buf][k][ty*TM+i] sea un
//                                    acceso secuencial en BM al variar i.
//      Bs[2][BK][BN_STRIDE]       -> B en orden natural con padding de fila
//                                    para evitar conflictos de banco al
//                                    leer 8 floats consecutivos por hilo.
//
//  Pipeline (double buffering):
//      Iter 0  : cp.async  buffer 0  ;  __pipeline_commit
//      Iter k  : cp.async  buffer ((k+1)%2)
//                __pipeline_commit
//                __pipeline_wait_prior(1)   // espera buffer (k%2) listo
//                __syncthreads()
//                compute(buffer k%2)
//      Iter ultima:  __pipeline_wait_prior(0); compute final.
//
//  --launch_bounds__(256, 2) sugiere al compilador limitar el numero de
//  registros para permitir al menos 2 bloques residentes por SM. En la
//  practica, ptxas suele asignar ~72-96 reg/hilo segun la version del
//  compilador, lo que produce 2-3 bloques/SM (24-32 warps residentes,
//  ~50-67% de ocupacion).
// ============================================================================

__global__ __launch_bounds__(NUM_THREADS, 2)
void sgemm_kernel(const float* __restrict__ A,
                  const float* __restrict__ B,
                  float*       __restrict__ C,
                  int M, int N, int K)
{
    // ---------------- Identificadores ---------------------------------------
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int tx  = threadIdx.x;                  // 0..15
    const int ty  = threadIdx.y;                  // 0..15
    const int tid = ty * BLOCK_DIM_X + tx;        // 0..255

    // ---------------- Memoria compartida con doble buffer -------------------
    __shared__ float As[2][BK][BM];               // 2 * 8 * 128 * 4 = 8 KB
    __shared__ float Bs[2][BK][BN_STRIDE];        // 2 * 8 * 132 * 4 ~ 8.25 KB

    // ---------------- Registros: acumulador 8x8 y operandos -----------------
    float c_reg[TM][TN] = {0.0f};
    float a_reg[TM];
    float b_reg[TN];

    // ---------------- Indices de carga global -> shared ---------------------
    //
    //  A: cada hilo carga 1 x float4 (4 floats consecutivos en K).
    //     BM*BK = 128*8 = 1024 floats = 256 hilos * 4 floats.
    //     innerRowA = tid / 2  (0..127)   fila en M
    //     innerColA = (tid % 2) * 4       offset en K (0 o 4)
    //
    //  B: cada hilo carga 1 x float4 (4 floats consecutivos en N).
    //     BK*BN = 8*128 = 1024 floats = 256 hilos * 4 floats.
    //     innerRowB = tid / 32 (0..7)     fila en K
    //     innerColB = (tid % 32) * 4      offset en N (0..124)
    // -----------------------------------------------------------------------
    const int innerRowA = tid >> 1;               // tid / 2
    const int innerColA = (tid & 1) << 2;         // (tid % 2) * 4
    const int innerRowB = tid >> 5;               // tid / 32
    const int innerColB = (tid & 31) << 2;        // (tid % 32) * 4

    // Punteros base de la fila/columna en memoria global de este bloque.
    const float* A_block = A + by * BM * K;
    const float* B_block = B + bx * BN;

    // ---------------- Helper: emite cp.async para un tile (A y B) -----------
    //  buf  : 0 o 1, indice del buffer destino en shared
    //  kStart: posicion en K (multiplo de BK)
    //
    //  Notas:
    //   - Para As (transpuesta) las direcciones destino de los 4 floats
    //     pertenecen a 4 filas distintas de As, por lo que NO se puede
    //     emitir una unica cp.async de 16 B (cp.async escribe verbatim).
    //     Se usan 4 cp.async de 4 B cada una.
    //   - Para Bs (no transpuesta) si se emite una cp.async de 16 B.
    // -----------------------------------------------------------------------
    auto issue_async_load = [&](int buf, int kStart) {
        // ---- A (transpuesta a As) ----
        const float* src_a = &A_block[innerRowA * K + kStart + innerColA];
        // 4 destinos en 4 filas distintas de As[buf][?][innerRowA]
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __pipeline_memcpy_async(
                &As[buf][innerColA + i][innerRowA],
                src_a + i,
                sizeof(float));
        }

        // ---- B (natural a Bs) ----
        const float* src_b = &B_block[(kStart + innerRowB) * N + innerColB];
        __pipeline_memcpy_async(
            &Bs[buf][innerRowB][innerColB],
            src_b,
            sizeof(float4));
    };

    // =================== PROLOGO: precargar buffer 0 ========================
    const int numTiles = K / BK;                  // se asume K % BK == 0

    issue_async_load(/*buf=*/0, /*kStart=*/0);
    __pipeline_commit();

    // =================== BUCLE PRINCIPAL =====================================
    for (int t = 0; t < numTiles - 1; ++t) {
        const int curBuf  = t & 1;
        const int nextBuf = curBuf ^ 1;

        // Lanzar la carga del siguiente tile (asincrona).
        issue_async_load(nextBuf, (t + 1) * BK);
        __pipeline_commit();

        // Esperar a que el tile actual (commit anterior) este listo.
        // Hay 2 commits en vuelo => esperar a que quede solo 1 pendiente.
        __pipeline_wait_prior(1);
        __syncthreads();

        // -------- Computo sobre el buffer actual (BK k-pasos) ---------------
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Cargar fragmento de A (8 floats consecutivos en M)
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[curBuf][k][ty * TM + i];
            }
            // Cargar fragmento de B (8 floats consecutivos en N)
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[curBuf][k][tx * TN + j];
            }
            // Producto exterior 8x8 acumulado en registros
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
    }

    // =================== EPILOGO: procesar el ultimo tile ===================
    {
        const int lastBuf = (numTiles - 1) & 1;
        __pipeline_wait_prior(0);    // esperar el unico commit restante
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a_reg[i] = As[lastBuf][k][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b_reg[j] = Bs[lastBuf][k][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
    }

    // =================== ESCRITURA DE C (float4) ============================
    const int cRow = by * BM + ty * TM;
    const int cCol = bx * BN + tx * TN;
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            float4 v;
            v.x = c_reg[i][j + 0];
            v.y = c_reg[i][j + 1];
            v.z = c_reg[i][j + 2];
            v.w = c_reg[i][j + 3];
            *reinterpret_cast<float4*>(&C[(cRow + i) * N + cCol + j]) = v;
        }
    }
}

// ============================================================================
//                            UTILIDADES DEL HOST
// ============================================================================

static void launch_sgemm(const float* dA, const float* dB, float* dC,
                         int M, int N, int K)
{
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);              // (16, 16, 1)
    dim3 grid (N / BN, M / BM, 1);                        // (N/128, M/128, 1)
    sgemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
}

// cuBLAS espera column-major; truco para fingir row-major: calcular B*A.
static void cublas_sgemm_rm(cublasHandle_t handle,
                            const float* dA, const float* dB, float* dC,
                            int M, int N, int K)
{
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,
                             dA, K,
                             &beta,
                             dC, N));
}

static void fill_random(float* p, size_t n, unsigned seed)
{
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        p[i] = (float)((rand() % 2001) - 1000) / 1000.0f;  // [-1, 1]
    }
}

static float max_abs_diff(const float* a, const float* b, size_t n)
{
    float maxd = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

// ============================================================================
//                                   MAIN
// ============================================================================
int main(int argc, char** argv)
{
    // Tamano por defecto (multiplos de 128 para encajar exactamente en los
    // tiles del kernel; sin codigo de borde para mantener el kernel limpio).
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

    printf("SGEMM  M=%d  N=%d  K=%d  (tile %dx%d, BK=%d, thread %dx%d)\n",
           M, N, K, BM, BN, BK, TM, TN);

    const size_t sA = (size_t)M * K;
    const size_t sB = (size_t)K * N;
    const size_t sC = (size_t)M * N;

    // ---- Host ----
    float* hA = (float*)std::malloc(sA * sizeof(float));
    float* hB = (float*)std::malloc(sB * sizeof(float));
    float* hC = (float*)std::malloc(sC * sizeof(float));
    float* hRef = (float*)std::malloc(sC * sizeof(float));
    fill_random(hA, sA, 1);
    fill_random(hB, sB, 2);

    // ---- Device ----
    float *dA, *dB, *dC, *dRef;
    CUDA_CHECK(cudaMalloc(&dA,   sA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB,   sB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC,   sC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRef, sC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sB * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // ---- Warm-up ----
    launch_sgemm(dA, dB, dC, M, N, K);
    cublas_sgemm_rm(handle, dA, dB, dRef, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Validacion numerica ----
    CUDA_CHECK(cudaMemcpy(hC,   dC,   sC * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hRef, dRef, sC * sizeof(float), cudaMemcpyDeviceToHost));
    float diff = max_abs_diff(hC, hRef, sC);
    printf("Max |diff| frente a cuBLAS: %.3e %s\n",
           diff, (diff < 1e-2f) ? "[OK]" : "[FALLO]");

    // ---- Benchmark ----
    const int iters = 20;
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Kernel propio
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) launch_sgemm(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_k;
    CUDA_CHECK(cudaEventElapsedTime(&ms_k, e0, e1));
    ms_k /= iters;

    // cuBLAS
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i)
        cublas_sgemm_rm(handle, dA, dB, dRef, M, N, K);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_c;
    CUDA_CHECK(cudaEventElapsedTime(&ms_c, e0, e1));
    ms_c /= iters;

    // RTX 4090 (Ada) FP32 CUDA-core peak: 82.6 TFLOP/s
    const double PEAK_TFLOPS_FP32 = 82.6;
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    const double tK = flops / (ms_k * 1.0e9);   // ms->s: /1e3, FLOP->TFLOP: /1e12  =>  /1e9
    const double tC = flops / (ms_c * 1.0e9);
    printf("Kernel propio : %7.3f ms   %8.2f TFLOP/s   (%.1f%% peak)\n",
           ms_k, tK, 100.0 * tK / PEAK_TFLOPS_FP32);
    printf("cuBLAS        : %7.3f ms   %8.2f TFLOP/s   (%.1f%% peak)\n",
           ms_c, tC, 100.0 * tC / PEAK_TFLOPS_FP32);
    printf("Ratio propio / cuBLAS: %.1f %%\n", 100.0 * tK / tC);

    // ---- Limpieza ----
    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dRef);
    std::free(hA); std::free(hB); std::free(hC); std::free(hRef);
    return 0;
}
