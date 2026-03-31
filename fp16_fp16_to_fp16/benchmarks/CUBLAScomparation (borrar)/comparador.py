Script que llame a 
./cublas/hgemm_cublas.cu
./hgemm_cublaslt_auto_tuning.cu
./hgemm_cublaslt_heuristic.cu

segun
    dims_base = [1024, 2046, 4096, 8192, 16384, 32768] 
    
    # Benchmark 1: Matrices Cuadradas (M = N = K)
    # Genera: (1024,1024,1024), (2046,2046,2046)...
    bench_1_combs = [("Square", d, d, d) for d in dims_base]
    
    # Benchmark 2: M y N varían, K fijo en 8192
    # Genera: (1024, 1024, 8192), (1024, 2046, 8192)...
    K_fixed = 8192
    bench_2_combs = []
    # Producto cartesiano solo de M y N
    #mn_combs = list(itertools.product(dims_base, dims_base)) 
    for i in dims_base:
        bench_2_combs.append(("Fixed_K", i, i, K_fixed))

    # Unimos ambas listas de tareas
    all_tasks = bench_1_combs + bench_2_combs


para: 
#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// from hgemm_cublas.cu
void init_cublas_handle();
void destroy_cublas_handle();
void hgemm_cublas_nn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);
void hgemm_cublas_tn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);

void init_cublaslt_handle_v1();
void destroy_cublaslt_handle_v1();
void hgemm_cublaslt_heuristic_nn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);
void hgemm_cublaslt_heuristic_tn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);

void init_cublaslt_handle_v2();
void destroy_cublaslt_handle_v2();
void find_best_algo_nn_v2_torch(int M, int N, int K);
void find_best_algo_tn_v2_torch(int M, int N, int K);
void hgemm_cublaslt_auto_tuning_nn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);
void hgemm_cublaslt_auto_tuning_tn(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);


void cuda_l2_a100_fp16(torch::Tensor a, torch::Tensor b, torch::Tensor b_col_major, torch::Tensor c);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // cuBLAS Tensor Cores
  TORCH_BINDING_COMMON_EXTENSION(init_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(destroy_cublas_handle)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_nn)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublas_tn)

  // cuBLAS LT Tensor Cores V1
  TORCH_BINDING_COMMON_EXTENSION(init_cublaslt_handle_v1)
  TORCH_BINDING_COMMON_EXTENSION(destroy_cublaslt_handle_v1)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublaslt_heuristic_nn)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublaslt_heuristic_tn)

  // cuBLAS LT Tensor Cores V2
  TORCH_BINDING_COMMON_EXTENSION(init_cublaslt_handle_v2)
  TORCH_BINDING_COMMON_EXTENSION(destroy_cublaslt_handle_v2)
  TORCH_BINDING_COMMON_EXTENSION(find_best_algo_nn_v2_torch)
  TORCH_BINDING_COMMON_EXTENSION(find_best_algo_tn_v2_torch)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublaslt_auto_tuning_nn)
  TORCH_BINDING_COMMON_EXTENSION(hgemm_cublaslt_auto_tuning_tn)

  // Cutlass
  TORCH_BINDING_COMMON_EXTENSION(cuda_l2_a100_fp16)
}
