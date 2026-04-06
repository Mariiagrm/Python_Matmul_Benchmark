
## 🚀 Extreme GEMM Optimization on RTX 4090 (Ada Lovelace)
This repository documents a comprehensive study to achieve the peak performance of the Ada Lovelace architecture (RTX 4090) in matrix multiplication operations (GEMM).

We compared the performance of different PyTorch compilation strategies (Eager, JIT, AOT) against native C++/CUDA implementations (cuBLAS).
- **First Version**: FP16/FP32
- **Second Version**: FP16/FP16

## 🔬 Testing Methodology
To ensure accurate measurements and avoid thermal throttling, all matrices were initialized to 0. Random data increases transistor switching activity, forcing the hardware to reduce clock frequencies due to power limitations.

The study is divided into two main phases:

**Phase 1: Square Matrices (Full Scaling)**
Analyze the hardware's response to incremental loads. Small (1024, 2046), medium (4096, 8192), and large (16384, 32768) dimensions were tested.

**Phase 2: Non-Square Matrices (Partial Load / Fixed-K)**
Designed to maintain sustained arithmetic intensity. The internal dimension was fixed at K=8192 to ensure Tensor Core saturation, while varying the M and N dimensions.

## 🛠️ Requirements and Dependencies
**Runoff Environment:**
- NVIDIA Container Toolkit: Required to expose the GPUs to the Docker container.
- Base Image: nvcr.io/nvidia/pytorch:23.11-py3 (Includes CUDA 12.0 and PyTorch 2.0+ pre-configured).

**Build Toolchain:**
- C++17 Compiler: gcc-10 and g++10
- Building Systems: cmake (>= 3.18), ninja-build, build-essential

**Core Libraries:**
- NVIDIA CUTLASS: C++ template library for high-performance linear algebra operations (compiled from source for 80 and 89 architectures).
- Python 3 and utilities (python3-dev, pip).

📂 Project Structure


      .
      ├── benchmarks_executor.sh        # 🧠 Main entry point: orchestrates all benchmark executions
      │
      ├── fp16_fp16_to_fp16             # 🔬 Experiments: FP16 → FP16 precision pipeline
      │   ├── benchmarks                # ⚙️ Core benchmark implementations
      │   │   ├── benchmark_aot_compile.py   # AOT (Ahead-Of-Time) compilation benchmark
      │   │   ├── benchmark_fp16             # Compiled binary for CUDA benchmark
      │   │   ├── benchmark_fp16.cu          # CUDA kernel implementation (FP16)
      │   │   ├── benchmark_fp16_fp16.py     # Python benchmark (FP16 → FP16)
      │   │   ├── benchmark_torch_compile.py # Torch compile benchmark
      │   │   ├── cuda_executor.sh           # Script to launch CUDA benchmarks
      │   │   ├── log                       # Execution logs
      │   │   └── torch_compile_debug       # Debug artifacts for torch.compile
      │   │
      │   ├── images                    # 🖼️ Generated images (plots, visual outputs)
      │   ├── nvidia_nsight             # 📊 Profiling outputs (Nsight Systems/Compute)
      │   ├── plots                     # 📈 Performance plots
      │   │   ├── compare_tflops_fixed_k.png
      │   │   └── compare_tflops_square.png
      │   │
      │   ├── results                   # 📄 Raw benchmark results (CSV format)
      │   │   ├── benchmarkCompleto.csv
      │   │   ├── rtx4090_benchmark_cuda.csv
      │   │   ├── rtx4090_benchmark_jit.csv
      │   │   ├── rtx4090_pytorch_eager.csv
      │   │   └── rtx4090_torch_aoti_benchmark.csv
      │   │
      │   ├── unique_results            # 🧹 Filtered/processed benchmark results
      │   │   ├── rtx4090_benchmark_jit.csv
      │   │   ├── rtx4090_benchmark_jit_optimized.csv
      │   │   ├── rtx4090_pytorch_eager_specific.csv
      │   │   └── rtx4090_torch_aoti_benchmark.csv
      │   │
      │   ├── unitary_benchmarks        # 🧪 Isolated benchmarks for individual testing
      │   │   ├── benchmark_aot_compile.py
      │   │   ├── benchmark_fp16
      │   │   ├── benchmark_fp16.cu
      │   │   ├── benchmark_fp16_fp16.py
      │   │   ├── benchmark_torch_compile.py
      │   │   └── cuda_executor.sh
      │   │
      │   └── utils                     # 🛠️ Utility scripts
      │       ├── plotCreate.py         # Plot generation
      │       └── sortBenchmark.py      # Result sorting/processing
      │
      ├── fp16_fp16_to_fp32             # 🔬 Experiments: FP16 → FP32 precision pipeline
      │   ├── analisis.md               # 📝 Analysis and notes for this configuration
      │   │
      │   ├── benchmarks                # ⚙️ Benchmark implementations
      │   │   ├── benchmark_aot_compile.py
      │   │   ├── benchmark_fp16
      │   │   ├── benchmark_fp16.cu
      │   │   ├── benchmark_fp16_fp32.py   # Python benchmark (FP16 → FP32)
      │   │   ├── benchmark_torch_compile.py
      │   │   ├── cuda_executor.sh
      │   │   ├── log
      │   │   └── mma-matmul            # 🧮 Advanced CUDA MMA (matrix multiply) experiments
       [Rest is simetric structure]
    
    
## 🚀 How to Run
To launch the complete benchmark suite, simply run the master script from the project root:

    ./benchmarks_executor.sh [OPTIONS]
### ⚙️ Available Options
**--a [fp16 | fp32]**
- Description: Specifies the architecture mode (precision).
Values:
- fp16 → Half precision (default)
- fp32 → Single precision

Example:
    ./script.sh --a fp32

**--profile**
Description: Enables profiling mode.
Behavior: Activates profiling tools such as Nsight Systems and Nsight Compute.
Example:
    ./script.sh --profile
    
**--sb**
Description: Skips benchmark execution.
Behavior: Prevents benchmarks from running (useful for faster runs or debugging).
Example:
    ./script.sh --sb

**-h**
Description: Displays help information.
Example:
    ./script.sh --help
 
 ## 🧪 Strategies Evaluated
**1. CUDA Native:** Maximum performance baseline using NVIDIA APIs (cuBLAS) directly from C++.

**2. PyTorch Eager:** Standard operation-by-operation execution.

**3. PyTorch JIT:** Uses torch.compile with:

- mode="max-autotune": Performs an exhaustive search by testing multiple tile configurations directly on the GPU to choose the absolute winner.
- dynamic=False: Disables dynamic shape inference to ensure maximum performance (assumes static sizes).

**4. PyTorch AOT:** Pre-compilation. Uses fullgraph=True to ensure that 100% of the model is compiled on the GPU, silently preventing any return to the Python interpreter.

**5. Benchmark mma-matmul (FP16/FP32 Precision):** 
This section evaluates the performance of High-Speed Matrix Multiplication (GEMM) using Tensor Cores on the NVIDIA Ada Lovelace architecture, based on the mma-matmul implementation.

The cublasGemmEx API, utilizing FP16 inputs and FP32 accumulation, serves as the primary performance reference. For a matrix of dimensions $M=N=K=4096$, cuBLAS achieves an execution time of $895\ \mu s$. 

To conduct a comprehensive comparison, a custom testbed (ejecutador.sh) was utilized to analyze a wide variety of matrix sizes.
The benchmark covers the evolutionary progression of all kernels developed in this project, including versions 0.x, 1.x, 2.x, and 3.x (specifically kernels 0, 1, 10, 11, 20, 21, 30, 31, 32, 33, and 34). This allows for a detailed observation of performance gains—from the initial "naive" implementation to advanced asynchronous pipelining.

## **First Version**: FP16/FP32

![results_scuare](https://github.com/Mariiagrm/Python_Matmul_Benchmark/blob/main/fp16_fp16_to_fp32/plots/compare_tflops_square.png)

![results_fixed_k](https://github.com/Mariiagrm/Python_Matmul_Benchmark/blob/main/fp16_fp16_to_fp32/plots/compare_tflops_fixed_k.png)


| Type    | M     | N     | K     | Time_ms           | TFLOPS             | Mode                         |
|---------|-------|-------|-------|-------------------|--------------------|------------------------------|
| Square  | 16384 | 16384 | 16384 | 50.952294921875   | 172.6338928539926  | rtx4090_torch_aoti_benchmark |
| Fixed_K | 32768 | 32768 | 8192  | 102.01292724609377| 172.45055621212586 | rtx4090_torch_aoti_benchmark |
| Square  | 32768 | 32768 | 32768 | 408.083154296875  | 172.43726783799482 | rtx4090_torch_aoti_benchmark |
| Fixed_K | 16384 | 16384 | 8192  | 25.530450439453126| 172.26670252192417 | rtx4090_torch_aoti_benchmark |
| Square  | 8192  | 8192  | 8192  | 6.6               | 166.5927           | benchmark_mma-matmul         |
| Fixed_K | 8192  | 8192  | 8192  | 6.6               | 166.5927           | benchmark_mma-matmul         |
| Fixed_K | 32768 | 32768 | 8192  | 107.585986328125  | 163.51744911053595 | rtx4090_pytorch_eager        |
| Fixed_K | 16384 | 16384 | 8192  | 26.90126953125    | 163.48843707896336 | rtx4090_pytorch_eager        |
| Square  | 32768 | 32768 | 32768 | 430.82796875      | 163.33374172951486 | rtx4090_pytorch_eager        |
| Square  | 16384 | 16384 | 16384 | 53.908994140625   | 163.16559346781426 | rtx4090_pytorch_eager        |
| Square  | 16384 | 16384 | 16384 | 53.91             | 163.1625           | benchmark_mma-matmul         |
| Square  | 8192  | 8192  | 8192  | 6.7856591796875   | 162.03460837929026 | rtx4090_pytorch_eager        |
| Fixed_K | 8192  | 8192  | 8192  | 6.78956298828125  | 161.94144301684088 | rtx4090_pytorch_eager        |
| Fixed_K | 4096  | 4096  | 8192  | 1.7139181518554687| 160.37983298468495 | rtx4090_pytorch_eager        |
| Fixed_K | 16384 | 16384 | 8192  | 27.43             | 160.3371           | benchmark_mma-matmul         |
| Square  | 32768 | 32768 | 32768 | 438.97            | 160.3042           | benchmark_mma-matmul         |
| Square  | 4096  | 4096  | 4096  | 0.8586966705322265| 160.05530030390628 | rtx4090_pytorch_eager        |
| Fixed_K | 2046  | 2046  | 8192  | 0.4285235214233398| 160.05031489565383 | rtx4090_pytorch_eager        |
| Fixed_K | 4096  | 4096  | 8192  | 1.718074951171875 | 159.99180172931898 | rtx4090_torch_aoti_benchmark |
| Square  | 4096  | 4096  | 4096  | 0.8640716552734375| 159.05967130527748 | rtx4090_torch_aoti_benchmark |
| Square  | 8192  | 8192  | 8192  | 7.004569244384766 | 156.97062723127863 | rtx4090_torch_aoti_benchmark |
| Fixed_K | 2046  | 2046  | 8192  | 0.4376678466796875| 156.7063357848058  | rtx4090_torch_aoti_benchmark |
| Fixed_K | 8192  | 8192  | 8192  | 7.073792266845703 | 155.4345372749102  | rtx4090_torch_aoti_benchmark |
| Fixed_K | 4096  | 4096  | 8192  | 1.77              | 155.2983           | benchmark_mma-matmul         |
| Fixed_K | 32768 | 32768 | 8192  | 114.66            | 153.4291           | benchmark_mma-matmul         |
| Square  | 4096  | 4096  | 4096  | 0.89738           | 153.1558           | benchmark_mma-matmul         |
| Square  | 2046  | 2046  | 2046  | 0.1132630443572998| 151.23720865177324 | rtx4090_pytorch_eager        |
| Fixed_K | 2046  | 2046  | 8192  | 0.46365           | 147.9248           | benchmark_mma-matmul         |
| Fixed_K | 1024  | 1024  | 8192  | 0.1180160045623779| 145.57236747427334 | rtx4090_pytorch_eager        |
| Square  | 2046  | 2046  | 2046  | 0.1209424018859863| 141.6342523786508  | rtx4090_torch_aoti_benchmark |
| Fixed_K | 1024  | 1024  | 8192  | 0.128089599609375 | 134.1238417201094  | rtx4090_torch_aoti_benchmark |
| Square  | 2046  | 2046  | 2046  | 0.14605           | 117.2858           | benchmark_mma-matmul         |



## **Second Version**: FP16/FP16

![results_scuare](https://github.com/Mariiagrm/Python_Matmul_Benchmark/blob/main/fp16_fp16_to_fp16/plots/compare_tflops_square.png)

![results_fixed_k](https://github.com/Mariiagrm/Python_Matmul_Benchmark/blob/main/fp16_fp16_to_fp16/plots/compare_tflops_fixed_k.png)

| Type     | M     | N     | K     | Time_ms           | TFLOPS              | Mode                               |
|----------|-------|-------|-------|-------------------|---------------------|------------------------------------|
| Fixed_K  | 2046  | 2046  | 8192  | 0.218757          | 313.523             | rtx4090_benchmark_cuda             |
| Fixed_K  | 2046  | 2046  | 8192  | 0.2234582328796386| 306.92681876233274  | rtx4090_pytorch_eager     |
| Square   | 32768 | 32768 | 32768 | 235.171           | 299.224             | rtx4090_benchmark_cuda             |
| Square   | 32768 | 32768 | 32768 | 235.449658203125  | 298.8695957968055   | rtx4090_torch_aoti_benchmark (fp16)|
| Square   | 32768 | 32768 | 32768 | 235.61572265625   | 298.6589493449383   | rtx4090_pytorch_eager     |
| Square   | 16384 | 16384 | 16384 | 29.808            | 295.091             | rtx4090_benchmark_cuda             |
| Square   | 16384 | 16384 | 16384 | 29.87230224609375 | 294.45648178517007  | rtx4090_pytorch_eager     |
| Square   | 16384 | 16384 | 16384 | 29.9085693359375  | 294.09942426229003  | rtx4090_torch_aoti_benchmark (fp16)|
| Fixed_K  | 2046  | 2046  | 8192  | 0.240199031829834 | 285.5353913024446   | rtx4090_torch_aoti_benchmark       |
| Square   | 16384 | 16384 | 16384 | 31.73125          | 277.2060042452787   | rtx4090_benchmark_jit     |
| Square   | 4096  | 4096  | 4096  | 0.5069100952148438| 271.1308272796367   | rtx4090_pytorch_eager     |
| Square   | 2046  | 2046  | 2046  | 0.0637328004837036| 268.771912453149    | rtx4090_pytorch_eager     |
| Square   | 4096  | 4096  | 4096  | 0.5204377746582032| 264.0833547531458   | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 32768 | 32768 | 8192  | 67.36967163085937 | 261.12916418547184  | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 32768 | 32768 | 8192  | 67.502548828125   | 260.615137499611    | rtx4090_pytorch_eager     |
| Fixed_K  | 32768 | 32768 | 8192  | 67.5091           | 260.59              | rtx4090_benchmark_cuda             |
| Fixed_K  | 16384 | 16384 | 8192  | 17.01529541015625 | 258.4760596327263   | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 16384 | 16384 | 8192  | 17.0625146484375  | 257.76074639190136  | rtx4090_pytorch_eager     |
| Fixed_K  | 8192  | 8192  | 8192  | 4.274585723876953 | 257.22062880488164  | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 16384 | 16384 | 8192  | 17.1194           | 256.904             | rtx4090_benchmark_cuda             |
| Fixed_K  | 4096  | 4096  | 8192  | 1.0751696014404295| 255.66004337896055  | rtx4090_pytorch_eager     |
| Square   | 8192  | 8192  | 8192  | 4.305270385742188 | 255.38735764825947  | rtx4090_torch_aoti_benchmark       |
Fixed_K|32768|32768|8192|69.21912231445313|254.152110806853|rtx4090_benchmark_jit|
Square|8192|8192|8192|4.326647644042969|254.1255304877516|rtx4090_pytorch_eager|
Fixed_K|4096|4096|8192|1.082449951171875|253.9405232051731|rtx4090_torch_aoti_benchmark|
Fixed_K|8192|8192|8192|4.3406298828125|253.30692951493344|rtx4090_pytorch_eager|

## Results Eager

![Eager Memory](/fp16_fp16_to_fp16/images/eager_memory.png)
![Eager Warp v2](/fp16_fp16_to_fp16/images/EAGER_warp(v2).png)
![GPU Speed of Light Eager](/fp16_fp16_to_fp16/images/GPU_speedofLight_eager.png)
![PM Sampling Eager](/fp16_fp16_to_fp16/images/PM_sampling_eager.png)

---

## Results JIT

![JIT Memory](/fp16_fp16_to_fp16/images/jit_memory.png)
![JIT Warp v2](/fp16_fp16_to_fp16/images/JIT_warp(v2).png)
![GPU Speed of Light JIT](/fp16_fp16_to_fp16/images/GPU_speedOfLight_jit.png)
![PM Sampling JIT](/fp16_fp16_to_fp16/images/PM_sampling_jit.png)

---

## Results AOT

![AOT Memory](/fp16_fp16_to_fp16/images/aot_memory.png)
![AOT Warp v2](/fp16_fp16_to_fp16/images/AOT_warp(v2).png)
![GPU Speed of Light AOT](/fp16_fp16_to_fp16/images/GPU_speedOfLight_aot.png)
![PM Sampling AOT](/fp16_fp16_to_fp16/images/PM_sampling_aot.png)

