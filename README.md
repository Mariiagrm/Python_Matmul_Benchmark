
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
    ├── benchmarks/
    │   ├── aoti_compiled_models/
    │   ├── benchmark_aot_compile.py     # Static Compilation (Ahead-Of-Time)
    │   ├── benchmark_fp16.cu            # Native CUDA/cuBLAS Implementation
    │   ├── benchmark_fp16               # Compiled CUDA Executable
    │   ├── benchmark_fp16_fp16.py       # Classic PyTorch Mode (Eager)
    │   ├── benchmark_torch_compile.py   # Dynamic Compilation (JIT/Dynamo)
    │   ├── CUBLAScomparation/           # [TODO]
    │   ├── cuda_executor.sh             # Exclusive launcher script for CUDA
    │   └── login/
    │       ├── AOT_records.py
    │       └── JIT_records.py
    ├── benchmark_executor.sh            # Master script to launch all tests
    ├── plots/                           # Automatically generated graphs
    │   ├── tflops_fixed_k_comparative.png
    │   └── tflops_square_comparative.png
    ├── results/                         # CSV files with raw data
    │   ├── full_benchmarks.csv
    │   ├── rtx4090_benchmark_cuda.csv
    │   ├── rtx4090_benchmark_jit_specific.csv
    │   ├── rtx4090_pytorch_eager_specific.csv
    │   └── rtx4090_torch_aoti_benchmark.csv
    └── utilities/
        ├── createGraph.py               # Script for data visualization
        └── sortBenchmark.py             # Script for cleaning and sorting CSVs
    
    
## 🚀 How to Run
To launch the complete benchmark suite, simply run the master script from the project root:

    bash run_benchmarks.sh

 
 ## 🧪 Strategies Evaluated
**1. CUDA Native (benchmark_fp16.cu):** Maximum performance baseline using NVIDIA APIs (cuBLAS) directly from C++.

**2. PyTorch Eager (benchmark_fp16_fp16.py):** Standard operation-by-operation execution.

**3. PyTorch JIT (benchmark_torch_compile.py):** Uses torch.compile with:

- mode="max-autotune": Performs an exhaustive search by testing multiple tile configurations directly on the GPU to choose the absolute winner.
- dynamic=False: Disables dynamic shape inference to ensure maximum performance (assumes static sizes).

**4. PyTorch AOT (benchmark_aot_compile.py):** Pre-compilation. Uses fullgraph=True to ensure that 100% of the model is compiled on the GPU, silently preventing any return to the Python interpreter.

## **First Version**: FP16/FP32

## **Second Version**: FP16/FP16

![results_scuare](https://github.com/Mariiagrm/Python_Matmul_Benchmark/blob/main/plots/compare_tflops_square.png)

![results_fixed_k](https://github.com/Mariiagrm/Python_Matmul_Benchmark/blob/main/plots/compare_tflops_fixed_k.png)

| Type     | M     | N     | K     | Time_ms           | TFLOPS              | Mode                               |
|----------|-------|-------|-------|-------------------|---------------------|------------------------------------|
| Fixed_K  | 2046  | 2046  | 8192  | 0.218757          | 313.523             | rtx4090_benchmark_cuda             |
| Fixed_K  | 2046  | 2046  | 8192  | 0.2234582328796386| 306.92681876233274  | rtx4090_pytorch_eager_specific     |
| Square   | 32768 | 32768 | 32768 | 235.171           | 299.224             | rtx4090_benchmark_cuda             |
| Square   | 32768 | 32768 | 32768 | 235.449658203125  | 298.8695957968055   | rtx4090_torch_aoti_benchmark (fp16)|
| Square   | 32768 | 32768 | 32768 | 235.61572265625   | 298.6589493449383   | rtx4090_pytorch_eager_specific     |
| Square   | 16384 | 16384 | 16384 | 29.808            | 295.091             | rtx4090_benchmark_cuda             |
| Square   | 16384 | 16384 | 16384 | 29.87230224609375 | 294.45648178517007  | rtx4090_pytorch_eager_specific     |
| Square   | 16384 | 16384 | 16384 | 29.9085693359375  | 294.09942426229003  | rtx4090_torch_aoti_benchmark (fp16)|
| Fixed_K  | 2046  | 2046  | 8192  | 0.240199031829834 | 285.5353913024446   | rtx4090_torch_aoti_benchmark       |
| Square   | 16384 | 16384 | 16384 | 31.73125          | 277.2060042452787   | rtx4090_benchmark_jit_specific     |
| Square   | 4096  | 4096  | 4096  | 0.5069100952148438| 271.1308272796367   | rtx4090_pytorch_eager_specific     |
| Square   | 2046  | 2046  | 2046  | 0.0637328004837036| 268.771912453149    | rtx4090_pytorch_eager_specific     |
| Square   | 4096  | 4096  | 4096  | 0.5204377746582032| 264.0833547531458   | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 32768 | 32768 | 8192  | 67.36967163085937 | 261.12916418547184  | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 32768 | 32768 | 8192  | 67.502548828125   | 260.615137499611    | rtx4090_pytorch_eager_specific     |
| Fixed_K  | 32768 | 32768 | 8192  | 67.5091           | 260.59              | rtx4090_benchmark_cuda             |
| Fixed_K  | 16384 | 16384 | 8192  | 17.01529541015625 | 258.4760596327263   | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 16384 | 16384 | 8192  | 17.0625146484375  | 257.76074639190136  | rtx4090_pytorch_eager_specific     |
| Fixed_K  | 8192  | 8192  | 8192  | 4.274585723876953 | 257.22062880488164  | rtx4090_torch_aoti_benchmark       |
| Fixed_K  | 16384 | 16384 | 8192  | 17.1194           | 256.904             | rtx4090_benchmark_cuda             |
| Fixed_K  | 4096  | 4096  | 8192  | 1.0751696014404295| 255.66004337896055  | rtx4090_pytorch_eager_specific     |
| Square   | 8192  | 8192  | 8192  | 4.305270385742188 | 255.38735764825947  | rtx4090_torch_aoti_benchmark       |
Fixed_K|32768|32768|8192|69.21912231445313|254.152110806853|rtx4090_benchmark_jit_specific|
Square|8192|8192|8192|4.326647644042969|254.1255304877516|rtx4090_pytorch_eager_specific|
Fixed_K|4096|4096|8192|1.082449951171875|253.9405232051731|rtx4090_torch_aoti_benchmark|
Fixed_K|8192|8192|8192|4.3406298828125|253.30692951493344|rtx4090_pytorch_eager_specific|
