

# High performance of CUTE and CUTE DSL

## How to achieve a higher performance using CUTE doing a Runtime Dispatch

## The latest of NVIDIA

Engeeniers of NVIDIA says that we  will not achieve significantly higher performance using straight-up CUTLASS (traditional C++) in comparison with a well-designed kernel using CuTe (if partitioning and pipeline optimization are performed) , since CUTLASS 3.0 onwards and later versions are built internally on top of CuTe abstractions. 


Also they proposes sighly advantages on using CUTE versus CutLass:
- **Tensor Core Efficiency** : CuTe allows utilization levels close to 95-98% of the theoretical peak on modern architectures (Blackwell or Hopper) for dense GEMM operations. In practice, a kernel written with CuTe can match the performance of cuBLAS for standard operations.
- **Ease of Optimization** : The main advantage of CuTe is that it handles the mathematical **bookkeeping** of layouts and thread partitioning algebraically. 
This allows for much faster iteration in critical optimizations such as:
    Swizzling shared memory to avoid bank conflicts.
    Asynchronous pipelining (using TMA on compatible architectures or cp.asyncon SM89).
- **Using CuTe DSL Python**: (introduced in CUTLASS 4.0), it can achieve C++-like efficiency with compilation times up to 100 times faster . Only a slight penalty is observed on very small problems due to initial synchronization costs but this  NVIDIA continues to optimize.

In all cases, this will improve peak performance, ultimately making it necessary as a last resort to hand-written (PTX) implementations that it can provide a 7-14% performance boost in very specific and combined kernels (such as Softmax + Top-K) but it adds significant debugging complexity and a lack of portability. 

## CUTE
 ### First Implementation

    --- Benchmark RTX 4090 CUTE (FP16 / Tensor Cores) ---
    Dimensiones (M x N x K): 16384 x 16384 x 16384

    Type        M       N       K       cuBLAS (TFLOPS)   CuTe (TFLOPS)
    -------------------------------------------------------------------------------
    Custom      16384   16384   16384   294.25            210.91
    

It it's interesting that the output  is 294.25 TFLOPS for cuBLAS and 210.91 TFLOPS for a custom CUTLASS/CuTe implementation.

Seeing my custom code run ~83 TFLOPS slower than cuBLAS can be frustrating, but this is a very expected result. Achieving ~63% of the RTX 4090's theoretical maximum (~330 TFLOPS dense FP16) but in comparison cuBLAS is hitting ~89% efficiency.


On the basis of the first version of my kernel that outperformed the CUTLASS baseline:
- **The Pipeline "Stages" Bottleneck** 
Is the major factor, for guarantees the safest shared memory limits and not produce internal out of memory errors, it strictly limited the CUTLASS configuration to 2 pipeline stages.

```cpp
using CutlassGemm = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock shape
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp shape
    cutlass::gemm::GemmShape<16, 8, 16>,     // Instruction shape
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2 // Stages (Restricted to 2 to safely fit in 48KB default shared memory)
>;
```


That is because matrix multiplication relies heavily on hiding memory latency. While the Tensor Cores are crunching math, the GPU needs to be fetching the next chunks of data from global memory into shared memory.

The collateral problem is that 2 stages mean the GPU can only queue up a tiny amount of memory in advance. cuBLAS, on the other hand, is likely using 3, 4, or even 5 stages.

To use more stages, the kernel requires more than the default 48KB of shared memory per block. cuBLAS dynamically requests extra shared memory from the driver at runtime, whereas our static CUTLASS template was constrained by default safety limits.

- **Sub-Optimal Tile Sizing**
In the code, I hardcoded a Tile Size of 128x128x32. That supposes that in a massive such as 16384×16384×16384 matrix, 128x128 is likely too small.

Wheras cuBLAS contains a massive database of "heuristics." Before it executes cublasHgemm, it looks at your exact matrix dimensions and GPU (RTX 4090, in my case), and instantly selects a highly tuned kernel—perhaps using a 256x128x64 or 128x256x64 tile size which results in far better L2 cache reuse.

- **Architecture Specificity (Sm80 vs Sm89)**
This shows the importance of compile with our exact arquitecture, the RTX 4090 uses the Ada Lovelace architecture (Sm89). Compilling the kernel targeting Sm80 (Ampere) misses out on a few Ada-specific hardware features.

Other asset for cuBLAS is that automatically dispatches an Sm89-native kernel compiled directly by NVIDIA engineers.

CuTe DSL allows low-level programming for NVIDIA Ampere, Hopper, and Ada architectures, including SM89. Usually it is used SM80 kernels as a fallback, which includes fp16 support, often tested on L4/RTX 4090 GPUs. However, employ SM89 functionality requires CUDA 12.4 or newer. 

- **Hand-Tuned Assembly (SASS)**
CUTLASS is a C++ template library that relies on the nvcc compiler to translate its instructions into PTX, and then into SASS (machine code). NVIDIA's cuBLAS engineers often write their most critical kernels directly in SASS, allowing them to perfectly schedule memory loads and math instructions on a clock-cycle-by-clock-cycle basis to prevent the Tensor Cores from been starving for data.

To  match cuBLAS with CUTLASS, we have to start tuning the software pipeline to utilize the massive 100KB+ (of each Streaming Multiprocessor) shared memory available on the 4090 GPU.

But it requires to write carefully  CUDA driver calls to unlock the RTX 4090's extended shared memory, allowing us to increase the pipeline stages and close that performance gap


### Upgrades to consider
To take advantage of all GPU's resources than our conservative first fallback configuration we need to do two things, increase the 
- Tile Size 
- Increase the Pipeline Stages.

However, doing this requires more than 48KB of Shared Memory per block. By default, the CUDA driver hard-caps shared memory at 48KB to ensure older programs don't crash. To use the massive 100KB+ shared memory available on  RTX 4090, we have to explicitly unlock it using a specific CUDA driver API call.

### Step 1: Upgrade the CUTLASS Configuration
We are going to change the GemmShape to process much larger chunks of the matrix at a time (256×128) and increase the Stages from 2 to 4. This will allow the Tensor Cores to chew through data while the pipeline fetches the next 3 blocks in the background.

Replace the CutlassGemm definition in your code with this:

    C++
    // High-Performance CUTLASS configuration
    using CutlassGemm = cutlass::gemm::device::Gemm<
        ElementInput, cutlass::layout::ColumnMajor,
        ElementInput, cutlass::layout::ColumnMajor,
        ElementOutput, cutlass::layout::ColumnMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,                     // Ampere/Ada Architecture
        cutlass::gemm::GemmShape<256, 128, 64>,  // Massive Threadblock Tile! (was 128x128x32)
        cutlass::gemm::GemmShape<64, 64, 64>,    // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 16>,     // MMA Instruction Shape
        EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        4                                        // Stages: Deep pipelining to hide latency!
    >;
### Step 2: Unlock the GPU's Shared Memory
Because our new configuration uses 4 stages of a massive 256×128 tile, it will require way more than 48KB of shared memory.

In the function of benchmark before calling our gemm operation, we have to insert the driver call ´cudaFuncSetAttribut´ with the amount of shared memory that they need:

    C++
        // 0. UNLOCK EXTENDED SHARED MEMORY
        // Calculate exactly how much shared memory this aggressive configuration needs
        int smem_size = int(sizeof(typename CutlassGemm::GemmKernel::SharedStorage));
        
        // If it needs more than the default 48KB, tell the driver to allow up to 100KB+
        if (smem_size > (48 * 1024)) {
            cudaError_t result = cudaFuncSetAttribute(
                cutlass::Kernel<typename CutlassGemm::GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size
            );
            if (result != cudaSuccess) {
                std::cerr << "Failed to unlock extended shared memory: " 
                        << cudaGetErrorString(result) << std::endl;
                return -1.0f;
            }
        }

This in comparison with before by loading a larger square  tile of the matrix into the SM, the Tensor Cores can reuse the same data many more times before having to go back to VRAM. This massively reduces memory bandwidth bottlenecks.

And related with the pipeline stages instead of just loading "A" while calculating "B", the GPU is now calculating the current tile, while simultaneously loading the next 3 tiles into a revolving queue in shared memory. The Tensor Cores will practically never sit idle waiting for data.

### The Final Step: Autotuning
cuBLAS is fast because it has an internal lookup table of the absolute perfect shapes for every dimension named [Nvidia matmul heuristics](https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/)

Now if we use a tile 256×128×64 with 4 stages doesn't immediately match cuBLAS, it just means we need a slightly different shape for a 16384 
3 matrix. We have to try tweaking the GemmShape (e.g., <128, 256, 64>, <256, 256, 64> with 3 stages, or <128, 128, 64> with 5 stages). This process of finding the optimal tile shape for a specific matrix size is called autotuning, and it is exactly how AI libraries (like PyTorch and vLLM) squeeze the maximum TFLOPS out of CuTe

As you have notice right now, it is a completely hadicap for who is doing a wide benchmarks (as a library) with size of matrix that doesn´t have anything in common, but ¿how really do autotuning to reach cuBLAS with cute?

This issue is the definite issue for cuBLAS is so dominant out-of-the-box! Doing it for arbitrary sizes is a massive headache.

The fundamental problem here is CUTLASS and CuTe are heavily dependent on C++ templates resolved at compile time, you cannot pass runtime variables into a GemmShape, compiler needs exact numbers before the program even runs.

To build a true autotuner that rivals cuBLAS, you have to use a technique called Runtime Dispatch as well as in the production environments of PyTorch, vLLM, and TensorRT.

### Runtime Dispatch

- Define a Search Space at Compile Time: We need to pre-compile a dozen different highly-optimized **CutlassGemm** configurations (different tile sizes and pipeline stages) into the binary. But it would be tricky and a complete  mess because it depends directly on the hardware of our GPU.

 The safety net we put in place worked perfectly, but I know exactly why it tripped. If you get a little too greedy with the pipeline stages, it will go out of memory. The **RTX 4090** uses the **Ada Lovelace** architecture, which has a hard hardware limit of **128 KB of Shared Memory (SMEM) per Threadblock**.

If we look at the math for `Gemm_128x256x64_4` (128x256 tile, $K=64$, 4 stages), it requires holding 4 distinct chunks of the matrix in memory at once. That comes out to roughly **192 KB** of shared memory. Because it asks for more memory than the GPU physically has, [CUTLASS's internal `can_implement()`](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cutlass_3x_backwards_compatibility.html\) check correctly rejected it. In fact, all three configurations I gave you exceeded the 128 KB limit, so the autotuner rejected all of them and threw a fatal error.

We have to be **constantly** aware of the **boundaries** of our GPU search space. Here are some corrected examples:

 Config 0: Standard Tile, 3 Stages (\~96 KB SMEM — Pushing Ada limits\!)

  * $(128 \times 64) + (64 \times 128) = 16,384$ elements
  * $16,384 \text{ elements} \times 2 \text{ bytes/element} = 32,768 \text{ bytes (32 KB)}$
  * $32 \text{ KB} \times 3 \text{ stages} = 96 \text{ KB}$



```cpp
using Gemm_128x128x64_3 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3
>;
```

 Config 1: Wide Tile, Thinner K, 3 Stages (\~72 KB SMEM)

```cpp
using Gemm_256x128x32_3 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 3
>;
```

 Config 2: Safe Fallback, 4 Stages (\~64 KB SMEM)

```cpp
using Gemm_128x128x32_4 = cutlass::gemm::device::Gemm<
    ElementInput, cutlass::layout::ColumnMajor,
    ElementInput, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4
>;
```

- The Benchmarking Loop in Run Time: When the user asks to multiply a specific $M \times N \times K$ matrix, the program loops through all the pre-compiled functions, runs a quick 5-iteration benchmark for each, and records the time.

- The Dispatch (Run Time): The program selects the kernel that returned the lowest time and executes the real payload.

        void autotune_and_run(int m, int n, int k, cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C) {
        
        // Create a list of our pre-compiled configurations
        std::vector<std::function<float()>> search_space = {
            [&]() { return test_config<Gemm_128x128x64_4>(m, n, k, A, B, C); },
            [&]() { return test_config<Gemm_256x128x64_3>(m, n, k, A, B, C); },
            [&]() { return test_config<Gemm_128x256x64_4>(m, n, k, A, B, C); }
        };

        float best_time = std::numeric_limits<float>::max();
        int best_idx = -1;

        std::cout << "Autotuning " << search_space.size() << " configurations...\n";

        // Fight it out!
        for (int i = 0; i < search_space.size(); ++i) {
            float time = search_space[i]();
            
            if (time > 0 && time < best_time) {
                best_time = time;
                best_idx = i;
            }
        }

        std::cout << "Best kernel found! Index: " << best_idx << " | Time: " << best_time << " ms\n";
        
        // Now you can permanently cache this `best_idx` for these specific (M,N,K) dimensions!
    }
    
    

So, we are forcing the ´nvcc´ compiler to generate SASS machine code for every single configuration in your search space.

By putting 50 configurations in search space, the benchmark program will take 5 to 10 minutes to compile, and the resulting .exe or .out file will be quite large. This is why PyTorch takes hours to compile from source. They are compiling hundreds of CUTLASS template combinations to ensure they have the absolute best kernel for whatever matrix size it throw at it.

To make this fully robust, we would ideally want to save the winning configuration in a unordered_map cache to permanently save the winning heuristic so we don't have to re-tune it every single time the program runs.


#### Unordered_map cache

To completely eliminate the "autotuning tax" on subsequent runs, we need to build a Persistent Heuristic Cache using an std::unordered_map to look up an $(M, N, K)$ combination is incredibly fast ($O(1)$ time complexity). The persits data we are going to use a .csv file.

For implement that there are some strocke brush strokes: 

- **Step 1**: Define the Key and the Hash Function An unordered_map uses a hash table under the hood. Out of the box, C++ doesn't know how to hash a custom 3D coordinate like $(M, N, K)$. We must define a struct for our dimensions and write a custom hash function for it.
- **Step 2**: Create a class that wraps the std::unordered_map. It will handle loading the saved heuristics from a file when your program starts, and writing new discoveries back to the file.
- **Step 3**: Integrating it into your AutotunerNow, we just wrap your benchmarking loop with this cache logic. If the map has the answer, we skip the 5-minute benchmark process entirely.

With that every time when we request a size of matrix that I have calculate before, it will take time to test all the options. The second time you run the script, it will take a few microseconds to read the .csv file, instantly select the winning C++ template, and bypass the benchmark phase.It is also a way of doing crowdsourcing when it is working on a team or have a cluster of 4090s, you can easily merge these files. You let different machines benchmark different dimensions and combine their findings into a file that everyone shares.

## Results

The results speak for themselves: by implementing a runtime dispatch and an autotuning cache, we have successfully matched—and even slightly exceeded—the cuBLAS baseline.

    --- Benchmark RTX 4090 CUTE (FP16 / Tensor Cores) ---
    Dimensiones (M x N x K): 16384 x 16384 x 16384

    Running cuBLAS baseline...
    [INFO] Cache file loaded successfully. (0 saved heuristics)
    [CACHE MISS] Autotuning 3 configurations...
    [AUTOTUNE COMPLETE] Best Kernel Index: 1 | Time: 29.7927 ms

    Type        M       N       K       cuBLAS (TFLOPS)   CuTe (TFLOPS)
    ...............................................................................
    Custom      16384   16384   16384   294.63            295.24

While achieving 295 TFLOPS is a massive win, this exercise highlights a major bottleneck: writing and tuning raw C++/CuTe kernels is fundamentally unscalable for everyday development. Squeezing out this performance requires a profound understanding of CUDA mechanics, shared memory limits, and underlying hardware architecture.

To solve this exact problem, the AI ecosystem has introduced Python-based Domain Specific Languages (DSLs) like CuTe DSL which promise to deliver performance equal to or better than custom CUDA C++, without the massive engineering overhead.

They claim to offer maximum performance with minimal code. Let's put that claim to the test.




## CUTE DSL

Based on the [NVIDIA CUTLASS Python (CuTe) DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html), implementing an autotuning workflow involves defining a search space of kernel configurations (like tile sizes and cluster parameters), benchmarking them, and caching the best-compiled kernel to avoid repeated Just-In-Time (JIT) compilation overhead.

### Implementation Guide
The overall strategy outlined by the CuTe DSL guidelines requires:

- Defining a Search Space: Determining the tile configurations (e.g., (128, 128, 32)) suitable for the hardware architecture.

- Dynamic Compilation: Using ´cute.compile´ to inject the tile settings into your JIT-compiled GEMM kernel.

- Benchmarking: Measuring the performance on a CUDA stream using warmup iterations to ensure accurate profiling.

- Caching: Storing the compiled_gemm object keyed by the matrix dimensions (M, N, K), as we did before but now we have a library that performs it for us.
 As demonstrated in the autotune_gemm function, we can use cute.compile() to compile a kernel once, cache the compiled result, and reuse the cached JIT executor for multiple kernel executions. It maintain a global  dictionary know as config_kernel_dict to cache the compiled GEMM kernels, where each key (kernel_cache_key) uniquely identifies a kernel based on its characteristics. 

