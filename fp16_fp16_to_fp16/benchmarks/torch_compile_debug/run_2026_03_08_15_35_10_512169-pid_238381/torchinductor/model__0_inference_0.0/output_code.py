# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import torch._inductor.kernel.mm_common
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import (
    grid,
    split_scan_grid,
    grid_combo_kernels,
    start_graph,
    end_graph,
    cooperative_reduction_grid,
)
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /tmp/torchinductor_root/my/cmytjbgduayzmybbxdc6aegll2wdg67a5qzzvp6k4eexna3xnlpg.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   matmul => mm
# Graph fragment:
#   %mm : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg1_1, %arg0_1), kwargs = {})
triton_tem_fused_mm_0 = async_compile.triton('triton_tem_fused_mm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=8,
    triton_meta={'signature': {'arg_A': '*fp16', 'arg_B': '*fp16', 'out_ptr0': '*fp16'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=128, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'kernel_name': 'triton_tem_fused_mm_0', 'backend_hash': 'D844C99EF7BE190A88A1CEF881166E6EF3ABD265FE42AE81084671AC40D8CEB9', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'coordinate_descent_tuning': True, 'coordinate_descent_search_radius': 1, 'coordinate_descent_check_all_directions': False},
)
@triton.jit
def triton_tem_fused_mm_0(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 64
    A = arg_A
    B = arg_B

    M = 1024
    N = 1024
    K = 1024
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 1024
    stride_ak = 1
    stride_bk = 1024
    stride_bn = 1

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and M >= BLOCK_M:
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and N >= BLOCK_N:
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        xindex = idx_n + 1024*idx_m
        a = tl.load(A + (xindex))

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        xindex = idx_n + 1024*idx_m
        b = tl.load(B + (xindex))
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + 1024*idx_m
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')
meta0 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg1_1, (1024, 1024), (1024, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float16)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_tem_fused_mm_0.run(arg1_1, arg0_1, buf0, grid=torch._inductor.kernel.mm_common.mm_grid(1024, 1024, meta0), stream=stream0)
        del arg0_1
        del arg1_1
    return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
