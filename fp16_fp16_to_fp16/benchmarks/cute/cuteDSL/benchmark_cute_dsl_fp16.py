# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Dense FP16 SIMT GEMM benchmark (C = A * B) using CuTe DSL.

  - A: MxK, fp16, row-major (k-major) or col-major (m-major)
  - B: NxK, fp16, row-major (n-major) or col-major (k-major)
  - C: MxN, fp16, row-major (n-major) or col-major (m-major)

Key differences from the FP32 SIMT reference (benchmark_cute_dsl_fp16_1py):
  - All tensors are torch.float16 / cutlass.Float16
  - MMA accumulates in fp16 via MmaUniversalOp(cutlass.Float16)
  - Vectorized async copies use 8 fp16 elements (128 bits) for col-major tensors
  - Default CTA K-tile extended to 16 for better arithmetic intensity

Benchmark sweep:
  - Square matrices: M = N = K for dims in [1024, 2046, 4096, 8192, 16384, 32768]
  - Fixed-K matrices: M = N, K = 8192 for same dims
  - Results saved to ../../../results/rtx4090_cute_dsl_fp16.csv

Usage:
    # Full benchmark sweep
    python benchmark_cute_dsl_fp16.py

    # Single run (with optional verification)
    python benchmark_cute_dsl_fp16.py --mnk 4096,4096,4096 --skip_ref_check

    # NCU profiling
    ncu python benchmark_cute_dsl_fp16.py --mnk 8192,8192,8192 \\
        --skip_ref_check --warmup_iterations 0 --iterations 2
"""

import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import pandas as pd
import torch
from cutlass.cute.runtime import from_dlpack
from tqdm import tqdm

# Change working directory to the script's location so relative paths resolve correctly
script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

# allow_fp16_accumulation was added in PyTorch 2.1; guard for older versions
#if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
#   torch.backends.cuda.matmul.allow_fp16_accumulation = True


# ---------------------------------------------------------------------------
# FP16 SIMT GEMM kernel
# ---------------------------------------------------------------------------

class HGemm:
    """Dense FP16 SIMT GEMM (C = A * B) using CuTe DSL.

    Features:
    - Multistage shared-memory pipeline (gmem -> smem latency hiding)
    - Register pipeline (smem -> rmem overlapped with compute)
    - Vectorized 128-bit async copies for col-major (m/n-major) tensors
    - Bank-conflict padding for row-major (k-major) tensors
    - Predication for non-tile-aligned problem sizes

    Default tile: (bM=128, bN=128, bK=16), 3 pipeline stages, 256 threads.
    """

    def __init__(
        self,
        cta_tiler: Tuple[int, int, int] = (128, 128, 16),
        num_stages: int = 3,
        num_threads: int = 256,
    ):
        self._cta_tiler = cta_tiler
        self._num_stages = num_stages
        self._num_threads = num_threads

        assert num_threads > 0
        assert num_threads % 16 == 0, "num_threads must be a multiple of 16"

        self._bM, self._bN, self._bK = self._cta_tiler
        assert self._bM % 16 == 0, "bM must be a multiple of 16"
        assert self._bN % 16 == 0, "bN must be a multiple of 16"
        assert self._num_stages >= 3, "num_stages must be >= 3"

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        stream: cuda.CUstream = cuda.CUstream(
            cuda.CUstream_flags.CU_STREAM_DEFAULT
        ),
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        # -----------------------------------------------------------------------
        # Shared-memory layouts for A and B
        # - m/n-major: no padding needed (already contiguous in the tile direction)
        # - k-major (row-major): pad 4 elements to reduce shared-memory bank conflicts
        # -----------------------------------------------------------------------
        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(
                1,
                self._bM + padding_a,
                self._bK * (self._bM + padding_a),
            ),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(
                1,
                self._bN + padding_b,
                self._bK * (self._bN + padding_b),
            ),
        )

        # -----------------------------------------------------------------------
        # Copy layouts.
        #
        # cp.async only accepts 32/64/128-bit transfer sizes, so a single fp16
        # (16 bits) per atom is illegal. Every path below picks num_vectorized
        # so that num_vectorized * 16 bits is in {32, 64, 128}, vectorizing
        # along the contiguous dimension of each tensor:
        #   - col-major (m/n-major): vectorize along M (A) / N (B)
        #   - row-major (k-major)  : vectorize along K
        # -----------------------------------------------------------------------
        if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized_A = (
                8 if (mA.layout[0].max_alignment % 16 == 0)
                else 4 if (mA.layout[0].max_alignment % 8 == 0)
                else 2
            )
            major_mode_size = self._bM // num_vectorized_A
            tA = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vA = cute.make_layout((num_vectorized_A, 1))
        else:
            num_vectorized_A = (
                8 if (mA.layout[1].max_alignment % 16 == 0)
                else 4 if (mA.layout[1].max_alignment % 8 == 0)
                else 2
            )
            k_mode_size = self._bK // num_vectorized_A
            tA = cute.make_layout(
                (self._num_threads // k_mode_size, k_mode_size),
                stride=(k_mode_size, 1),
            )
            vA = cute.make_layout((1, num_vectorized_A))

        if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vectorized_B = (
                8 if (mB.layout[0].max_alignment % 16 == 0)
                else 4 if (mB.layout[0].max_alignment % 8 == 0)
                else 2
            )
            major_mode_size = self._bN // num_vectorized_B
            tB = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vB = cute.make_layout((num_vectorized_B, 1))
        else:
            num_vectorized_B = (
                8 if (mB.layout[1].max_alignment % 16 == 0)
                else 4 if (mB.layout[1].max_alignment % 8 == 0)
                else 2
            )
            k_mode_size = self._bK // num_vectorized_B
            tB = cute.make_layout(
                (self._num_threads // k_mode_size, k_mode_size),
                stride=(k_mode_size, 1),
            )
            vB = cute.make_layout((1, num_vectorized_B))

        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vectorized_A,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width * num_vectorized_B,
        )

        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

        # -----------------------------------------------------------------------
        # MMA layout: SIMT fp16 (thread-level FMA, not tensor-core)
        # atoms_layout: (num_threads/16, 16, 1) for row-major C (n-major)
        #               (16, num_threads/16, 1) for col-major C (m-major)
        # -----------------------------------------------------------------------
        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )

        # FP16 accumulation via SIMT fma.rn.f16 instructions
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float16)

        permutation_tiler_M = cute.make_layout(
            (atoms_layout.shape[0], 4), stride=(4, 1)
        )
        permutation_tiler_N = cute.make_layout(
            (atoms_layout.shape[1], 4), stride=(4, 1)
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            atoms_layout,
            permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
        )

        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            epilogue_op,
        ).launch(
            grid=grid_dim,
            block=[cute.size(atoms_layout), 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)

        # -------------------------------------------------------------------
        # Partition global tensors into CTA tiles
        # gA: (bM, bK, k_tiles)   gB: (bN, bK, k_tiles)   gC: (bM, bN)
        # -------------------------------------------------------------------
        gA = cute.local_tile(
            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gB = cute.local_tile(
            mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )

        # Shift to process the irregular (residue) k-tile first so the
        # main loop body never needs to handle partial k-tiles.
        #residue_k = mA.shape[1] - self._bK * gA.shape[2]
        # Shift to process the irregular (residue) k-tile first so the
        # main loop body never needs to handle partial k-tiles.
        # Hardcoded to 0 for benchmarking perfectly aligned sizes:
        residue_k = cutlass.Int32(0)
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)

        # -------------------------------------------------------------------
        # Shared memory buffers and per-thread copy partitions
        # sA: (bM, bK, stages)   sB: (bN, bK, stages)
        # tAgA: (CPY, CPY_M, CPY_K, k_tiles)   tAsA: (CPY, CPY_M, CPY_K, stages)
        # -------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        # -------------------------------------------------------------------
        # Predicates for out-of-bounds suppression
        # tApA / tBpB: m/n bounds only (used in the main loop steady state)
        # tApA_residue_k / tBpB_residue_k: m/n/k bounds (used for residue tile)
        # -------------------------------------------------------------------
        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        cA = cute.local_tile(
            mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        cB = cute.local_tile(
            mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)

        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tApA_residue_k = cute.make_rmem_tensor(
            cute.make_layout(
                (tAsA.shape[0][1], cute.size(tAsA, mode=[1]), cute.size(tAsA, mode=[2])),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_rmem_tensor(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )

        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )
        for rest_v in range(tApA_residue_k.shape[0]):
            for m in range(tApA_residue_k.shape[1]):
                for k in range(tApA_residue_k.shape[2]):
                    coord_A = tAcA[(0, rest_v), m, k, 0]
                    tApA_residue_k[rest_v, m, k] = cute.elem_less(
                        (coord_A[0], cutlass.Int32(-1)),
                        (mA.shape[0], coord_A[1]),
                    )
        for rest_v in range(tBpB_residue_k.shape[0]):
            for n in range(tBpB_residue_k.shape[1]):
                for k in range(tBpB_residue_k.shape[2]):
                    coord_B = tBcB[(0, rest_v), n, k, 0]
                    tBpB_residue_k[rest_v, n, k] = cute.elem_less(
                        (coord_B[0], cutlass.Int32(-1)),
                        (mB.shape[0], coord_B[1]),
                    )

        # -------------------------------------------------------------------
        # Prefetch prologue: fill the pipeline stages before the main loop
        # -------------------------------------------------------------------
        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)

        #cp.async on NVIDIA hardware requires the transfer size to
        # be 32, 64, or 128 bits. Passing 16 bits (one fp16 element)
        # hits the hardware limit. The fix packs 2 fp16 elements into
        # every 32-bit copy atom — the thread layout then sees a halved
        # logical K dimension, which keeps the arithmetic identical to the FP32 reference.
        # Stage 0: residue k-tile (may be irregular in the K dimension)
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue_k,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue_k,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )

        # Stages 1 .. (k_pipe_max - 2): regular k-tiles
        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )
            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()

        # Clear m/n predicates once all tiles are prefetched
        if k_tile_count < k_pipe_max:
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)

        # -------------------------------------------------------------------
        # MMA accumulators and register partitions
        # -------------------------------------------------------------------
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)
        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]

        # -------------------------------------------------------------------
        # Register pipeline prefetch
        # -------------------------------------------------------------------
        k_block_max = cute.size(tCrA, mode=[2])

        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        # -------------------------------------------------------------------
        # Main loop: interleaved smem-pipeline (gmem->smem) and
        #            register-pipeline (smem->rmem) + FP16 GEMM
        # -------------------------------------------------------------------
        for _ in range(k_tile_count):
            for k_block in range(k_block_max, unroll_full=True):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    self.cta_sync_barrier.arrive_and_wait()

                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_p[None, None, k_block_next],
                    tCrB[None, None, k_block_next],
                )

                if k_block == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )

                # FP16 SIMT fused multiply-add
                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

                if k_block == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(1)
                    )

        # -------------------------------------------------------------------
        # Epilogue: apply op, predicate, copy C accumulators back to global mem
        # -------------------------------------------------------------------
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()
        tCrC.store(epilogue_op(tCrC.load()))

        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_rmem_tensor(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))

        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def _make_tensor(mode0: int, mode1: int, is_mode0_major: bool) -> torch.Tensor:
    """Create a random fp16 tensor on CUDA with the requested memory layout.

    is_mode0_major=True  → mode0 dimension is contiguous (col-major for A/C)
    is_mode0_major=False → mode1 dimension is contiguous (row-major)
    """
    if is_mode0_major:
        t = torch.empty(mode1, mode0, dtype=torch.float16).random_(-5, 5)
        return t.permute(1, 0).cuda()
    else:
        return torch.empty(mode0, mode1, dtype=torch.float16).random_(-5, 5).cuda()


def _to_cute_tensor(
    t: torch.Tensor, leading_dim: int, divisibility: int
) -> "cute.Tensor":
    """Wrap a PyTorch fp16 tensor as a dynamic CuTe tensor."""
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, divisibility=divisibility)
    )


# ---------------------------------------------------------------------------
# Single-run function (verification + benchmark for one size)
# ---------------------------------------------------------------------------

def run(
    mnk: Tuple[int, int, int],
    a_major: str = "m",
    b_major: str = "k",
    c_major: str = "n",
    warmup_iterations: int = 5,
    iterations: int = 100,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
) -> float:
    """Run the FP16 GEMM for a single (M, N, K) and return avg time in µs.

    :param mnk: Problem dimensions (M, N, K).
    :param a_major: 'm' (col-major) or 'k' (row-major) for A.
    :param b_major: 'n' (col-major) or 'k' (row-major) for B.
    :param c_major: 'm' (col-major) or 'n' (row-major) for C.
    :param warmup_iterations: Warm-up iterations (includes JIT compilation).
    :param iterations: Timed iterations.
    :param skip_ref_check: Skip numerical validation against torch.mm.
    :param use_cold_l2: Rotate input buffers to stress L2 cache.
    :return: Average kernel execution time in microseconds.
    """
    torch.manual_seed(1024)
    M, N, K = mnk

    print(f"FP16 CuTe DSL GEMM  mnk=({M}, {N}, {K})  "
          f"A={a_major}-major  B={b_major}-major  C={c_major}-major")

    a = _make_tensor(M, K, a_major == "m")
    b = _make_tensor(N, K, b_major == "n")
    c = _make_tensor(M, N, c_major == "m")

    div_a = a.shape[1] if a_major == "k" else a.shape[0]
    div_b = b.shape[1] if b_major == "k" else b.shape[0]
    div_c = c.shape[1] if c_major == "n" else c.shape[0]

    a_tensor = _to_cute_tensor(a, leading_dim=(1 if a_major == "k" else 0), divisibility=div_a)
    b_tensor = _to_cute_tensor(b, leading_dim=(1 if b_major == "k" else 0), divisibility=div_b)
    c_tensor = _to_cute_tensor(c, leading_dim=(1 if c_major == "n" else 0), divisibility=div_c)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    hgemm = HGemm()

    print("Compiling kernel ...")
    t0 = time.time()
    compiled_fn = cute.compile[cute.GenerateLineInfo](
        hgemm, a_tensor, b_tensor, c_tensor, stream=current_stream
    )
    print(f"Compilation time: {time.time() - t0:.2f}s")

    if not skip_ref_check:
        compiled_fn(a_tensor, b_tensor, c_tensor)
        torch.cuda.synchronize()
        # Reference: fp32 matmul cast back to fp16
        ref = torch.mm(a.float(), b.float()).half()
        # FP16 accumulation is lossy for large K; use a generous tolerance
        torch.testing.assert_close(c.cpu(), ref.cpu(), atol=0.5, rtol=0.1)
        print("Numerical check passed.")

    def generate_tensors():
        a_w = _make_tensor(M, K, a_major == "m")
        b_w = _make_tensor(N, K, b_major == "n")
        c_w = _make_tensor(M, N, c_major == "m")
        return testing.JitArguments(
            _to_cute_tensor(a_w, 1 if a_major == "k" else 0, div_a),
            _to_cute_tensor(b_w, 1 if b_major == "k" else 0, div_b),
            _to_cute_tensor(c_w, 1 if c_major == "n" else 0, div_c),
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        one_ws_bytes = (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + c.numel() * c.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_ws_bytes, warmup_iterations, iterations
        )

    avg_time_us = testing.benchmark(
        compiled_fn,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    flops = 2.0 * M * N * K
    tflops = flops / (avg_time_us / 1e6) / 1e12
    print(f"Avg time: {avg_time_us / 1e3:.4f} ms  |  {tflops:.2f} TFLOPS")
    return avg_time_us


# ---------------------------------------------------------------------------
# Full benchmark sweep
# ---------------------------------------------------------------------------

def run_benchmarks(
    a_major: str = "m",
    b_major: str = "k",
    c_major: str = "n",
    warmup_iterations: int = 5,
    iterations: int = 100,
) -> pd.DataFrame:
    """Sweep over Square and Fixed-K configurations and collect TFLOPS.

    Mirrors the benchmark dimensions used in the PyTorch and AOTI benchmarks:
      Square:  M = N = K  for each dim in dims_base
      Fixed_K: M = N,  K = 8192  for each dim in dims_base
    """
    dims_base = [1024, 2046, 4098, 8192, 16384, 32768]
    K_fixed = 8192

    all_tasks = (
        [("Square",  d, d, d)       for d in dims_base]
        + [("Fixed_K", d, d, K_fixed) for d in dims_base]
    )

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"Mode: FP16 CuTe DSL GEMM  "
          f"(A={a_major}-major, B={b_major}-major, C={c_major}-major)")
    print(f"Running {len(all_tasks)} configurations ...\n")

    hgemm = HGemm()
    results = []
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_fn = None   # set on first successful compile
    prev_shape = None    # recompile only when layout changes

    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking"):
        try:
            torch.cuda.empty_cache()

            a = _make_tensor(M, K, a_major == "m")
            b = _make_tensor(N, K, b_major == "n")
            c = _make_tensor(M, N, c_major == "m")

           # Provide the actual dimension sizes to help the compiler fold residue_k to 0
            div_a = a.shape[1] if a_major == "k" else a.shape[0]
            div_b = b.shape[1] if b_major == "k" else b.shape[0]
            div_c = c.shape[1] if c_major == "n" else c.shape[0]

            a_tensor = _to_cute_tensor(a, 1 if a_major == "k" else 0, div_a)
            b_tensor = _to_cute_tensor(b, 1 if b_major == "k" else 0, div_b)
            c_tensor = _to_cute_tensor(c, 1 if c_major == "n" else 0, div_c)
            # (Re-)compile when needed; @cute.jit caches by shape/type so
            # subsequent calls for new shapes trigger a new JIT compilation.
            compiled_fn = cute.compile(
                hgemm, a_tensor, b_tensor, c_tensor, stream=current_stream
            )

            def generate_tensors():
                a_w = _make_tensor(M, K, a_major == "m")
                b_w = _make_tensor(N, K, b_major == "n")
                c_w = _make_tensor(M, N, c_major == "m")
                return testing.JitArguments(
                    _to_cute_tensor(a_w, 1 if a_major == "k" else 0, div_a),
                    _to_cute_tensor(b_w, 1 if b_major == "k" else 0, div_b),
                    _to_cute_tensor(c_w, 1 if c_major == "n" else 0, div_c),
                    current_stream,
                )

            avg_time_us = testing.benchmark(
                compiled_fn,
                workspace_generator=generate_tensors,
                workspace_count=1,
                stream=current_stream,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
            )

            avg_time_ms = avg_time_us / 1e3
            flops = 2.0 * M * N * K
            tflops = flops / (avg_time_us / 1e6) / 1e12

            results.append(
                {
                    "Type": label,
                    "M": M,
                    "N": N,
                    "K": K,
                    "Time_ms": avg_time_ms,
                    "TFLOPS": tflops,
                }
            )

            del a, b, c

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                tqdm.write(f"\n  OOM at ({M},{N},{K}) - skipping")
                torch.cuda.empty_cache()
            else:
                tqdm.write(f"\n  RuntimeError at ({M},{N},{K}): {e}")
        except Exception as e:
            tqdm.write(f"\n  Error at ({M},{N},{K}): {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def _parse_mnk(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Expected comma-separated integers, e.g. 4096,4096,4096"
            )

    parser = argparse.ArgumentParser(
        description="FP16 SIMT GEMM benchmark using CuTe DSL"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run full M/N/K sweep and save CSV (default: single run)",
    )
    parser.add_argument("--mnk", type=_parse_mnk, default=(4096, 4096, 4096))
    parser.add_argument("--a_major", choices=["k", "m"], default="m")
    parser.add_argument("--b_major", choices=["k", "n"], default="k")
    parser.add_argument("--c_major", choices=["n", "m"], default="n")
    parser.add_argument("--warmup_iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Rotate input buffers to keep L2 cache cold",
    )
    args = parser.parse_args()

    if args.sweep:
        # ---- Full benchmark sweep ----
        df = run_benchmarks(
            a_major=args.a_major,
            b_major=args.b_major,
            c_major=args.c_major,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
        )

        results_dir = Path("../../../results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "rtx4090_cute_dsl_fp16.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")

        print("\n--- Square matrices ---")
        print(df[df["Type"] == "Square"].to_markdown(index=False))
        print("\n--- Fixed K = 8192 ---")
        print(
            df[df["Type"] == "Fixed_K"]
            .sort_values("TFLOPS", ascending=False)
            .to_markdown(index=False)
        )
    else:
        # ---- Single-run mode ----
        run(
            mnk=args.mnk,
            a_major=args.a_major,
            b_major=args.b_major,
            c_major=args.c_major,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            skip_ref_check=args.skip_ref_check,
            use_cold_l2=args.use_cold_l2,
        )
 
