# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
FP16 dense GEMM benchmark for NVIDIA RTX 4090 (Ada, sm_89) using CuTe DSL.

Implementación basada en la guía general del CuTe Python DSL:
    https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general

Características del kernel:
  - SIMT FP16 GEMM: C(MxN) = A(MxK) * B(NxK), todas las matrices fp16
  - Multistage shared-memory pipeline (cp.async) para ocultar latencia GMEM->SMEM
  - Register pipeline: SMEM -> registros se solapa con compute
  - Vectorización 128b en cp.async cuando el alineamiento lo permite
  - Padding en SMEM para reducir bank conflicts en layouts row-major (k-major)
  - Predicación para tamaños no múltiplos del tile

Configuración por defecto: tile (bM=128, bN=128, bK=16), 3 stages, 256 threads.

Uso:
    # Sweep completo (Square + Fixed_K) y guardado en CSV
    python benchmark_cute_dsl_co.py --sweep

    # Ejecución única
    python benchmark_cute_dsl_co.py --mnk 8192,8192,8192 --skip_ref_check

    # Profiling con NCU
    ncu python benchmark_cute_dsl_co.py --mnk 8192,8192,8192 \\
        --skip_ref_check --warmup_iterations 0 --iterations 2

Variables de entorno recomendadas:
    export CUDA_TOOLKIT_PATH=/usr/local/cuda
    export CUTE_DSL_ARCH=sm_89   # RTX 4090 (Ada Lovelace)
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

script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)


# ---------------------------------------------------------------------------
# FP16 SIMT GEMM kernel (CuTe DSL)
# ---------------------------------------------------------------------------

class HGemm:
    """GEMM denso FP16 (C = A * B) para Ada (sm_89) con CuTe DSL."""

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
        assert num_threads % 16 == 0, "num_threads debe ser múltiplo de 16"

        self._bM, self._bN, self._bK = self._cta_tiler
        assert self._bM % 16 == 0, "bM debe ser múltiplo de 16"
        assert self._bN % 16 == 0, "bN debe ser múltiplo de 16"
        assert self._num_stages >= 3, "num_stages debe ser >= 3"

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

        # ---- Layouts SMEM (con padding si es k-major para evitar bank conflicts) ----
        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, self._bM + padding_a, self._bK * (self._bM + padding_a)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, self._bN + padding_b, self._bK * (self._bN + padding_b)),
        )

        # ---- Vectorización cp.async (32/64/128 bits permitidos) ----
        # FP16 = 16 bits → vectorizamos 2/4/8 elementos por copia.
        # ---- Eleccion de num_vec sin "break" (compatible con @cute.jit) ----
        # Para que el atom cp.async-128b sea valido el num_vec*sizeof(elem) debe
        # ser <= 128 bits Y la dimension contigua del tile debe ser multiplo de
        # num_vec. Para FP16 los candidatos son 8 (128b), 4 (64b), 2 (32b), 1.
        # Ada/sm89 + bK=16 con cp.async puede fallar la verificacion de 128b en
        # algunos tiles, asi que CAPAMOS a 4 cuando la dim contigua < 32 elementos.
        def _pick_num_vec(elem_bits, contig_dim):
            cap_128 = 128 // elem_bits          # 8 para fp16
            # cap defensivo: si la dim contigua es chica, no fuerces 128b
            cap_safe = 8 if contig_dim >= 32 else 4
            cap = min(cap_128, cap_safe)
            if cap >= 8 and (contig_dim % 8 == 0): return 8
            if cap >= 4 and (contig_dim % 4 == 0): return 4
            if cap >= 2 and (contig_dim % 2 == 0): return 2
            return 1

        if cutlass.const_expr(self.a_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vec_A = _pick_num_vec(mA.element_type.width, self._bM)
            m_size = self._bM // num_vec_A
            tA = cute.make_layout(
                (self._num_threads // m_size, m_size), stride=(m_size, 1)
            )
            vA = cute.make_layout((num_vec_A, 1))
        else:
            num_vec_A = _pick_num_vec(mA.element_type.width, self._bK)
            k_size = self._bK // num_vec_A
            tA = cute.make_layout(
                (self._num_threads // k_size, k_size), stride=(k_size, 1)
            )
            vA = cute.make_layout((1, num_vec_A))

        if cutlass.const_expr(self.b_major_mode == utils.LayoutEnum.COL_MAJOR):
            num_vec_B = _pick_num_vec(mB.element_type.width, self._bN)
            n_size = self._bN // num_vec_B
            tB = cute.make_layout(
                (self._num_threads // n_size, n_size), stride=(n_size, 1)
            )
            vB = cute.make_layout((num_vec_B, 1))
        else:
            num_vec_B = _pick_num_vec(mB.element_type.width, self._bK)
            k_size = self._bK // num_vec_B
            tB = cute.make_layout(
                (self._num_threads // k_size, k_size), stride=(k_size, 1)
            )
            vB = cute.make_layout((1, num_vec_B))

        atom_async_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec_A,
        )
        atom_async_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mB.element_type,
            num_bits_per_copy=mB.element_type.width * num_vec_B,
        )
        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

        # ---- Tiled MMA (SIMT FP16, no tensor-core en este ejemplo) ----
        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if cutlass.const_expr(self.c_major_mode == utils.LayoutEnum.COL_MAJOR):
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )
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
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma, epilogue_op,
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
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)

        # ---- Particionado de tensores globales en tiles del CTA ----
        gA = cute.local_tile(
            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gB = cute.local_tile(
            mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )

        # Tamaños del benchmark son múltiplos del tile → residue_k = 0
        residue_k = cutlass.Int32(0)
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)

        # ---- Buffers en SMEM y particiones por hilo ----
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        # ---- Predicados para tamaños no múltiplos del tile ----
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

        # ---- Prefetch prologue: rellenar las stages antes del mainloop ----
        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)

        # Stage 0: residue tile
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

        # Stages 1..k_pipe_max-2: tiles regulares
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

        if k_tile_count < k_pipe_max:
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)

        # ---- Acumuladores MMA y particionado de fragmentos ----
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

        # ---- Prefetch register pipeline ----
        k_block_max = cute.size(tCrA, mode=[2])
        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            self.cta_sync_barrier.arrive_and_wait()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        # ---- Mainloop: smem-pipeline + register-pipeline + MMA ----
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

        # ---- Epilogue: aplicar epilogue_op y escribir C con predicación ----
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
# Helpers tensores
# ---------------------------------------------------------------------------

def _make_tensor(mode0: int, mode1: int, is_mode0_major: bool) -> torch.Tensor:
    if is_mode0_major:
        t = torch.empty(mode1, mode0, dtype=torch.float16).random_(-5, 5)
        return t.permute(1, 0).cuda()
    return torch.empty(mode0, mode1, dtype=torch.float16).random_(-5, 5).cuda()


def _to_cute_tensor(t: torch.Tensor, leading_dim: int, divisibility: int):
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(mode=leading_dim, divisibility=divisibility)
    )


# ---------------------------------------------------------------------------
# Ejecución única (verificación + benchmark de un tamaño)
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
    torch.manual_seed(1024)
    M, N, K = mnk
    print(f"[CuTe DSL FP16 GEMM] mnk=({M},{N},{K})  "
          f"A={a_major}-major B={b_major}-major C={c_major}-major")

    a = _make_tensor(M, K, a_major == "m")
    b = _make_tensor(N, K, b_major == "n")
    c = _make_tensor(M, N, c_major == "m")

    div_a = a.shape[1] if a_major == "k" else a.shape[0]
    div_b = b.shape[1] if b_major == "k" else b.shape[0]
    div_c = c.shape[1] if c_major == "n" else c.shape[0]

    """con cute.compile reutilizar los mismos a_t/b_t/c_t entre múltiples tiles rompía la inferencia de alineamiento"""

    a_t = _to_cute_tensor(a, 1 if a_major == "k" else 0, div_a)
    b_t = _to_cute_tensor(b, 1 if b_major == "k" else 0, div_b)
    c_t = _to_cute_tensor(c, 1 if c_major == "n" else 0, div_c)

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    hgemm = HGemm()
    print("Compilando kernel ...")
    t0 = time.time()
    compiled_fn = cute.compile[cute.GenerateLineInfo](
        hgemm, a_t, b_t, c_t, stream=current_stream
    )
    print(f"Compilation time: {time.time() - t0:.2f}s")

    if not skip_ref_check:
        compiled_fn(a_t, b_t, c_t)
        torch.cuda.synchronize()
        ref = torch.mm(a.float(), b.float()).half()
        torch.testing.assert_close(c.cpu(), ref.cpu(), atol=0.5, rtol=0.1)
        print("Verificación numérica OK.")

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
# Sweep completo (Square + Fixed_K)
# ---------------------------------------------------------------------------

# Configuraciones de tile a explorar (bM, bN, bK, num_stages, num_threads).
# Filtramos aquellas que excedan ~96 KB de SMEM en sm_89 (RTX 4090).
TILE_CONFIGS: Tuple[Tuple[int, int, int, int, int], ...] = (
    (128, 128, 16, 3, 256),
    (128, 256, 16, 3, 256),
    (256, 128, 16, 3, 256),
    (128, 128, 32, 3, 256),
    (128, 128, 16, 4, 256),
)


def _smem_bytes(bM: int, bN: int, bK: int, stages: int) -> int:
    # 2 buffers (A,B) * fp16 (2 bytes) * stages * (M+N) * K + padding ~ aproximación
    return 2 * 2 * stages * (bM + bN) * bK


def run_benchmarks(
    a_major: str = "m",
    b_major: str = "k",
    c_major: str = "n",
    warmup_iterations: int = 5,
    iterations: int = 100,
    tune_tiles: bool = False,
) -> pd.DataFrame:
    """Sweep:
      - Square : M = N = K para cada d en dims_base
      - Fixed_K: M = N, K = 8192 para cada d en dims_base
    Si tune_tiles=True, prueba varios tiles por (M,N,K) y guarda el mejor.
    """
    #dims_base = [1024, 2048, 4096, 8192, 16384, 32768]
    #K_fixed = 8192

    dims_base = [8192]
    K_fixed = 8192

    bench_1 = [("Square", d, d, d) for d in dims_base]
    bench_2 = [("Fixed_K", d, d, K_fixed) for d in dims_base]
    all_tasks = bench_1 + bench_2

    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"Mode: CuTe DSL FP16 GEMM "
          f"(A={a_major}-major, B={b_major}-major, C={c_major}-major)")
    tile_list = TILE_CONFIGS if tune_tiles else (TILE_CONFIGS[0],)
    print(f"Ejecutando {len(all_tasks)} configuraciones × {len(tile_list)} tiles ...\n")

    results = []
    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    for label, M, N, K in tqdm(all_tasks, desc="Benchmarking"):
        try:
            torch.cuda.empty_cache()

            a = _make_tensor(M, K, a_major == "m")
            b = _make_tensor(N, K, b_major == "n")
            c = _make_tensor(M, N, c_major == "m")

            div_a = a.shape[1] if a_major == "k" else a.shape[0]
            div_b = b.shape[1] if b_major == "k" else b.shape[0]
            div_c = c.shape[1] if c_major == "n" else c.shape[0]

            a_t = _to_cute_tensor(a, 1 if a_major == "k" else 0, div_a)
            b_t = _to_cute_tensor(b, 1 if b_major == "k" else 0, div_b)
            c_t = _to_cute_tensor(c, 1 if c_major == "n" else 0, div_c)

            best = None  # (tflops, time_ms, cfg)
            for (bM, bN, bK, stages, nthreads) in tile_list:
                if M % bM != 0 or N % bN != 0:
                    continue  # tile incompatible con la forma
                if _smem_bytes(bM, bN, bK, stages) > 96 * 1024:
                    continue  # excede SMEM disponible en sm_89
                try:
                    hgemm = HGemm(
                        cta_tiler=(bM, bN, bK),
                        num_stages=stages,
                        num_threads=nthreads,
                    )
                    compiled_fn = cute.compile(
                        hgemm, a_t, b_t, c_t, stream=current_stream
                    )

                    def generate_tensors(_M=M, _N=N, _K=K,
                                         _da=div_a, _db=div_b, _dc=div_c):
                        a_w = _make_tensor(_M, _K, a_major == "m")
                        b_w = _make_tensor(_N, _K, b_major == "n")
                        c_w = _make_tensor(_M, _N, c_major == "m")
                        return testing.JitArguments(
                            _to_cute_tensor(a_w, 1 if a_major == "k" else 0, _da),
                            _to_cute_tensor(b_w, 1 if b_major == "k" else 0, _db),
                            _to_cute_tensor(c_w, 1 if c_major == "n" else 0, _dc),
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
                    tflops = (2.0 * M * N * K) / (avg_time_us / 1e6) / 1e12
                    if best is None or tflops > best[0]:
                        best = (tflops, avg_time_ms, (bM, bN, bK, stages, nthreads))
                except Exception as e:
                    tqdm.write(f"  tile {(bM,bN,bK,stages,nthreads)} fallo en "
                               f"({M},{N},{K}): {e}")

            if best is not None:
                tflops, avg_time_ms, cfg = best
                results.append({
                    "Type": label, "M": M, "N": N, "K": K,
                    "Time_ms": avg_time_ms, "TFLOPS": tflops,
                    "Tile": f"{cfg[0]}x{cfg[1]}x{cfg[2]}",
                    "Stages": cfg[3], "Threads": cfg[4],
                })

            del a, b, c

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                tqdm.write(f"\n  OOM en ({M},{N},{K}) - skip")
                torch.cuda.empty_cache()
            else:
                tqdm.write(f"\n  RuntimeError en ({M},{N},{K}): {e}")
        except Exception as e:
            tqdm.write(f"\n  Error en ({M},{N},{K}): {e}")

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
                "Formato esperado: enteros separados por coma, p.ej. 4096,4096,4096"
            )

    parser = argparse.ArgumentParser(
        description="Benchmark FP16 GEMM con CuTe DSL para RTX 4090 (sm_89)"
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep completo M/N/K y guardar CSV")
    parser.add_argument("--tune_tiles", action="store_true",
                        help="Probar varios tiles por (M,N,K) y elegir el mejor")
    parser.add_argument("--mnk", type=_parse_mnk, default=(4096, 4096, 4096))
    parser.add_argument("--a_major", choices=["k", "m"], default="m")
    parser.add_argument("--b_major", choices=["k", "n"], default="k")
    parser.add_argument("--c_major", choices=["n", "m"], default="n")
    parser.add_argument("--warmup_iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true", default=False)
    args = parser.parse_args()

    if args.sweep:
        df = run_benchmarks(
            a_major=args.a_major,
            b_major=args.b_major,
            c_major=args.c_major,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            tune_tiles=args.tune_tiles,
        )

        results_dir = Path("../../../results")
        results_dir.mkdir(parents=True, exist_ok=True)
        out_csv = results_dir / "rtx4090_cute_dsl_co.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nResultados guardados en: {out_csv}")

        if df.empty or "Type" not in df.columns:
            print("\n[!] No hay resultados (todas las configuraciones fallaron).")
        else:
            sq = df[df["Type"] == "Square"]
            fk = df[df["Type"] == "Fixed_K"]
            print("\n--- Square ---")
            print(sq.to_markdown(index=False) if not sq.empty else "(vacío)")
            print("\n--- Fixed K = 8192 ---")
            if not fk.empty:
                print(
                    fk.sort_values("TFLOPS", ascending=False)
                    .to_markdown(index=False)
                )
            else:
                print("(vacío)")
    else:
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
        print("PASS")