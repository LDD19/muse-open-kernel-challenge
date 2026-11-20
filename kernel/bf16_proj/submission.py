# okc: problem=muse_bf16_proj

from torch.utils.cpp_extension import load_inline as _ldd19_raw_inline
def _ldd19_inline(*args, **kw):
    kw['extra_cflags'] = [*kw.get('extra_cflags', []), '-fno-gnu-unique']
    kw['extra_cuda_cflags'] = [*kw.get('extra_cuda_cflags', []), '-Xcompiler=-fno-gnu-unique']
    return _ldd19_raw_inline(*args, **kw)
#
# Source: https://github.com/NathenFernandes/muse-bf16-proj (v18 submission.py, judged 1.5186x,
#   blob sha256 111630ff363e5988ca57f28434a2e1a7e9c1d524079c64727b52c6aa287dcb44, commit 86aff75)
# Author: NathenFernandes (Nathen Fernandes)
# License: MIT License. v18's code below is unchanged except for the v19 changes listed after the license texts.
#
# MIT License
#
# Copyright (c) 2026 Nathen Fernandes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Portions of this file (marked "chenwainuo/okc" below: the rq split-K register-A wgmma kernel, its 12-bit weight pack
# helpers _pack_nat / _c12 / _unc12 / _rqpack / _rqunpack / _rqesc, the rr FP32 router GEMV and the unpooler m=128
# cuBLASLt descriptor choice) are adapted from https://github.com/chenwainuo/okc (commit 785c2c01dc21,
# results/proj-rsi/releases/pw45-v1-835acb90b47b/submission.py), written by chenwainuo (https://github.com/chenwainuo),
# licensed under the Apache License, Version 2.0 (the "License"); you may not use those portions except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 . Unless required by applicable law
# or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing
# permissions and limitations under the License.
# Modifications (v19, 2026-10-27): the rq/rr host binding takes raw pointers and the stream (pybind11 only, no torch
# headers); only the instances rq<32,5,1,0,0,0,35>, rq<128,5,1,0,0,0,3> and rr<1,1,1024> are built; routes are fixed
# per (layer, m) instead of baked in prepare; validation follows the v18 one-hot / random checks.
#
# v19 changes to the MIT code above: prepare() no longer builds families that serve none of the 30 scored and 9 test
# shapes (see V19_PRUNED below); routes added for qkv_gate/pooler/unpooler m=32, qkv_gate/pooler m=128 (rq), router m=1
# (rr) and a different pinned cuBLASLt descriptor for unpooler m=128. Every other route is v18's, unchanged.
#
# LDD19 (2026-10-30, the "ldd19r" section at the end of the file): routes for qkv_gate m=1 and the unpooler slice
# m=1 (cgemv), pooler m=32 and unpooler m=32 (sk2) and router m=256 (rk). The sk2 and rk kernels, the shared device
# helpers, cgemv and the host packing helpers (sk2_tileunits / sk2_compress / sk2_stream / sk2_table /
# sk2_balance_perm / cg_compress / _sk2_geom) are from team vjeux's muse_bf16_proj.py, https://github.com/vjeux/okc-2026
# (commit f35a41ee1c13), Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0; distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND). The unpooler m=32 sk2 configuration u32_r96 and the
# idea of independent cgemv accumulators (NACC) are team alygg77's, https://github.com/alygg77/okc-muse-spark-kernels
# (Apache License 2.0). Modified by LDD19: pybind11-only raw-pointer host launchers, rk reads v18's packed router
# layout, per-shape fixed routes with prepare-time validation; every change is marked "LDD19".
#
# Readable-source edition of v25 (agents/bf16_readable/build.py, 2026-11-02): every CUDA/C++ source is a plain string
# literal; nothing is compressed or encoded. The routes, kernels and template instances that v25's dispatch never reaches
# for any (layer, m) were removed (see agents/bf16_readable/REPORT.md); every kept kernel source is unchanged.
# Since then v19's rq extension holds rq<32,5,1,0,0,0,35> and rr<1,1,1024> only, the chenwainuo factory (team
# keell0renz's selective routes) keeps its gv qkv_gate m=8 instance and the rq128 d2s5hzwP variant, and the v10_tune,
# v10_m32x, v10_warps, v11/v12 smtrace1, v14 qkvbig, bundle qkv8 / routerwide, bundle2 qkv16 / pooler32 / mw32 and
# decodediet qkv1 families (all shadowed by a route above them or stubbed out by v19 / LDD19) are gone.
#
# LDD19 optimizer (2026-11-03, EAT iteration 1, all in the ldd19r section, each behind an LDD19R switch): pooler m=1 and o m=1
# on vjeux's cgemv (configs 2x512 and 2x64), the unpooler slice m=1 on cgemv 1 row x 128 threads, o m=32 on vjeux's sk2
# config o32h_r100 (vjeux/okc-2026 via keell0renz 8e30e1b, Apache-2.0), and router m=1 on cgemv_hl, our derivative of
# vjeux's cgemv that streams the router's BF16 hi + lo pair as two 12-bit planes with FP32 accumulation. vjeux's rgemv
# (Apache-2.0) is compiled but switched off (rg1 / rg8 = 0).
"""Dense projections: small_m (12-bit lossless or BF16 register-direct GEMV) for the shapes it claims, then the
register-direct swap-AB GEMV for small m, 12-bit lossless pooler weights at m = 1, and a TMA + wgmma router for
64 < m <= 256. Shapes no kernel claims fall back to SGLang's own dispatch."""

import hashlib
from typing import TypedDict

import torch
import torch.nn.functional as F
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


# ---- small_m/small_m_kernels.py ----


SM_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SMD __device__ __forceinline__

SMD int4 sm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint4 sm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint32_t sm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMD uint32_t sm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMD void sm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMD void sm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = sm_prmt(T0, T1, cm);
  const uint32_t hhi = sm_prmt(T0, T1, cm >> 16);
  a[0] = sm_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = sm_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = sm_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = sm_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void sm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMD void sm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) sm_never(p);
}

SMD uint32_t sm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMD uint32_t sm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMD uint32_t sm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMD void sm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMD void sm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMD void sm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMD void sm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMD void sm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMD void sm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMD bool sm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMD uint64_t sm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
SMD void sm_wait_bar(uint32_t bar) {
  if (sm_mbar_try(bar)) return;
  const uint64_t t0 = sm_gtime();
  while (!sm_mbar_try(bar))
    if (sm_gtime() - t0 > 20000000ull) __trap();
}

SMD void sm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMD float4 sm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMD void sm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMD void sm_store(float* p, float v) { *p = v; }

template <int FMT> struct SmFmt;
template <> struct SmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct SmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct SmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct SmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMD void sm_decode(const int4 (&w)[SmFmt<FMT>::NL], uint32_t (&A)[SmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < SmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < SmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      sm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      sm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      sm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      sm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMD void sm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMD void sm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) sm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
__global__ void __launch_bounds__(NTMAX) sm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = SmFmt<FMT>::NL, NA = SmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = sm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = sm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        sm_mbar_init(bar, 1);
        sm_mbar_expect(bar, parts * RT * TV * 128u);
        sm_fence_init();
      }
      sm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = sm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = sm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = sm_ldx(xp);
          xv[slot][n][1] = sm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    sm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          sm_order_point(m, outp);
        } else {
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) sm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          sm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) sm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  sm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) sm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          sm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) sm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                sm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) sm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            sm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            sm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) sm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            sm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = sm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else sm_st_async4(sm_mapa(sm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            sm_wait_bar(bar);
            sm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            sm_cl_wait();
            const uint32_t rbar = sm_mapa(bar, 0);
            sm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else sm_st_async4(sm_mapa(sm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            sm_wait_bar(bar);
            sm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int sm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
static int sm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = sm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = sm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
'''

SM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
SM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
SM_ESC_CAP = 32


def sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7))
        cases.append(f"    case {i}: return sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return SM_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def sm_build(instances, name="small_m_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return _ldd19_inline(name, SM_CPP + extra_cpp, cuda_sources=sm_cuda_source(instances) + extra_cuda,
                       functions=["sm_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def sm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def _to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def _to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(SM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def sm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def sm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [sm_units(hi), sm_units(lo)]
    else:
        mats = [sm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = sm_c12_bytes(v)
        if not torch.equal(sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(sm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = _to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.sm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# ---- deliverable API ----
# Exact token counts take precedence over token-tile entries. Geometry and reduction mapping are H100-validated;
# the compressed m=16/32 configurations have not yet been measured on the H200 judge.
SM_TABLE_M = {
    ("router", 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)),
}
SM_TABLE_MT = {
    ("pooler", 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)),
    ("qkv_gate", 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)),
    ("o", 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("o", 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("unpooler", 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)),
    ("router", 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)),
}
SM_MAX_M = 32

SM_TABLE_M.update({
    ("router", 8): SM_TABLE_MT[("router", 1)],
    ("qkv_gate", 8): SM_TABLE_MT[("qkv_gate", 1)],
    ("pooler", 8): SM_TABLE_MT[("pooler", 1)],
    ("o", 8): SM_TABLE_MT[("o", 1)],
    ("unpooler", 1): SM_TABLE_MT[("unpooler", 1)],
    ("unpooler", 8): SM_TABLE_MT[("unpooler", 1)],
})
SM_TABLE_MT[('pooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT[('pooler', 4)] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT[('qkv_gate', 2)] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
SM_TABLE_MT[('unpooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))
# v19: only the small_m entries that serve one of the judged shapes are built (tests m 7 pooler and m 5
# unpooler; every scored shape is claimed by a later family)
SM_TABLE_M = {}
SM_TABLE_MT = {key: SM_TABLE_MT[key] for key in (('pooler', 1), ('unpooler', 1))}


def sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def sm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = sm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = sm_verify_onehot(ext, inst_id[inst], sp, sm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_small_m(weights, table_m=None, table_mt=None, name="small_m_gemv"):
    table_m = dict(SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if sm_bf16_twin(inst):
            instances.append(sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = sm_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and sm_bf16_twin(inst):
            plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, sm_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    return st


def run_small_m(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= SM_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", sm_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != SM_LAYER_K[layer] or not x.is_cuda:
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if sm_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out


# ---- router/router_kernels.py ----


def _mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


ROUTER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t rr_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void rr_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void rr_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void rr_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t rr_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void rr_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 rr_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void rr_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t rr_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void rr_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}
@@MMA@@

template<int BN,int S,int ST,int SW,int RED,int PIPE>
__global__ __launch_bounds__((PIPE&2)?256:128) void rr_kernel(const __grid_constant__ CUtensorMap tw,
                                               const __grid_constant__ CUtensorMap tx, float* out, int m) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int WB=NR*128, XB=TM*128, SB=WB*2+XB;
  constexpr int NF=64*BN, NV=NF/4, OWN=NV/S;
  extern __shared__ __align__(1024) unsigned char sm[];
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, rank=blockIdx.x%S;
  const int row=blockIdx.x/S*NR, tok=blockIdx.y*TM;
  const uint32_t base=rr_smem(sm), bars=base+ST*SB, empty=bars+ST*8, rb=(empty+((PIPE&2)?ST*8:0)+15)&~15u;
  const uint32_t recv=rb+16;
  const int kb=rank*(128/S);
  auto load=[&](int i) {
    const int st=i%ST;
    const uint32_t b=bars+st*8, d=base+st*SB;
    rr_expect(b,SB);
    rr_tma(d,&tw,0,((row/NR)*128+kb+i)*NR*2,b);
    rr_tma(d+2*WB,&tx,(kb+i)*64,tok,b);
  };
  if(tid==((PIPE&2)?128:0)) {
#pragma unroll
    for(int i=0;i<ST;i++) { rr_init(bars+i*8,1); if constexpr(PIPE&2) rr_init(empty+i*8,4); }
    asm volatile("fence.proxy.async.shared::cta;" :::"memory");
    if constexpr(PIPE&4) {
    if constexpr(RED>=1) {
      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
#pragma unroll
    for(int i=0;i<ST && i<128/S;i++) load(i);
    if constexpr(!(PIPE&4)) {
    if constexpr(RED>=1) {
      rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");rr_expect(rb,(S-1)*OWN*16);
      asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");
    }
    }
  }
  __syncthreads();
  if constexpr(RED>=1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  if constexpr(PIPE&2) {
    if(tid>=128) {
      if(tid==128) {
        for(int i=ST;i<128/S;i++) {
          rr_wait(empty+(i%ST)*8,((i/ST)-1)&1);
          load(i);
        }
      }
      if constexpr(RED>=1) asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
      if constexpr(RED!=2) rr_sync();
      return;
    }
  }
  float acc[BN/2];
#pragma unroll
  for(int j=0;j<BN/2;j++) asm volatile("mov.b32 %0, 0;":"=f"(acc[j]));
  int prev=0;
#pragma unroll 1
  for(int i=0;i<128/S;i++) {
    const int st=i%ST;
    rr_wait(bars+st*8,(i/ST)&1);
    const uint32_t w=base+st*SB, x=w+2*WB;
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    asm volatile("wgmma.fence.sync.aligned;" :::"memory");
#pragma unroll
    for(int q=0;q<4;q++) {
      const uint64_t dx=rr_desc(x+q*32),dh=rr_desc(w+q*32),dl=rr_desc(w+WB+q*32);
      if constexpr(SW) { @@CALL_H@@; @@CALL_L@@; }
      else { @@CALL_NH@@; @@CALL_NL@@; }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" :::"memory");
    if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" :::"memory");
    else asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("":"+f"(acc[j])::"memory");
    if constexpr(PIPE&2) {
      if constexpr(PIPE&1) {
        if(i>0 && prev+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+(prev%ST)*8):"memory");
      } else {
        if(i+ST<128/S && lane==0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(empty+st*8):"memory");
      }
    } else if constexpr(PIPE&1) {
      if(i>0 && prev+ST<128/S && tid==0) load(prev+ST);
    } else {
      if(i+ST<128/S && tid==0) load(i+ST);
    }
    prev=i;
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" :::"memory");
  auto emit=[&](int v,float4 z) {
    const int t=v%128, j=v/128, ln=t&31, wr=t>>5;
    const int ar=wr*16+(ln>>2),bc=j*8+(ln&3)*2;
    const float vv[4]={z.x,z.y,z.z,z.w};
#pragma unroll
    for(int c=0;c<4;c++) {
      const int a=ar+(c/2)*8, b=bc+(c&1);
      const int tr=tok+(SW?b:a), wc=row+(SW?a:b);
      if(tr<m) out[tr*256+wc]=vv[c];
    }
  };
  if constexpr(S==1) {
#pragma unroll
    for(int j=0;j<BN/8;j++) emit(j*128+tid,make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]));
  } else if constexpr(RED==0) {
    float4* part=(float4*)sm;
#pragma unroll
    for(int j=0;j<BN/8;j++) part[j*128+tid]=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
    rr_sync();
    for(int t=tid;t<OWN;t+=128) {
      const int v=rank*OWN+t;
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rr_add(sum,rr_pull(rr_map(base+v*16,r)));
      emit(v,sum);
    }
    rr_sync();
  } else {
    asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
#pragma unroll
    for(int j=0;j<BN/8;j++) {
      const int v=j*128+tid, dst=v/OWN, local=v%OWN;
      const float4 val=make_float4(acc[4*j],acc[4*j+1],acc[4*j+2],acc[4*j+3]);
      const uint32_t addr=recv+(rank*OWN+local)*16;
      if(dst==rank) *(float4*)(sm+(addr-base))=val;
      else rr_push(rr_map(addr,dst),val,rr_map(rb,dst));
    }
    if constexpr(PIPE&2) asm volatile("bar.sync 1, 128;" :::"memory"); else __syncthreads();
    rr_wait(rb,0);
    for(int t=tid;t<OWN;t+=128) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int r=0;r<S;r++) rr_add(sum,*(float4*)(sm+recv-base+(r*OWN+t)*16));
      emit(rank*OWN+t,sum);
    }
    if constexpr(RED!=2) rr_sync();
  }
}

typedef CUresult (*RR_Encode)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
 const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static int rr_map_host(CUtensorMap* tm,int64_t p,int inner,int outer,int box) {
  static RR_Encode enc=nullptr;
  if(!enc) { cudaDriverEntryPointQueryResult q; if(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&enc,12000,cudaEnableDefault,&q)!=cudaSuccess||!enc) return -100; }
  cuuint64_t dims[2]={(uint64_t)inner,(uint64_t)outer},stride[1]={(uint64_t)inner*2};
  cuuint32_t sz[2]={64,(uint32_t)box},es[2]={1,1};
  return (int)enc(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)p,dims,stride,sz,es,CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

template<int BN,int S,int ST,int SW,int RED,int PIPE>
int rr_launch_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  constexpr int NR=SW?64:BN, TM=SW?BN:64;
  constexpr int bytes=ST*(2*NR+TM)*128+ST*8*((PIPE&2)?2:1)+48+(RED>=1?64*BN*4:0);
  auto fn=rr_kernel<BN,S,ST,SW,RED,PIPE>;
  static bool configured=false;
  if(!configured) {
    cudaError_t e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    if(e!=cudaSuccess) return -(int)e;
    e=cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
    if(e!=cudaSuccess)return -(int)e;
    configured=true;
  }
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};
  cfg.gridDim=dim3(256/NR*S,(m+TM-1)/TM,1);cfg.blockDim=dim3((PIPE&2)?256:128);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];attr[0].id=cudaLaunchAttributeClusterDimension;attr[0].val.clusterDim={S,1,1};
  if(S>1){cfg.attrs=attr;cfg.numAttrs=1;}
  if(query==1){int n=0;cudaError_t e=cudaOccupancyMaxActiveClusters(&n,fn,&cfg);return e==cudaSuccess?n:-(int)e;}
  CUtensorMap tw,tx;
  int e=rr_map_host(&tw,w,64,256*128*2,NR*2);if(e)return e;
  e=rr_map_host(&tx,x,8192,m,TM);if(e)return e;
  cudaError_t err=cudaLaunchKernelEx(&cfg,fn,tw,tx,(float*)out,m);
  return err==cudaSuccess?0:(int)err;
}
int rr_launch(int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
 switch(id){@@CASES@@ default:return -99;}
}
@@SCALAR@@
'''

ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def build_router(configs):
    src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join(_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rr_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
    name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return _ldd19_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch','rr_scalar','rr_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rr_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rr_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


# Configuration: WGMMA N, K splits, stages, swap operands, reduction, pipeline flags.
# Reduction 0 pulls, 1 pushes with a final barrier, 2 relies on destination byte completion.
# Pipeline bits: 1 keeps one MMA group in flight, 2 adds a producer, 4 initializes reduction first.
ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]


def prepare_router(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = prepare_candidates(weights, ROUTER_CONFIGS)
    st['small'] = {(16,8):pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return run_candidate(x,st,-1)
    if m <= 16:
        return run_small_candidate(x,st)
    return run_candidate(x,st,0 if m <= 32 else 1 if m <= 128 else 2)


ROUTER_SCALAR = r'''
template<int M, int NT, int ST>
__global__ void rr_scalar_kernel(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int rr_scalar(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=rr_scalar_kernel<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
'''

ROUTER_SMALL = r'''
template<int MT,int S,int WK>
__global__ __launch_bounds__(WK*32) void rr_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  constexpr int KU=128/S/WK, NI=MT*32;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+S*NI;
  const uint32_t rb=rr_smem(sm+(S+WK)*NI*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank==0 && tid==0) {rr_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  if(rank==0 && warp==0) __syncwarp();
  if(rank==0 && tid==0) {rr_expect(rb,(S-1)*NI*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  for(int v=tid;v<NI;v+=WK*32) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) rr_add(sum,part[k*NI+v]);
    if(rank==0)recv[v]=sum;
    else rr_push(rr_map(rr_smem(recv+rank*NI+v),0),sum,rr_map(rb,0));
  }
  if(rank==0) {
    __syncthreads();rr_wait(rb,0);
    for(int v=tid;v<NI;v+=WK*32) {
      float4 sum=make_float4(0,0,0,0);
#pragma unroll
      for(int s=0;s<S;s++) rr_add(sum,recv[s*NI+v]);
      const int ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
      if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
      if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
    }
  }
}

template<int MT,int S,int WK>
int rr_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=rr_small_kernel<MT,S,WK>;
  constexpr int bytes=(S+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int rr_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return rr_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return rr_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return rr_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

'''


def pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rr_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out


# ---- olayer/olayer_kernels.py ----


OLAYER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
@@MMA@@

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      __nv_bfloat162 lo=__floats2bfloat162_rn(a[4*j],a[4*j+1]);
      __nv_bfloat162 hi=__floats2bfloat162_rn(a[4*j+2],a[4*j+3]);
      int rb=((tid%128)/32)*16+((tid%16)/8)*8,cb=(tid/128)*BN+8*j+(tid%8);
      uint32_t d=base+2*(cb*72+rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};" :: "r"(d),"r"(*(uint32_t*)&lo),"r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  for(int q=tid;q<T*(R/8);q+=TH) {
    int token=q/(R/8),row=(q%(R/8))*8;
    if(mt*T+token<m) *(uint4*)(out+(long)(mt*T+token)*8192+nt*R+row)=*(uint4*)((__nv_bfloat16*)sm+token*72+row);
  }
}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
'''

OLAYER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
'''


def olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join(f'%{i}' for i in range(n // 2))
        outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
        }}''')
    cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
    calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
    return OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)


def build_olayer(configs):
    for cfg in configs:
        bn,bk,st,pipe,pack,hint,*groups = cfg
        tw = groups[0] if groups else 1
        if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = olayer_source(configs)
    name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return _ldd19_inline(name,OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True,with_pytorch_error_handling=False)


def pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
    tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
    return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()


OLAYER_NEARBY_CONFIGS = {
    bn: (bn,128,4,3,1,1) if bn<=128 else (bn,64,min(8,227000//((64+bn)*128+16)),3,1,1)
    for bn in range(72,249,8)
}
OLAYER_CONFIGS = [(32,128,8,1,1,1),(128,128,4,3,1,1)]  # v19: the o m=32 and m=128 configs only
# LDD19 (integrator, iteration 5): o m=121..128 on BK 64 x 8 stages (work_integrator/probeL1.json); 0 = (128,128,4,3,1,1)
LDD19_OL128 = 1
if LDD19_OL128:
    OLAYER_CONFIGS = [(32,128,8,1,1,1),(128,64,8,3,1,1)]
    OLAYER_NEARBY_CONFIGS[128] = (128,64,8,3,1,1)


def prepare_olayer_candidates(weights,configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled':True,'ext':build_olayer(configs),'configs':configs,'weight':weight,
            'packed':pack_olayer(weight) if any(c[4]==1 for c in configs) else None,
            'tiled32':weight.reshape(256,32,16,64).permute(0,2,1,3).contiguous() if any(c[4]==2 for c in configs) else None}


def run_olayer_candidate(x,st,index) -> torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack==2 else st['packed'] if pack==1 else st['weight']
    out = torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    rc = st['ext'].ol_launch(index,x.data_ptr(),weight.data_ptr(),out.data_ptr(),x.shape[0],torch.cuda.current_stream().cuda_stream,0)
    if rc:
        raise RuntimeError(f'olayer launch failed: {rc}, config={st["configs"][index]}')
    return out


def prepare_olayer(weights) -> dict:
    weight = weights.get('o')
    if weight is None or weight.shape != (8192,1024) or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(weight.device) != (9,0):
        return {'enabled':False}
    st = prepare_olayer_candidates(weights,OLAYER_CONFIGS)
    st['token_configs'] = {bn:OLAYER_CONFIGS.index(cfg) for bn,cfg in OLAYER_NEARBY_CONFIGS.items() if cfg in OLAYER_CONFIGS}
    for index,cfg in enumerate(OLAYER_CONFIGS):
        m = cfg[0]*(cfg[6] if len(cfg)>6 else 1)
        run_olayer_candidate(torch.zeros((m,1024),dtype=torch.bfloat16,device=weight.device),st,index)
    torch.cuda.synchronize(weight.device)
    return st


def run_olayer(x,layer,n,st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'o' or n != 8192:
        return None
    if x.ndim != 2 or x.shape[1] != 1024 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr()%16 or not 1 <= x.shape[0] <= 256:
        return None
    m = x.shape[0]
    if 65<=m<=224:
        index = st['token_configs'].get(((m+7)//8)*8)
        return None if index is None else run_olayer_candidate(x,st,index)
    return None

# ---- cpstage/cpstage_kernels.py ----


tt_lt__LT_CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <vector>

#define LT_OK(call) TORCH_CHECK((call) == CUBLAS_STATUS_SUCCESS, "cuBLASLt failure: ", #call)

using Config = std::array<int64_t, 9>;

Config config_of(const cublasLtMatmulAlgo_t& a) {
  Config c{};
  for (int i = 0; i < 9; ++i) {
    size_t written = 0;
    if (i >= 7) {
      uint16_t value = 0;
      cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i,
                                          &value, sizeof(value), &written);
      c[i] = value;
    } else {
      uint32_t value = 0;
      cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i,
                                          &value, sizeof(value), &written);
      c[i] = value;
    }
  }
  return c;
}

bool set_config(cublasLtMatmulAlgo_t& a, int attr, int value) {
  if (attr >= 7) {
    uint16_t v = value;
    return cublasLtMatmulAlgoConfigSetAttribute(&a,
      (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
  }
  uint32_t v = value;
  return cublasLtMatmulAlgoConfigSetAttribute(&a,
    (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}

std::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
  size_t bytes = 0;
  cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &bytes);
  if (!bytes) return {0};
  std::vector<uint32_t> values(bytes / sizeof(uint32_t));
  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, values.data(), bytes, &bytes) != CUBLAS_STATUS_SUCCESS)
    return {0};
  return values;
}

uint32_t cap_int(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
  uint32_t v = 0;
  size_t written = 0;
  cublasLtMatmulAlgoCapGetAttribute(&a, attr, &v, sizeof(v), &written);
  return v;
}

struct LtContext {
  cublasLtHandle_t handle = nullptr;
  torch::Tensor workspace;
  explicit LtContext(torch::Tensor ws) : workspace(ws) {
    c10::cuda::CUDAGuard guard(ws.device());
    LT_OK(cublasLtCreate(&handle));
  }
  ~LtContext() { if (handle) cublasLtDestroy(handle); }
};

struct LtPlan {
  std::shared_ptr<LtContext> context;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;
  torch::Tensor weight, bias;
  int m, n, k, layout, epilogue;
  std::vector<cublasLtMatmulAlgo_t> algorithms;
  std::vector<size_t> workspaces;
  std::set<Config> seen;
  std::vector<std::vector<int64_t>> diagnostics;
  int checked = 0, heuristic_count = 0, id_count = 0;

  LtPlan(std::shared_ptr<LtContext> ctx, torch::Tensor w, int rows, int cols,
         int inner, int variant, int epi)
      : context(ctx), weight(w), m(rows), n(cols), k(inner), layout(variant), epilogue(epi) {
    c10::cuda::CUDAGuard guard(w.device());
    LT_OK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    int8_t fast = 0;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
    bool is_row = layout % 2;
    bool is_packed = layout >= 2;
    cublasOperation_t trans_a = !is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b = is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    if (is_row) {
      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, m, k, k));
      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, is_packed ? k : n,
                                      is_packed ? n : k, is_packed ? n : k));
      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, m, n, n));
      cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
      for (auto d : {a_layout, b_layout, d_layout})
        LT_OK(cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    } else {
      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, is_packed ? n : k,
                                      is_packed ? k : n, is_packed ? n : k));
      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, k, m, k));
      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, n, m, n));
    }
    cublasLtEpilogue_t e = epi ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    if (epi) {
      bias = torch::zeros({std::max(m, n)}, w.options());
      void* ptr = bias.data_ptr();
      LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &ptr, sizeof(ptr)));
    }
  }

  ~LtPlan() {
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
  }

  bool add(const cublasLtMatmulAlgo_t& algo) {
    Config c = config_of(algo);
    if (seen.count(c)) return false;
    seen.insert(c);
    ++checked;
    if (diagnostics.size() < 32) {
      uint64_t flags = 0;
      size_t written = 0;
      int status = cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
        &flags, sizeof(flags), &written);
      std::vector<int64_t> row(c.begin(), c.end());
      row.push_back(flags);
      row.push_back(status);
      diagnostics.push_back(row);
    }
    if (c[3] != CUBLASLT_REDUCTION_SCHEME_NONE && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
      return false;
    if (c[2] > 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return false;
    uint64_t flags = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                         &flags, sizeof(flags), &written) != CUBLAS_STATUS_SUCCESS) return false;
    if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)
      return false;
    // Algo 66 reports INPUT_16F even for CUDA_R_16BF descriptors in cuBLASLt 13.
    // The descriptors fix BF16 storage; reject TF32 and require FP32 accumulators.
    if (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)
      return false;
    cublasLtMatmulHeuristicResult_t result{};
    if (cublasLtMatmulAlgoCheck(context->handle, operation, a_layout, b_layout,
                               d_layout, d_layout, &algo, &result) != CUBLAS_STATUS_SUCCESS) return false;
    if (result.state != CUBLAS_STATUS_SUCCESS || result.workspaceSize > size_t(context->workspace.numel())) return false;
    algorithms.push_back(algo);
    workspaces.push_back(result.workspaceSize);
    return true;
  }

  int restore(std::vector<int64_t> values) {
    TORCH_CHECK(values.size() == 9, "expected nine cuBLASLt configuration fields");
    cublasLtMatmulAlgo_t algo{};
    if (cublasLtMatmulAlgoInit(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, values[0], &algo) != CUBLAS_STATUS_SUCCESS) return -1;
    for (int i = 1; i < 9; ++i) {
      if (!set_config(algo, i, values[i]) && values[i] != 0) return -1;
    }
    if (!add(algo)) return -1;
    return algorithms.size() - 1;
  }

  void enumerate(bool exhaustive) {
    c10::cuda::CUDAGuard guard(weight.device());
    cublasLtMatmulPreference_t pref;
    LT_OK(cublasLtMatmulPreferenceCreate(&pref));
    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                             &reduction, sizeof(reduction)));
    for (size_t ws : {size_t(0), size_t(4 << 20), size_t(32 << 20), size_t(128 << 20)}) {
      ws = std::min(ws, size_t(context->workspace.numel()));
      LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      std::vector<cublasLtMatmulHeuristicResult_t> result(256);
      int count = 0;
      auto rc = cublasLtMatmulAlgoGetHeuristic(context->handle, operation, a_layout, b_layout,
                                               d_layout, d_layout, pref, 256, result.data(), &count);
      if (rc != CUBLAS_STATUS_SUCCESS) continue;
      heuristic_count += count;
      for (int i = 0; i < count; ++i) if (result[i].state == CUBLAS_STATUS_SUCCESS) add(result[i].algo);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!exhaustive) return;
    std::array<int, 256> ids{};
    LT_OK(cublasLtMatmulAlgoGetIds(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
      CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids.size(), ids.data(), &id_count));
    for (int i = 0; i < id_count; ++i) {
      cublasLtMatmulAlgo_t a{};
      if (cublasLtMatmulAlgoInit(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
          CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[i], &a) != CUBLAS_STATUS_SUCCESS) continue;
      auto tiles = cap_list(a, CUBLASLT_ALGO_CAP_TILE_IDS);
      auto stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);
      for (auto tile : tiles) for (auto stage : stages) {
        auto b = a;
        if (!set_config(b, 1, tile) || !set_config(b, 6, stage)) continue;
        add(b);
        if (tiles.size() > 100) {
          for (int cluster : {2, 3, 4, 5, 6, 7, 8, 9, 11, 12}) {
            auto s = b;
            if (set_config(s, 8, cluster)) add(s);
          }
        }
      }
    }
  }

  void expand(std::vector<int> indices) {
    for (int index : indices) {
      auto a = algorithms.at(index);
      if (cap_int(a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT) &&
          (cap_int(a, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for (int split : {2, 3, 4, 6, 8, 12, 16}) {
          auto b = a;
          if (set_config(b, 2, split) && set_config(b, 3, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) add(b);
        }
      }
      auto stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);
      for (auto stage : stages) {
        auto b = a;
        if (set_config(b, 6, stage)) add(b);
      }
      for (int swizzle = 0; swizzle <= int(cap_int(a, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT)); ++swizzle) {
        for (int custom = 0; custom <= int(cap_int(a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX)); ++custom) {
          for (int cluster : {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12}) {
            auto b = a;
            if (set_config(b, 4, swizzle) && set_config(b, 5, custom) && set_config(b, 8, cluster)) add(b);
          }
        }
      }
    }
  }

  std::vector<std::vector<int64_t>> metadata() const {
    std::vector<std::vector<int64_t>> result;
    for (size_t i = 0; i < algorithms.size(); ++i) {
      Config c = config_of(algorithms[i]);
      std::vector<int64_t> row(c.begin(), c.end());
      row.push_back(workspaces[i]);
      result.push_back(row);
    }
    return result;
  }

  int run(torch::Tensor x, torch::Tensor out, int index) {
    const float alpha = 1.0f, beta = 0.0f;
    const void* a = layout % 2 ? x.data_ptr() : weight.data_ptr();
    const void* b = layout % 2 ? weight.data_ptr() : x.data_ptr();
    return int(cublasLtMatmul(context->handle, operation, &alpha, a, a_layout, b, b_layout,
      &beta, out.data_ptr(), d_layout, out.data_ptr(), d_layout, &algorithms.at(index),
      context->workspace.data_ptr(), workspaces.at(index), at::cuda::getCurrentCUDAStream(x.get_device())));
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("version", []() { return cublasLtGetVersion(); });
  pybind11::class_<LtContext, std::shared_ptr<LtContext>>(m, "Context")
    .def(pybind11::init<torch::Tensor>());
  pybind11::class_<LtPlan, std::shared_ptr<LtPlan>>(m, "Plan")
    .def(pybind11::init<std::shared_ptr<LtContext>, torch::Tensor, int, int, int, int, int>())
    .def("enumerate", &LtPlan::enumerate).def("expand", &LtPlan::expand)
    .def("metadata", &LtPlan::metadata).def("restore", &LtPlan::restore).def("run", &LtPlan::run)
    .def_readonly("checked", &LtPlan::checked).def_readonly("heuristic_count", &LtPlan::heuristic_count)
    .def_readonly("diagnostics", &LtPlan::diagnostics)
    .def_readonly("id_count", &LtPlan::id_count);
}
'''
tt_lt__LT_EXT = None
tt_lt__LT_WORKSPACE_BYTES = 128 << 20
tt_lt__LT_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
tt_lt__LT_H200_FIXED = {(256, 'o'): (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5]), (256, 'unpooler'): (3, 0, [66, 20, 1, 0, 0, 4, 35, 0, 2]), (128, 'qkv_gate'): (0, 0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (256, 'qkv_gate'): (0, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (128, 'pooler'): (2, 0, [66, 15, 1, 0, 0, 4, 35, 0, 2]), (256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2])}
tt_lt__LT_H200_VERSION = 130101

def tt_lt__lt_build():
    global tt_lt__LT_EXT
    if tt_lt__LT_EXT is None:
        tt_lt__LT_EXT = _ldd19_inline(name='muse_cublaslt_v1', cpp_sources=tt_lt__LT_CPP, with_cuda=True, extra_cflags=['-O3', '-std=c++20'], extra_ldflags=['-l:libcublasLt.so.13'], verbose=False)
    return tt_lt__LT_EXT

def tt_lt__lt_state(weights: dict) -> dict:
    ext = tt_lt__lt_build()
    device = weights['qkv_gate'].device
    workspace = torch.empty(tt_lt__LT_WORKSPACE_BYTES, device=device, dtype=torch.uint8)
    return {'ext': ext, 'version': ext.version(), 'context': ext.Context(workspace), 'configs': {}, 'weights': {name: weights[name].contiguous() for name in tt_lt__LT_LAYERS}, 'packed': {}, 'workspace': workspace}

def tt_lt__lt_plan(st: dict, layer: str, m: int, layout: int, epilogue: int):
    w = st['weights'][layer]
    n, k = w.shape
    if layout >= 2:
        if layer not in st['packed']:
            st['packed'][layer] = w.t().contiguous()
        w = st['packed'][layer]
    return st['ext'].Plan(st['context'], w, m, n, k, layout, epilogue)

def tt_lt_prepare_cublaslt(weights: dict) -> dict:
    device = weights['qkv_gate'].device
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0) or not tt_lt__LT_H200_FIXED:
        return {'configs': {}}
    st = tt_lt__lt_state(weights)
    if tt_lt__LT_H200_VERSION is not None and st['version'] != tt_lt__LT_H200_VERSION:
        return {'configs': {}}
    for key, descriptor in tt_lt__LT_H200_FIXED.items():
        m, layer = key
        layout, epilogue, config = descriptor
        plan = tt_lt__lt_plan(st, layer, m, layout, epilogue)
        index = plan.restore(list(config))
        if index < 0 or plan.metadata()[index][:9] != list(config):
            continue
        x = torch.zeros((m, st['weights'][layer].shape[1]), device=device, dtype=torch.bfloat16)
        out = torch.empty((m, st['weights'][layer].shape[0]), device=device, dtype=torch.bfloat16)
        if plan.run(x, out, index) == 0:
            st['configs'][key] = (plan, index)
    torch.cuda.synchronize(device)
    return st

def tt_lt_run_cublaslt(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    entry = st['configs'].get((x.shape[0], layer))
    if entry is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    w = st['weights'][layer]
    if x.device != w.device or w.dtype != torch.bfloat16 or tuple(w.shape) != (n, x.shape[1]) or x.data_ptr() % 256:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = entry[0].run(x, out, entry[1])
    if rc:
        raise RuntimeError(f'cuBLASLt launch failed: status {rc}')
    return out


V9_SELECTED_ROUTES = [{'m': 32, 'layer': 'o', 'name': 'ol_1', 'module': 'ol_v8', 'config': [32, 128, 8, 1, 1, 1]}, {'m': 32, 'layer': 'pooler', 'name': 'cps', 'module': 'cps_v8'}, {'m': 32, 'layer': 'unpooler', 'name': 'mw', 'module': 'mw'}, {'m': 128, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 128, 'layer': 'unpooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'qkv_gate', 'name': 'lt', 'module': 'lt'}]


def candidate_sglang(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor:
    if layer != 'router':
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return F.linear(x.float(), state['router_weight'])
    output = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32)
    output += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32)
    return output


def candidate_v8(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    output = None
    if layer == 'router':
        output = run_router(x, layer, n, state['router'])
    elif layer == 'o' and 65 <= m <= 224:
        output = run_olayer(x, layer, n, state['olayer'])
    if output is None and m <= 32:
        output = run_small_m(x, layer, n, state['sm'])
    return candidate_sglang(x, layer, n, state) if output is None else output


def candidate_base(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    if layer == 'router' and x.shape[0] <= 16:
        output = run_router(x, layer, n, state['router'])
        return candidate_sglang(x, layer, n, state) if output is None else output
    return candidate_v8(data, state)


def candidate_v9(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    if layer == 'o' and m == 32 and n == 8192 and x.shape[1] == 1024 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
        output = run_olayer_candidate(x, state['olayer'], OLAYER_CONFIGS.index((32, 128, 8, 1, 1, 1)))
        if output is not None:
            return output
    if layer == 'pooler' and m == 128:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 128:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 256:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'qkv_gate' and m == 256:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    return candidate_base(data, state)


tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SMALLM2D __device__ __forceinline__

SMALLM2D int4 smallm2_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMALLM2D uint4 smallm2_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMALLM2D uint32_t smallm2_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMALLM2D uint32_t smallm2_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMALLM2D void smallm2_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMALLM2D void smallm2_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = smallm2_prmt(T0, T1, cm);
  const uint32_t hhi = smallm2_prmt(T0, T1, cm >> 16);
  a[0] = smallm2_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = smallm2_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = smallm2_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = smallm2_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void smallm2_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMALLM2D void smallm2_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) smallm2_never(p);
}

SMALLM2D uint32_t smallm2_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMALLM2D uint32_t smallm2_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMALLM2D uint32_t smallm2_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMALLM2D void smallm2_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMALLM2D void smallm2_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMALLM2D void smallm2_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMALLM2D void smallm2_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMALLM2D void smallm2_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMALLM2D void smallm2_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMALLM2D bool smallm2_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMALLM2D uint64_t smallm2_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
SMALLM2D void smallm2_wait_bar(uint32_t bar) {
  if (smallm2_mbar_try(bar)) return;
  const uint64_t t0 = smallm2_gtime();
  while (!smallm2_mbar_try(bar))
    if (smallm2_gtime() - t0 > 20000000ull) __trap();
}

SMALLM2D void smallm2_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMALLM2D float4 smallm2_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMALLM2D void smallm2_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMALLM2D void smallm2_store(float* p, float v) { *p = v; }

template <int FMT> struct Smallm2Fmt;
template <> struct Smallm2Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Smallm2Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Smallm2Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Smallm2Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMALLM2D void smallm2_decode(const int4 (&w)[Smallm2Fmt<FMT>::NL], uint32_t (&A)[Smallm2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Smallm2Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Smallm2Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      smallm2_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      smallm2_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      smallm2_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      smallm2_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMALLM2D void smallm2_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMALLM2D void smallm2_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) smallm2_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) smallm2_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Smallm2Fmt<FMT>::NL, NA = Smallm2Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = smallm2_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = smallm2_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smallm2_mbar_init(bar, 1);
        smallm2_mbar_expect(bar, parts * RT * TV * 128u);
        smallm2_fence_init();
      }
      smallm2_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = smallm2_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = smallm2_ldx(xp);
          xv[slot][n][1] = smallm2_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = smallm2_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smallm2_mbar_init(bar, 1);
        smallm2_mbar_expect(bar, parts * RT * TV * 128u);
        smallm2_fence_init();
      }
      smallm2_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);

    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = smallm2_ldx(xp), v = smallm2_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) smallm2_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) smallm2_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  smallm2_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) smallm2_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          smallm2_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) smallm2_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                smallm2_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) smallm2_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            smallm2_cl_wait();
            if (t < TV) {
              const uint32_t rbar = smallm2_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            smallm2_wait_bar(bar);
            smallm2_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            smallm2_cl_wait();
            const uint32_t rbar = smallm2_mapa(bar, 0);
            smallm2_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            smallm2_wait_bar(bar);
            smallm2_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int smallm2_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
static int smallm2_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = smallm2_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = smallm2_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<int M, int NT, int ST>
__global__ void smallm2_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int smallm2_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=smallm2_router_scalar<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
'''
tt_v10_sm2_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v10_sm2_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_sm2_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v10_sm2_SMALLM2_ESC_CAP = 32


def tt_v10_sm2_smallm2_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return tt_v10_sm2_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v10_sm2_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _ldd19_inline(name, tt_v10_sm2_SMALLM2_CPP + extra_cpp, cuda_sources=tt_v10_sm2_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v10_sm2_smallm2_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_sm2_smallm2_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_sm2_smallm2_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_sm2_smallm2_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_sm2_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_sm2_smallm2_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v10_sm2_smallm2_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v10_sm2_smallm2_c12_unpack(data, base, esc):
    """Inverse of smallm2_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_v10_sm2_SMALLM2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v10_sm2_smallm2_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_sm2_smallm2_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_sm2_smallm2_units(hi), tt_v10_sm2_smallm2_units(lo)]
    else:
        mats = [tt_v10_sm2_smallm2_units(w)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_sm2_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_sm2_smallm2_c12_bytes(v)
        if not torch.equal(tt_v10_sm2_smallm2_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(tt_v10_sm2_smallm2_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_sm2_smallm2_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_sm2_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_sm2_SMALLM2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_sm2_smallm2_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)


def tt_v10_sm2_smallm2_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v10_sm2_smallm2_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v10_sm2_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_sm2_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_sm2_smallm2_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def tt_v10_sm2_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = tt_v10_sm2_smallm2_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = tt_v10_sm2_smallm2_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = tt_v10_sm2_smallm2_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = tt_v10_sm2_smallm2_verify_onehot(ext, inst_id[inst], sp, tt_v10_sm2_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v10_sm2_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
tt_v10_sm2_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def tt_v10_sm2_prepare_smallm2(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in tt_v10_sm2_SMALLM2_ROUTES.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_final_v3')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

def tt_v10_sm2_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != tt_v10_sm2_SMALLM2_LAYER_K[layer]:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = tt_v10_sm2_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
tt_v10_sm2_SMALLM2_ROUTES = {key: value for key, value in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}


V10_ROUTES = [{'m': 8, 'layer': 'pooler', 'name': 'smallm2', 'module': 'v10_sm2'}, {'m': 16, 'layer': 'qkv_gate', 'name': 'warps', 'module': 'v10_warps'}, {'m': 256, 'layer': 'o', 'name': 'm256tune', 'module': 'v10_tune', 'config': [256, 64, 5, 7, 1, 1, 1]}, {'m': 32, 'layer': 'qkv_gate', 'name': 'm32x', 'module': 'v10_m32x'}, {'m': 32, 'layer': 'pooler', 'name': 'm32x', 'module': 'v10_m32x'}]


def candidate_v10(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    output = None
    if layer == 'pooler' and m == 8:
        output = tt_v10_sm2_run_smallm2(x, layer, n, state['v10_sm2'])
    return candidate_v9(data, state) if output is None else output


V11_ROUTES = [{'m': 32, 'layer': 'unpooler', 'name': 'm32x_unpooler', 'module': 'v11_m32x'}]


V12_SM2_O1 = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1))}

def prepare_v12_o1(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in V12_SM2_O1.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_o1_v12')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

V12_ROUTES = [{'m': 1, 'layer': 'unpooler', 'name': 'smtrace1', 'module': 'v12_smtrace1'}, {'m': 1, 'layer': 'o', 'name': 'smallm2_o1', 'module': 'v12_o1'}]


def candidate_v12(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'o' and x.shape[0] == 1:
        output = tt_v10_sm2_run_smallm2(x,layer,n,state['v12_o1'])
        if output is not None:
            return output
    return candidate_v10(data,state)


tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SMD __device__ __forceinline__

SMD int4 smtrace_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint4 smtrace_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMD uint32_t smtrace_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMD uint32_t smtrace_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMD void smtrace_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMD void smtrace_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = smtrace_prmt(T0, T1, cm);
  const uint32_t hhi = smtrace_prmt(T0, T1, cm >> 16);
  a[0] = smtrace_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = smtrace_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = smtrace_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = smtrace_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void smtrace_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMD void smtrace_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) smtrace_never(p);
}

SMD uint32_t smtrace_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMD uint32_t smtrace_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMD uint32_t smtrace_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMD void smtrace_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMD void smtrace_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMD void smtrace_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMD void smtrace_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMD void smtrace_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMD void smtrace_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMD bool smtrace_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMD uint64_t smtrace_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
SMD void smtrace_wait_bar(uint32_t bar) {
  if (smtrace_mbar_try(bar)) return;
  const uint64_t t0 = smtrace_gtime();
  while (!smtrace_mbar_try(bar))
    if (smtrace_gtime() - t0 > 20000000ull) __trap();
}

SMD void smtrace_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMD float4 smtrace_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMD void smtrace_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMD void smtrace_store(float* p, float v) { *p = v; }

template <int FMT> struct SmFmt;
template <> struct SmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct SmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct SmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct SmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMD void smtrace_decode(const int4 (&w)[SmFmt<FMT>::NL], uint32_t (&A)[SmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < SmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < SmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      smtrace_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      smtrace_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      smtrace_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      smtrace_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMD void smtrace_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMD void smtrace_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) smtrace_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
__global__ void __launch_bounds__(NTMAX) smtrace_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = SmFmt<FMT>::NL, NA = SmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = smtrace_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = smtrace_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smtrace_mbar_init(bar, 1);
        smtrace_fence_init();
      }
      if (rank == 0) {
        __syncthreads();
        if(tid==0) {
          const uint32_t parts=RED==2?(uint32_t)((S-1)*NW):(uint32_t)((S-1)*WR);
          smtrace_mbar_expect(bar,parts*RT*TV*128u);
        }
      }
      smtrace_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = smtrace_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = smtrace_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = smtrace_ldx(xp);
          xv[slot][n][1] = smtrace_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    smtrace_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          smtrace_order_point(m, outp);
        } else {
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) smtrace_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          smtrace_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) smtrace_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  smtrace_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) smtrace_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          smtrace_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) smtrace_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                smtrace_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) smtrace_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            smtrace_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            smtrace_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) smtrace_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            smtrace_cl_wait();
            if (t < TV) {
              const uint32_t rbar = smtrace_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            smtrace_wait_bar(bar);
            smtrace_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            smtrace_cl_wait();
            const uint32_t rbar = smtrace_mapa(bar, 0);
            smtrace_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else smtrace_st_async4(smtrace_mapa(smtrace_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            smtrace_wait_bar(bar);
            smtrace_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int smtrace_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS>
static int smtrace_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = smtrace_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = smtrace_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
'''
tt_v13_smtrace8_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v13_smtrace8_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v13_smtrace8_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v13_smtrace8_SMTRACE_ESC_CAP = 32


def tt_v13_smtrace8_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v13_smtrace8_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return _ldd19_inline(name, tt_v13_smtrace8_SMTRACE_CPP + extra_cpp, cuda_sources=tt_v13_smtrace8_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v13_smtrace8_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v13_smtrace8__smtrace_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v13_smtrace8__smtrace_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v13_smtrace8_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v13_smtrace8__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v13_smtrace8_smtrace_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v13_smtrace8__smtrace_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc):
    """Inverse of smtrace_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_v13_smtrace8_SMTRACE_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v13_smtrace8_smtrace_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v13_smtrace8_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v13_smtrace8_smtrace_units(hi), tt_v13_smtrace8_smtrace_units(lo)]
    else:
        mats = [tt_v13_smtrace8_smtrace_units(w)]
    if fmt in (0, 2):
        data = torch.cat([tt_v13_smtrace8_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v13_smtrace8_smtrace_c12_bytes(v)
        if not torch.equal(tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(tt_v13_smtrace8_smtrace_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v13_smtrace8_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v13_smtrace8_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v13_smtrace8_SMTRACE_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v13_smtrace8__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
tt_v13_smtrace8_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
tt_v13_smtrace8_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
tt_v13_smtrace8_SMTRACE_MAX_M = 32
tt_v13_smtrace8_SMTRACE_TABLE_M.update({('router', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1]})
tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def tt_v13_smtrace8_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v13_smtrace8_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v13_smtrace8_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v13_smtrace8_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v13_smtrace8_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = tt_v13_smtrace8_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = tt_v13_smtrace8_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = tt_v13_smtrace8_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id[inst], sp, tt_v13_smtrace8_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v13_smtrace8_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def tt_v13_smtrace8_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(tt_v13_smtrace8_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(tt_v13_smtrace8_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = tt_v13_smtrace8_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, tt_v13_smtrace8_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if tt_v13_smtrace8_smtrace_bf16_twin(inst):
            instances.append(tt_v13_smtrace8_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = tt_v13_smtrace8_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and tt_v13_smtrace8_smtrace_bf16_twin(inst):
            plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, tt_v13_smtrace8_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def tt_v13_smtrace8_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= tt_v13_smtrace8_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', tt_v13_smtrace8_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != tt_v13_smtrace8_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    if tt_v13_smtrace8_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
tt_v13_smtrace8_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}
# LDD19 (integrator, iteration 5): unpooler m=8 slice on a re-swept smtrace geometry (work_integrator/probeU.py);
# LDD19_SMT8 = 0 restores Nathen's (RT1 ST2 RED2, WR1 WK16 KU2 S2)
LDD19_SMT8 = 1
if LDD19_SMT8:
    tt_v13_smtrace8_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 3, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def tt_v13_smtrace8_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in tt_v13_smtrace8_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in tt_v13_smtrace8_SMTRACE_ROUTES.items() if key[0] in usable}
    return tt_v13_smtrace8_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

def tt_v13_smtrace8_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in tt_v13_smtrace8_SMTRACE_ROUTES:
        return None
    return tt_v13_smtrace8_smtrace_run_core(x, layer, n, st)

V13_ROUTES = [{'m': 8, 'layer': 'unpooler', 'name': 'smtrace8', 'module': 'v13_smtrace8'}]


def candidate_v13(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'unpooler' and x.shape[0] == 8:
        output = tt_v13_smtrace8_run_smtrace(x,layer,n,state['v13_smtrace8'])
        if output is not None:
            return output
    return candidate_v12(data,state)


V14_ROUTES = [{'m': 128, 'layer': 'qkv_gate', 'name': 'qkvbig', 'module': 'v14_qkv'}]


V15_ROUTES = [{'m': 256, 'layer': 'unpooler', 'name': 'ltsearch_transposed', 'module': 'lt', 'descriptor': [66, 20, 1, 0, 0, 4, 35, 0, 2], 'layout': 3, 'epilogue': 0, 'algorithm_workspace_bytes': 0}]


def candidate_v15(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'unpooler' and x.shape[0] == 256:
        output = tt_lt_run_cublaslt(x,layer,n,state['new_lt'])
        if output is not None:
            return output
    return candidate_v13(data,state)


_bundle_base_kernel = candidate_v15


dd_decodediet_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}


dd_decodediet_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def dd_decodediet_dd_cap2_b_cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


dd_decodediet_CPSSM_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define CPSSD __device__ __forceinline__

CPSSD int4 dd_cap2_b_cpssm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint4 dd_cap2_b_cpssm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint32_t dd_cap2_b_cpssm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

CPSSD uint32_t dd_cap2_b_cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

CPSSD void dd_cap2_b_cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
CPSSD void dd_cap2_b_cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x5140u);
  a[1] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x7362u);
  a[2] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x5140u);
  a[3] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x7362u);
}

__device__ __noinline__ void dd_cap2_b_cpssm_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
CPSSD void dd_cap2_b_cpssm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) dd_cap2_b_cpssm_never(p);
}

CPSSD uint32_t dd_cap2_b_cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

CPSSD uint32_t dd_cap2_b_cpssm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

CPSSD uint32_t dd_cap2_b_cpssm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

CPSSD void dd_cap2_b_cpssm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

CPSSD void dd_cap2_b_cpssm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

CPSSD void dd_cap2_b_cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
CPSSD void dd_cap2_b_cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
CPSSD void dd_cap2_b_cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

CPSSD void dd_cap2_b_cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

CPSSD bool dd_cap2_b_cpssm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

CPSSD uint64_t dd_cap2_b_cpssm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
CPSSD void dd_cap2_b_cpssm_wait_bar(uint32_t bar) {
  if (dd_cap2_b_cpssm_mbar_try(bar)) return;
  const uint64_t t0 = dd_cap2_b_cpssm_gtime();
  while (!dd_cap2_b_cpssm_mbar_try(bar))
    if (dd_cap2_b_cpssm_gtime() - t0 > 20000000ull) __trap();
}

CPSSD void dd_cap2_b_cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

CPSSD float4 dd_cap2_b_cpssm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

CPSSD void dd_cap2_b_cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
CPSSD void dd_cap2_b_cpssm_store(float* p, float v) { *p = v; }

template <int FMT> struct CpsSmFmt;
template <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
CPSSD void dd_cap2_b_cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      dd_cap2_b_cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      dd_cap2_b_cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      dd_cap2_b_cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      dd_cap2_b_cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
CPSSD void dd_cap2_b_cpssm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
CPSSD void dd_cap2_b_cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) dd_cap2_b_cpssm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
__global__ void __launch_bounds__(NTMAX) dd_cap2_b_cpssm_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = dd_cap2_b_cpssm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = dd_cap2_b_cpssm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        dd_cap2_b_cpssm_mbar_init(bar, 1);
        dd_cap2_b_cpssm_mbar_expect(bar, parts * RT * TV * 128u);
        dd_cap2_b_cpssm_fence_init();
      }
      dd_cap2_b_cpssm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = dd_cap2_b_cpssm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4* rawring = reinterpret_cast<int4*>(smem);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=dd_cap2_b_cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = dd_cap2_b_cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = dd_cap2_b_cpssm_ldx(xp);
          xv[slot][n][1] = dd_cap2_b_cpssm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    dd_cap2_b_cpssm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          dd_cap2_b_cpssm_order_point(m, outp);
        } else {
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=dd_cap2_b_cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) dd_cap2_b_cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          dd_cap2_b_cpssm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) dd_cap2_b_cpssm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  dd_cap2_b_cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) dd_cap2_b_cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          dd_cap2_b_cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) dd_cap2_b_cpssm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + rawbytes);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) dd_cap2_b_cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            dd_cap2_b_cpssm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            dd_cap2_b_cpssm_wait_bar(bar);
            dd_cap2_b_cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            dd_cap2_b_cpssm_cl_wait();
            const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);
            dd_cap2_b_cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            dd_cap2_b_cpssm_wait_bar(bar);
            dd_cap2_b_cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int dd_cap2_b_cpssm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
static int dd_cap2_b_cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = dd_cap2_b_cpssm_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = dd_cap2_b_cpssm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}
'''
dd_decodediet_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
dd_decodediet_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
dd_decodediet_CPSSM_ESC_CAP = 32

def dd_decodediet_dd_cap2_b_cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = dd_decodediet_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = dd_decodediet_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def dd_decodediet_dd_cap2_b_cpssm_table(base):
    return base * 16843009

def dd_decodediet_dd_cap2_b_cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [dd_decodediet_dd_cap2_b_cpssm_units(hi), dd_decodediet_dd_cap2_b_cpssm_units(lo)]
    else:
        mats = [dd_decodediet_dd_cap2_b_cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v)
        if not torch.equal(dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(dd_decodediet_dd_cap2_b_cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def dd_decodediet_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, dd_decodediet_CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > dd_decodediet_CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = dd_decodediet_cpstage___to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
dd_decodediet_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
dd_decodediet_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
dd_decodediet_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
dd_decodediet_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
dd_decodediet_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
dd_decodediet_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def dd_decodediet_dd_cap2_b_cpssm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def dd_decodediet_dd_cap2_b_cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def dd_decodediet_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = dd_decodediet_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = dd_decodediet_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = dd_decodediet_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def dd_decodediet_cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def dd_decodediet_cpstage___to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
dd_decodediet_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def dd_decodediet_cps_build(family, instances, depth):
    is32 = False
    prefix = 'dd_cap2_b_cpssm_'
    template = dd_decodediet_CPSSM_CUDA_TEMPLATE
    cpp = dd_decodediet_CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return _ldd19_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

# LDD19 (integrator, iteration 5): pooler / unpooler m=16 plans with m fixed at compile time; 0 = bucket plan only
LDD19_CPS16 = 1
LDD19_CPS16U = 1   # LDD19 (sweeper, iteration 6): unpooler m=16 plan on WR1 WK4 KU16 (work_sweeper/probe6.json)

def dd_decodediet_prepare_cps(weights, depth=None, layers=dd_decodediet_CPS_LAYERS, buckets=(2, 4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
        for bucket, selected_depth in ((2, 2), (4, 1)):
            if bucket in buckets and selected_layers:
                part = dd_decodediet_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0, 1, 2, 3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm',):
        table = dd_decodediet_CPSSM_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and (layer in dd_decodediet_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and ((layer, bucket) in dd_decodediet_CPS32_TABLE_MT):
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
            entries.append(((layer, 'mt', bucket), (inst, cfg)))
            if LDD19_CPS16 and bucket == 2 and layer in ('pooler', 'unpooler') and inst[7] == 0:   # LDD19 (integrator, iteration 5)
                if LDD19_CPS16U and layer == 'unpooler' and tuple(cfg) == (2, 4, 16, 1):   # LDD19 (sweeper, iteration 6)
                    entries.append(((layer, 'm', 16), (inst[:7] + (16,) + inst[8:10] + (1, 4, 16, 1), (1, 4, 16, 1))))
                else:
                    entries.append(((layer, 'm', 16), (inst[:7] + (16,) + inst[8:], cfg)))
        if include_policy and family == 'sm':
            for (layer, m), (inst, cfg) in dd_decodediet_CPSSM_TABLE_M.items():
                if layer not in dd_decodediet_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'm', m), (inst, cfg)))
        if not entries:
            continue
        twin = dd_decodediet_dd_cap2_b_cpssm_bf16_twin
        instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
        ext = dd_decodediet_cps_build(family, instances, depth)
        ids = {inst: i for i, inst in enumerate(instances)}
        plan_fn = dd_decodediet_dd_cap2_b_cpssm_plan
        packs, streams = ({}, {})
        for key, (inst, cfg) in entries:
            sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
            if sp is not None:
                st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
    return st

def dd_decodediet_cps_launch(sp, x, out, n, policy=0, query=0):
    fn = dd_decodediet_dd_cap2_b_cpssm_launch_raw
    return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def dd_decodediet_run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', dd_decodediet_dd_cap2_b_cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != dd_decodediet_CPS32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    rc = dd_decodediet_cps_launch(sp, x, out, n, st.get('policy', 0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out


def dd_prepare_decodediet(weights) -> dict:
    routes = dd_DECODEDIET_ROUTES
    state = {'routes': frozenset(routes)}
    if not routes:
        return state
    if any((m == 16 and layer in ('pooler', 'unpooler') for layer, m in routes)):
        layers = tuple((layer for layer, m in routes if m == 16 and layer in ('pooler', 'unpooler')))
        state['cps'] = dd_decodediet_prepare_cps(weights, layers=layers, buckets=(2,))
    return state

def dd_run_decodediet(x, layer, n, state) -> torch.Tensor | None:
    m = x.shape[0] if x.ndim == 2 else 0
    if (layer, m) not in state.get('routes', ()):
        return None
    return dd_decodediet_run_cps(x, layer, n, state['cps'])


rt__CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void routertail_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_expect(uint32_t b, int n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b),"r"(n):"memory"); }
RD void routertail_wait(uint32_t b, int ph) {
  asm volatile("{ .reg .pred p; wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1; @!p bra wait; }" ::"r"(b),"r"(ph):"memory");
}
RD void routertail_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" :::"memory"); }
RD uint32_t routertail_map(uint32_t a, int r) { uint32_t b; asm("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(b):"r"(a),"r"(r));return b; }
RD void routertail_push(uint32_t d, float4 v, uint32_t b) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(d),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w),"r"(b):"memory");
}
RD float4 routertail_pull(uint32_t d) {
  float4 v; asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(d):"memory");return v;
}
RD void routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3}], [%4];" ::"r"(dst),"l"(tm),"r"(k),"r"(r),"r"(bar):"memory");
}

#ifndef LDD19_RT_CM
#define LDD19_RT_CM 1
#endif
template<int MT,int S,int WK,bool FM=false>   // LDD19 (escape_lab): FM = compile-time m == 8 * MT
__global__ __launch_bounds__(WK*32) void routertail_small_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m) {
  if constexpr (FM) m = MT * 8;
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][8];uint4 xx[KU][MT][2];
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<8;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*8+j)*32+lane;
      asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];":"=r"(w[k][j].x),"=r"(w[k][j].y),"=r"(w[k][j].z),"=r"(w[k][j].w):"l"(p));
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(xx[k][n][h].x),"=r"(xx[k][n][h].y),"=r"(xx[k][n][h].z),"=r"(xx[k][n][h].w):"l"(p+h*8));
    }
  }
  if(rank<P && tid==0) {routertail_init(rb,1);asm volatile("fence.proxy.async.shared::cta;" :::"memory");}
  __syncthreads();
  if(rank<P && tid==0) {routertail_expect(rb,PARTS*OWN*16);asm volatile("fence.mbarrier_init.release.cluster;" :::"memory");}
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" :::"memory");
  float c[2][MT][4]={};
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          const int4 a=w[k][q+h*4];
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       :"+f"(c[h][n][0]),"+f"(c[h][n][1]),"+f"(c[h][n][2]),"+f"(c[h][n][3]):"r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[2*q]),"r"(b[2*q+1]));
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  asm volatile("barrier.cluster.wait.acquire.aligned;" :::"memory");
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    routertail_push(routertail_map(routertail_smem(recv+rank*OWN+local),dst),sum,routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }

}

template<int MT,int S,int WK,bool FM=false>
int routertail_small_t(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto fn=routertail_small_kernel<MT,S,WK,FM>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  if(query==2){cudaFuncAttributes fa;cudaFuncGetAttributes(&fa,fn);return fa.numRegs*1000+fa.localSizeBytes;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m);
}
int routertail_small(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t stream,int query) {
#if LDD19_RT_CM   // LDD19 (escape_lab): m = 8 / 16 / 32 on compile-time instances
#define RSMF(S,W) if(s==S && wk==W) {if(m==16)return routertail_small_t<2,S,W,true>(w,x,out,m,stream,query);}
RSMF(8,8) RSMF(16,8)
if(s==16 && wk==4) {if(m==16)return routertail_small_t<2,16,4,true>(w,x,out,m,stream,query);if(m==32)return routertail_small_t<4,16,4,true>(w,x,out,m,stream,query);}   // LDD19 (escape_lab)
#undef RSMF
#endif
#define RSM(S,W) if(s==S && wk==W) {if(m<=8)return routertail_small_t<1,S,W>(w,x,out,m,stream,query);if(m<=16)return routertail_small_t<2,S,W>(w,x,out,m,stream,query);if(m<=32)return routertail_small_t<4,S,W>(w,x,out,m,stream,query);}
RSM(8,8) RSM(16,8) RSM(16,4)
return -99;
}

'''
# LDD19 (router_owner, iteration 2): router m=8 on routertail S=8 WK=8 (128 CTAs, its own pack; probe1: 5.15-5.20 vs
# 5.50 us), router m=32 on routertail MT=4 (6.37-6.42 vs rr_kernel 6.58). Both validated in prepare() (every one-hot
# column bit-exact against hi + lo, random rows vs FP64); a failure keeps the previous route.
LDD19_RT8_S8 = True
LDD19_RT32 = True
LDD19_RT_WK4 = {32}   # LDD19 (escape_lab): router m on routertail S=16 WK=4 (own pack)
rt__CPP = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def rt__pack(w, s=16, wk=8):
    import torch
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)

    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def rt_prepare_routertail_candidates(weights) -> dict:
    import torch
    w = weights.get('router')
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    name = 'routertail_' + hashlib.sha256(rt__CUDA.encode()).hexdigest()[:12]
    ext = _ldd19_inline(name, rt__CPP, cuda_sources=rt__CUDA, functions=['routertail_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
    st = {'enabled': True, 'ext': ext, 'weight': w.contiguous(), 'packed': rt__pack(w)}
    for m in (8, 16, 17):
        x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
        rt_run_routertail_candidate(x, st)
    torch.cuda.synchronize(w.device)
    if LDD19_RT8_S8:  # LDD19 (router_owner): m <= 8 on S=8 WK=8
        p8 = rt__pack(w, 8, 8)
        if rt_ldd19_verify(st, p8, 8, 8, 8):
            st['packed8'] = p8
    st['rt32'] = bool(LDD19_RT32) and rt_ldd19_verify(st, st['packed'], 16, 8, 32)  # LDD19 (router_owner)
    if LDD19_RT_WK4:  # LDD19 (escape_lab): S=16 WK=4 routes, each validated (one-hot bit-exact, FP64 rows)
        p4 = rt__pack(w, 16, 4)
        st['wk4'] = {m for m in LDD19_RT_WK4 if rt_ldd19_verify(st, p4, 16, 4, m)}
        if st['wk4']:
            st['packed4'] = p4
    torch.cuda.synchronize(w.device)
    return st

def rt_ldd19_verify(st, packed, s, wk, m):
    """LDD19 (router_owner): every one-hot K column bit-exact against hi + lo (in FP32), random rows vs FP64."""
    import torch
    w = st['weight']
    hi = w.to(torch.bfloat16)
    exact = hi.float() + (w - hi.float()).to(torch.bfloat16).float()
    stream = torch.cuda.current_stream().cuda_stream
    eye = torch.eye(8192, dtype=torch.bfloat16, device=w.device).view(8192 // m, m, 8192)
    got = torch.empty((8192 // m, m, 256), dtype=torch.float32, device=w.device)
    for i in range(8192 // m):
        if st['ext'].routertail_small(packed.data_ptr(), eye[i].data_ptr(), got[i].data_ptr(), m, s, wk, stream, 0):
            return False
    if not torch.equal(got.view(8192, 256), exact.t()):
        return False
    g = torch.Generator(device='cpu').manual_seed(1234)
    x = torch.randn((m, 8192), generator=g).to(device=w.device, dtype=torch.bfloat16)
    x[0, 77] = 64.0
    out = torch.full((m, 256), float('nan'), dtype=torch.float32, device=w.device)
    if st['ext'].routertail_small(packed.data_ptr(), x.data_ptr(), out.data_ptr(), m, s, wk, stream, 0):
        return False
    ref = x.double() @ w.double().T
    return bool(torch.isfinite(out).all()) and float((out.double() - ref).norm() / ref.norm()) < 2.5e-05

def rt_run_routertail_candidate(x, st):
    import torch
    if not st.get('enabled') or x.ndim != 2 or x.shape[1] != 8192 or (not 2 <= len(x) <= 32):
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != st['weight'].device:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((len(x), 256), device=x.device, dtype=torch.float32)
    if len(x) in st.get('wk4', ()):  # LDD19 (escape_lab): S=16 WK=4
        rc = st['ext'].routertail_small(st['packed4'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 16, 4, torch.cuda.current_stream().cuda_stream, 0)
    elif len(x) <= 8 and 'packed8' in st:  # LDD19 (router_owner): S=8 WK=8 for m <= 8
        rc = st['ext'].routertail_small(st['packed8'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 8, 8, torch.cuda.current_stream().cuda_stream, 0)
    else:
        rc = st['ext'].routertail_small(st['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 16, 8, torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routertail launch failed: {rc}')
    return out


mt_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
mt_W0_DC2_ESC_CAP = 32

def mt_wp0_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def mt_wp0_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def mt_wp0_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def mt_wp0_bf16_bytes(v):
    r, c = v.shape[:2]
    words = mt_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def mt_wp0_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = mt_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def mt_wp0_c12_unpack(data, base, esc, decode=0):
    """Inverse of wp0_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def mt_wp0_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def mt_wp0_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [mt_wp0_units(hi), mt_wp0_units(lo)]
    else:
        mats = [mt_wp0_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([mt_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = mt_wp0_c12_bytes(v, decode)
        if not torch.equal(mt_wp0_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(mt_wp0_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def mt_wp0_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def mt_wp0_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = mt_wp0_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, mt_W0_DC2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > mt_W0_DC2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = mt_wp0_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
mt_M16TRACE_ROUTES = {'o': {'cfg': (2, 4, 4, 1), 'inst': (1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), 'decode': 0, 'rows': 8192, 'k': 1024, 'variant': 'next_early'}}
mt_M16TRACE_SOURCES = {'o': ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m16trace_59fcb5b72994_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n', r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define M32D __device__ __forceinline__

M32D int4 m16trace_59fcb5b72994_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint4 m16trace_59fcb5b72994_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint32_t m16trace_59fcb5b72994_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

M32D uint32_t m16trace_59fcb5b72994_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

M32D void m16trace_59fcb5b72994_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
M32D void m16trace_59fcb5b72994_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = m16trace_59fcb5b72994_prmt(T0, T1, cm);
  const uint32_t hhi = m16trace_59fcb5b72994_prmt(T0, T1, cm >> 16);
  a[0] = m16trace_59fcb5b72994_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = m16trace_59fcb5b72994_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = m16trace_59fcb5b72994_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = m16trace_59fcb5b72994_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void m16trace_59fcb5b72994_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
M32D void m16trace_59fcb5b72994_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) m16trace_59fcb5b72994_never(p);
}

M32D uint32_t m16trace_59fcb5b72994_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

M32D uint32_t m16trace_59fcb5b72994_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

M32D uint32_t m16trace_59fcb5b72994_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

M32D void m16trace_59fcb5b72994_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

M32D void m16trace_59fcb5b72994_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

M32D void m16trace_59fcb5b72994_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
M32D void m16trace_59fcb5b72994_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
M32D void m16trace_59fcb5b72994_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

M32D void m16trace_59fcb5b72994_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

M32D bool m16trace_59fcb5b72994_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

M32D uint64_t m16trace_59fcb5b72994_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
M32D void m16trace_59fcb5b72994_wait_bar(uint32_t bar) {
  if (m16trace_59fcb5b72994_mbar_try(bar)) return;
  const uint64_t t0 = m16trace_59fcb5b72994_gtime();
  while (!m16trace_59fcb5b72994_mbar_try(bar))
    if (m16trace_59fcb5b72994_gtime() - t0 > 20000000ull) __trap();
}

M32D void m16trace_59fcb5b72994_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

M32D float4 m16trace_59fcb5b72994_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

M32D void m16trace_59fcb5b72994_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
M32D void m16trace_59fcb5b72994_store(float* p, float v) { *p = v; }

template <int FMT> struct M32Fmt;
template <> struct M32Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct M32Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct M32Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct M32Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
M32D void m16trace_59fcb5b72994_decode(const int4 (&w)[M32Fmt<FMT>::NL], uint32_t (&A)[M32Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < M32Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < M32Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      m16trace_59fcb5b72994_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      m16trace_59fcb5b72994_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      m16trace_59fcb5b72994_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      m16trace_59fcb5b72994_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
M32D void m16trace_59fcb5b72994_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
M32D void m16trace_59fcb5b72994_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) m16trace_59fcb5b72994_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
__global__ void __launch_bounds__(NTMAX) m16trace_59fcb5b72994_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = M32Fmt<FMT>::NL, NA = M32Fmt<FMT>::NA, MTT = MT * TG;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int wbytes = XM == 4 ? CWR * CWK * RT * NA * 2048 : 0;
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = m16trace_59fcb5b72994_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = m16trace_59fcb5b72994_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        m16trace_59fcb5b72994_mbar_init(bar, 1);
        m16trace_59fcb5b72994_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        m16trace_59fcb5b72994_fence_init();
      }
      m16trace_59fcb5b72994_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = m16trace_59fcb5b72994_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = m16trace_59fcb5b72994_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = m16trace_59fcb5b72994_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if (XM != 4 || tg == 0) w[slot][j][l] = m16trace_59fcb5b72994_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = m16trace_59fcb5b72994_ldx(xp);
          xv[slot][n][1] = m16trace_59fcb5b72994_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    m16trace_59fcb5b72994_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          m16trace_59fcb5b72994_order_point(m, outp);
        } else {
          int4 current[RT][NL];
#pragma unroll
          for(int j=0;j<RT;++j)
#pragma unroll
            for(int z=0;z<NL;++z)current[j][z]=w[s][j][z];
          if(si+ST<KU)load(s,si+ST);
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=m16trace_59fcb5b72994_ldx(xp), z1=m16trace_59fcb5b72994_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            m16trace_59fcb5b72994_order_point(m, outp);
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) m16trace_59fcb5b72994_decode<FMT>(current[j], A[j], T0, T1, U0, U1);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }

          m16trace_59fcb5b72994_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) m16trace_59fcb5b72994_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  m16trace_59fcb5b72994_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = m16trace_59fcb5b72994_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  m16trace_59fcb5b72994_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) m16trace_59fcb5b72994_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = m16trace_59fcb5b72994_ldx(xp); z1 = m16trace_59fcb5b72994_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  m16trace_59fcb5b72994_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) m16trace_59fcb5b72994_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          m16trace_59fcb5b72994_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) m16trace_59fcb5b72994_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                m16trace_59fcb5b72994_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) m16trace_59fcb5b72994_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) m16trace_59fcb5b72994_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            m16trace_59fcb5b72994_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            m16trace_59fcb5b72994_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) m16trace_59fcb5b72994_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            m16trace_59fcb5b72994_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = m16trace_59fcb5b72994_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else m16trace_59fcb5b72994_st_async4(m16trace_59fcb5b72994_mapa(m16trace_59fcb5b72994_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            m16trace_59fcb5b72994_wait_bar(bar);
            if constexpr (RED == 4) m16trace_59fcb5b72994_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else m16trace_59fcb5b72994_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            m16trace_59fcb5b72994_cl_wait();
            const uint32_t rbar = m16trace_59fcb5b72994_mapa(bar, 0);
            m16trace_59fcb5b72994_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else m16trace_59fcb5b72994_st_async4(m16trace_59fcb5b72994_mapa(m16trace_59fcb5b72994_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            m16trace_59fcb5b72994_wait_bar(bar);
            m16trace_59fcb5b72994_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int m16trace_59fcb5b72994_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
static int m16trace_59fcb5b72994_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = m16trace_59fcb5b72994_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * M32Fmt<FMT>::NA * 2048;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = m16trace_59fcb5b72994_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int m16trace_59fcb5b72994_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
case 0: return m16trace_59fcb5b72994_launch_t<1,2,1,1,0,0,256,0,8192,1024,2,4,4,1,7,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);
case 1: return m16trace_59fcb5b72994_launch_t<0,2,1,1,0,0,256,0,8192,1024,2,4,4,1,7,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);
    default: return -99;
  }
}
''', 'm16trace_59fcb5b72994_launch')}

def mt__launch(fn, index, plan, x, out, query=0) -> int:
    return fn(index, plan['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(), x.shape[0], plan['rows'], plan['k'], *plan['cfg'], plan['t01'], plan['u01'], torch.cuda.current_stream().cuda_stream, query)

def mt__onehot(fn, index, plan, weight) -> bool:
    x = torch.zeros((16, plan['k']), device=weight.device, dtype=torch.bfloat16)
    out = torch.empty((16, plan['rows']), device=weight.device, dtype=torch.bfloat16)
    ids = torch.arange(16, device=weight.device)
    for column in range(0, plan['k'], 16):
        x.zero_()
        x[ids, column + ids] = 1
        if mt__launch(fn, index, plan, x, out):
            return False
        if not torch.equal(out, weight[:, column:column + 16].t()):
            return False
    return True

def mt_prepare_m16trace(weights) -> dict:
    state = {'plans': {}}
    for layer, route in mt_M16TRACE_ROUTES.items():
        weight = weights.get(layer)
        if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16:
            continue
        if weight.ndim != 2 or tuple(weight.shape) != (route['rows'], route['k']):
            continue
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            continue
        cpp, cuda, name = mt_M16TRACE_SOURCES[layer]
        ext = _ldd19_inline(name, cpp, cuda_sources=cuda, functions=[name], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        fn = getattr(ext, name)
        for index, fmt in enumerate((1, 0)):
            pk = mt_wp0_pack_format(weight, fmt, decode=route['decode'])
            if pk is None:
                continue
            sp = mt_wp0_stream(pk, 1, *route['cfg'])
            if sp is None:
                continue
            plan = dict(sp, **route, fn=fn, index=index)
            x = torch.empty((16, route['k']), device=weight.device, dtype=torch.bfloat16)
            out = torch.empty((16, route['rows']), device=weight.device, dtype=torch.bfloat16)
            attr = mt__launch(fn, index, plan, x, out, 2)
            if attr < 0 or attr % 1000:
                continue
            if not mt__onehot(fn, index, plan, weight):
                continue
            state['plans'][layer] = plan
            break
    return state

def mt_run_m16trace(x, layer, n, state) -> torch.Tensor | None:
    if x.ndim != 2 or x.shape[0] != 16 or (not x.is_cuda) or (x.dtype != torch.bfloat16) or (not x.is_contiguous()):
        return None
    plan = state['plans'].get(layer)
    if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((16, n), device=x.device, dtype=torch.bfloat16)
    rc = mt__launch(plan['fn'], plan['index'], plan, x, out)
    if rc:
        raise RuntimeError(f'm16trace launch failed: {rc}')
    return out
dd_DECODEDIET_ROUTES = (('pooler',16),('unpooler',16))

def prepare_bundle(weights) -> dict:
    return {'dd': dd_prepare_decodediet(weights), 'rt': rt_prepare_routertail_candidates(weights), 'mt': mt_prepare_m16trace(weights)}

def run_bundle(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if layer in ('pooler', 'unpooler') and m == 16:
        return dd_run_decodediet(x, layer, n, st['dd'])
    if layer == 'router' and m in (8, 16):
        return rt_run_routertail_candidate(x, st['rt']) if n == 256 else None
    if layer == 'router' and m == 32 and n == 256 and st['rt'].get('rt32'):  # LDD19 (router_owner): MT=4 routertail
        return rt_run_routertail_candidate(x, st['rt'])
    if layer == 'o' and m == 16:
        return mt_run_m16trace(x, layer, n, st['mt'])
    return None


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    output = None
    if 'bundle' in state:
        output = run_bundle(data['x'], data['config']['layer'], data['config']['n'], state['bundle'])
    return _bundle_base_kernel(data, state) if output is None else output


_b2_base_kernel = kernel_fn


b2o_OLAYER_CUDA = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t r3_o32_o_direct_a_ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void r3_o32_o_direct_a_ol_wait(uint32_t b,int ph) {

  asm volatile("{ .reg .pred ready, expired; .reg .b64 start, now, elapsed;\n"
    "mbarrier.try_wait.parity.shared::cta.b64 ready,[%0],%1; @ready bra DONE;\n"
    "mov.u64 start, %%globaltimer;\n"
    "WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready,[%0],%1; @ready bra DONE;\n"
    "mov.u64 now, %%globaltimer; sub.u64 elapsed,now,start;\n"
    "setp.gt.u64 expired,elapsed,20000000; @expired trap; bra WAIT;\n"
    "DONE: }" :: "r"(b), "r"(ph) : "memory");

}
OD void r3_o32_o_direct_a_ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t r3_o32_o_direct_a_ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void r3_o32_o_direct_a_ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
@@MMA@@

template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void r3_o32_o_direct_a_ol_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=r3_o32_o_direct_a_ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto load=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) r3_o32_o_direct_a_ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else r3_o32_o_direct_a_ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
    r3_o32_o_direct_a_ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) load(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {r3_o32_o_direct_a_ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      r3_o32_o_direct_a_ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=r3_o32_o_direct_a_ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=r3_o32_o_direct_a_ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        @@CALL@@
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) r3_o32_o_direct_a_ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    int row=(tid%128/32)*16+(tid%32)/4;
    int col=2*(tid%4);
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      #pragma unroll
      for(int h=0;h<2;h++) {
        int token=mt*T+(tid/128)*BN+col+8*j;
        if(token<m) out[(long)token*8192+nt*R+row+8*h]=__float2bfloat16_rn(a[4*j+2*h]);
        if(token+1<m) out[(long)(token+1)*8192+nt*R+row+8*h]=__float2bfloat16_rn(a[4*j+2*h+1]);
      }
    }
  }

}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int r3_o32_o_direct_a_ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int r3_o32_o_direct_a_ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=r3_o32_o_direct_a_ol_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=r3_o32_o_direct_a_ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=r3_o32_o_direct_a_ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  fn<<<dim3(8192/R,(m+T-1)/T),128*TW+((PIPE&2)?128:0),SMEM,(cudaStream_t)stream>>>(tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  return int(cudaGetLastError());
}
int r3_o32_o_direct_a_ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {@@CASES@@ default:return -99;}
}
'''
b2o_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_o32_o_direct_a_ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def b2o_olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void r3_o32_o_direct_a_ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return r3_o32_o_direct_a_ol_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) r3_o32_o_direct_a_ol_mma{n}(a,aw,ax);' for n in ns))
    return b2o_OLAYER_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def b2o_build_olayer(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (0, 1, 2, 3)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = b2o_olayer_source(configs)
    name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return _ldd19_inline(name, b2o_OLAYER_CPP, cuda_sources=source, functions=['r3_o32_o_direct_a_ol_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2o_pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64, device=weight.device)[None, :] ^ (torch.arange(64, device=weight.device)[:, None] & 7) * 8
    tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
    return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()

def b2o_prepare_olayer_candidates(weights, configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled': True, 'ext': b2o_build_olayer(configs), 'configs': configs, 'weight': weight, 'packed': b2o_pack_olayer(weight) if any((c[4] == 1 for c in configs)) else None, 'tiled32': weight.reshape(256, 32, 16, 64).permute(0, 2, 1, 3).contiguous() if any((c[4] == 2 for c in configs)) else None}

def b2o_run_olayer_candidate(x, st, index) -> torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack == 2 else st['packed'] if pack == 1 else st['weight']
    out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
    rc = st['ext'].r3_o32_o_direct_a_ol_launch(index, x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f"olayer launch failed: {rc}, config={st['configs'][index]}")
    return out


def run_bundle2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    m, k = x.shape
    if layer == 'o' and m == 32 and (n == 8192) and (k == 1024) and (x.device == st['o']['weight'].device):
        return b2o_run_olayer_candidate(x, st['o'], 0)
    return None


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    out = None
    if 'bundle2' in state:
        out = run_bundle2(data['x'], data['config']['layer'], data['config']['n'], state['bundle2'])
    return _b2_base_kernel(data, state) if out is None else out


from concurrent.futures import ThreadPoolExecutor as _PreparePool

def prepare(weights: Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    device = weights['qkv_gate'].device
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = (weights['router']-hi.float()).to(torch.bfloat16).t().contiguous()
    state['router_weight'] = weights['router']
    torch.cuda.synchronize(device)
    jobs = v19_jobs(weights)
    def run_job(fn, selected_weights, options):
        with torch.cuda.device(device):
            return fn(selected_weights, **options)
    with _PreparePool(max_workers=4) as pool:
        pending = [(name, pool.submit(run_job, fn, selected_weights, options))
                   for name, fn, selected_weights, options in jobs]
        for name, future in pending:
            state[name] = future.result()
    for routes, fn in ((V9_SELECTED_ROUTES,candidate_v9),(V10_ROUTES,candidate_v10),
                       (V11_ROUTES,candidate_v10),(V12_ROUTES,candidate_v12),
                       (V13_ROUTES,candidate_v13),(V14_ROUTES,candidate_v13),
                       (V15_ROUTES,candidate_v15)):
        for route in routes:
            m, layer = route['m'], route['layer']
            weight = weights[layer]
            n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
            data = {'x':torch.zeros((m,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
                    'config':{'m':m,'n':n,'k':weight.shape[1],'layer':layer}}
            fn(data, state)
    torch.cuda.synchronize(device)
    return state


V18S_CUDA_TEMPLATE = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SM3D __device__ __forceinline__

SM3D int4 v18s_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint4 v18s_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint32_t v18s_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SM3D uint32_t v18s_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SM3D void v18s_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = v18s_prmt(T0, T1, cm);
  const uint32_t hhi = v18s_prmt(T0, T1, cm >> 16);
  a[0] = v18s_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = v18s_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = v18s_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = v18s_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void v18s_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SM3D void v18s_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) v18s_never(p);
}

SM3D uint32_t v18s_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SM3D uint32_t v18s_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SM3D uint32_t v18s_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SM3D void v18s_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SM3D void v18s_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SM3D void v18s_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SM3D void v18s_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SM3D void v18s_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SM3D void v18s_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SM3D bool v18s_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SM3D uint64_t v18s_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
SM3D void v18s_wait_bar(uint32_t bar) {
  if (v18s_mbar_try(bar)) return;
  const uint64_t t0 = v18s_gtime();
  while (!v18s_mbar_try(bar))
    if (v18s_gtime() - t0 > 20000000ull) __trap();
}

SM3D void v18s_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SM3D float4 v18s_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SM3D void v18s_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SM3D void v18s_store(float* p, float v) { *p = v; }

template <int FMT> struct Sm3Fmt;
template <> struct Sm3Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Sm3Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Sm3Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Sm3Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SM3D void v18s_decode(const int4 (&w)[Sm3Fmt<FMT>::NL], uint32_t (&A)[Sm3Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Sm3Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Sm3Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      v18s_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      v18s_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      v18s_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      v18s_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SM3D void v18s_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) v18s_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) v18s_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Sm3Fmt<FMT>::NL, NA = Sm3Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = v18s_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = v18s_ldx(xp);
          xv[slot][n][1] = v18s_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = v18s_ldx(xp), v = v18s_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) v18s_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  v18s_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) v18s_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          v18s_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) v18s_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                v18s_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) v18s_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            v18s_cl_wait();
            if (t < TV) {
              const uint32_t rbar = v18s_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            v18s_cl_wait();
            const uint32_t rbar = v18s_mapa(bar, 0);
            v18s_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int v18s_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
static int v18s_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = v18s_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = v18s_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<int M, int NT, int ST>
__global__ void v18s_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int v18s_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=v18s_router_scalar<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
'''

V18S_CPP = r"""
int v18s_router_launch(long,long,long,int,long,int);
#include <pybind11/pybind11.h>
#include <stdint.h>
int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

V18S_ESC_CAP = 32

def v18s_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst + (0,) * (15 - len(inst))))
        cases.append(f"    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return V18S_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))

def v18s_build(instances, name="v18s_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return _ldd19_inline(name, V18S_CPP + extra_cpp, cuda_sources=v18s_cuda_source(instances) + extra_cuda,
                       functions=["v18s_launch", "v18s_router_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def v18s_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def v18s_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def v18s_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def v18s_bf16_bytes(v):
    r, c = v.shape[:2]
    words = v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def v18s_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = v18s_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def v18s_c12_unpack(data, base, esc):
    """Inverse of v18s_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(V18S_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def v18s_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def v18s_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [v18s_units(hi), v18s_units(lo)]
    else:
        mats = [v18s_units(w)]
    if fmt in (0, 2):
        data = torch.cat([v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = v18s_c12_bytes(v)
        if not torch.equal(v18s_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(v18s_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def v18s_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def v18s_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = v18s_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, V18S_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > V18S_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = v18s_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp["fmt"] == 4:
        return ext.v18s_router_launch(sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                     torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18s_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

def v18s_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def v18s_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def v18s_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = v18s_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = v18s_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = v18s_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = v18s_verify_onehot(ext, inst_id[inst], sp, v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


V18_SMALL = {
    ('pooler',1):((1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,6),(2,16,8,1)),
    ('o',8):((1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6),(1,8,2,1)),
}
# LDD19 (integrator, iteration 5): o m=8 with m fixed at compile time (CM = 8; the launcher rejects any other m,
# which then falls back); work_integrator/probeO1.json. LDD19_O8CM = 0 restores CM = 0.
LDD19_O8CM = 1
if LDD19_O8CM:
    V18_SMALL[('o', 8)] = ((1, 1, 1, 2, 0, 0, 256, 8, 8192, 1024, 1, 8, 2, 1, 6), (1, 8, 2, 1))


def run_v18(x,layer,n,st):
    if x.ndim!=2 or x.dtype!=torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr()%16:
        return None
    m,k=x.shape
    plan=st['plans'].get((layer,m))
    if plan is None or n!=plan['rows'] or k!=(8192 if layer=='pooler' else 1024) or x.device!=plan['wp'].device:
        return None
    out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    rc=v18s_launch_raw(st['ext'],plan['inst_id'],plan,x,out,n,plan['cfg'])
    if rc:raise RuntimeError(f'v18 selected launch failed: {rc}')
    return out

_v18_base_prepare=prepare
_v18_base_kernel=kernel_fn

def prepare(weights: Weights) -> dict:
    state=_v18_base_prepare(weights)
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state

def kernel_fn(data: Inputs,state: dict) -> torch.Tensor:
    out=run_v18(data['x'],data['config']['layer'],data['config']['n'],state['v18'])
    return _v18_base_kernel(data,state) if out is None else out


# ======================================================================================================== v19
# v18 (above) plus: dead builds dropped from prepare() (V19_PRUNED), and new routes served ahead of v18's chain for
# exact (layer, m) only. Every new route is validated in prepare() (12-bit pack round trip, one-hot columns bit-exact,
# random rows with a spike row and outlier columns against an FP64 reference); a route that fails keeps v18's kernel.

# ---- chenwainuo/okc (Apache-2.0, see the notice at the top of the file): the rq kernel, binding changed for v19
# LDD19: switch the rr22 merge off before prepare(); the v8 router kernel stays its fallback.
LDD19_RR22 = True

V19_RQCU = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint4 rq_ldg(const void* a) {  // v21: volatile, so it stays at kernel entry
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@

template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int SMEM = 1024 + ST * (XB + WB) + 24 * ST + 16;
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
__global__ void __launch_bounds__(384, 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && ST == 5 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && (FL == 65538 || FL == 65539)), "EH backport approved M128/ST5/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  // v21 RG (FL bit 24): W stage 0 goes straight into each math warp's registers (ld.global.nc.v4 at kernel entry,
  // before any TMA is issued); the W lane / EARLY thread only arrives (no bytes) on stage 0's full barrier for W
  constexpr bool RG = (FL & 16777216) != 0;
  static_assert(!RG || (GS == 1 && WM == 0 && PRB == 0 && INS == 0), "RG: plain variants only");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + ST * WB;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  uint32_t ec = warp < NW ? __ldg(el + lane) : 0u;
  uint4 QR0 = make_uint4(0, 0, 0, 0), QR1 = QR0, QR2 = QR0;  // v21 RG: this warp's stage-0 W fragment
  if constexpr (RG) if (warp < NW && KCS > 0) {
    const uint8_t* wsrc = p.W + ((size_t)(p.walias ? nb % p.walias : nb) * p.KC + s0) * 12288 + hh * 6144 + warp * 1536 + lane * 16;
    QR0 = rq_ldg(wsrc); QR1 = rq_ldg(wsrc + 512); QR2 = rq_ldg(wsrc + 1024);
  }
  if (tid == (EARLY ? 256 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < ST && s < KCS; s++) {
        if (RG && s == 0) rq_arrive(full);  // v21 RG: W stage 0 comes through the math warps' registers
        else rq_expect(full + 8 * s, WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * s, XB);
        if (RG && s == 0) {
        } else if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  }
  if (tid == 320 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  if (warp >= 8) {  // ---------------- producer warpgroup: W lane (warp 8), x lane (warp 9)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == 8 && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;  // EARLY: stages 0 .. ST-1 already issued (slot 0 of round 1 next)
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s++) {
        if (s >= ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        if (RG && s == 0) {
          rq_arrive(full);  // v21 RG: no bytes for W stage 0
        } else {
          rq_expect(full + 8 * slot, WB);
          if (HINT) rq_bulkh(wring + slot * WB, src + (size_t)s * 12288, WB, full + 8 * slot, pol);
          else rq_bulk(wring + slot * WB, src + (size_t)s * 12288, WB, full + 8 * slot);
        }
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > ST ? KCS - ST : 0); s < KCS; s++) rq_wait(full + 8 * (s % ST), (s / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == 9 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;  // EARLY: stages 0 .. ST-1 already issued (slot 0 of round 1 next)
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s++) {
        if (s >= ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : xempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, XB);
        if (HINT) rq_tileh(xring + slot * XB, &tx, (s0 + s) * 64, 0, full + 8 * slot, pol);
        else rq_tile(xring + slot * XB, &tx, (s0 + s) * 64, 0, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > ST ? KCS - ST : 0); s < KCS; s++) rq_wait(full + 8 * (s % ST), (s / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= 10) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - 320, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : s % ST;
        if (!RES) rq_wait(full + 8 * slot, (s / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j++) if (GS * G + j < KCS) rq_arrive(wempty + 8 * ((GS * G + j) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j++) if (GS * G + j < KCS) rq_arrive(xempty + 8 * ((GS * G + j) % ST));
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : s % ST) * XB) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    if constexpr (RG) {  // v21 RG: stage 0 (GS == 1): wait for its x box, W from the registers loaded at entry
      rq_wait(full, 0);
      Q[0][0] = QR0; Q[0][1] = QR1; Q[0][2] = QR2;
      __syncwarp();
      if (lane == 0) rq_arrive(wempty);
    } else ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
static int rq_go(RqP& p, const void* x, void* out, int KS, int info, cudaStream_t st) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x, p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out, p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3(384), C::SMEM, st>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
// v19 binding: raw pointers + stream (the caller checks shapes, dtypes and contiguity)
int64_t v19_rqrun(int64_t v, int64_t W, int64_t NB, int64_t KC, int64_t x, int64_t m, int64_t K, int64_t out, int64_t N,
                  int64_t esc, int64_t ecap, int64_t cnt, int64_t ncnt, int64_t KS, int64_t Tb, int64_t sink, int64_t stream,
                  int64_t info) {
  RqP p = {};
  if (!info) {
    if (K != KC * 64 || N != NB * 128 || m < 1 || ecap < 32 || ecap % 32) return -11;
    p.W = (const uint8_t*)W; p.esc = (const uint32_t*)esc; p.ecap = (int)ecap;
    p.out = (__nv_bfloat16*)out; p.cnt = (int*)cnt; p.ncnt = (int)ncnt; p.prof = nullptr; p.sink = (uint32_t*)sink;
    p.m = (int)m; p.K = (int)K; p.N = (int)N; p.KC = (int)KC; p.NB = (int)NB; p.walias = 0; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
// chenwainuo/okc pw45 rr<M, R, NT>: FP32 router GEMV, small m; CTA = R full rows, NT threads split K, fp32 weights read once
__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int M, int R, int NT = 512>
__global__ void __launch_bounds__(NT) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[NT / 32][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  constexpr int NI = 2048 / NT;  // float4 per thread per row (K = 8192)
  float4 wv[R][NI];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < NI; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * NT); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < NI; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * NT);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162*)&xv.x), b = __bfloat1622float2(*(const __nv_bfloat162*)&xv.y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r][q] = fmaf(wv[r][i].w, b.y, fmaf(wv[r][i].z, b.x, fmaf(wv[r][i].y, a.y, fmaf(wv[r][i].x, a.x, acc[r][q]))));
      }
    }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      float v = acc[r][q];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) red[w][r * M + q] = v;
    }
  __syncthreads();
  if (t < R * M && t % M < m) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
}

int64_t v19_rr(int64_t W, int64_t x, int64_t out, int64_t m, int64_t N, int64_t K, int64_t stream) {
  if (K != 8192 || m != 1) return -1;
  rr<1, 1, 1024><<<(int)N, 1024, 0, (cudaStream_t)stream>>>((const float4*)W, (const __nv_bfloat16*)x, (float*)out, (int)m, (int)N, (int)K);
  return (int64_t)cudaGetLastError();
}
"""


# ---- chenwainuo/okc (Apache-2.0): wgmma helper and the 12-bit pack / escape-list helpers, unchanged
def _rqwg(n):
    """rq_wgmma<n>: wgmma.mma_async m64n<n>k16 f32 += bf16 (A: 4 registers) x bf16 (B: smem descriptor), scale-d from a
    register (0 on the CTA's first k16: no accumulator zero-fill), scale-a = scale-b = 1, trans-b = 0"""
    nd = n // 2
    d = ", ".join("%%%d" % i for i in range(nd))
    a = ", ".join("%%%d" % (nd + i) for i in range(4))
    outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
    return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
             "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
             "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
             "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
            % (n, nd, nd + 5, n, d, a, nd + 4, outs))


# code-word bit of each value's 3-bit offset and of its sign, per decode variant (see dec0 / dec1)
_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


def _c12(fr, dv=1):
    """bf16 fragments [T, KC, 4, 32, 8] -> (uint8 [T, KC, 3, 32, 16], table, escapes [E, 6] (tile, ch, j, lane, v, xor)) or None"""
    T, KC = fr.shape[:2]
    dev = fr.device
    b = fr.reshape(T, KC, 4, 32, 8).view(torch.int16).int() & 0xffff
    h7, sg = (b >> 8) & 127, b >> 15
    win = torch.bincount(h7.flatten(), minlength=128).unfold(0, 8, 1).sum(1)
    B = int(win.argmax())
    d = h7 - B
    inw = (d >= 0) & (d < 8)
    d = torch.where(inw, d, 0).long()
    C = ((d << torch.tensor(_OS[dv], device=dev)) | (sg.long() << torch.tensor(_SS[dv], device=dev))).sum(-1)
    C = torch.where(C >= 2 ** 31, C - 2 ** 32, C).int()
    lo = (b & 255).to(torch.uint8)
    p0 = lo[:, :, 0:2].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p1 = lo[:, :, 2:4].permute(0, 1, 3, 2, 4).reshape(T, KC, 32, 16)
    p2 = C.permute(0, 1, 3, 2).contiguous().view(torch.uint8).view(T, KC, 32, 16)
    data = torch.stack((p0, p1, p2), 2).contiguous()
    ei = (~inw).nonzero()
    esc = torch.cat((ei, (h7[tuple(ei.t())] ^ B)[:, None].long()), 1)
    tab = B * 0x01010101 if dv else sum((B + i) << (8 * i) for i in range(8))
    if not torch.equal(_unc12(data, tab, esc, dv), b):
        return None
    return data, tab - 2 ** 64 if tab >= 2 ** 63 else tab, esc


def _unc12(data, tab, esc, dv):
    """independent decoder (mirrors the kernel's byte permutes) for the exactness proof"""
    T, KC = data.shape[:2]
    dev = data.device
    w = data.view(torch.int32).view(T, KC, 3, 32, 4).long() & 0xffffffff
    L = torch.stack((w[:, :, 0, :, 0], w[:, :, 0, :, 1], w[:, :, 0, :, 2], w[:, :, 0, :, 3], w[:, :, 1, :, 0], w[:, :, 1, :, 1], w[:, :, 1, :, 2], w[:, :, 1, :, 3]), 2).view(T, KC, 4, 2, 32)
    C = w[:, :, 2].permute(0, 1, 3, 2)
    off = (C[..., None] >> torch.tensor(_OS[dv], device=dev)) & 7
    sgn = (C[..., None] >> torch.tensor(_SS[dv], device=dev)) & 1
    tb = torch.tensor([(tab >> (8 * i)) & 255 for i in range(8)], device=dev)
    hi = ((tab & 255) + off if dv else tb[off]) | (sgn << 7)
    lob = (L.permute(0, 1, 2, 4, 3)[..., None] >> (8 * torch.arange(4, device=dev))) & 255
    val = (hi << 8) | lob.reshape(T, KC, 4, 32, 8)
    if len(esc):
        t, c, j, l, vv, x = esc.t()
        val[t, c, j, l, vv] ^= x << 8
    return val.int()


def _pack_nat(w):
    """pw10 _pack_nat: bf16 [N, K] -> A fragments [N/16, K/64, 4 steps, 32 lanes, 8] in natural k order (wgmma B = x
    in natural k): step q, lane g*4+t, value v = (kh, half, e) holds W[16T + 8*half + g][64c + 16q + 8*kh + 2t + e]"""
    N, K = w.shape
    W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)  # T, half, g, c, q, kh, t, e
    return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


def _rqpack(w):
    """12-bit pack re-blocked to 128-row blocks: [N/128][K/64][8 tiles][1536 B] (one 12,288 B bulk copy per stage),
    the decode table word, and the escape rows (tile, chunk, step, lane, value, xor); None if the pack is not exact"""
    N, K = w.shape
    c = _c12(_pack_nat(w), 1)
    if c is None or N % 128:
        return None
    data, tab, esc = c
    T, KC = data.shape[:2]
    lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
    return lay, tab, esc


def _rqunpack(lay, tab, esc, N, K):
    """independent check of the re-blocking: unpack [NB][KC][8][1536] with _unc12 and undo _pack_nat -> bf16 [N, K]"""
    NB, KC = lay.shape[:2]
    data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
    v = _unc12(data, tab, esc, 1)  # [T, KC, 4, 32, 8] int32 bf16 bits
    W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)  # T, c, q, g, t, kh, half, e
    return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)  # int32 bf16 bits (compare with w.view(int16) & 0xffff)


def _rqesc(esc6, NB, KC, KS, dev):
    """escape lists for grid (NB, KS): one fixed-stride slot of CAP entries per (ks, nb, warp) (CAP a multiple of 32,
    > the longest list, so every list ends in a sentinel), + 32 trailing sentinels; int32 [KS*NB*8*CAP + 32]"""
    G = KS * NB * 8
    if len(esc6):
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = tile // 8, tile % 8
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = (u << 17) | (ln << 12) | ((q * 4 + v // 2) << 8) | ((v % 2) << 7) | xr
        gw = (ks * NB + nb) * 8 + wt
        o = torch.argsort(gw * 65536 + u)
        gw, ent = gw[o], ent[o]
        rank = torch.arange(len(gw), device=dev) - torch.searchsorted(gw, gw)
        cap = (int(torch.bincount(gw, minlength=G).max()) + 1 + 31) // 32 * 32
    else:
        cap = 32
    arr = torch.full((G * cap + 32,), 0xffffffff, dtype=torch.long, device=dev)
    if len(esc6):
        arr[gw * cap + rank] = ent
    arr = torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int()
    return arr.contiguous(), torch.zeros(1, dtype=torch.int32, device=dev)


RQTOL = {1: 4e-3, 2: 5e-3, 3: 6e-3, 4: 6e-3, 5: 7e-3, 6: 7e-3, 7: 8e-3, 8: 8e-3, 9: 8e-3}  # chenwainuo/okc: overall rel L2 bound per KS


# ---- v19 glue
V19_STEPS = ['lt', 'prune', 'rq', 'rr']
# rq instance: t32s5hze = rq<32,5,1,0,0,0,35> (NT 32, 5 slots, L2 hints, in-kernel zeroing, early first loads), the
# pick of chenwainuo's judge run d5a125fc
V19_RQ_VARIANTS = [(32, 5, 1, 0, 0, 0, 35)]
# LDD19: routes (layer, m) -> [(instance, K split)]. Only qkv_gate m=32 is served by rq here: pooler / unpooler m=32 are
# ldd19r's sk2 routes and qkv_gate / pooler m=128 are the chenwainuo route above v19 (v25 built and validated them unused).
V19_RQ_ROUTES = {('qkv_gate', 32): [(0, 7)]}
V19_RR = 'rr' in V19_STEPS  # router m=1: rr<1,1,1024> FP32 GEMV


def v19_jobs(weights):
    """prepare's jobs (a pool of 4 threads): the LDD19 / v19 / v22 builds and the families that serve a route."""
    return [
        ('ldd19r', prepare_ldd19r, weights, {}),
        ('v22', prepare_v22, weights, {}),
        ('v19', prepare_v19, weights, {}),
        ('bundle', prepare_bundle, weights, {}),
        ('new_lt', tt_lt_prepare_cublaslt, weights, {}),
        ('bundle2', ldd19r_prepare_bundle2, weights, {}),
        ('v18', ldd19r_prepare_v18_extra, weights, {}),
        ('sm', prepare_small_m, weights, {}),
        ('router', prepare_router, weights, {}),
        ('olayer', prepare_olayer, weights, {}),
        ('v10_sm2', tt_v10_sm2_prepare_smallm2, weights, {}),
        ('v12_o1', prepare_v12_o1, weights, {}),
        ('v13_smtrace8', tt_v13_smtrace8_prepare_smtrace, weights, {}),
    ]


def v19_build():
    cases = "\n".join("    case %d: return rq_go<%s>(p, (const void*)x, (void*)out, (int)KS, (int)info, (cudaStream_t)stream);"
                      % (i, ", ".join(map(str, v))) for i, v in enumerate(V19_RQ_VARIANTS))
    src = V19_RQCU.replace("@@WGMMA@@", "".join(_rqwg(n) for n in sorted({v[0] for v in V19_RQ_VARIANTS}))).replace("@@CASES@@", cases)
    cpp = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t v19_rqrun(" + ", ".join(["int64_t"] * 18) + ");\n"
           "int64_t v19_rr(" + ", ".join(["int64_t"] * 7) + ");\n")
    return _ldd19_inline('v19_rq_' + hashlib.sha256(src.encode()).hexdigest()[:12], cpp, cuda_sources=src,
                       functions=['v19_rqrun', 'v19_rr'],
                       extra_cuda_cflags=['-O3', '-DNDEBUG', '--expt-relaxed-constexpr', '-gencode=arch=compute_90a,code=sm_90a'],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def v19_rqcall(st, p, x, out):
    return st['fn'](p['vi'], p['lay'].data_ptr(), p['NB'], p['KC'], x.data_ptr(), x.shape[0], p['K'], out.data_ptr(), p['n'],
                    p['esc'].data_ptr(), p['ecap'], p['cnt'].data_ptr(), p['cnt'].numel(), p['KS'], p['tab'],
                    st['sink'].data_ptr(), torch.cuda.current_stream(x.device).cuda_stream, 0)


def v19_rq_verify(st, p, w, m):
    """v18-style checks at m exactly: a zero input over a NaN-filled output (the in-kernel zeroing), every K column as a
    one-hot row (bit-exact against W, counters back to 0), random rows (normal, then 1% outlier columns x30) with a
    16.0 spike row: overall rel L2 <= chenwainuo's per-KS bound (<= 8e-3, judge 1e-2), every row <= 1e-2, spike <= 2.5e-4."""
    n, K = w.shape
    dev = w.device
    out = torch.full((m, n), float('nan'), dtype=torch.bfloat16, device=dev)
    rc = v19_rqcall(st, p, torch.zeros((m, K), dtype=torch.bfloat16, device=dev), out)
    if rc:
        return False, f'launch rc {rc}'
    if not bool((out == 0).all()):
        return False, 'zero input: output not zeroed'
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev)
    got = torch.empty((K // m, m, n), dtype=torch.bfloat16, device=dev)
    for i in range(K // m):
        if v19_rqcall(st, p, eye[i * m:(i + 1) * m], got[i]) != 0:
            return False, 'one-hot launch failed'
    if not torch.equal(got.view(K, n), w.t()):
        return False, 'one-hot not exact'
    del eye, got
    if int(p['cnt'].abs().sum()):
        return False, 'counters not reset'
    tol = RQTOL[p['KS']]
    gen = torch.Generator(device=dev).manual_seed(11)
    worst = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[:K // 100]] *= dist
        x[m - 1].zero_()
        x[m - 1, (11 * 131 + m) % K] = 16.0
        x = x.to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
        if v19_rqcall(st, p, x, o) != 0:
            return False, 'random launch failed'
        diff = o.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        err = float(diff.norm() / ref.norm())
        worst = max(worst, err)
        if not bool(torch.isfinite(o).all()) or err > tol or float(rows.max()) > 0.01 or float(rows[m - 1]) > 0.00025:
            return False, f'random dist {dist}: rel L2 {err:.2e} (bound {tol:.0e}) row max {float(rows.max()):.2e} spike {float(rows[m - 1]):.2e}'
    if int(p['cnt'].abs().sum()):
        return False, 'counters not reset'
    return True, f'ok (rel L2 {worst:.1e}, bound {tol:.0e})'


def v19_prepare_rq_layer(st, w, layer):
    w = w.contiguous()
    n, K = w.shape
    dev = w.device
    pk = _rqpack(w)
    if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
        st['log'].append(f'rq {layer}: 12-bit pack not exact, v18 routes kept')
        return
    lay, tab, esc6 = pk
    NB, KC = lay.shape[:2]
    for (ly, m), cands in sorted(V19_RQ_ROUTES.items()):
        if ly != layer:
            continue
        for vi, KS in cands:
            gx = NB * (2 if V19_RQ_VARIANTS[vi][6] & 16 else 1)  # HALF instances: 64-row CTAs
            tag = f'rq {layer} m={m} rq{V19_RQ_VARIANTS[vi]} KS={KS} escapes {len(esc6)}'
            if gx * KS > 132:
                st['log'].append(f'{tag}: grid not one wave')
                continue
            esc, _ = _rqesc(esc6, NB, KC, KS, dev)
            nl = KS * NB * 8
            if (esc.numel() - 32) % nl:
                st['log'].append(f'{tag}: bad escape list')
                continue
            p = {'vi': vi, 'lay': lay, 'NB': NB, 'KC': KC, 'K': K, 'n': n, 'esc': esc, 'ecap': (esc.numel() - 32) // nl,
                 'cnt': torch.zeros(2 * gx, dtype=torch.int32, device=dev), 'KS': KS, 'tab': int(tab), 'm': m}
            ok, msg = v19_rq_verify(st, p, w, m)
            st['log'].append(f'{tag}: {msg}' + ('' if ok else ', next candidate'))
            if ok:
                st['plans'][(layer, m)] = p
                break


def v19_prepare_rr(st, w):
    """router m=1 on rr<1,1,1024> (FP32 weights as they are): every column one-hot bit-exact, then v18's random check
    (router tolerance 1e-4: overall <= 2.5e-5, the spike row <= 2.5e-4) on a plain row and on a spike row."""
    w = w.contiguous()
    n, K = w.shape
    dev = w.device
    if w.dtype != torch.float32 or K != 8192:
        st['log'].append('rr: unexpected router weights')
        return
    stream = torch.cuda.current_stream(dev).cuda_stream
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev)
    got = torch.empty((K, n), dtype=torch.float32, device=dev)
    for k in range(K):
        if st['rr'](w.data_ptr(), eye[k].data_ptr(), got[k].data_ptr(), 1, n, K, stream) != 0:
            st['log'].append('rr: launch failed')
            return
    if not torch.equal(got, w.t()):
        st['log'].append('rr: one-hot not exact, v18 route kept')
        return
    del eye, got
    gen = torch.Generator(device=dev).manual_seed(11)
    worst = 0.0
    for spike in (False, True):
        x = torch.randn((1, K), generator=gen, device=dev)
        if spike:
            x.zero_()
            x[0, (11 * 131 + 1) % K] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w.double().t()).float().double()
        o = torch.empty((1, n), dtype=torch.float32, device=dev)
        if st['rr'](w.data_ptr(), x.data_ptr(), o.data_ptr(), 1, n, K, stream) != 0:
            st['log'].append('rr: launch failed')
            return
        err = float((o.double() - ref).norm() / ref.norm())
        worst = max(worst, err)
        if not bool(torch.isfinite(o).all()) or err > (0.00025 if spike else 0.0001 / 4):
            st['log'].append(f'rr: random rel L2 {err:.2e}, v18 route kept')
            return
    st['rrw'] = w
    st['rr_ok'] = True
    st['log'].append(f'rr router m=1: ok (rel L2 {worst:.1e})')


def prepare_v19(weights):
    st = {'plans': {}, 'log': []}
    dev = weights['qkv_gate'].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        return st
    try:
        ext = v19_build()
    except Exception as error:  # noqa: BLE001 - a failed build keeps every v18 route
        st['log'].append(f'build failed: {type(error).__name__}: {str(error)[-300:]}')
        return st
    st['ext'], st['fn'], st['rr'] = ext, ext.v19_rqrun, ext.v19_rr
    st['sink'] = torch.zeros(4, dtype=torch.int32, device=dev)
    for layer in sorted({layer for layer, _ in V19_RQ_ROUTES}):
        v19_prepare_rq_layer(st, weights[layer], layer)
    if V19_RR:
        v19_prepare_rr(st, weights['router'])
    torch.cuda.synchronize(dev)
    return st


def run_v19(x, layer, n, st):
    if not st or x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if layer == 'router':
        if m != 1 or not st.get('rr_ok') or n != 256 or x.shape[1] != 8192 or x.device != st['rrw'].device:
            return None
        out = torch.empty((1, 256), dtype=torch.float32, device=x.device)
        rc = st['rr'](st['rrw'].data_ptr(), x.data_ptr(), out.data_ptr(), 1, 256, 8192, torch.cuda.current_stream(x.device).cuda_stream)
        return out if rc == 0 else None
    p = st['plans'].get((layer, m))
    if p is None or n != p['n'] or x.shape[1] != p['K'] or x.device != p['lay'].device:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    return out if v19_rqcall(st, p, x, out) == 0 else None


_v19_base_prepare = prepare
_v19_base_kernel = kernel_fn


def prepare(weights: Weights) -> dict:
    state = _v19_base_prepare(weights)
    st = state.get('v19')
    if st:
        for line in st['log']:
            print('v19 ' + line, flush=True)
        routes = list(st['plans']) + ([('router', 1)] if st.get('rr_ok') else [])
        for layer, m in routes:  # one warm-up call per new route, as v18's layers do
            w = weights[layer]
            data = {'x': torch.zeros((m, w.shape[1]), dtype=torch.bfloat16, device=w.device),
                    'config': {'m': m, 'n': w.shape[0], 'k': w.shape[1], 'layer': layer}}
            kernel_fn(data, state)
        torch.cuda.synchronize(weights['qkv_gate'].device)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    out = run_v19(data['x'], data['config']['layer'], data['config']['n'], state.get('v19'))
    return _v19_base_kernel(data, state) if out is None else out


# ======================================================================================================== LDD19 ldd19r
# LDD19: five routes from team vjeux's muse_bf16_proj.py (Apache-2.0, see the notice at the top of the file), served ahead
# of the v19 chain for exact (layer, m) only:
#   qkv_gate m=1 and the unpooler slice m=1 on cgemv; pooler m=32 (pooler32_r144) and unpooler m=32 (alygg77's u32_r96)
#   on sk2; router m=256 on rk; optional: qkv_gate m=16 on sk2 (qkv16t_r200, escape-balanced rows).
# Each route is validated in prepare() (zero input over a NaN output, every K column one-hot bit-exact, random rows with
# outlier columns and a spike row against FP64); a route that fails keeps v19's kernel. When a route is on, the layout it
# replaces is not built (prepare() memory stays flat): qkv m=1 drops v18's decodediet qkv1 layout, unpooler m=1 drops the
# v12 smtrace1 layouts (unpooler m=8 runs on v13's own), unpooler m=32 drops the rq unpooler pack, pooler m=32 drops its
# rq route (the pooler pack stays for m=128) and the dead v17 bundle2 pooler / unpooler m=32 layouts, qkv m=16 drops
# bundle2's qkv16 layout. rk reuses v18's packed router layout.
LDD19R = {'qkv1': 1, 'un1': 1, 'pool32': 1, 'un32': 1, 'rt256': 1, 'qkv16': 1, 'dead32': 1}   # LDD19: route switches
# LDD19 (optimizer, 2026-11-03): vjeux's m = 1 specialists: pooler m=1 and o m=1 on cgemv ('pool1', 'o1'), router m=1 on
# rgemv (plain FP32 weights, one row per CTA, 'rg1'); 'rg8': the same rgemv with 8 token accumulators for router m=8.
LDD19R.update({'pool1': 1, 'o1': 1, 'rg1': 0, 'rg8': 0})
LDD19R.update({'cgh1': 1})   # LDD19 (optimizer): router m=1 on cgemv_hl (12-bit hi + lo)
LDD19R_CGH_CFG = 2   # LDD19 (optimizer): cgemv_hl launch config (0: 2 x 512, 1: 2 x 256, 2: 1 x 512)
LDD19R.update({'sk_o32': 1})   # LDD19 (optimizer): sk2 routes   # LDD19 (optimizer): rg1 tied (1.000), rg8 lost (0.660), ab_1_h1.json
# LDD19: cgemv configs (K, rows per CTA, threads, independent accumulators); per layer the pinned pick (index)
LDD19R_CG_CONFIGS = [(8192, 2, 256, 1), (8192, 2, 512, 1), (8192, 2, 512, 2),
                     (4096, 2, 128, 1), (4096, 2, 256, 1), (4096, 2, 128, 4),
                     (1024, 2, 64, 1), (4096, 1, 128, 4)]   # LDD19 (optimizer): index 6 = vjeux's o m=1 config (2 rows x 64 threads)
LDD19R_CG_PICK = {'qkv_gate': 1, 'unpooler': 7, 'pooler': 1, 'o': 6}  # LDD19 (optimizer): pooler 2x512 = vjeux's pick (NACC 2 and o 4x64 lost, ab_2_h2b.json), o 2x64   # LDD19: picked by graph timing on an H200 (qkv 2x512, slice 2x128 NACC 4)
# LDD19: sk2 configs (vjeux's tuple: name, layer, n_rows, R rows/CTA, NT, NWR, WSK, UPW, NACC, CL), tokr, switch
LDD19R_SK2 = [(('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1), 32, 'pool32'),
              (('u32_r96', 'unpooler', 8192, 32, 2, 2, 2, 2, 1, 1), 32, 'un32'),
              (('qkv16t_r200', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 2, 1), 16, 'qkv16'),
              # LDD19 (optimizer, 2026-11-03): more of vjeux's sk2 configs (vjeux/okc-2026 via keell0renz 8e30e1b,
              # Apache-2.0), each behind its own switch in LDD19R
              (('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1), 32, 'sk_o32'),
              (('o16t_r68', 'o', 8192, 32, 2, 2, 1, 4, 2, 1), 16, 'sk_o16'),   # LDD19 (smallm_owner, iteration 3; harvester): vjeux's o m=16 config
              ]
LDD19R.update({'sk_o16': 1})   # LDD19 (harvester, from smallm_owner): o m=16 on sk2 o16t_r68
# LDD19 (sweeper, iteration 6): router m=128 on rk with S = 16 (ring depth LDD19_RK_QS128) instead of rr22; 0 = rr22
LDD19_RK128 = 1
LDD19R_RK128_S = 16
LDD19R_RK_S = 8   # rk K splits at m = 256 (vjeux's RK_SPLITS[4]): grid 4 x 4 x 8 = 128 CTAs, one per SM, all resident

LDD19R_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

namespace ldd19r {
// ---- vjeux/okc-2026 f35a41e (Apache-2.0): device helpers, verbatim
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }



// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n selp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(ok) : "r"(bar), "r"(parity), "r"(0x989680u) : "memory");
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}\n" :: "r"(bar), "r"(bytes) : "memory");
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}


__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2) : "memory");
}

// ---- vjeux/okc-2026 f35a41e (Apache-2.0): sk2 kernel, verbatim


// LDD19 (timeline_lab): LDD19_SK2_SPIN 1 = the producer / escape-warp ring waits spin instead of sleeping (in-kernel
// timeline: the escape warps' per-stage slot release is on the ring's critical path); 0 = vjeux's sleeping wait
#ifndef LDD19_SK2_SPIN
#define LDD19_SK2_SPIN 0
#endif
#ifndef LDD19_SK2_EARLY
#define LDD19_SK2_EARLY 0
#endif
#ifndef LDD19_SK2_INITFIRST
#define LDD19_SK2_INITFIRST 0
#endif
#ifndef LDD19_SK2_EARLYREL
#define LDD19_SK2_EARLYREL 1
#endif
#if LDD19_SK2_SPIN
#define SK2_WAIT(bar, par) mbar_wait((bar), (par))
#else
#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#endif
#define SK2_DIAG 0
#ifndef LDD19_SK2_CM
#define LDD19_SK2_CM 0
#endif
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
#if LDD19_SK2_CM   // LDD19 (escape_lab): compile-time token count on the non-packed instances (host asserts M == TOKR)
  if constexpr (!PK) M = TOKR;
#endif
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
#if LDD19_SK2_INITFIRST   // LDD19 (timeline_lab): all CL == 1 barriers before the single pre-copy fence
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
    }
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
#if LDD19_SK2_EARLY
    if constexpr (CL == 1) {   // LDD19 (timeline_lab): the rest of the ring's first fill goes out before the CTA barrier
      if (!LDD19_SK2_INITFIRST) {
        for (int i = 1; i < NSTAGE; ++i) mbar_init(full0 + 8 * i, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      for (int st = 1; st < NSTAGE && st < nst; ++st) {
        const uint32_t s0 = sbase + st * SST;
        mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      }
    }
#endif
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1 && !LDD19_SK2_INITFIRST) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1 && !LDD19_SK2_INITFIRST) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
#if LDD19_SK2_EARLY   // LDD19 (timeline_lab): the full barriers were initialised before the early issue
      for (int i = 1; i < NSTAGE; ++i) mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
#else
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
#endif
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = (LDD19_SK2_EARLY && CL == 1) ? NSTAGE : 1; st < nst; ++st) {   // LDD19 (timeline_lab)
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
#if LDD19_SK2_EARLYREL   // LDD19 (timeline_lab): release first, the skip walk reads only ptab_s
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
#else
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#endif
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const uint32_t s0 = sbase + slot * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(lw[nt].x), "=r"(lw[nt].y), "=r"(lw[nt].z), "=r"(lw[nt].w) : "r"(wb + nt * TU + lane * 16));
          asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(cw[nt].x), "=r"(cw[nt].y) : "r"(wb + nt * TU + 512 + lane * 8));
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][0]), "=r"(b[nt][1]), "=r"(b[nt][2]), "=r"(b[nt][3]) : "r"(a0));
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(b[nt][4]), "=r"(b[nt][5]), "=r"(b[nt][6]), "=r"(b[nt][7]) : "r"(a0 + 16));
          if constexpr (HL) {
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][0]), "=r"(bl[nt][1]), "=r"(bl[nt][2]), "=r"(bl[nt][3]) : "r"(a0 + 1024));
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bl[nt][4]), "=r"(bl[nt][5]), "=r"(bl[nt][6]), "=r"(bl[nt][7]) : "r"(a0 + 1040));
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0), "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])), "r"(qbar_r) : "memory");
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                     :: "r"(a0 + 8 * RP * 4), "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])), "r"(qbar_r) : "memory");
      }
    return;
  }
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
}


// ---- vjeux/okc-2026 f35a41e (Apache-2.0): cgemv; LDD19: NACC independent accumulator chains (alygg77's idea;
// NACC = 1 is vjeux's kernel exactly)
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K, int NACC>
__global__ void __launch_bounds__(NTH + 32) cgemv_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS][NACC];   // LDD19: NACC chains per row (group 2j + g goes to chain (2j + g) % NACC)
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int c = 0; c < NACC; ++c) acc[r][c] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        float v = acc[r][(2 * j + g) % NACC];   // LDD19
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
        acc[r][(2 * j + g) % NACC] = v;   // LDD19
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r][0];
#pragma unroll
    for (int c = 1; c < NACC; ++c) v += acc[r][c];   // LDD19: fixed order
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}
// ---- vjeux/okc-2026 f35a41e (Apache-2.0): rk kernel; LDD19: producer reads rwdl's packed hi | lo layout (one box)

// ============================================================================ rk: router GEMM, m 33..256 (sm_90)
// out[M, 256] (FP32) = x[M, 8192] (BF16) @ (hi + lo)[256, 8192]^T on router_cat = [hi | lo] [256, 16384] (BF16).
// CTA (192 threads) = 1 consumer warpgroup (SS-wgmma m64n64k16 on the hi tile, then the lo tile, of each 64-k block)
// + 1 producer warp (TMA, RK_QS-stage ring of 24 KB: hi 8 KB | lo 8 KB | x 8 KB, 128B swizzle) + 1 helper warp.
// Tile = 64 W rows x 64 tokens; split-K over S CTAs per tile (grid 4 x ntt x S <= 128: one CTA per SM, all resident).
// Each CTA ADDS its partial tile into the FP32 output (smem-transposed, 16 B red.add.v4.f32): the reduction is
// ORDER-DEPENDENT (run-to-run differences ~1e-7 relative, inside the router tolerance; POD item 6). The output is
// cleared inside the kernel: every CTA's helper warp reads the grid counter of this grid size (generation =
// value / grid: earlier launches' counts are complete and this CTA has not counted yet), clears its slice of out,
// counts it (release), waits (acquire, 2 s limit -> trap) until every CTA has counted, and releases the consumers'
// adds through a CTA mbarrier. Monotonic counters (one per token-tile count = per grid size): no host reset.
#ifndef LDD19_RK_QS   // LDD19 (sweeper, iteration 6): rk ring depth re-swept (work_sweeper/probe1.json); 6 = vjeux's
#define LDD19_RK_QS 8
#endif
#define RK_QS LDD19_RK_QS
#ifndef LDD19_RK_QS128   // LDD19 (sweeper, iteration 6): ring depth of the m <= 128 instance (router m=128, S = 16)
#define LDD19_RK_QS128 6
#endif
__device__ __forceinline__ uint64_t rk_desc(uint32_t addr) {   // K-major tile, 128 B rows, 128B swizzle, SBO 1024
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)64 << 32) | (1ull << 62);
}
__device__ __forceinline__ void rk_mbar_init(uint32_t bar, uint32_t cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(cnt)); }
__device__ __forceinline__ void rk_mbar_expect(uint32_t bar, uint32_t bytes) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory"); }
__device__ __forceinline__ void rk_mbar_arrive(uint32_t bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory"); }
__device__ __forceinline__ void rk_mbar_wait(uint32_t bar, uint32_t ph) {
  asm volatile("{\n.reg .pred P1;\nRKW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@!P1 bra RKW_%=;\n}\n" :: "r"(bar), "r"(ph) : "memory");
}
__device__ __forceinline__ void rk_tma2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void rk_wgmma(float d[4][8], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;\n"
               : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[0][4]), "+f"(d[0][5]), "+f"(d[0][6]), "+f"(d[0][7]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[1][4]), "+f"(d[1][5]), "+f"(d[1][6]), "+f"(d[1][7]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[2][4]), "+f"(d[2][5]), "+f"(d[2][6]), "+f"(d[2][7]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[3][4]), "+f"(d[3][5]), "+f"(d[3][6]), "+f"(d[3][7])
               : "l"(da), "l"(db));
}

template <int QS_T = RK_QS>   // LDD19 (sweeper): ring depth as a template parameter (was RK_QS)
__global__ void __launch_bounds__(192, 1) rk_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
                                                     float* __restrict__ out, int M, int S, int ntt, int* __restrict__ zcnt) {
  constexpr int BM = 64, BN = 64, NJ = 4, QS = QS_T;
  constexpr int W_BYTES = 2 * BM * 128, ST = W_BYTES + BN * 128;
  extern __shared__ __align__(1024) unsigned char rk_smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(rk_smem);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t fullb = sbase + QS * ST, emptyb = fullb + 8 * QS, zbar = emptyb + 8 * QS;
  const int rt = blockIdx.x & 3, tt = (blockIdx.x >> 2) % ntt, sp = (blockIdx.x >> 2) / ntt;
  const int kb0 = (sp * 128) / S, nk = ((sp + 1) * 128) / S - kb0;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < QS; ++i) { rk_mbar_init(fullb + 8 * i, 1); rk_mbar_init(emptyb + 8 * i, 1); }
    rk_mbar_init(zbar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 5) {   // helper: clear this CTA's slice of out, count it, wait for every slice, release the adds
    int target = 0;
    if (lane == 0) {
      int v;
      asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
      target = (v / (int)gridDim.x + 1) * (int)gridDim.x;
    }
    const int n4 = M * 64;   // float4s in out
    const int per = (n4 + (int)gridDim.x - 1) / (int)gridDim.x;
    const int z0 = blockIdx.x * per, z1 = min(n4, z0 + per);
    for (int i = z0 + lane; i < z1; i += 32) reinterpret_cast<float4*>(out)[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(zcnt) : "memory");
      unsigned long long t0, t1;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(zcnt) : "memory");
        if (v - target >= 0) break;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        if (t1 - t0 > 2000000000ull) __trap();   // 2 s: a clear that never lands is a bug, never a slow node
        __nanosleep(128);
      }
      rk_mbar_arrive(zbar);
    }
    return;
  }
  if (warp == 4) {   // producer
    if (lane == 0) {
      for (int i = 0; i < nk; ++i) {
        const int s = i % QS;
        if (i >= QS) rk_mbar_wait(emptyb + 8 * s, ((i / QS) - 1) & 1);
        const uint32_t fb = fullb + 8 * s, st = sbase + s * ST;
        rk_mbar_expect(fb, ST);
        const int kb = kb0 + i;
        rk_tma2d(st, &mapW, fb, 0, (rt * 128 + kb) * 128);   // LDD19: hi | lo boxes of rwdl's packed layout
        rk_tma2d(st + W_BYTES, &mapX, fb, kb * 64, tt * BN);
      }
    }
    return;
  }
  // consumer warpgroup
  float d[NJ][8];
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) d[j][e] = 0.f;
  for (int i = 0; i < nk; ++i) {
    const int s = i % QS;
    rk_mbar_wait(fullb + 8 * s, (i / QS) & 1);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    const uint32_t a = sbase + s * ST, b = a + W_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      rk_wgmma(d, rk_desc(a + 32 * k16), rk_desc(b + 32 * k16));
      rk_wgmma(d, rk_desc(a + BM * 128 + 32 * k16), rk_desc(b + 32 * k16));
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
    if (i > 0 && tid == 0) rk_mbar_arrive(emptyb + 8 * ((i - 1) % QS));   // stage i-1's wgmmas are done
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+f"(d[j][e]) :: "memory");
  // partial tile -> shared memory [token][row] (the ring is idle: every stage was waited for), then 16 B adds
  constexpr int TP = 68;
  float* T = reinterpret_cast<float*>(rk_smem + (sbase - sraw));
  const int rl = (tid >> 5) * 16 + ((tid & 31) >> 2);
#pragma unroll
  for (int j = 0; j < NJ; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int cl = j * 16 + h * 8 + (tid & 3) * 2;
      T[cl * TP + rl] = d[j][4 * h]; T[cl * TP + rl + 8] = d[j][4 * h + 2];
      T[(cl + 1) * TP + rl] = d[j][4 * h + 1]; T[(cl + 1) * TP + rl + 8] = d[j][4 * h + 3];
    }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  rk_mbar_wait(zbar, 0);   // every CTA's slice of out is clear (acquired by the helper warp)
  for (int idx = tid; idx < BN * 16; idx += 128) {
    const int tk = idx >> 4, qd = idx & 15;
    const int c = tt * BN + tk;
    if (c < M) {
      const float4 v = *reinterpret_cast<const float4*>(T + tk * TP + qd * 4);
      asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};"
                   :: "l"(out + (size_t)c * 256 + rt * BM + qd * 4), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
    }
  }
}


// ---- vjeux/okc-2026 (Apache-2.0, via keell0renz 8e30e1b): rgemv, router GEMV on the FP32 weights read directly
// (8.4 MB, no hi/lo split), ROWS rows per CTA, 256 threads; thread i owns k = 8i + 2048j (j < 4): every weight and x
// load is issued before the first FMA; warp butterfly then a fixed-order cross-warp sum. LDD19 (optimizer): copied for
// router m=1 (MMAX 1) and m=8 (MMAX 8); unchanged but for the namespace.
template <int ROWS, int MMAX>
__global__ void __launch_bounds__(256) rgemv_kernel(const float* __restrict__ W, const __nv_bfloat16* __restrict__ x,
                                                    float* __restrict__ out, int M, int N) {
  constexpr int K = 8192;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  float4 w[ROWS][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float4* p = reinterpret_cast<const float4*>(W + (size_t)(r0 + r) * K + j * 2048 + tid * 8);
      w[r][2 * j] = __ldg(p); w[r][2 * j + 1] = __ldg(p + 1);
    }
  float acc[ROWS][MMAX];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int t = 0; t < MMAX; ++t) {
    if (t >= M) break;
    uint4 xv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) xv[j] = __ldg(reinterpret_cast<const uint4*>(x + (size_t)t * K + j * 2048 + tid * 8));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv[j]);
      float xf[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); xf[2 * e] = f.x; xf[2 * e + 1] = f.y; }
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const float4 a = w[r][2 * j], b = w[r][2 * j + 1];
        float v = acc[r][t];
        v = fmaf(a.x, xf[0], v); v = fmaf(a.y, xf[1], v); v = fmaf(a.z, xf[2], v); v = fmaf(a.w, xf[3], v);
        v = fmaf(b.x, xf[4], v); v = fmaf(b.y, xf[5], v); v = fmaf(b.z, xf[6], v); v = fmaf(b.w, xf[7], v);
        acc[r][t] = v;
      }
    }
  }
  __shared__ float red[ROWS][MMAX][8];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < MMAX; ++t) {
      float v = acc[r][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * MMAX) {
    const int r = tid / MMAX, t = tid % MMAX;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q < 8; ++q) v += red[r][t][q];
      out[(size_t)t * N + r0 + r] = v;
    }
  }
}

// ---- LDD19 (optimizer, 2026-11-03): cgemv_hl, vjeux's cgemv (above) for the FP32 router at m = 1 as a BF16 hi + lo pair:
// W' = [hi | lo] [N, 16384] in the 12-bit planes, each half with its own exponent window (base4 hi / lo); x is read
// twice (k & 8191); FP32 accumulation, FP32 output. Escapes: the CTA table holds columns 0..16383, delta per half's base.
template <int ROWS, int NTH>
__global__ void __launch_bounds__(NTH + 32) cgemv_hl_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    float* __restrict__ out, uint32_t base4h, uint32_t base4l) {
  constexpr int K = 16384, KX = 8192;
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 2 && (KX % (16 * NTH)) == 0, "cgemv_hl: each j chunk lies in one half");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const int col = en.x & 0xFFFF;
        const float c = __bfloat162float(x[col & (KX - 1)]) * esc_delta((uint32_t)en.y, (col >= KX ? base4l : base4h) & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER / 2][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER / 2; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const int jx = j % (PER / 2);
    const uint32_t b4 = j < PER / 2 ? base4h : base4l;
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[jx][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
#pragma unroll
      for (int g = 0; g < 2; ++g) {
        float v = acc[r];
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, b4);
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, b4);
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
        acc[r] = v;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = v;
  }
}
}  // namespace ldd19r

// ---- LDD19: host side (pybind11 only, raw pointers + stream, error codes instead of exceptions)
typedef CUresult (*Ldd19rEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static Ldd19rEnc ldd19r_enc() {
  static Ldd19rEnc e = nullptr;
  if (!e) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) != cudaSuccess) p = nullptr;
#else
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q) != cudaSuccess) p = nullptr;
#endif
    if (q != cudaDriverEntryPointSuccess) p = nullptr;
    e = reinterpret_cast<Ldd19rEnc>(p);
  }
  return e;
}
// vjeux's make_map_x3d: x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}, 128B swizzle
static int ldd19r_map_x3d(CUtensorMap* mp, const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  Ldd19rEnc enc = ldd19r_enc();
  if (!enc) return -3;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  return enc(mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -21;
}
// vjeux's rk_map with the box height as a parameter: [rows, cols] bf16, box {64, box_rows}, 128B swizzle
static int ldd19r_map_2d(CUtensorMap* mp, const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  Ldd19rEnc enc = ldd19r_enc();
  if (!enc) return -3;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  return enc(mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
static int ldd19r_attr(const void* kern, int query) {   // query 1 = registers, 2 = local (spill) bytes, 3 = static smem
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, kern) != cudaSuccess) { cudaGetLastError(); return -1; }
  return query == 1 ? a.numRegs : query == 2 ? (int)a.localSizeBytes : (int)a.sharedSizeBytes;
}
static int ldd19r_rc() { const cudaError_t e = cudaGetLastError(); return e == cudaSuccess ? 0 : 1000 + (int)e; }

// vjeux's launch_sk2 (CL = 1 only), as a raw-pointer launcher
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT, bool PK>
static int ldd19r_sk2_go(int64_t Ws, int64_t x, int64_t out, int64_t ptab, int64_t pmax, int64_t base4, int M, int N, int K,
                         cudaStream_t st, int query) {
  static_assert(CL == 1, "ldd19r: sk2 CL = 1 only");
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (size_t)pmax * 8 + 128;
  auto kern = ldd19r::sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
  if (query) return query == 4 ? (int)smem : query == 5 ? NSTAGE : ldd19r_attr((const void*)kern, query);
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) { cudaGetLastError(); return -5; }
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes) != cudaSuccess) {
      cudaGetLastError();
      return -6;
    }
    attr_done = true;
  }
  if (N % R) return -7;
#if LDD19_SK2_CM   // LDD19 (escape_lab): the non-packed instance is compiled for M == TOKR
  if (!PK && M != TOKR) return -8;
#endif
  CUtensorMap xm;
  const int rc = ldd19r_map_x3d(&xm, (const void*)x, (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
  if (rc) return rc;
  kern<<<N / R, (NW + 1 + NESC) * 32, smem, st>>>(xm, reinterpret_cast<const uint8_t*>(Ws), reinterpret_cast<const __nv_bfloat16*>(x),
      reinterpret_cast<OutT*>(out), reinterpret_cast<const int2*>(ptab), (int)pmax, (uint32_t)base4, M, N, K, 0, nullptr);
  return ldd19r_rc();
}

int64_t ldd19r_sk2(int64_t cfg, int64_t Ws, int64_t x, int64_t out, int64_t ptab, int64_t pmax, int64_t base4, int64_t M,
                   int64_t N, int64_t K, int64_t stream, int64_t query) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  @@SK2_DISPATCH@@
  return -2;
}

template <int ROWS, int NTH, int K, int NACC>
static int ldd19r_cg_go(int64_t L, int64_t C, int64_t x, int64_t etab, int64_t pmax, int64_t out, int64_t base4, int N,
                        cudaStream_t st, int query) {
  auto kern = ldd19r::cgemv_kernel<ROWS, NTH, K, NACC>;
  if (query) return ldd19r_attr((const void*)kern, query);
  if (N % ROWS) return -7;
  kern<<<N / ROWS, NTH + 32, 0, st>>>(reinterpret_cast<const uint8_t*>(L), reinterpret_cast<const uint8_t*>(C),
      reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const int2*>(etab), (int)pmax,
      reinterpret_cast<__nv_bfloat16*>(out), (uint32_t)base4);
  return ldd19r_rc();
}

int64_t ldd19r_cg(int64_t cfg, int64_t L, int64_t C, int64_t x, int64_t etab, int64_t pmax, int64_t out, int64_t base4,
                  int64_t N, int64_t stream, int64_t query) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  @@CG_DISPATCH@@
  return -2;
}

// LDD19 (optimizer): cgemv_hl host launcher (router m = 1), cfg 0: 2 rows x 512 threads, 1: 2 x 256, 2: 1 x 512
template <int ROWS, int NTH>
static int ldd19r_cgh_go(int64_t L, int64_t C, int64_t x, int64_t etab, int64_t pmax, int64_t out, int64_t b4h, int64_t b4l,
                         int N, cudaStream_t st, int query) {
  auto kern = ldd19r::cgemv_hl_kernel<ROWS, NTH>;
  if (query) return ldd19r_attr((const void*)kern, query);
  if (N % ROWS) return -7;
  kern<<<N / ROWS, NTH + 32, 0, st>>>(reinterpret_cast<const uint8_t*>(L), reinterpret_cast<const uint8_t*>(C),
      reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const int2*>(etab), (int)pmax,
      reinterpret_cast<float*>(out), (uint32_t)b4h, (uint32_t)b4l);
  return ldd19r_rc();
}

int64_t ldd19r_cgh(int64_t cfg, int64_t L, int64_t C, int64_t x, int64_t etab, int64_t pmax, int64_t out, int64_t b4h,
                   int64_t b4l, int64_t N, int64_t stream, int64_t query) {
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (cfg == 0) return ldd19r_cgh_go<2, 512>(L, C, x, etab, pmax, out, b4h, b4l, (int)N, st, (int)query);
  if (cfg == 1) return ldd19r_cgh_go<2, 256>(L, C, x, etab, pmax, out, b4h, b4l, (int)N, st, (int)query);
  if (cfg == 2) return ldd19r_cgh_go<1, 512>(L, C, x, etab, pmax, out, b4h, b4l, (int)N, st, (int)query);
  return -2;
}

// LDD19 (optimizer): rgemv host launcher, W [N, 8192] fp32, x [M, 8192] bf16, out [M, N] fp32; M = 1 or 2..8
int64_t ldd19r_rg(int64_t W, int64_t x, int64_t out, int64_t M, int64_t N, int64_t stream, int64_t query) {
  auto kern = M <= 1 ? ldd19r::rgemv_kernel<1, 1> : ldd19r::rgemv_kernel<1, 8>;
  if (query) return ldd19r_attr((const void*)kern, (int)query);
  if (M < 1 || M > 8) return -4;
  kern<<<(int)N, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const float*>(W),
      reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<float*>(out), (int)M, (int)N);
  return ldd19r_rc();
}

// vjeux's rk host function: W = rwdl's packed router [wrows = 65536, 64] bf16 (LDD19), x [M, 8192] bf16, out [M, 256] fp32
template <int QS>   // LDD19 (sweeper): one launcher per ring depth
static int64_t ldd19r_rk_go(int64_t W, int64_t wrows, int64_t x, int64_t out, int64_t M, int64_t S, int64_t zc, int64_t stream,
                  int64_t query) {
  const int smem = 1024 + QS * 24576 + 8 * (2 * QS + 1) + 64;
  if (query) return query == 4 ? smem : ldd19r_attr((const void*)ldd19r::rk_kernel<QS>, (int)query);
  if (M < 1 || M > 256 || wrows != 4 * 128 * 2 * 64) return -4;
  const int ntt = (int)((M + 63) / 64);
  const int grid = 4 * ntt * (int)S;
  static int nsm = 0;
  if (!nsm) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev); }
  if (grid > nsm || 128 % S) return -8;   // every CTA must be resident (one per SM): the in-kernel clear waits for all
  CUtensorMap mw, mx;
  int rc = ldd19r_map_2d(&mw, (const void*)W, (uint64_t)wrows, 64, 128);
  if (rc) return rc;
  rc = ldd19r_map_2d(&mx, (const void*)x, (uint64_t)M, 8192, 64);
  if (rc) return rc;
  static bool attr = false;
  if (!attr) {
    if (cudaFuncSetAttribute(ldd19r::rk_kernel<QS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
      cudaGetLastError();
      return -6;
    }
    attr = true;
  }
  ldd19r::rk_kernel<QS><<<grid, 192, smem, reinterpret_cast<cudaStream_t>(stream)>>>(mw, mx, reinterpret_cast<float*>(out), (int)M,
      (int)S, ntt, reinterpret_cast<int*>(zc) + ntt);
  return ldd19r_rc();
}
int64_t ldd19r_rk(int64_t W, int64_t wrows, int64_t x, int64_t out, int64_t M, int64_t S, int64_t zc, int64_t stream,
                  int64_t query) {
  if (M <= 128) return ldd19r_rk_go<LDD19_RK_QS128>(W, wrows, x, out, M, S, zc, stream, query);   // LDD19 (sweeper)
  return ldd19r_rk_go<RK_QS>(W, wrows, x, out, M, S, zc, stream, query);
}
"""

LDD19R_CPP = ("#include <pybind11/pybind11.h>\n#include <stdint.h>\n"
              "int64_t ldd19r_sk2(" + ", ".join(["int64_t"] * 12) + ");\n"
              "int64_t ldd19r_cg(" + ", ".join(["int64_t"] * 11) + ");\n"
              "int64_t ldd19r_rk(" + ", ".join(["int64_t"] * 9) + ");\n"
              "int64_t ldd19r_rg(" + ", ".join(["int64_t"] * 7) + ");\n"
              "int64_t ldd19r_cgh(" + ", ".join(["int64_t"] * 12) + ");\n")   # LDD19 (optimizer)

# ---- vjeux/okc-2026 f35a41e (Apache-2.0): packing helpers and sk2 geometry, verbatim but for the ldd19r_ prefix
LDD19R_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}


def ldd19r_exponent_window(e71):
    hist = torch.bincount(e71.flatten(), minlength=128)
    win = torch.stack([hist[b:b + 8].sum() for b in range(128 - 8 + 1)])
    base = int(win.argmax())
    return base, (e71 - base).clamp(0, 8 - 1)


def ldd19r_escapes(e71, base):
    return (e71 < base) | (e71 > base + 8 - 1)


def ldd19r_sk2_geom(cfg, tokr):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    hl = layer == "router"
    tu = 2048 if hl else 768
    ups = WSK * UPW
    wby = NWR * WSK * UPW * NT * tu
    xby = ups * tokr * 128
    sst = (xby + wby + 1023) // 1024 * 1024
    nst = -(-(LDD19R_SK2_K[layer] // (ups * 64)) // CL)
    mt = 2 if tokr == 32 else 1
    toks = 16 * mt
    cp = R + 1
    nesc = 4 if not hl else 1
    rp = R + ((40 - (R % 32)) % 32)
    extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
             + 64 * 8 + 128)
    cap_kb = 200
    tail = name.rsplit("_r", 1)
    if len(tail) == 2 and tail[1].isdigit():
        cap_kb = int(tail[1])
    ring = min(cap_kb * 1024, 232_448 - extra)
    nstage = max(2, min(16, nst, ring // sst))
    return hl, nstage


def ldd19r_sk2_tileunits(w):
    n, k = w.shape
    t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
    return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def ldd19r_sk2_compress(w):
    n, k = w.shape
    tu = ldd19r_sk2_tileunits(w)
    bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi = bits >> 8
    low = bits & 0xFF
    e71 = hi & 0x7F
    sign = hi >> 7
    base, o = ldd19r_exponent_window(e71)
    code = (sign << 3) | o
    esc = ldd19r_escapes(e71, base)
    if int(esc.sum()) > n * k // 200:
        return None
    T, U = n // 8, k // 64
    c8 = code.to(torch.uint8)
    w0 = c8[..., 0:4] | (c8[..., 4:8] << 4)
    w1 = c8[..., 8:12] | (c8[..., 12:16] << 4)
    codes = torch.cat([w0, w1], dim=-1)
    packed = torch.cat([low.to(torch.uint8).reshape(T, U, 512), codes.reshape(T, U, 256)], dim=-1).contiguous()
    idx = esc.nonzero()
    dec_bits = ((((sign << 7) | (base + o)) << 8) | low)[esc]
    wdec = dec_bits.to(torch.int16).view(torch.bfloat16).float()
    wtrue = tu[esc].float()
    delta = wtrue - wdec
    if not bool(((wdec + delta) == wtrue).all()):
        return None
    lane, v = idx[:, 2], idx[:, 3]
    rows = idx[:, 0] * 8 + (lane >> 2)
    cols = idx[:, 1] * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
    wbits = (tu[esc].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).contiguous()
    return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "wbits": wbits}


def ldd19r_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
    dev = tub.device
    U = tub.shape[1]
    ups = WSK * UPW
    ar = lambda v: torch.arange(v, device=dev)  # noqa: E731
    s = ar(U // ups).view(-1, 1, 1, 1, 1)
    c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
    w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
    uu = ar(UPW).view(1, 1, 1, -1, 1)
    nt = ar(NT).view(1, 1, 1, 1, -1)
    tile = c * (R // 8) + (w // WSK) * NT + nt
    unit = s * ups + (w % WSK) * UPW + uu
    return tub[tile, unit].contiguous()


def ldd19r_sk2_balance_perm(w, n_rows, R):
    wl = w[:n_rows]
    bits = wl.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e71 = (bits >> 8) & 0x7F
    base, _ = ldd19r_exponent_window(e71)
    cnt = ldd19r_escapes(e71, base).sum(1)
    G4 = 4
    tcnt = cnt.view(n_rows // G4, G4).sum(1)
    order = torch.argsort(tcnt, descending=True, stable=True)
    TB = R // G4
    nb = n_rows // R
    i = torch.arange(n_rows // G4, device=w.device)
    rnd, pos = i // nb, i % nb
    blk = torch.where(rnd % 2 == 0, pos, nb - 1 - pos)
    tperm = torch.empty(n_rows // G4, dtype=torch.long, device=w.device)
    tperm[blk * TB + rnd] = order
    return (tperm.view(-1, 1) * G4 + torch.arange(G4, device=w.device).view(1, G4)).reshape(-1)


def ldd19r_sk2_table_add_perm(tab, pmax, perm, R):
    ncta = tab.shape[0]
    pm2 = pmax + R // 2
    pm2 += pm2 % 2
    out = torch.zeros(ncta, pm2, 2, dtype=torch.int32, device=tab.device)
    out[:, :pmax] = tab
    out[:, pmax:pmax + R // 2] = perm.to(torch.int32).view(ncta, R // 2, 2)
    out[:, 0, 1] = pmax
    return out.contiguous(), pm2


def ldd19r_sk2_table(comp, n_cfg, R):
    dev = comp["rows"].device
    rows, cols, delta = comp["rows"], comp["cols"], comp["wbits"]
    keep = rows < n_cfg
    rows, cols, delta = rows[keep], cols[keep], delta[keep]
    ncta = n_cfg // R
    cta = rows // R
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max().item()) + 1 if rows.numel() else 1
    pmax += pmax % 2
    order = torch.argsort(cta.long() * 65536 + cols.long())
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
    start[1:] = torch.cumsum(counts, 0)
    slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, slot, 0] = (((rows % R) << 16) | cols).to(torch.int32)
    tab[cta, slot, 1] = delta.to(torch.int32)
    return tab.contiguous(), pmax


def ldd19r_cg_compress(w, rows_per_cta):
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    hi, low = bits >> 8, bits & 0xFF
    e71, sign = hi & 0x7F, hi >> 7
    base, o = ldd19r_exponent_window(e71)
    code = ((sign << 3) | o).to(torch.uint8)
    esc = ldd19r_escapes(e71, base)
    if int(esc.sum()) > n * k // 200:
        return None
    g = code.view(n, k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k // 2).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    dec = ((((sign[rows, cols] << 7) | (base + o[rows, cols])) << 8) | low[rows, cols]).to(torch.int16).view(torch.bfloat16).float()
    wtrue = w[rows, cols].float()
    delta = (wtrue - dec).contiguous()
    if not bool(((dec + delta) == wtrue).all()):
        return None
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=w.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=w.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": base * 0x01010101}


def ldd19r_cgh_compress(hi, lo, rows_per_cta):
    """LDD19 (optimizer): cg_compress for [hi | lo] [n, 2k], each half with its own exponent window; escape table entries
    hold columns 0..2k-1 (the kernel picks the half's base by column)."""
    n, k = hi.shape
    lows, codes, escs, bitss, bases = [], [], [], [], []
    for w in (hi, lo):
        bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        h8, low = bits >> 8, bits & 0xFF
        e71, sign = h8 & 0x7F, h8 >> 7
        base, o = ldd19r_exponent_window(e71)
        lows.append(low); codes.append(((sign << 3) | o).to(torch.uint8)); escs.append(ldd19r_escapes(e71, base))
        bitss.append(bits); bases.append(base)
    low, code, esc, bits = (torch.cat(t, 1) for t in (lows, codes, escs, bitss))
    if int(esc.sum()) > n * 2 * k // 200:
        return None
    g = code.view(n, 2 * k // 8, 8)
    C = (g[:, :, 0:4] | (g[:, :, 4:8] << 4)).reshape(n, k).contiguous()
    L = low.to(torch.uint8).contiguous()
    rows, cols = esc.nonzero(as_tuple=True)
    ncta = n // rows_per_cta
    cta = rows // rows_per_cta
    counts = torch.bincount(cta, minlength=ncta)
    pmax = int(counts.max()) + 1 if rows.numel() else 1
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(rows.numel(), device=hi.device) - start[cta] + 1
    tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=hi.device)
    tab[:, 0, 0] = counts.to(torch.int32)
    tab[cta, pos, 0] = (((rows - cta * rows_per_cta) << 16) | cols).to(torch.int32)
    tab[cta, pos, 1] = bits[rows, cols].to(torch.int32)
    return {"L": L, "C": C, "tab": tab.contiguous(), "pmax": pmax, "base4": bases[0] * 0x01010101,
            "base4l": bases[1] * 0x01010101}


# ---- LDD19: build, validation, prepare, dispatch
def ldd19r_source():
    sk2 = []
    for ci, (cfg, tokr, _) in enumerate(LDD19R_SK2):
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        hl, nstage = ldd19r_sk2_geom(cfg, tokr)
        mt = 2 if tokr == 32 else 1
        nacc = 1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC   # vjeux's _sk2_src rule
        targs = f"{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, {str(not hl).lower()}, {str(hl).lower()}, {nacc}, {CL}, " \
                f"{'float' if hl else '__nv_bfloat16'}"
        sk2.append(f"if (cfg == {ci}) {{ if ({'true' if tokr == 8 else 'false'} || M < {tokr}) return ldd19r_sk2_go<{targs}, true>(Ws, x, out, "
                   f"ptab, pmax, base4, (int)M, (int)N, (int)K, st, (int)query); return ldd19r_sk2_go<{targs}, false>(Ws, x, out, ptab, "
                   f"pmax, base4, (int)M, (int)N, (int)K, st, (int)query); }}")
    cg = [f"if (cfg == {i}) return ldd19r_cg_go<{r}, {t}, {k}, {a}>(L, C, x, etab, pmax, out, base4, (int)N, st, (int)query);"
          for i, (k, r, t, a) in enumerate(LDD19R_CG_CONFIGS)]
    return LDD19R_CUDA.replace("@@SK2_DISPATCH@@", "\n  ".join(sk2)).replace("@@CG_DISPATCH@@", "\n  ".join(cg))


def ldd19r_build():
    src = ldd19r_source()
    return _ldd19_inline('ldd19r_' + hashlib.sha256(src.encode()).hexdigest()[:12], LDD19R_CPP, cuda_sources=src,
                       functions=['ldd19r_sk2', 'ldd19r_cg', 'ldd19r_rk', 'ldd19r_rg', 'ldd19r_cgh'],
                       extra_cuda_cflags=['-O3', '-DNDEBUG', '-gencode=arch=compute_90a,code=sm_90a'],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def ldd19r_call(st, r, x, out):
    """Launch route r on x into out (current stream); 0 on success."""
    s = torch.cuda.current_stream(x.device).cuda_stream
    if r['kind'] == 'cg':
        return st['ext'].ldd19r_cg(r['cfg'], r['L'].data_ptr(), r['C'].data_ptr(), x.data_ptr(), r['tab'].data_ptr(), r['pmax'],
                                   out.data_ptr(), r['base4'], r['n'], s, 0)
    if r['kind'] == 'sk2':
        return st['ext'].ldd19r_sk2(r['cfg'], r['ws'].data_ptr(), x.data_ptr(), out.data_ptr(), r['tab'].data_ptr(), r['pmax'],
                                    r['base4'], x.shape[0], r['n'], r['K'], s, 0)
    if r['kind'] == 'cgh':   # LDD19 (optimizer)
        return st['ext'].ldd19r_cgh(r['cfg'], r['L'].data_ptr(), r['C'].data_ptr(), x.data_ptr(), r['tab'].data_ptr(), r['pmax'],
                                    out.data_ptr(), r['base4'], r['base4l'], r['n'], s, 0)
    if r['kind'] == 'rg':   # LDD19 (optimizer)
        return st['ext'].ldd19r_rg(r['w'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], r['n'], s, 0)
    return st['ext'].ldd19r_rk(r['w'].data_ptr(), r['w'].shape[0], x.data_ptr(), out.data_ptr(), x.shape[0], r.get('S', LDD19R_RK_S),
                               r['zc'].data_ptr(), s, 0)


def ldd19r_verify(st, r, w_eff, m):
    """v19-style checks at m exactly. w_eff: [n, K] weights the route must reproduce (BF16; router: hi + lo in FP32).
    Zero input over a NaN output; every K column as a one-hot row, bit-exact against w_eff; random rows (normal, then 1%
    outlier columns x30) with a 16.0 spike row against FP64: overall rel L2 <= tol, every row <= 5 tol, spike row exact
    to 2.5e-4 (router: tolerances / 100)."""
    n, K = w_eff.shape
    dev = w_eff.device
    od = torch.float32 if r['kind'] in ('rk', 'rg', 'cgh') else torch.bfloat16   # LDD19 (optimizer): rg, cgh
    out = torch.full((m, n), float('nan'), dtype=od, device=dev)
    if ldd19r_call(st, r, torch.zeros((m, K), dtype=torch.bfloat16, device=dev), out):
        return False, 'launch failed'
    if not bool((out == 0).all()):
        return False, 'zero input: output not zero'
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev)
    got = torch.empty((K // m, m, n), dtype=od, device=dev)
    for i in range(K // m):
        if ldd19r_call(st, r, eye[i * m:(i + 1) * m], got[i]):
            return False, 'one-hot launch failed'
    exact = w_eff.t().to(od)
    if not (torch.equal(got.view(K, n), exact) or (r['kind'] == 'cgh' and float((got.view(K, n) - exact).abs().max()) <= 1e-6 * float(exact.abs().max()))):
        # LDD19 (optimizer): cgh adds an escaped element's exact correction in the escape warp, so (decoded hi + lo) + delta
        # can round differently in the last FP32 bits (seen: 9 of 2.1M entries); bound the absolute error instead
        bad = int((got.view(K, n) != exact).sum())
        return False, f'one-hot not exact ({bad} of {K * n})'
    del eye, got
    tol = 2.5e-5 if r['kind'] in ('rk', 'rg', 'cgh') else 5e-3   # LDD19 (optimizer): rg, cgh
    spike_tol = 2.5e-5 if r['kind'] in ('rk', 'rg', 'cgh') else 2.5e-4   # rk: hi + lo vs the FP32 weight (~2^-17 relative)
    gen = torch.Generator(device=dev).manual_seed(11)
    worst = 0.0
    wd = r['wref'].double()
    for dist in (1.0, 30.0, 0.0):   # 0.0: m = 1 only, the spike row alone (at m > 1 it is the last row of every input)
        if dist == 0.0 and m > 1:
            continue
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist == 30.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[:K // 100]] *= dist
        if m > 1 or dist == 0.0:
            x[m - 1].zero_()
            x[m - 1, (11 * 131 + m) % K] = 16.0
        x = x.to(torch.bfloat16)
        ref = x.double() @ wd.t()
        o = torch.empty((m, n), dtype=od, device=dev)
        if ldd19r_call(st, r, x, o):
            return False, 'random launch failed'
        diff = o.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        err = float(diff.norm() / ref.norm())
        worst = max(worst, err)
        spk = float(rows[m - 1]) if (m > 1 or dist == 0.0) else 0.0
        if not bool(torch.isfinite(o).all()) or err > tol or float(rows.max()) > 5 * tol or spk > spike_tol:
            return False, (f'random dist {dist}: rel L2 {err:.2e} (bound {tol:.0e}) row max {float(rows.max()):.2e} '
                           f'spike {float(rows[m - 1]):.2e}')
    return True, f'ok (rel L2 {worst:.1e})'


def ldd19r_add(st, key, r, w_eff, m):
    try:
        ok, msg = ldd19r_verify(st, r, w_eff, m)
    except Exception as error:  # noqa: BLE001 - a failing route keeps v19's kernel
        ok, msg = False, f'{type(error).__name__}: {str(error)[-200:]}'
    st['log'].append(f"{r['kind']} {key[0]} m={key[1]} {r.get('name', '')}: {msg}" + ('' if ok else ', v19 route kept'))
    if ok:
        st['routes'][key] = r


def prepare_ldd19r(weights):
    """Prepare-pool job: build the extension, pack and validate the cgemv and sk2 routes (rk is set up after the pool,
    on v18's packed router layout)."""
    st = {'routes': {}, 'log': []}
    dev = weights['qkv_gate'].device
    if torch.cuda.get_device_capability(dev) != (9, 0):
        return st
    try:
        st['ext'] = ldd19r_build()
    except Exception as error:  # noqa: BLE001 - a failed build keeps every v19 route
        st['log'].append(f'build failed: {type(error).__name__}: {str(error)[-300:]}')
        return st
    for layer, sw in (('qkv_gate', 'qkv1'), ('unpooler', 'un1'), ('pooler', 'pool1'), ('o', 'o1')):   # LDD19 (optimizer): +pool1, o1
        if not LDD19R.get(sw):
            continue
        ci = LDD19R_CG_PICK[layer]
        K, rows, nth, nacc = LDD19R_CG_CONFIGS[ci]
        w = weights[layer]
        w = (w[:1024] if layer == 'unpooler' else w).contiguous()
        c = ldd19r_cg_compress(w, rows) if w.shape[1] == K else None
        if c is None:
            st['log'].append(f'cg {layer}: 12-bit packing not usable, v19 route kept')
            continue
        r = dict(kind='cg', name=f'cg{ci}', cfg=ci, n=w.shape[0], K=K, wref=w, **c)
        ldd19r_add(st, (layer, 1), r, w, 1)
    wr = weights['router']   # LDD19 (optimizer): router m=1 on cgemv_hl (12-bit hi + lo planes)
    if LDD19R.get('cgh1') and wr.dtype == torch.float32 and tuple(wr.shape) == (256, 8192):
        rows = (2, 2, 1)[LDD19R_CGH_CFG]
        hi = wr.to(torch.bfloat16)
        lo = (wr - hi.float()).to(torch.bfloat16)
        c = ldd19r_cgh_compress(hi, lo, rows)
        if c is None:
            st['log'].append('cgh router: 12-bit packing not usable, v19 route kept')
        else:
            weff = hi.float() + lo.float()
            ldd19r_add(st, ('router', 1), dict(kind='cgh', name=f'cgh{LDD19R_CGH_CFG}', cfg=LDD19R_CGH_CFG, n=256, K=8192,
                                               wref=wr, **c), weff, 1)
    if wr.dtype == torch.float32 and tuple(wr.shape) == (256, 8192):
        wr = wr.contiguous()
        for m, sw in ((1, 'rg1'), (8, 'rg8')):
            if LDD19R.get(sw):
                ldd19r_add(st, ('router', m), dict(kind='rg', name=f'rg{m}', n=256, K=8192, w=wr, wref=wr), wr, m)
    for ci, (cfg, tokr, sw) in enumerate(LDD19R_SK2):
        if not LDD19R.get(sw):
            continue
        name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
        w = weights[layer][:n_rows].contiguous()   # LDD19 (optimizer): [:n_rows] for the unpooler slice (same for others)
        perm = None
        if layer == 'qkv_gate':   # vjeux's SK2_BALANCE (qkv only): 4-row groups dealt so every block has ~equal escapes
            perm = ldd19r_sk2_balance_perm(w, n_rows, R)
            comp = ldd19r_sk2_compress(w[:n_rows][perm].contiguous())
        else:
            comp = ldd19r_sk2_compress(w)
        if comp is None:
            st['log'].append(f'sk2 {layer}: 12-bit packing not usable, v19 route kept')
            continue
        tab, pmax = ldd19r_sk2_table(comp, n_rows, R)
        if perm is not None:
            tab, pmax = ldd19r_sk2_table_add_perm(tab, pmax, perm, R)
        ws = ldd19r_sk2_stream(comp['packed'], n_rows, R, NT, NWR, WSK, UPW)
        r = dict(kind='sk2', name=name, cfg=ci, n=n_rows, K=w.shape[1], ws=ws, tab=tab, pmax=pmax, base4=comp['base4'], wref=w)
        del comp
        ldd19r_add(st, (layer, tokr), r, w, tokr)
    torch.cuda.synchronize(dev)
    return st


def ldd19r_prepare_rk(state, weights):
    st = state.get('ldd19r')
    if not st or 'ext' not in st or not LDD19R.get('rt256'):
        return
    w = weights['router']
    if w.dtype != torch.float32 or tuple(w.shape) != (256, 8192):
        st['log'].append('rk: unexpected router weights')
        return
    rw = (state.get('bundle') or {}).get('rw') or {}
    packed = rw.get('packed') if rw.get('enabled') else None
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    if packed is None or tuple(packed.shape) != (65536, 64):   # rwdl not built: the same layout, kept by this route
        packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)
    r = dict(kind='rk', name=f'rk_s{LDD19R_RK_S}', w=packed, n=256, K=8192, wref=w,
             zc=torch.zeros(8, dtype=torch.int32, device=w.device))
    ldd19r_add(st, ('router', 256), r, hi.float() + lo.float(), 256)
    if LDD19_RK128:   # LDD19 (sweeper): router m=128 on rk, S = 16, its own zeroing counters; validated like m=256
        r128 = dict(kind='rk', name=f'rk_s{LDD19R_RK128_S}', w=packed, n=256, K=8192, wref=w, S=LDD19R_RK128_S,
                    zc=torch.zeros(8, dtype=torch.int32, device=w.device))
        ldd19r_add(st, ('router', 128), r128, hi.float() + lo.float(), 128)


def ldd19r_prepare_v18_extra(weights):
    """v18's prepare_v18_extra without the qkv_gate m=1 layout (served by cgemv)."""
    instances = list(dict.fromkeys(inst for original, cfg in V18_SMALL.values()
                                   for inst in (original, v18s_bf16_twin(original))))
    ext = v18s_build(instances, 'v18_small_selected')
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams, plans = {}, {}, {}
    for (layer, m), (inst, cfg) in V18_SMALL.items():
        key = (layer, 'm', m)
        plan = v18s_plan(ext, ids, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None:
            plan = v18s_plan(ext, ids, packs, streams, weights[layer], layer, key, v18s_bf16_twin(inst), cfg)
        if plan is not None:
            plans[layer, m] = plan
    return {'ext': ext, 'plans': plans}


def ldd19r_prepare_bundle2(weights) -> dict:
    """v17's prepare_bundle2 cut to its one served route, o m=32 (qkv m=16 is ldd19r's sk2, pooler / unpooler m=32 are
    ldd19r's sk2 routes)."""
    return {'o': b2o_prepare_olayer_candidates(weights, [(32, 128, 8, 1, 1, 1)])}


def run_ldd19r(x, layer, n, st):
    if not st or x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    r = st['routes'].get((layer, x.shape[0]))
    if r is None or n != r['n'] or x.shape[1] != r['K'] or x.device != r['wref'].device:
        return None
    out = torch.empty((x.shape[0], n), dtype=torch.float32 if r['kind'] in ('rk', 'rg', 'cgh') else torch.bfloat16, device=x.device)
    return out if ldd19r_call(st, r, x, out) == 0 else None


_ldd19r_base_prepare = prepare
_ldd19r_base_kernel = kernel_fn


def prepare(weights: Weights) -> dict:
    state = _ldd19r_base_prepare(weights)
    ldd19r_prepare_rk(state, weights)
    st = state.get('ldd19r')
    if st:
        for line in st['log']:
            print('ldd19r ' + line, flush=True)
        for layer, m in list(st['routes']):   # one warm-up call per new route
            w = weights[layer]
            n = st['routes'][(layer, m)]['n']
            data = {'x': torch.zeros((m, w.shape[1]), dtype=torch.bfloat16, device=w.device),
                    'config': {'m': m, 'n': n, 'k': w.shape[1], 'layer': layer}}
            kernel_fn(data, state)
        torch.cuda.synchronize(weights['qkv_gate'].device)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    out = run_ldd19r(data['x'], data['config']['layer'], data['config']['n'], state.get('ldd19r'))
    return _ldd19r_base_kernel(data, state) if out is None else out


# ======================================================================================================== v22
# LDD19: v22 (2026-10-28): router m=128 on rr22 (ours): grid 8 row tiles x 16 K slices = 128 CTAs; each CTA bursts its 64 KB
# hi | lo weight slice and its x slice at entry, wgmma m64n128k16 as blocks land, hi + lo summed in registers, one fp32
# TMA tensor reduce-add of its 32 x 128 partial into an in-kernel-zeroed out (per-tile counters, self-resetting).
# Validated in prepare() (zeroing, every K column one-hot bit-exact against hi + lo, random rows vs FP64); on any failure
# the route is dropped and the v8 router kernel (rr_kernel) serves router m=128.
V22_RR22_CU = r'''
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>
#define R22D __device__ __forceinline__
R22D uint32_t r22_s(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
R22D void r22_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory"); }
R22D void r22_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(tx) : "memory"); }
R22D void r22_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
R22D void r22_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nR22W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra R22W_%=;\n}\n" ::"r"(b), "r"(ph) : "memory");
}
R22D void r22_bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
R22D void r22_tile(uint32_t dst, const void* tm, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(tm), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
R22D uint64_t r22_desc(uint32_t a) { return (uint64_t)((a & 0x3ffff) >> 4) | ((uint64_t)64 << 32) | ((uint64_t)1 << 62); }
R22D unsigned long long r22_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
R22D void r22_mma64(float* d, uint64_t a, uint64_t b) { asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 1, 1, 1, 0, 0;" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) : "l"(a), "l"(b)); }
R22D void r22_mma128(float* d, uint64_t a, uint64_t b) { asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1, 0, 0;" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) : "l"(a), "l"(b)); }
R22D void r22_mma256(float* d, uint64_t a, uint64_t b) { asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 1, 1, 1, 0, 0;" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]), "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]), "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]), "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]), "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]), "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]), "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]), "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]), "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127]) : "l"(a), "l"(b)); }

struct R22P { const uint8_t* W; float* out; int* cnt; unsigned long long* prof; int m; };

template <int T, int NKB, int ST, int FL>
__global__ void __launch_bounds__(160, 1) r22k(const R22P p, const __grid_constant__ CUtensorMap tx, const __grid_constant__ CUtensorMap to) {
  constexpr int KS = 128 / NKB, WB = 8192, XB = T * 128, SB = WB + XB, NA = T / 2, NS = ST < NKB ? ST : NKB;
  constexpr bool RED = (FL & 1) != 0, PROF = (FL & 256) != 0, NOEPI = (FL & 512) != 0, NOMMA = (FL & 1024) != 0;
  // LDD19: C2 (FL & 2): clusters of 2 along K; rank r owns tokens [r T / 2, (r + 1) T / 2) of the pair's tile and receives
  // the peer's partial of them by st.async into RV [32 rows][RS] (its own region after the stages), then reduce-adds T / 2
  constexpr bool C2 = (FL & 2) != 0, PW = (FL & 16) != 0, SWZ = (FL & 32) != 0;
  static_assert(!(PW && (C2 || RED)), "PW: TMA path only");
  static_assert(!(SWZ && (C2 || RED || PW)), "SWZ: plain TMA path only");
  constexpr int RS = T / 2 + 4, OT = C2 ? T / 2 : T;  // RV row stride (floats), tokens reduce-added per CTA
  extern __shared__ __align__(1024) uint8_t smraw[];
  __shared__ __align__(8) uint64_t bars[2 * NS];
  __shared__ uint32_t zflag;
  __shared__ __align__(8) uint64_t rbar;
  const uint32_t sb = (r22_s(smraw) + 1023u) & ~1023u;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rt = blockIdx.x, ks = blockIdx.y, tz = blockIdx.z, tok0 = tz * T, cta = (tz * KS + ks) * 8 + rt;
  const int ci = tz * 8 + rt;  // this tile's counters: cnt[ci] (zeroed slices), cnt[64 + ci] (pollers)
  const uint32_t full = r22_s(bars), empty = full + 8 * NS, zf = r22_s(&zflag), rb = r22_s(&rbar), rv = sb + NS * SB;
  const uint8_t* wsrc = p.W + ((size_t)rt * 128 + (size_t)ks * NKB) * WB;
  if (PROF && tid == 0) p.prof[cta * 8 + 0] = r22_gt();
  auto issue = [&](int i) {
    const int s = i % NS;
    const uint32_t st = sb + s * SB;
    r22_expect(full + 8 * s, SB);
    r22_bulk(st, wsrc + (size_t)i * WB, WB, full + 8 * s);
    r22_tile(st + WB, &tx, (ks * NKB + i) * 64, tok0, full + 8 * s);
  };
  if (tid == 128) {  // LDD19: init + the whole first burst before the block barrier (the math warps only wait after it)
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&tx)) : "memory");
    for (int s = 0; s < NS; s++) { r22_init(full + 8 * s, 1); r22_init(empty + 8 * s, 4); }
    asm volatile("st.shared.u32 [%0], 0;" ::"r"(zf) : "memory");
    if (C2) { r22_init(rb, 1); r22_expect(rb, 32 * (T / 2) * 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if constexpr ((FL & 12) == 0) {
      for (int i = 0; i < NS; i++) issue(i);
    } else {  // LDD19: FL 4: every W block first, then the x boxes; FL 8: x boxes first
      for (int i = 0; i < NS; i++) r22_expect(full + 8 * i, SB);
      for (int h = 0; h < 2; h++)
        for (int i = 0; i < NS; i++) {
          if ((h == 0) == ((FL & 4) != 0)) r22_bulk(sb + i * SB, wsrc + (size_t)i * WB, WB, full + 8 * i);
          else r22_tile(sb + i * SB + WB, &tx, (ks * NKB + i) * 64, tok0, full + 8 * i);
        }
    }
    if (!RED) asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&to)) : "memory");
  }
  __syncthreads();
  if (C2) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");  // my rb is ready (waited at the push)
  if (warp == 4) {  // ---- producer warp: ring refills, then the zeroing of this CTA's token rows and the counter poll
    if (C2) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (lane == 0)
      for (int i = NS; i < NKB; i++) {
        r22_wait(empty + 8 * (i % NS), ((i / NS) - 1) & 1);
        issue(i);
      }
    if (NOEPI) return;
    constexpr int TZ = T / KS;  // token rows zeroed by this CTA (of its row tile's 32 columns)
    for (int e = lane; e < TZ * 8; e += 32) {
      const int tok = tok0 + ks * TZ + e / 8;
      if (tok < p.m) {
        float* d = p.out + (size_t)tok * 256 + rt * 32 + (e % 8) * 4;
        asm volatile("st.global.v4.f32 [%0], {%1, %1, %1, %1};" ::"l"(d), "f"(0.f) : "memory");
      }
    }
    if (!RED) asm volatile("fence.proxy.async.global;" ::: "memory");  // generic zero stores -> the TMA reduce-adds
    __syncwarp();
    if (lane == 0) {
      asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p.cnt + ci) : "memory");
      uint32_t c = 0;
      for (int it = 0; it < (1 << 22); it++) {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(c) : "l"(p.cnt + ci) : "memory");
        if ((int)c >= KS) break;
        __nanosleep(64);
      }
      if ((int)c < KS) asm volatile("trap;");  // grid not co-resident: fail loudly, never reduce into unzeroed memory
      if (!RED) asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("st.release.cta.shared.u32 [%0], 1;" ::"r"(zf) : "memory");
      if (atomicAdd(p.cnt + 64 + ci, 1) == KS - 1) { p.cnt[ci] = 0; p.cnt[64 + ci] = 0; }  // last poller resets both
    }
    return;
  }
  // ---- consumer warpgroup (warps 0..3)
  float acc[NA];
#pragma unroll
  for (int j = 0; j < NA; j++) asm volatile("mov.b32 %0, 0;" : "=f"(acc[j]));
#pragma unroll 1
  for (int i = 0; i < NKB; i++) {
    const int s = i % NS;
    r22_wait(full + 8 * s, (i / NS) & 1);
    if (PROF && tid == 0) { if (i == 0) p.prof[cta * 8 + 1] = r22_gt(); if (i == NKB - 1) p.prof[cta * 8 + 2] = r22_gt(); }
    if (!NOMMA) {
      const uint32_t w = sb + s * SB, x = w + WB;
#pragma unroll
      for (int j = 0; j < NA; j++) asm volatile("" : "+f"(acc[j])::"memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
      for (int q = 0; q < 4; q++) {
        if constexpr (T == 64) r22_mma64(acc, r22_desc(w + q * 32), r22_desc(x + q * 32));
        else if constexpr (T == 128) r22_mma128(acc, r22_desc(w + q * 32), r22_desc(x + q * 32));
        else r22_mma256(acc, r22_desc(w + q * 32), r22_desc(x + q * 32));
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
#pragma unroll
      for (int j = 0; j < NA; j++) asm volatile("" : "+f"(acc[j])::"memory");
    }
    if (NS < NKB && i >= 1 && i - 1 + NS < NKB && lane == 0) r22_arrive(empty + 8 * ((i - 1) % NS));
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
  for (int j = 0; j < NA; j++) asm volatile("" : "+f"(acc[j])::"memory");
  if (PROF && tid == 0) p.prof[cta * 8 + 3] = r22_gt();
  if (NOEPI) return;
  // hi + lo: real row 8 warp + g, tokens 8 j + 2 t (+1); partial staged as [OT][32] fp32 over stage 0 (every stage is done)
  float* P = (float*)(smraw + (sb - r22_s(smraw)));
  const int g = lane >> 2, t = lane & 3, r = 8 * warp + g;
  int t0 = 0;  // first token reduce-added by this CTA
  if constexpr (PW) {  // LDD19: FL 16: each warp stages its 8 rows as a dense [T][8] block and reduce-adds it on its own
    float* Pw = P + warp * T * 8;
#pragma unroll
    for (int j = 0; j < T / 8; j++) {
      Pw[(8 * j + 2 * t) * 8 + g] = acc[4 * j] + acc[4 * j + 2];
      Pw[(8 * j + 2 * t + 1) * 8 + g] = acc[4 * j + 1] + acc[4 * j + 3];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncwarp();
    if (lane == 0) {
      uint32_t f = 0;
      for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(f) : "r"(zf) : "memory");
      if (!f) asm volatile("trap;");
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];"
                   ::"l"(reinterpret_cast<uint64_t>(&to)), "r"(rt * 32 + 8 * warp), "r"(tok0), "r"(sb + warp * T * 32) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    if (PROF && tid == 0) p.prof[cta * 8 + 4] = r22_gt();
    return;
  } else if constexpr (!C2 && SWZ) {  // LDD19: FL 32: [T][32] fp32 staged 128B-swizzled (chunk r / 4 of row tok at
    // (r / 4) ^ (tok % 8)), bank-conflict free; the out tensor map uses SWIZZLE_128B to match
#pragma unroll
    for (int j = 0; j < T / 8; j++) {
      const int k0 = 8 * j + 2 * t, k1 = k0 + 1;
      P[k0 * 32 + ((((r >> 2) ^ (k0 & 7))) << 2) + (r & 3)] = acc[4 * j] + acc[4 * j + 2];
      P[k1 * 32 + ((((r >> 2) ^ (k1 & 7))) << 2) + (r & 3)] = acc[4 * j + 1] + acc[4 * j + 3];
    }
  } else if constexpr (!C2) {
#pragma unroll
    for (int j = 0; j < T / 8; j++) {
      P[(8 * j + 2 * t) * 32 + r] = acc[4 * j] + acc[4 * j + 2];
      P[(8 * j + 2 * t + 1) * 32 + r] = acc[4 * j + 1] + acc[4 * j + 3];
    }
  } else {
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");  // the peer's rb is initialised
    uint32_t prv, prb;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(prv) : "r"(rv), "r"(rank ^ 1));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(prb) : "r"(rb), "r"(rank ^ 1));
    const float* R = (const float*)(smraw + (rv - r22_s(smraw)));
    auto epi = [&]<int MY>() {  // compile-time halves: register indices stay static
      constexpr int PJ = (1 - MY) * (T / 16), MJ = MY * (T / 16);
#pragma unroll
      for (int jj = 0; jj < T / 16; jj++) {  // push the peer's half
        const int j = PJ + jj;
        const float v0 = acc[4 * j] + acc[4 * j + 2], v1 = acc[4 * j + 1] + acc[4 * j + 3];
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                     ::"r"(prv + (r * RS + 8 * jj + 2 * t) * 4), "f"(v0), "f"(v1), "r"(prb) : "memory");
      }
      if (PROF && tid == 0) p.prof[cta * 8 + 5] = r22_gt();
      r22_wait(rb, 0);  // the peer's half of my tokens landed
      if (PROF && tid == 0) p.prof[cta * 8 + 6] = r22_gt();
#pragma unroll
      for (int jj = 0; jj < T / 16; jj++) {
        const int j = MJ + jj;
        const float2 q = *(const float2*)(R + r * RS + 8 * jj + 2 * t);
        P[(8 * jj + 2 * t) * 32 + r] = acc[4 * j] + acc[4 * j + 2] + q.x;
        P[(8 * jj + 2 * t + 1) * 32 + r] = acc[4 * j + 1] + acc[4 * j + 3] + q.y;
      }
    };
    if (rank == 0) epi.template operator()<0>();
    else epi.template operator()<1>();
    t0 = (int)rank * (T / 2);
  }
  if (!RED) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (!RED) {
    if (tid == 0) {
      uint32_t f = 0;
      for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(f) : "r"(zf) : "memory");
      if (!f) asm volatile("trap;");
      asm volatile("fence.proxy.async.global;" ::: "memory");
      asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];"
                   ::"l"(reinterpret_cast<uint64_t>(&to)), "r"(rt * 32), "r"(tok0 + t0), "r"(sb) : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
    }
  } else {
    if (tid == 0) {
      uint32_t f = 0;
      for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];" : "=r"(f) : "r"(zf) : "memory");
      if (!f) asm volatile("trap;");
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int mm = (p.m < tok0 + t0 + OT ? p.m : tok0 + t0 + OT) - tok0 - t0;
    for (int v = tid; v < mm * 8; v += 128) {
      const float4 a = *(const float4*)(P + v * 4);
      float* d = p.out + (size_t)(tok0 + t0 + (v >> 3)) * 256 + rt * 32 + (v & 7) * 4;
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(d), "f"(a.x), "f"(a.y), "f"(a.z), "f"(a.w) : "memory");
    }
  }
  if (PROF && tid == 0) p.prof[cta * 8 + 4] = r22_gt();
}

typedef CUresult (*R22Enc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                           const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                           CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static R22Enc r22_enc() {
  static R22Enc e = nullptr;
  if (!e) {
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", (void**)&e, 12000, cudaEnableDefault, &q) != cudaSuccess) e = nullptr;
  }
  return e;
}
template <int T, int NKB, int ST, int FL>
static int r22_go(const R22P& p, const void* x, int64_t stream, int query) {
  constexpr bool C2 = (FL & 2) != 0;
  constexpr int NS = ST < NKB ? ST : NKB, SMEM = NS * (8192 + T * 128) + 1024 + (C2 ? 32 * (T / 2 + 4) * 4 : 0);
  auto k = r22k<T, NKB, ST, FL>;
  if (query) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    return query == 1 ? a.numRegs : query == 2 ? (int)a.localSizeBytes : SMEM;
  }
  if (p.m < 1 || (p.m + T - 1) / T > 8) return -4;
  R22Enc enc = r22_enc();
  if (!enc) return -3;
  CUtensorMap tx, to;
  {
    cuuint64_t dims[2] = {8192, (cuuint64_t)p.m}, str[1] = {8192 * 2};
    cuuint32_t box[2] = {64, (cuuint32_t)T}, es[2] = {1, 1};
    if (enc(&tx, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)x, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -21;
  }
  {
    cuuint64_t dims[2] = {256, (cuuint64_t)p.m}, str[1] = {256 * 4};
    cuuint32_t box[2] = {(FL & 16) ? 8u : 32u, (cuuint32_t)(C2 ? T / 2 : T)}, es[2] = {1, 1};
    if (enc(&to, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)p.out, dims, str, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            (FL & 32) ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return -22;
  }
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(8, 128 / NKB, (p.m + T - 1) / T);
  cfg.blockDim = dim3(160, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = C2 ? 2 : 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, k, p, tx, to);
  return e == cudaSuccess ? 0 : (int)e;
}

int64_t v22_rr22(int64_t v, int64_t W, int64_t x, int64_t out, int64_t cnt, int64_t prof, int64_t m, int64_t stream, int64_t query) {
  R22P p;
  p.W = (const uint8_t*)W; p.out = (float*)out; p.cnt = (int*)cnt; p.prof = (unsigned long long*)prof; p.m = (int)m;
  switch (v) {
    case 0: return r22_go<128, 8, 8, 32>(p, (const void*)x, stream, (int)query);
    default: return -99;
  }
}
'''
V22_RR22_INST = [(128, 8, 8, 32)]  # LDD19: rr22 r22k<T, NKB, ST, FL> instances, case i
V22_ROUTES = {('router', 128): 0}  # LDD19: (layer, m) -> rr22 instance


def v22_pack(w):
    """LDD19: router W fp32 [256, 8192] -> uint8 [8 row tiles][128 k-blocks][8192 B]: per block 64 bf16 rows (j = 16 w + 8 h
    + g: real row 32 rt + 8 w + g, h 0 = hi, 1 = lo) x 64 k, 128-byte swizzled K-major (chunk c of row j at c ^ (j % 8))"""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    S = torch.stack([hi.view(8, 4, 8, 128, 64), lo.view(8, 4, 8, 128, 64)], dim=3)  # rt, w, g, h, kb, c
    S = S.permute(0, 4, 1, 3, 2, 5).reshape(8, 128, 64, 8, 8).contiguous()  # rt, kb, j, chunk, e
    j = torch.arange(64, device=w.device)
    idx = (torch.arange(8, device=w.device)[None, :] ^ (j % 8)[:, None]).view(1, 1, 64, 8, 1).expand(8, 128, 64, 8, 8)
    return torch.gather(S, 3, idx).contiguous().view(torch.uint8).view(8, 128, 8192), hi, lo


def v22_unpack(P):
    """LDD19: independent inverse of v22_pack -> hi, lo bf16 [256, 8192]"""
    S = P.view(torch.bfloat16).view(8, 128, 64, 8, 8)
    j = torch.arange(64, device=P.device)
    idx = (torch.arange(8, device=P.device)[None, :] ^ (j % 8)[:, None]).view(1, 1, 64, 8, 1).expand(8, 128, 64, 8, 8)
    out = torch.empty_like(S)
    out.scatter_(3, idx, S)
    S = out.view(8, 128, 4, 2, 8, 64).permute(0, 2, 4, 3, 1, 5).reshape(256, 2, 8192)
    return S[:, 0].contiguous(), S[:, 1].contiguous()


def v22_build():
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t v22_rr22(' + ', '.join(['int64_t'] * 9) + ');\n'
    return _ldd19_inline('v22_rr22_' + hashlib.sha256(V22_RR22_CU.encode()).hexdigest()[:12], cpp, cuda_sources=V22_RR22_CU,
                       functions=['v22_rr22'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def v22_call(st, vi, x, out):
    return st['fn'](vi, st['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), st['cnt'].data_ptr(), 0, x.shape[0],
                    torch.cuda.current_stream(x.device).cuda_stream, 0)


def v22_verify(st, vi, m, w):
    """LDD19: at m exactly: a zero input over a NaN-filled output (the in-kernel zeroing), every K column as a one-hot row
    (bit-exact against hi + lo in fp32), random rows (normal, then 1% outlier columns x30) with a 16.0 spike row against
    an FP64 reference: overall rel L2 <= 2.5e-5 (judge 1e-4), every row and the spike row <= 2.5e-4; counters back to 0."""
    n, K = w.shape
    dev = w.device
    out = torch.full((m, n), float('nan'), dtype=torch.float32, device=dev)
    rc = v22_call(st, vi, torch.zeros((m, K), dtype=torch.bfloat16, device=dev), out)
    if rc:
        return False, f'launch rc {rc}'
    if not bool((out == 0).all()):
        return False, 'zero input: output not zeroed'
    whl = (st['hi'].float() + st['lo'].float()).t().contiguous()  # exact: hi + lo spans <= 24 bits
    eye = torch.eye(K, dtype=torch.bfloat16, device=dev)
    got = torch.empty((K // m, m, n), dtype=torch.float32, device=dev)
    for i in range(K // m):
        if v22_call(st, vi, eye[i * m:(i + 1) * m], got[i]) != 0:
            return False, 'one-hot launch failed'
    if not torch.equal(got.view(K, n), whl):
        return False, 'one-hot not exact'
    del eye, got, whl
    if int(st['cnt'].abs().sum()):
        return False, 'counters not reset'
    gen = torch.Generator(device=dev).manual_seed(11)
    worst = 0.0
    for dist in (1.0, 30.0):
        x = torch.randn((m, K), generator=gen, device=dev)
        if dist != 1.0:
            x[:, torch.randperm(K, generator=gen, device=dev)[:K // 100]] *= dist
        x[m - 1].zero_()
        x[m - 1, (11 * 131 + m) % K] = 16.0
        x = x.to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        o = torch.empty((m, n), dtype=torch.float32, device=dev)
        if v22_call(st, vi, x, o) != 0:
            return False, 'random launch failed'
        diff = o.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        err = float(diff.norm() / ref.norm())
        worst = max(worst, err)
        if not bool(torch.isfinite(o).all()) or err > 2.5e-5 or float(rows.max()) > 2.5e-4:
            return False, f'random dist {dist}: rel L2 {err:.2e} row max {float(rows.max()):.2e}'
    if int(st['cnt'].abs().sum()):
        return False, 'counters not reset'
    return True, f'ok (rel L2 {worst:.1e})'


def prepare_v22(weights):
    st = {'routes': {}, 'log': []}
    w = weights.get('router')
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or not w.is_cuda:
        return st
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return st
    w = w.contiguous()
    try:
        ext = v22_build()
    except Exception as error:  # noqa: BLE001 - a failed build keeps the v18 router route
        st['log'].append(f'build failed: {type(error).__name__}: {str(error)[-300:]}')
        return st
    wp, hi, lo = v22_pack(w)
    uh, ul = v22_unpack(wp)
    if not (torch.equal(uh.view(torch.int16), hi.view(torch.int16)) and torch.equal(ul.view(torch.int16), lo.view(torch.int16))):
        st['log'].append('rr22 pack round trip failed, v18 route kept')
        return st
    st.update(ext=ext, fn=ext.v22_rr22, wp=wp, hi=hi, lo=lo, dev=w.device,
              cnt=torch.zeros(128, dtype=torch.int32, device=w.device))
    for (layer, m), vi in sorted(V22_ROUTES.items()):
        try:
            ok, msg = v22_verify(st, vi, m, w)
        except RuntimeError as error:  # LDD19: a failed check keeps the v18 route
            ok, msg = False, f'{type(error).__name__}: {str(error)[-200:]}'
        st['log'].append(f'rr22{tuple(V22_RR22_INST[vi])} {layer} m={m}: {msg}' + ('' if ok else ', v18 route kept'))
        if ok:
            st['routes'][(layer, m)] = vi
    del st['hi'], st['lo']
    torch.cuda.synchronize(w.device)
    return st


def run_v22(x, layer, n, st):
    if not LDD19_RR22 or not st or not st.get('routes') or x.ndim != 2:  # LDD19: switch-off fallback
        return None
    vi = st['routes'].get((layer, x.shape[0]))
    if vi is None or n != 256 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != st['dev']:
        return None
    out = torch.empty((x.shape[0], 256), dtype=torch.float32, device=x.device)
    return out if v22_call(st, vi, x, out) == 0 else None


_v22_base_prepare = prepare
_v22_base_kernel = kernel_fn


def prepare(weights: Weights) -> dict:  # LDD19: v22 wrapper (the build ran in v19_jobs' pool as job 'v22')
    state = _v22_base_prepare(weights)
    st = state.get('v22')
    if st:
        for line in st['log']:
            print('v22 ' + line, flush=True)
        for layer, m in st['routes']:  # one warm-up call per new route
            w = weights[layer]
            data = {'x': torch.zeros((m, w.shape[1]), dtype=torch.bfloat16, device=w.device),
                    'config': {'m': m, 'n': w.shape[0], 'k': w.shape[1], 'layer': layer}}
            kernel_fn(data, state)
        torch.cuda.synchronize(weights['qkv_gate'].device)
    return state


def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:  # LDD19: v22 routes ahead of v21's chain
    if (LDD19_RK128 and data['config']['layer'] == 'router' and data['x'].ndim == 2 and data['x'].shape[0] == 128
            and ('router', 128) in ((state.get('ldd19r') or {}).get('routes') or {})):   # LDD19 (sweeper): rk serves it
        return _v22_base_kernel(data, state)
    out = run_v22(data['x'], data['config']['layer'], data['config']['n'], state.get('v22'))
    return _v22_base_kernel(data, state) if out is None else out

# LDD19: selective routes adapted from team keell0renz (Bohdan Agarkov),
# copyright 2026, Apache-2.0, commit 8e30e1be994db4310794b8d04e90d18d895deb84.
# chenwainuo/okc 3d418e1 and vjeux/okc-2026 (Apache-2.0) are upstream sources;
# Lt wrapper/pins credit alygg77 and Matsvei (kernarc matvey).
# Keep LICENSE.keell0renz.txt and NOTICE.keell0renz.txt with this file.
_ldd19_ext_table = {'qkv_gate:8':'chenwainuo', 'qkv_gate:128':'chenwainuo',
    'pooler:128':'chenwainuo', 'unpooler:128':'chenwainuo', 'o:256':'ag'}
# LDD19 (sweeper, iteration 6): qkv_gate m=32 on the backend's rq source with the cursor-escape FL (work_sweeper/probe3.json:
# 11.65 -> 11.34 us standalone); 0 keeps v19's rq<32,5,1,0,0,0,35>
LDD19_RQ32 = 1
# LDD19 (integrator, iteration 6): rq m=128 with FL bit 5 EARLY on the cursor FL (42999971); 0 keeps 42999939
LDD19_RQ128E = 1
if LDD19_RQ32:
    _ldd19_ext_table['qkv_gate:32'] = 'chenwainuo'
def _ldd19_make_chenwainuo():
    r'''Stream-K skinny GEMM: out[m, n] = x @ W[:n].T for m <= 32 (and the router at any m).
W is repacked into mma.sync A-fragment order (2 KB contiguous per warp per 64-wide K chunk); every warp streams an
equal contiguous range of (tile, chunk) units; tiles split across warps meet through fp32 atomics and the last
arriver writes the output. The FP32 router runs as a BF16 hi + lo pair with FP32 accumulation.
v13/v14: up to 32 tokens, the BF16 projections and the router (12-bit hi + lo) run a register-direct GEMV on losslessly 12-bit-packed weights (see GVCU),
launch geometry picked per shape in prepare() by timing against the v12 path.
pw11 ("gvx"): r6-c12 unchanged (every r6 path is still baked and kept as the per-shape baseline "r6"), plus gv geometries
with 64-144 rows per CTA (x fetched once per 128 rows instead of once per 16) and a cross-CTA K split of 2/4/8, reduced
either through cluster DSMEM (path suffix a, r6's own S > 1 epilogue) or through an fp32 workspace + last-arriving CTA
(suffix b, V bit 2, no cluster); suffix -s = x in a shared-memory slab instead of through L1; the router gets the
12-bit codec on both hi and lo. Each new path is validated (one-hot x8 exact, random rows) and timed against the r6
choice in a second bake (BAKE / PICK lines); kernel_fn takes the pw11 pick when it exists, else r6's dispatch.
pw12: the same gv kernel with W streamed through a private per-warp cp.async.bulk smem ring (V bit 3, paths gb-*);
decode, mma, escapes, reduction and epilogue unchanged. PROBE gvbulk lines compare it with r6 on pooler m=1.
pw18 (= pw17 with codec E v3): r6 unchanged (its 12-bit paths, bake and dispatch are SASS-identical) plus codec E paths
(gv FMT 4, "ge-*"). The code length is fixed by a code's first 3 bits (LOP3 + PRMT on an 8-byte table), so the per-lane
bit-position chain has no memory access. The exponent pair value comes from a 256-entry smem LUT (single or lane-replicated)
off the chain. There are 1, 2 or 4 independent substreams per lane, and refill words come through a conflict-free
[substream][slot][lane] 4 B cp.async ring and are prefetched. 11.07 / 11.17 / 11.32 bits/weight (K 1 / 2 / 4) against 12
(proj_rsi/tools/CODEC_E_LAYOUT.md v3). Each pack is proven bit-exact by the reference unpacker before any E kernel runs. Each
path is validated like every second-bake path and timed against r6's pick (BAKE / PICK). PROBE ecod lines time the best E path
against r6 on pooler m=1 and m=16, alternated.
pw29: pw23 plus wgmma router paths at m = 128 / 256 (rwk, its own sm_90a extension), baked against pw23's rs / rp / rv
paths and r6's cuBLASLt path.
pw31 (scored merge): pw27h (P1: flag-zeroed split-K router m = 8/16/32 and unpooler m <= 8 on r6's gv, rr6 at router
m = 1) + pw29h (wgmma split-K router m = 128 / 256 with self-zeroing); the two regions are disjoint (3-way merge on pw23).
pw34 = pw33 + rs128 (audits/m128-plan.md): pw23 unchanged plus a swap-AB register-A split-K wgmma kernel for m exactly 128 / 256 on the
pooler, qkv_gate, unpooler and o layers: 128 W rows x all tokens per CTA, two math warpgroups decode the lossless 12-bit
pack (pw10 _pack_nat + _c12, re-blocked to 12 KB per 128-row k64 stage) into wgmma A registers, x is the smem B operand
(2D TMA, SW128), one full barrier per ring slot shared by a W bulk lane and an x TMA lane, K split over KS CTAs on a
one-wave grid, partials summed by cp.reduce.async.bulk.tensor bf16 add into out, which the kernel zeroes itself (each
CTA its slice, release counter, polled off the critical path; no memset node, no cluster). Built in a background thread
during r6's phase; BAKE / PICK lines time it against r6's own path per (layer, m); any failure leaves r6's choices.'''
    # LDD19 readable edition: of the published file's routes this backend serves only qkv_gate m=8 (gv) and qkv_gate /
    # pooler / unpooler m=128 (rq, "RS128"); v25 always resolved them to the picks pinned below. Its cuBLAS(Lt) tuner,
    # flush / fixed-cost probes, ge / gx / rs / rw / rq32 bakes and their kernels never served a route and are removed.
    import torch
    GVCU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

struct G { const uint4* W; const uint4* W2; const __nv_bfloat16* x; void* out; const uint32_t* esc; float4* ws; int* cnt; int m, N, K, KC; uint32_t T0, T1, U0, U1; };

__device__ __forceinline__ uint4 gnc(const void* p) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p);  // pw16 x load policy (defined below)
__device__ __forceinline__ uint4 gx(const void* p) {
  uint4 v; asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v;
}
template <int XP> __device__ __forceinline__ uint4 gxp(const void* p) {
  uint4 v;
  if constexpr (XP == 1) asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else if constexpr (XP == 2) asm volatile("ld.global.nc.L1::evict_last.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else v = gx(p);
  return v;
}
__device__ __forceinline__ uint4 lds4(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void hmma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// V & 1 == 0: nibble v = offset of value v (byte-permute table T0:T1 = base .. base + 7), signs at bits 3, 11, 19, 27 (values 0-3)
// and 7, 15, 23, 31 (values 4-7): 11 ops / 8 values
__device__ __forceinline__ void dec0(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hl = prmt(T0, T1, cm) | ((C << 4) & 0x80808080u), hh = prmt(T0, T1, cm >> 16) | (C & 0x80808080u);
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
// V & 1 == 1: C byte i: bits 0-2 offset of value i, bit 7 its sign, bits 4-6 offset of value 4 + i, bit 3 the sign of value 4 + (i + 3) % 4,
// so (C & 0x87878787) + base and the same on C rotated by 4 are the high bytes of values 0-3 and 4-7 (window base <= 120)
__device__ __forceinline__ void dec1(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = prmt(L0, hl, 0x5140); a[1] = prmt(L0, hl, 0x7362); a[2] = prmt(L1, hh, 0x5140); a[3] = prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void sto(__nv_bfloat16* d, float v) { *d = __float2bfloat16(v); }
__device__ __forceinline__ void sto(float* d, float v) { *d = v; }
template <typename O>
__device__ __forceinline__ void emit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  sto(o, v.x); sto(o + 8, v.z);
  if (tok + 1 < p.m) { sto(o + p.N, v.y); sto(o + p.N + 8, v.w); }
}
// pw27: split-K partial added into an output the caller zeroed (a memset node in the same graph); fp32 (router) or bf16
__device__ __forceinline__ void reda(float* d, float v) { atomicAdd(d, v); }
__device__ __forceinline__ void reda2(__nv_bfloat16* d, float a, float b) {  // native packed bf16x2 reduction (no return value)
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  asm volatile("red.global.add.noftz.bf16x2 [%0], %1;\n" :: "l"(d), "r"(*(const uint32_t*)&h) : "memory");
}
template <typename O>
__device__ __forceinline__ void redemit(const G& p, int idx, int row0, float4 v, int MT) {
  const int l = idx & 31, r = idx >> 5, n = r % MT, row = row0 + (r / MT) * 16 + (l >> 2), tok = n * 8 + 2 * (l & 3);
  O* o = (O*)p.out + (size_t)tok * p.N + row;
  if constexpr (std::is_same<O, float>::value) {
    reda(o, v.x); reda(o + 8, v.z);
    if (tok + 1 < p.m) { reda(o + p.N, v.y); reda(o + p.N + 8, v.w); }
  } else {  // bf16: rows (g, g+1) of lanes l, l ^ 4 (same tokens, same liveness) as one native packed bf16x2 red.add
    const unsigned am = __activemask();
    const float px = __shfl_xor_sync(am, v.x, 4), py = __shfl_xor_sync(am, v.y, 4), pz = __shfl_xor_sync(am, v.z, 4), pw = __shfl_xor_sync(am, v.w, 4);
    if (((l >> 2) & 1) == 0) {  // even row g: the pair starts at an even column (4-B aligned: row0 % 16 == 0, N even)
      reda2(o, v.x, px); reda2(o + 8, v.z, pz);
      if (tok + 1 < p.m) { reda2(o + p.N, v.y, py); reda2(o + p.N + 8, v.w, pw); }
    }
  }
}
__device__ __forceinline__ void add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }
__device__ __forceinline__ bool live(int idx, int MT, int m) { return ((idx >> 5) % MT) * 8 + 2 * (idx & 3) < m; }

// ---- pw18: codec E v3, routine copied from proj_rsi/tools/codec_e.py CODEC_E_CUH ----
// ---- codec E v3 device decode (proj_rsi/tools/CODEC_E_LAYOUT.md v3) ------------------------------------------------
// K = 1, 2 or 4 independent substreams per lane; substream s owns steps j in [s*4/K, (s+1)*4/K) of every unit.
// Per lane and substream: lo, hi = 64-bit bit buffer (next code at bit 0 of lo); V = valid bits mod 128 (bits 8+ carry
// garbage from V -= len; only (V & 0x60) and (V & 31) are read); pb = bytes taken from the substream's lane stream;
// iss = words issued into the ring; nw = prefetched refill word.
// Code length: the first 3 bits of every code determine its length, so len = byte (lo & 7) of the 8-byte table (t0, t1):
// LOP3 + PRMT. The bit-position chain (LOP3 -> PRMT -> SHF) has no memory access. The value (both exponents) comes from
// lut[lo & 0xff] (LDS, off the chain); A = lop3(e, 0x807f807f, plane bytes, 0xb8) (bits 0-6 and 15 from the plane).
// Stream: per segment, K substream regions of P * 32 words; word q of lane l of substream s at soff + (s * P + q) * 32 + l
// (4 B pieces: coalesced 128 B per warp, padding to a word). Ring per warp and substream: ES = ED / K word slots x 32
// lanes ([s][slot][lane], 128 B per slot: one bank per lane, conflict-free); word q in slot q % ES.
// Per unit: ce_issue for each substream (NI = 4 / K predicated 4 B cp.async), one commit_group, wait_group(2) (the
// prologue issues 3 rounds); then ce_unit. A substream's first refill word of the unit is read after the wait, the
// others are prefetched one step ahead (same landed set, no issue in between).
struct CeS { uint32_t lo, hi, V, pb, iss, nw; };
__device__ __forceinline__ uint32_t ce_lds(uint32_t a) { uint32_t v; asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a)); return v; }
__device__ __forceinline__ uint32_t ce_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ uint32_t ce_sel(uint32_t e, uint32_t d) {  // (e & ~0x807f807f) | (d & 0x807f807f): immLut b ? c : a = 0xb8
  uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xb8;" : "=r"(r) : "r"(e), "r"(0x807f807fu), "r"(d)); return r;
}
// smem address of the word at substream byte pb: rl = ring_s + lane * 4, ring_s aligned to ES * 128
template <int ES>
__device__ __forceinline__ uint32_t ce_addr(uint32_t rl, uint32_t pb) { return rl | ((pb << 5) & ((ES - 1) << 7)); }
// NI predicated 4 B cp.async; gsrc = stream + (soff[gw] + s * P * 32 + lane) * 4 (word 0 of this lane and substream)
template <int ES, int NI>
__device__ __forceinline__ void ce_issue(CeS& b, uint32_t P, const uint8_t* gsrc, uint32_t rl) {
#pragma unroll
  for (int k = 0; k < NI; k++) {
    const bool go = b.iss < P && b.iss < (b.pb >> 2) + ES;
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n @p cp.async.ca.shared.global [%0], [%1], 4; }\n"
                 :: "r"(rl | ((b.iss % ES) << 7)), "l"(gsrc + (size_t)b.iss * 128), "r"((uint32_t)go));
    b.iss += go;
  }
}
__device__ __forceinline__ void ce_refill(CeS& b, uint32_t w) {
  const bool need = (b.V & 0x60u) == 0u;
  w = need ? w : 0u;
  const uint32_t at = b.V & 31u;
  b.lo |= w << at; b.hi |= __funnelshift_l(w, 0u, at);
  b.V += need ? 32u : 0u; b.pb += need ? 4u : 0u;
}
// one unit: the K substreams' chains interleaved (independent), 16 A registers in r6 order A[4 j + i];
// rl = ring of substream 0 + lane * 4 (substream s at + s * ES * 128); LS = LUT index shift (2 single, 7 lane-replicated)
template <int ES, int LS, int K>
__device__ __forceinline__ void ce_unit(CeS* b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint4 q0, uint4 q1, uint32_t* A) {
  constexpr int NS = 4 / K;
  const uint32_t L[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
  for (int t = 0; t < NS; t++) {
#pragma unroll
    for (int s = 0; s < K; s++) {
      const uint32_t rs = rl + s * ES * 128;
      ce_refill(b[s], t == 0 ? ce_lds(ce_addr<ES>(rs, b[s].pb)) : b[s].nw);
      if (t + 1 < NS) b[s].nw = ce_lds(ce_addr<ES>(rs, b[s].pb));
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int s = 0; s < K; s++) {
        const int j = s * NS + t;
        const uint32_t len = ce_prmt(t0, t1, b[s].lo & 7u);
        const uint32_t e = ce_lds(lut + ((b[s].lo & 0xffu) << LS));
        b[s].lo = __funnelshift_r(b[s].lo, b[s].hi, len); b[s].hi = __funnelshift_r(b[s].hi, 0u, len); b[s].V -= len;
        A[4 * j + i] = ce_sel(e, ce_prmt(L[2 * j + (i >> 1)], 0u, (i & 1) ? 0x3322u : 0x1100u));
      }
  }
}
// escapes (r6 walk, unchanged except the XOR field): A[r] ^= (nx & 0xff) << (7 + 16 * ((nx >> 10) & 1))
// pw20: one step t (4 A registers) of ce_unit<ES, LS, 1>, identical operations in identical order (refill, prefetch for
// step t + 1, 4 pairs), so a caller can decode and consume a unit step by step with 4 A registers live instead of 16
template <int ES, int LS>
__device__ __forceinline__ void ce_step1(CeS& b, uint32_t lut, uint32_t rl, uint32_t t0, uint32_t t1, uint32_t Lw0, uint32_t Lw1, int t, uint32_t* A) {
  ce_refill(b, t == 0 ? ce_lds(ce_addr<ES>(rl, b.pb)) : b.nw);
  if (t < 3) b.nw = ce_lds(ce_addr<ES>(rl, b.pb));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const uint32_t len = ce_prmt(t0, t1, b.lo & 7u);
    const uint32_t e = ce_lds(lut + ((b.lo & 0xffu) << LS));
    b.lo = __funnelshift_r(b.lo, b.hi, len); b.hi = __funnelshift_r(b.hi, 0u, len); b.V -= len;
    A[i] = ce_sel(e, ce_prmt((i >> 1) ? Lw1 : Lw0, 0u, (i & 1) ? 0x3322u : 0x1100u));
  }
}

// pw12: per-warp private bulk-copy ring for W (V bit 3). V bits 4-5: slots NS = 2 << code; bits 6-7: chunks per slot
// CU = 1 << code; bit 8: refill without fence.proxy.async (the slot's values are consumed by the decode, in program
// order, before the __syncwarp and the refill; CUTLASS-style release). Lane 0 of each warp fills its own slots with cp.async.bulk (one copy per row tile and matrix, CU
// contiguous 64-k chunks) on a per-slot mbarrier; the warp waits on it, reads its fragments with ld.shared.v4 at the
// offsets r6's ld.global.v4 used, and lane 0 refills the slot after the warp has decoded its last chunk.
template <int FMT, int MT, int WR, int WK, int KU, int S, int RT, int XS, int V>
struct GL {
  static constexpr int NL = FMT == 4 ? 2 : ((FMT & 1) ? 3 : 4), NA = (FMT == 2 || FMT == 3) ? 2 : 1, NW = WR * WK, NI = WR * RT * MT * 32;
  // pw18 codec E v3 (FMT 4): V bits 22-23 = log2 substreams CK (1/2/4); bit 19 = ring 32 words per lane (else 16), split
  // into ES = CED / CK slots per substream; bit 20 = LUT replicated per lane (32 KB, conflict-free); smem = per-warp ring
  // (2048-aligned) + LUT, after the slab / reduction buffers
  static constexpr int CK = 1 << ((V >> 22) & 3), CED = ((V >> 19) & 1) ? 32 : 16, CES = CED / CK;
  static constexpr int CLUT = FMT == 4 ? (((V >> 20) & 1) ? 32768 : 1024) : 0;
  static constexpr int CAL = CES * 128 > 2048 ? CES * 128 : 2048;  // ring alignment (ring_s | slot offset)
  static constexpr int CRING = FMT == 4 ? NW * 32 * CED * 4 + CAL : 0;  // + CAL: alignment slack
  static constexpr bool BK = (V & 8) != 0;
  static constexpr int CU = BK ? 1 << ((V >> 6) & 3) : 1, G = KU / CU, NS0 = BK ? 2 << ((V >> 4) & 3) : 1;
  static constexpr int NS = NS0 < G ? NS0 : G, SB = RT * NA * CU * NL * 512;
  static constexpr int SLAB = XS == 1 ? WK * KU * MT * 8 * 128 : 0, REDB = (WK > 1 || S > 1) ? (WK + 1) * NI * 16 : 0;
  static constexpr int BAR = BK ? ((NW * NS * 8 + 127) / 128) * 128 : 0, RING = BK ? NW * NS * SB : 0;
  static constexpr int SMEM = SLAB + REDB + BAR + RING + CLUT + CRING;
};
__device__ __forceinline__ void gxwait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nGXW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra GXW;\n}\n" :: "r"(b), "r"(ph) : "memory");
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv(G p) {
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {  // pw20 lite (V bit 24): K = 1, decode and consume one k16 step at a time
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "pw20 lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // pw18: one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router m=128/256 (V bit 25): the decoded W chunk (hi + lo) in registers
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          if (lane == (int)((nx >> 15) & 31)) {
            const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = CE ? (nx & 0xffu) << (7 + ((nx >> 10) & 1) * 16) : (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
            for (int h = 0; h < NA; h++)
#pragma unroll
              for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
          }
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}

// Predicated format1 repair; source inputs are copied before any output write.
__device__ __forceinline__ void gv_fix_pred(uint32_t (&a)[16],uint32_t r,uint32_t xr,int lane,uint32_t target,uint32_t hm){
  asm volatile("{ .reg .b32 ri, mask; .reg .pred pl, pa, pr; mov.b32 ri, %16; mov.b32 mask, %17; setp.eq.u32 pl, %18, %19; setp.eq.and.u32 pa, %20, 0, pl; setp.eq.and.u32 pr, ri, 0, pa; @pr xor.b32 %0, %0, mask; setp.eq.and.u32 pr, ri, 1, pa; @pr xor.b32 %1, %1, mask; setp.eq.and.u32 pr, ri, 2, pa; @pr xor.b32 %2, %2, mask; setp.eq.and.u32 pr, ri, 3, pa; @pr xor.b32 %3, %3, mask; setp.eq.and.u32 pr, ri, 4, pa; @pr xor.b32 %4, %4, mask; setp.eq.and.u32 pr, ri, 5, pa; @pr xor.b32 %5, %5, mask; setp.eq.and.u32 pr, ri, 6, pa; @pr xor.b32 %6, %6, mask; setp.eq.and.u32 pr, ri, 7, pa; @pr xor.b32 %7, %7, mask; setp.eq.and.u32 pr, ri, 8, pa; @pr xor.b32 %8, %8, mask; setp.eq.and.u32 pr, ri, 9, pa; @pr xor.b32 %9, %9, mask; setp.eq.and.u32 pr, ri, 10, pa; @pr xor.b32 %10, %10, mask; setp.eq.and.u32 pr, ri, 11, pa; @pr xor.b32 %11, %11, mask; setp.eq.and.u32 pr, ri, 12, pa; @pr xor.b32 %12, %12, mask; setp.eq.and.u32 pr, ri, 13, pa; @pr xor.b32 %13, %13, mask; setp.eq.and.u32 pr, ri, 14, pa; @pr xor.b32 %14, %14, mask; setp.eq.and.u32 pr, ri, 15, pa; @pr xor.b32 %15, %15, mask; }" : "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7]), "+r"(a[8]), "+r"(a[9]), "+r"(a[10]), "+r"(a[11]), "+r"(a[12]), "+r"(a[13]), "+r"(a[14]), "+r"(a[15]) : "r"(r),"r"(xr),"r"(lane),"r"(target),"r"(hm));
}

template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
__global__ void __launch_bounds__(WR * WK * 32, ((V >> 12) & 7) ? ((V >> 12) & 7) : (XS == 2 && WR * WK * 32 < 512 ? 512 / (WR * WK * 32) : 1)) gv_predxor(G p) {
  static_assert((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1), "GVpredxor only seven reviewed format1 templates");
  // pw14 probes only (never registered as paths): V bits 9-11 = ladder level LAD (1: W loads only, 2: + x loads,
  // 3: + decode, 4: + mma; no reduction / store); bits 12-14 = __launch_bounds__ minBlocks override (register cap).
  constexpr int LAD = (V >> 9) & 7;
  constexpr bool LX = LAD != 1, LDEC = LAD == 0 || LAD >= 3, LMMA = LAD == 0 || LAD == 4;
  uint32_t sk = 0;
  constexpr bool C12 = FMT & 1, HL = FMT == 2 || FMT == 3, CE = FMT == 4;
  constexpr int NL = CE ? 2 : (C12 ? 3 : 4), NA = HL ? 2 : 1, NW = WR * WK, NT = NW * 32, NI = WR * RT * MT * 32, CH = WK * KU, TK = MT * 8;
  constexpr int NC = (MT * RT * NA >= 2 || (CE && ((V >> 24) & 1))) ? 1 : 2, XB = XS ? 1 : ST;  // pw20 lite: one accumulator set (register budget)  // XS 2: x straight from global at use (L1), no ring
  using O = typename std::conditional<HL, float, __nv_bfloat16>::type;
  extern __shared__ __align__(16) uint8_t smem[];
  float4* red = (float4*)(smem + (XS == 1 ? CH * TK * 128 : 0));
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, wr = w % WR, wk = w / WR, g = lane >> 2, t = lane & 3;
  constexpr bool XZ = ((V >> 29) & 1) != 0;  // P1 (V bit 29): grid.z plane 0 = zeroer CTAs (one per tile), the K splits follow
  const int bz = blockIdx.x, bx = blockIdx.y, by = (int)blockIdx.z - (XZ ? 1 : 0), GX = gridDim.y;
  if constexpr (XZ) {
    if (by < 0) {  // dispatched first (lowest z): zero this tile of the output, publish, exit; no W, no x
      O* o = (O*)p.out + (size_t)bz * TK * p.N + bx * WR * RT * 16;
      const int R = WR * RT * 16, mm = min(TK, p.m - bz * TK);
      for (int i = threadIdx.x; i < mm * R; i += WR * WK * 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
      __threadfence(); __syncthreads();
      if (threadIdx.x == 0) atomicAdd(p.cnt + 2 * (bz * GX + bx) + 1, 1);
      return;
    }
  }
  p.x += (size_t)bz * TK * p.K; p.out = (uint8_t*)p.out + (size_t)bz * TK * p.N * sizeof(O); p.m = min(TK, p.m - bz * TK);
  const int cb = by * CH, c0 = cb + wk * KU, row0 = bx * WR * RT * 16;
  const size_t wo = ((size_t)(bx * WR + wr) * RT * p.KC + c0) * (NL * 32) + lane, js = (size_t)p.KC * NL * 32;
  const __nv_bfloat16* xp[MT];
#pragma unroll
  for (int n = 0; n < MT; n++) xp[n] = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + c0 * 64 + t * 16;
  uint4 wb[ST][RT][NA][NL], xb[XB][MT][2];
  using L = GL<FMT, MT, WR, WK, KU, S, RT, XS, V>;
  static_assert(KU % L::CU == 0, "chunks per slot must divide KU");
  auto ld = [&](int s, int u) {
    if constexpr (!L::BK) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
#pragma unroll
        for (int i = 0; i < NL; i++) wb[s][j][h][i] = gnc((h ? p.W2 : p.W) + wo + j * js + (u * NL + i) * 32);
    }
    if (!XS && LX) {
#pragma unroll
      for (int n = 0; n < MT; n++) { xb[s % XB][n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xb[s % XB][n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
  };
  uint32_t bbar = 0, bring = 0;
  auto bissue = [&](int slot, int grp) {  // lane 0 only: fill one slot with slot-group grp (CU chunks per row tile / matrix)
    const uint32_t bar = bbar + slot * 8;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "n"(L::SB) : "memory");
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                     :: "r"(bring + slot * L::SB + (j * NA + h) * L::CU * NL * 512),
                        "l"((h ? p.W2 : p.W) + (wo - lane) + j * js + (size_t)grp * L::CU * NL * 32), "n"(L::CU * NL * 512), "r"(bar) : "memory");
  };
  if constexpr (L::BK) {
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    bbar = s0 + w * L::NS * 8;
    bring = s0 + L::BAR + w * L::NS * L::SB;
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < L::NS; k++) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bbar + k * 8));
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#pragma unroll
      for (int k = 0; k < L::NS; k++) bissue(k, k);
    }
    __syncwarp();
  }
  int tk = 0;
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // P1 (V bit 27): no memset node; the split-0 CTA of the tile zeroes it, then raises a flag
    if (w == 0) {
      const int tile = bz * GX + bx, R = WR * RT * 16;
      if (lane == 0) tk = atomicAdd(p.cnt + 2 * tile, 1);
      if (by == 0 && !XZ) {
        O* o = (O*)p.out + row0;
        for (int i = lane; i < p.m * R; i += 32) sto(o + (size_t)(i / R) * p.N + i % R, 0.f);
        __threadfence(); __syncwarp();
        if (lane == 0) atomicAdd(p.cnt + 2 * tile + 1, 1);
      }
    }
  }
#pragma unroll
  for (int s = 0; s < ST; s++) if (s < KU) ld(s, s);
  CeS ceb[CE ? L::CK : 1];
  uint32_t cP = 0u, crl = 0u, clut = 0u, ct0 = 0u, ct1 = 0u;
  const uint8_t* cgs = nullptr;
  if constexpr (CE) {  // pw18: segment start, ring prologue (3 issue rounds, one commit group each), LUT + length table
#pragma unroll
    for (int s = 0; s < L::CK; s++) ceb[s] = CeS{0u, 0u, 0u, 0u, 0u, 0u};
    const uint32_t gw = (uint32_t)((by * GX + bx) * NW + w);
    const uint32_t* soff = (const uint32_t*)(((uint64_t)p.U1 << 32) | p.U0);
    const uint32_t s0 = __ldg(soff + gw);
    cP = (__ldg(soff + gw + 1) - s0) / (32u * L::CK);
    cgs = (const uint8_t*)p.W2 + ((size_t)s0 + lane) * 4;  // word 0 of this lane, substream 0 (substream s: + s * P * 128 B)
    const uint32_t cs00 = (uint32_t)__cvta_generic_to_shared(smem) + L::SLAB + L::REDB;
    const uint32_t cs0 = (cs00 + L::CAL - 1u) & ~(uint32_t)(L::CAL - 1);  // ring_s | slot offset needs ES * 128 alignment
    crl = cs0 + (w * 32 * L::CED + lane) * 4;
    clut = cs0 + NW * 32 * L::CED * 4 + (((V >> 20) & 1) ? lane * 4 : 0);
    const uint32_t* gl = (const uint32_t*)(((uint64_t)p.T1 << 32) | p.T0);
    ct0 = __ldg(gl + 256); ct1 = __ldg(gl + 257);
#pragma unroll
    for (int k = 0; k < 3; k++) {
#pragma unroll
      for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    uint32_t* sl = (uint32_t*)(smem + L::SLAB + L::REDB + (cs0 - cs00) + NW * 32 * L::CED * 4);
    constexpr int NE = (256 + NT - 1) / NT;  // LUT entries per thread: all loads in flight at once, then the stores
    uint32_t lv[NE];
#pragma unroll
    for (int k = 0; k < NE; k++) lv[k] = threadIdx.x + k * NT < 256 ? __ldg(gl + threadIdx.x + k * NT) : 0u;
#pragma unroll
    for (int k = 0; k < NE; k++) {
      const int x = threadIdx.x + k * NT;
      if (x < 256) {
        if constexpr (((V >> 20) & 1) != 0) {  // entry x for lane l at word 32 x + l: 8 x 16 B per entry
#pragma unroll
          for (int q = 0; q < 8; q++) ((uint4*)(sl + x * 32))[q] = make_uint4(lv[k], lv[k], lv[k], lv[k]);
        } else sl[x] = lv[k];
      }
    }
    __syncthreads();
  }
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem);
  if (XS == 1) {
    for (int e = threadIdx.x; e < CH * TK * 8; e += NT) {
      const int s8 = e & 7, tok = (e >> 3) % TK, ch = e / (TK * 8);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sb + ((ch * TK + tok) * 8 + (s8 ^ (tok & 7))) * 16),
                   "l"(p.x + (size_t)min(tok, p.m - 1) * p.K + (cb + ch) * 64 + s8 * 8));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  }
  uint32_t e = ~0u, nx = ~0u;
  int ep = 0;
  if ((C12 || CE) && LDEC) { e = p.esc[((size_t)(by * GX + bx) * NW + w) * 32 + lane]; nx = __shfl_sync(~0u, e, 0); }
  if (XS == 1) __syncthreads();
  float acc[RT][MT][NC][4];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++)
#pragma unroll
      for (int c = 0; c < NC; c++) acc[j][n][c][0] = acc[j][n][c][1] = acc[j][n][c][2] = acc[j][n][c][3] = 0.f;
  auto step = [&](const int u, const int s) {
    if constexpr (CE && ((V >> 24) & 1)) {  // pw20 lite (V bit 24): K = 1, decode and consume one k16 step at a time
      static_assert(RT == 1 && NA == 1 && XS != 1 && L::CK == 1, "pw20 lite path: RT 1, one substream, x not in a slab");
      ce_issue<L::CES, 4>(ceb[0], cP, cgs, crl);
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 2;\n" ::: "memory");
      const uint4 q0 = wb[s][0][0][0], q1 = wb[s][0][0][1];
      uint32_t B[MT][8];
#pragma unroll
      for (int n = 0; n < MT; n++) {
        uint4 b0, b1;
        if (XS == 2) { b0 = gxp<(V >> 15) & 3>(xp[n] + u * 64); b1 = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
        else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
        B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
      }
      if (u + ST < KU) ld(s, u + ST);
      const uint32_t Lw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int jj = 0; jj < 4; jj++) {
        uint32_t A4[4];
        ce_step1<L::CES, ((V >> 20) & 1) ? 7 : 2>(ceb[0], clut, crl, ct0, ct1, Lw[2 * jj], Lw[2 * jj + 1], jj, A4);
        uint32_t n3 = nx;  // this unit's escape run (warp-uniform), applied to step jj's registers
        int e3 = ep;
        while ((n3 >> 20) == (uint32_t)u) {
          if (lane == (int)((n3 >> 15) & 31) && (int)((n3 >> 13) & 3) == jj) {
            const uint32_t r = (n3 >> 11) & 3, xr = (n3 & 0xffu) << (7 + ((n3 >> 10) & 1) * 16);
#pragma unroll
            for (int i = 0; i < 4; i++) if (i == (int)r) A4[i] ^= xr;
          }
          n3 = ++e3 < 32 ? __shfl_sync(~0u, e, e3) : ~0u;
        }
        if (jj == 3) { nx = n3; ep = e3; }
#pragma unroll
        for (int n = 0; n < MT; n++) hmma(acc[0][n][jj % NC], A4, B[n][2 * jj], B[n][2 * jj + 1]);
      }
      return;
    }
    uint32_t A[RT][NA][16], B[MT][8];
    uint4 xd[XS == 2 ? MT : 1][2];
    if (XS == 2 && LX && !((V >> 25) & 1)) {  // pw21: the router token loop loads x at use
#pragma unroll
      for (int n = 0; n < MT; n++) { xd[n][0] = gxp<(V >> 15) & 3>(xp[n] + u * 64); xd[n][1] = gxp<(V >> 15) & 3>(xp[n] + u * 64 + 8); }
    }
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int h = 0; h < NA; h++) {
        uint4 qv[NL];
        if constexpr (L::BK) {
          const int grp = u / L::CU, ci = u % L::CU, slot = grp % L::NS;
          if (ci == 0 && j == 0 && h == 0) gxwait(bbar + slot * 8, (grp / L::NS) & 1);
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = lds4(bring + slot * L::SB + ((j * NA + h) * L::CU + ci) * NL * 512 + i * 512 + lane * 16);
        } else {
#pragma unroll
          for (int i = 0; i < NL; i++) qv[i] = wb[s][j][h][i];
        }
        const uint4* q = qv;
        if constexpr (!LDEC) {  // ladder L0/L1: consume the W words with a cheap XOR, no decode
#pragma unroll
          for (int i = 0; i < NL; i++) sk ^= q[i].x ^ q[i].y ^ q[i].z ^ q[i].w;
        } else if constexpr (CE) {  // pw18: one issue round + commit per unit (u, j); the words this unit reads have landed
#pragma unroll
          for (int s = 0; s < L::CK; s++) ce_issue<L::CES, 4 / L::CK>(ceb[s], cP, cgs + (size_t)s * cP * 128, crl + s * L::CES * 128);
          asm volatile("cp.async.commit_group;\n" ::: "memory");
          asm volatile("cp.async.wait_group 2;\n" ::: "memory");
          ce_unit<L::CES, ((V >> 20) & 1) ? 7 : 2, L::CK>(ceb, clut, crl, ct0, ct1, q[0], q[1], A[j][h]);
        } else if (C12) {
          const uint32_t T0 = h ? p.U0 : p.T0, T1 = h ? p.U1 : p.T1;
          if (V & 1) {
            dec1(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec1(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec1(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec1(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          } else {
            dec0(q[0].x, q[0].y, q[2].x, T0, T1, A[j][h]); dec0(q[0].z, q[0].w, q[2].y, T0, T1, A[j][h] + 4);
            dec0(q[1].x, q[1].y, q[2].z, T0, T1, A[j][h] + 8); dec0(q[1].z, q[1].w, q[2].w, T0, T1, A[j][h] + 12);
          }
        } else {
#pragma unroll
          for (int i = 0; i < 4; i++) { A[j][h][4 * i] = q[i].x; A[j][h][4 * i + 1] = q[i].y; A[j][h][4 * i + 2] = q[i].z; A[j][h][4 * i + 3] = q[i].w; }
        }
      }
    if constexpr (((V >> 25) & 1) != 0) {  // pw21 router m=128/256 (V bit 25): the decoded W chunk (hi + lo) in registers
      // multiplies every n8 token tile of the CTA in turn; tile n's x fragment is loaded at use (ld.global.nc, L1 shared
      // by the CTA's row-tile warps); one accumulator per (row tile, token tile)
      static_assert(XS == 2 && !L::BK, "token loop: x at use, register W ring");
      if (u + ST < KU) ld(s, u + ST);
      if ((C12 || CE) && LDEC) {
#pragma unroll
        for (int j = 0; j < RT; j++)
          while ((nx >> 20) == (uint32_t)(u * RT + j)) {
            if (lane == (int)((nx >> 15) & 31)) {
              const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
#pragma unroll
              for (int h = 0; h < NA; h++)
#pragma unroll
                for (int i = 0; i < 16; i++) if (i == (int)r && h == (int)hm) A[j][h][i] ^= xr;
            }
            nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
          }
      }
#pragma unroll
      for (int n = 0; n < MT; n++) {
        const __nv_bfloat16* xq = p.x + (size_t)min(n * 8 + g, p.m - 1) * p.K + (c0 + u) * 64 + t * 16;
        const uint4 b0 = gx(xq), b1 = gx(xq + 8);
        const uint32_t Bq[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int q = 0; q < 4; q++)
#pragma unroll
          for (int j = 0; j < RT; j++)
#pragma unroll
            for (int h = 0; h < NA; h++) hmma(acc[j][n][0], A[j][h] + 4 * q, Bq[2 * q], Bq[2 * q + 1]);
      }
      return;
    }
    if constexpr (L::BK) {  // the slot's last chunk is decoded into registers: refill it with the group NS ahead
      const int grp = u / L::CU;
      if (u % L::CU == L::CU - 1 && grp + L::NS < L::G) {
        __syncwarp();
        if (lane == 0) {
          if constexpr (!(V & 256)) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");  // V bit 8: no proxy fence
          bissue(grp % L::NS, grp + L::NS);
        }
      }
    }
#pragma unroll
    for (int n = 0; n < MT; n++) {
      uint4 b0, b1;
      if (XS == 1) {
        const uint32_t a = sb + ((wk * KU + u) * TK + n * 8 + g) * 128;
        b0 = lds4(a + (((2 * t) ^ g) << 4)); b1 = lds4(a + (((2 * t + 1) ^ g) << 4));
      } else if (XS == 2) { b0 = xd[n % (XS == 2 ? MT : 1)][0]; b1 = xd[n % (XS == 2 ? MT : 1)][1]; }
      else { b0 = xb[s % XB][n][0]; b1 = xb[s % XB][n][1]; }
      B[n][0] = b0.x; B[n][1] = b0.y; B[n][2] = b0.z; B[n][3] = b0.w; B[n][4] = b1.x; B[n][5] = b1.y; B[n][6] = b1.z; B[n][7] = b1.w;
    }
    if constexpr (LAD == 2 || LAD == 3) {  // ladder L1/L2: consume x (and the decoded A) instead of the mma
#pragma unroll
      for (int n = 0; n < MT; n++)
#pragma unroll
        for (int i = 0; i < 8; i++) sk ^= B[n][i];
      if constexpr (LAD == 3) {
#pragma unroll
        for (int j = 0; j < RT; j++)
#pragma unroll
          for (int h = 0; h < NA; h++)
#pragma unroll
            for (int i = 0; i < 16; i++) sk ^= A[j][h][i];
      }
    }
    if (u + ST < KU) ld(s, u + ST);
    if ((C12 || CE) && LDEC) {
#pragma unroll
      for (int j = 0; j < RT; j++)
        while ((nx >> 20) == (uint32_t)(u * RT + j)) {
          const uint32_t r = (nx >> 11) & 15, hm = (nx >> 9) & 1, xr = (nx & 0x7fu) << (8 + ((nx >> 10) & 1) * 16);
          gv_fix_pred(A[j][0],r,xr,lane,(nx>>15)&31,hm);
          nx = ++ep < 32 ? __shfl_sync(~0u, e, ep) : ~0u;
        }
    }
    if constexpr (LMMA) {
#pragma unroll
    for (int q = 0; q < 4; q++)
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int h = 0; h < NA; h++) hmma(acc[j][n][(q * NA + h) % NC], A[j][h] + 4 * q, B[n][2 * q], B[n][2 * q + 1]);
    }
  };
  if (V & 2) {  // rolled: small code (cold instruction fetch), no cross-group load hoisting
#pragma unroll 1
    for (int u0 = 0; u0 < KU; u0 += ST)
#pragma unroll
      for (int s = 0; s < ST; s++) if (u0 + s < KU) step(u0 + s, s);
  } else {
#pragma unroll
    for (int u = 0; u < KU; u++) step(u, u % ST);
  }
  if constexpr (LAD != 0) {  // ladder: keep the work alive with a store that never happens in practice, then exit
    if constexpr (LAD == 4) {
#pragma unroll
      for (int j = 0; j < RT; j++)
#pragma unroll
        for (int n = 0; n < MT; n++)
#pragma unroll
          for (int c = 0; c < NC; c++) sk ^= __float_as_uint(acc[j][n][c][0]) ^ __float_as_uint(acc[j][n][c][1]) ^ __float_as_uint(acc[j][n][c][2]) ^ __float_as_uint(acc[j][n][c][3]);
    }
    if (sk == (p.T0 ^ 0x9e3779b9u)) ((float*)p.ws)[threadIdx.x] = __uint_as_float(sk);
    return;
  }
  float4 f[RT][MT];
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) {
      f[j][n] = make_float4(acc[j][n][0][0], acc[j][n][0][1], acc[j][n][0][2], acc[j][n][0][3]);
#pragma unroll
      for (int c = 1; c < NC; c++) add4(f[j][n], make_float4(acc[j][n][c][0], acc[j][n][c][1], acc[j][n][c][2], acc[j][n][c][3]));
    }
  if (WK == 1 && S == 1) {
#pragma unroll
    for (int j = 0; j < RT; j++)
#pragma unroll
      for (int n = 0; n < MT; n++) if (n * 8 + 2 * t < p.m) emit<O>(p, ((wr * RT + j) * MT + n) * 32 + lane, row0, f[j][n], MT);
    return;
  }
  float4* cs = red + WK * NI;
#pragma unroll
  for (int j = 0; j < RT; j++)
#pragma unroll
    for (int n = 0; n < MT; n++) red[wk * NI + ((wr * RT + j) * MT + n) * 32 + lane] = f[j][n];
  if constexpr (((V >> 27) & 1) != 0 || XZ) {  // acquire-poll the tile's zero flag (H100 probe: 0.35 us under volatile + fence)
    if (threadIdx.x == 0) {
      const int* fl = p.cnt + 2 * (bz * GX + bx) + 1, tgt = tk / S + 1;
      unsigned spins_ = 0;  // critic (pw27h): bounded wait; a stuck zeroer aborts visibly instead of hanging to the timeout
      while (true) { int v_; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v_) : "l"(fl) : "memory"); if (v_ >= tgt) break; if (++spins_ > (1u << 24)) asm volatile("trap;"); }
    }
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (((V >> 28) & 1) != 0) asm volatile("griddepcontrol.wait;\n" ::: "memory");  // P1 (V bit 28): PDL zero kernel done
#endif
  __syncthreads();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = red[idx];
#pragma unroll
    for (int q = 1; q < WK; q++) add4(v, red[q * NI + idx]);
    if (S == 1) emit<O>(p, idx, row0, v, MT);
    else if constexpr (((V >> 26) & 15) != 0) redemit<O>(p, idx, row0, v, MT);  // pw27: K split reduced by red.add, no cluster
    else {
      cs[idx] = v;
      if constexpr ((V & 4) != 0) __stcg(p.ws + ((size_t)(bz * GX + bx) * S + by) * NI + idx, v);  // pw11 (B): partial to its ws slot
    }
  }
  if (S == 1) return;
  if constexpr (((V >> 26) & 15) != 0) return;
  if constexpr ((V & 4) != 0) {  // pw11 (B): no cluster; the last-arriving CTA of the n-block sums the KS partials from L2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // release this thread's partial stores (lighter than __threadfence's fence.sc)
    __syncthreads();
    __shared__ int gxlast;
    if (threadIdx.x == 0) gxlast = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
    __syncthreads();
    if (!gxlast) return;
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");  // acquire the other CTAs' partials
    const float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
    for (int idx = threadIdx.x; idx < NI; idx += NT) {
      if (!live(idx, MT, p.m)) continue;
      float4 v = cs[idx];
#pragma unroll
      for (int q = 0; q < S; q++) if (q != by) add4(v, __ldcg(wsp + q * NI + idx));
      emit<O>(p, idx, row0, v, MT);
    }
    if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;  // self-cleaning for graph replay
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
  constexpr int PER = (NI + S - 1) / S;
  const uint32_t ca = (uint32_t)__cvta_generic_to_shared(cs);
  for (int idx = rk * PER + threadIdx.x; idx < min(NI, (int)(rk + 1) * PER); idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v[S];
#pragma unroll
    for (int q = 0; q < S; q++) {
      uint32_t ra; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(ca + idx * 16), "r"(q));
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];\n" : "=f"(v[q].x), "=f"(v[q].y), "=f"(v[q].z), "=f"(v[q].w) : "r"(ra) : "memory");
    }
#pragma unroll
    for (int q = 1; q < S; q++) add4(v[0], v[q]);
    emit<O>(p, idx, row0, v[0], MT);
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  float4* wsp = p.ws + (size_t)(bz * GX + bx) * S * NI;
  for (int idx = threadIdx.x; idx < NI; idx += NT) wsp[by * NI + idx] = cs[idx];
  __threadfence();
  __syncthreads();
  __shared__ int last;
  if (threadIdx.x == 0) last = atomicAdd(p.cnt + bz * GX + bx, 1) == S - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int idx = threadIdx.x; idx < NI; idx += NT) {
    if (!live(idx, MT, p.m)) continue;
    float4 v = __ldcg(wsp + idx);
    for (int q = 1; q < S; q++) add4(v, __ldcg(wsp + q * NI + idx));
    emit<O>(p, idx, row0, v, MT);
  }
  if (threadIdx.x == 0) p.cnt[bz * GX + bx] = 0;
#endif
  }
}


__global__ void p1zero(uint4* o, int n16) {  // P1 (V bit 28): the PDL primary, lets the gv grid launch at once
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n");
#endif
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) o[i] = make_uint4(0u, 0u, 0u, 0u);
}
template <int FMT, int MT, int ST, int WR, int WK, int KU, int S, int RT, int XS, int V>
static int64_t gvl(const G& p, cudaStream_t st, int occ) {
  constexpr int NI = WR * RT * MT * 32;
  auto k = gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  if constexpr ((FMT == 1 && MT == 1 && ST == 1 && WR == 2 && WK == 16 && KU == 8 && S == 1 && RT == 1 && XS == 0 && V == 3) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 0) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 2 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1) ||
      (FMT == 1 && MT == 2 && ST == 2 && WR == 1 && WK == 4 && KU == 16 && S == 1 && RT == 1 && XS == 2 && V == 1) ||
      (FMT == 1 && MT == 1 && ST == 2 && WR == 1 && WK == 8 && KU == 16 && S == 1 && RT == 1 && XS == 0 && V == 1)) {
    const bool pooler = p.N == 4096 && p.K == 8192 && ((MT == 1 && (p.m == 1 || p.m == 8)) || (MT == 2 && p.m == 16));
    const bool unpooler = MT == 2 && p.N == 8192 && p.K == 4096 && p.m == 16;
    if (pooler || unpooler) k = gv_predxor<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>;
  }
  if (p.N % (16 * WR * RT) || p.K != S * WK * KU * 64 || p.m < 1) return -1;
  const size_t sm = (XS == 1 ? (size_t)WK * KU * MT * 8 * 128 : 0) + ((WK > 1 || S > 1) ? (size_t)(WK + 1) * NI * 16 : 0)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::BAR + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::RING)
                  + (size_t)(GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CLUT + GL<FMT, MT, WR, WK, KU, S, RT, XS, V>::CRING);
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    if ((V >> 17) & 1) { cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 0); cudaGetLastError(); }  // pw16
    if (S > 8) { cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaGetLastError(); }
    once = true;
  }
  static int cc = 0;
  if (!cc) { int dev; cudaGetDevice(&dev); cudaDeviceGetAttribute(&cc, cudaDevAttrComputeCapabilityMajor, dev); }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.m + 8 * MT - 1) / (8 * MT), p.N / (16 * WR * RT), S + (((V >> 29) & 1) ? 1 : 0)); cfg.blockDim = dim3(WR * WK * 32); cfg.dynamicSmemBytes = sm; cfg.stream = st;
  cudaLaunchAttribute at[1];
  const bool clu = S > 1 && cc >= 9 && !(V & 4) && !((V >> 26) & 15);  // pw11: V bit 2 = global last-arriver reduction, no cluster
  if (occ == 2) { cudaFuncAttributes fa; if (cudaFuncGetAttributes(&fa, k) != cudaSuccess) { cudaGetLastError(); return -1; } return fa.numRegs; }  // pw14
  if (occ) {  // pw11: how many CTAs of this launch can be co-resident (clusters: cudaOccupancyMaxActiveClusters)
    int n = 0;
    if (clu) {
      at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { cudaGetLastError(); return -1; }
      return (int64_t)n * S;
    }
    int dev, sms; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, WR * WK * 32, sm) != cudaSuccess) { cudaGetLastError(); return -1; }
    return (int64_t)n * sms;
  }
  if (clu) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 1; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = S; cfg.attrs = at; cfg.numAttrs = 1; }
  if constexpr (((V >> 28) & 1) != 0) {
    const int n16 = (int)(((size_t)p.m * p.N * sizeof(typename std::conditional<FMT == 2 || FMT == 3, float, __nv_bfloat16>::type)) / 16);
    p1zero<<<(n16 + 255) / 256 < 8 ? (n16 + 255) / 256 : 8, 256, 0, st>>>((uint4*)p.out, n16);
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1; cfg.attrs = at; cfg.numAttrs = 1;
  }
  const cudaError_t r = cudaLaunchKernelEx(&cfg, k, p);
  if (r != cudaSuccess) cudaGetLastError();
  return (int64_t)r;
}

// FP32 router, small m: CTA = R full rows (no cross-CTA reduction), 512 threads split K, fp32 weights read once
template <int M, int R, int NT = 512>
__global__ void __launch_bounds__(NT) rr(const float4* W, const __nv_bfloat16* x, float* out, int m, int N, int K) {
  __shared__ float red[NT / 32][R * M];
  const int t = threadIdx.x, lane = t & 31, w = t >> 5, K4 = K >> 2, row0 = blockIdx.x * R;
  constexpr int NI = 2048 / NT;  // float4 per thread per row (K = 8192)
  float4 wv[R][NI];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int i = 0; i < NI; i++) { const uint4 u = gnc(W + (size_t)(row0 + r) * K4 + t + i * NT); wv[r][i] = make_float4(__uint_as_float(u.x), __uint_as_float(u.y), __uint_as_float(u.z), __uint_as_float(u.w)); }
  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) acc[r][q] = 0.f;
#pragma unroll
  for (int i = 0; i < NI; i++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      if (q < m) {
        const uint2 xv = __ldg((const uint2*)(x + (size_t)q * K) + t + i * NT);
        const float2 a = __bfloat1622float2(*(const __nv_bfloat162*)&xv.x), b = __bfloat1622float2(*(const __nv_bfloat162*)&xv.y);
#pragma unroll
        for (int r = 0; r < R; r++) acc[r][q] = fmaf(wv[r][i].w, b.y, fmaf(wv[r][i].z, b.x, fmaf(wv[r][i].y, a.y, fmaf(wv[r][i].x, a.x, acc[r][q]))));
      }
    }
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int q = 0; q < M; q++) {
      float v = acc[r][q];
#pragma unroll
      for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) red[w][r * M + q] = v;
    }
  __syncthreads();
  if (t < R * M && t % M < m) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < NT / 32; i++) v += red[i][t];
    out[(size_t)(t % M) * N + row0 + t / M] = v;
  }
}

// ---------------------------------------------------------------- pw22: router m = 128 / 256 as a split-K GEMM
// out[tok][n] (fp32) = sum_k x[tok][k] (hi[n][k] + lo[n][k]), hi / lo = the BF16 split of the FP32 router weights, both
// products accumulated in fp32 by the same mma.sync (as SGLang above 8 tokens). CTA (n tile of BN, K split of KR, token
// block of 128): every x row slice (128 x KR) and hi / lo row slice (BN x KR) is requested up front with 16-B cp.async
// into XOR-swizzled smem (16-B chunk c of row r at r*KR*2 + ((c ^ (r & 7)) << 4): ldmatrix rows hit distinct banks),
// one wait, then KR/16 k16 steps of ldmatrix.x4 + mma.sync m16n8k16 (A = x rows, B = W rows: both K-major). 8 warps:
// warp w owns tokens 32 (w & 3) .. +31 and n 0.5 BN (w >> 2) .. +0.5 BN. Rows >= m are zero-filled (src-size 0).
// RED 0: red.global.add.v2.f32 into an output zeroed by the caller (a memset node in the same graph);
// RED 1: fp32 partial to a workspace slot, per-(token block, n tile) counter, the last arriving CTA sums the KS slots.
__device__ __forceinline__ void rs_cpa(uint32_t dst, const void* src, uint32_t sz) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(sz) : "memory");
}
__device__ __forceinline__ void rs_ldm4(uint32_t a, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void rs_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" :: "l"(a), "f"(x), "f"(y) : "memory");
}
template <int BN, int KR, int RED>
__global__ void __launch_bounds__(256, 1) rsk(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                            float4* ws, int* cnt, int m, int N, int K) {
  constexpr int TOK = 128, CH = KR / 8, WN = BN / 2, NT8 = WN / 8, RB = KR * 2;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOK * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOK;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
  for (int i = tid; i < TOK * CH; i += 256) {
    const int r = i / CH, c = i - r * CH, tok = t0 + r;
    rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
  }
  for (int i = tid; i < BN * CH; i += 256) {
    const int r = i / CH, c = i - r * CH;
    const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
    const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
    rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  const int wt = w & 3, wn = w >> 2;
  float acc[2][NT8][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;                 // A: row (lane & 15) of the m16 tile, k chunk + (lane >> 4)
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;  // B: n row of the n16 pair, k chunk + (j & 1)
#pragma unroll 4
  for (int q = 0; q < KR / 16; q++) {
    uint32_t a[2][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
      const int r = wt * 32 + mi * 16 + ar, c = 2 * q + ac;
      rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
    }
#pragma unroll
    for (int h = 0; h < 2; h++)
#pragma unroll
      for (int nb = 0; nb < NT8 / 2; nb++) {
        const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
        uint32_t b[4];
        rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
        for (int mi = 0; mi < 2; mi++) {
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
            : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
        }
      }
  }
  if constexpr (RED == 0) {
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) {
        const int tok = t0 + wt * 32 + mi * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
        if (tok < m) rs_red2(out + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
        if (tok + 8 < m) rs_red2(out + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
      }
  } else {
    constexpr int NF = 2 * NT8 * 32 * 8;  // float4 fragments per CTA partial (8 warps)
    float4* slot = ws + ((size_t)(tb * NTL + nt) * KS + ks) * NF;
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
      for (int j = 0; j < NT8; j++) __stcg(slot + ((w * 2 + mi) * NT8 + j) * 32 + lane, make_float4(acc[mi][j][0], acc[mi][j][1], acc[mi][j][2], acc[mi][j][3]));
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const float4* base = ws + (size_t)(tb * NTL + nt) * KS * NF;
    for (int i = tid; i < NF; i += 256) {
      float4 v = __ldcg(base + i);
      for (int k2 = 1; k2 < KS; k2++) { const float4 u = __ldcg(base + (size_t)k2 * NF + i); v.x += u.x; v.y += u.y; v.z += u.z; v.w += u.w; }
      const int l = i & 31, f = i >> 5, j = f % NT8, mi = (f / NT8) & 1, ww = f / NT8 / 2;
      const int tok = t0 + (ww & 3) * 32 + mi * 16 + (l >> 2), n = n0 + (ww >> 2) * WN + j * 8 + 2 * (l & 3);
      if (tok < m) { out[(size_t)tok * N + n] = v.x; out[(size_t)tok * N + n + 1] = v.y; }
      if (tok + 8 < m) { out[(size_t)(tok + 8) * N + n] = v.z; out[(size_t)(tok + 8) * N + n + 1] = v.w; }
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;  // self-cleaning for graph replay
  }
}
// ---------------------------------------------------------------- pw23: pipelined router split-K (rsk2)
// rsk with (1) the K slice loaded in NSTG commit groups, all issued up front; stage s is multiplied after
// cp.async.wait_group(NSTG - 1 - s) + a CTA barrier, so the mma of stage s overlaps the landing of s + 1 ..;
// (2) TOKS = 128 or 256 tokens per CTA: at m = 256 one CTA holds both 128-token blocks and every ldmatrix'd W fragment
// feeds MI = TOKS / 64 m16 tiles per warp (the W chain is not doubled); (3) RED 0 = red.add into an output zeroed by a
// memset node; RED 2 = red.add into a persistent fp32 workspace (zero at prepare), then per output tile a counter: the
// CTA that completes the tile copies it ws -> out and re-zeroes its ws tile and the counter (self-cleaning).
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
__global__ void __launch_bounds__(256, MINB) rsk2(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out,
                                                 float* ws, int* cnt, int m, int N, int K) {
  constexpr int CH = KR / 8, CS = CH / NSTG, WN = BN / 2, NT8 = WN / 8, RB = KR * 2, MI = TOKS / 64, QS = KR / 16 / NSTG;
  static_assert(CH % NSTG == 0 && NT8 % 2 == 0, "stages / n tiles");
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sx = (uint32_t)__cvta_generic_to_shared(smem), sh = sx + TOKS * RB, sl = sh + BN * RB;
  const int nt = blockIdx.x, ks = blockIdx.y, tb = blockIdx.z, NTL = gridDim.x, KS = gridDim.y;
  const int n0 = nt * BN, k0 = ks * KR, t0 = tb * TOKS;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {  // stage s = 16-B chunks [s CS, (s + 1) CS) of every row
    for (int i = tid; i < TOKS * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS, tok = t0 + r;
      rs_cpa(sx + r * RB + ((c ^ (r & 7)) << 4), x + (size_t)min(tok, m - 1) * K + k0 + c * 8, tok < m ? 16u : 0u);
    }
    for (int i = tid; i < BN * CS; i += 256) {
      const int r = i / CS, c = s * CS + i - (i / CS) * CS;
      const size_t o = (size_t)(n0 + r) * K + k0 + c * 8;
      const uint32_t d = r * RB + ((c ^ (r & 7)) << 4);
      rs_cpa(sh + d, hi + o, 16u); rs_cpa(sl + d, lo + o, 16u);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  const int wt = w & 3, wn = w >> 2;
  float acc[MI][NT8][4];
#pragma unroll
  for (int i = 0; i < MI; i++)
#pragma unroll
    for (int j = 0; j < NT8; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  const int ar = lane & 15, ac = lane >> 4;
  const int bj = lane >> 3, br = (bj >> 1) * 8 + (lane & 7), bc = bj & 1;
#pragma unroll
  for (int s = 0; s < NSTG; s++) {
    if (s == 0) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG - 1) : "memory");
    else if (s == 1) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 1 ? NSTG - 2 : 0) : "memory");
    else if (s == 2) asm volatile("cp.async.wait_group %0;\n" :: "n"(NSTG > 2 ? NSTG - 3 : 0) : "memory");
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
#pragma unroll 2
    for (int qq = 0; qq < QS; qq++) {
      const int q = s * QS + qq;
      uint32_t a[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; mi++) {  // warp wt owns tokens 32 wt .. of each 128-token block: mi = (block, m16)
        const int r = (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + ar, c = 2 * q + ac;
        rs_ldm4(sx + r * RB + ((c ^ (r & 7)) << 4), a[mi]);
      }
#pragma unroll
      for (int h = 0; h < 2; h++)
#pragma unroll
        for (int nb = 0; nb < NT8 / 2; nb++) {
          const int r = wn * WN + nb * 16 + br, c = 2 * q + bc;
          uint32_t b[4];
          rs_ldm4((h ? sl : sh) + r * RB + ((c ^ (r & 7)) << 4), b);
#pragma unroll
          for (int mi = 0; mi < MI; mi++) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb][0]), "+f"(acc[mi][2 * nb][1]), "+f"(acc[mi][2 * nb][2]), "+f"(acc[mi][2 * nb][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[0]), "r"(b[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
              : "+f"(acc[mi][2 * nb + 1][0]), "+f"(acc[mi][2 * nb + 1][1]), "+f"(acc[mi][2 * nb + 1][2]), "+f"(acc[mi][2 * nb + 1][3])
              : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[2]), "r"(b[3]));
          }
        }
    }
  }
  float* dst = RED == 0 ? out : ws;
#pragma unroll
  for (int mi = 0; mi < MI; mi++)
#pragma unroll
    for (int j = 0; j < NT8; j++) {
      const int tok = t0 + (mi >> 1) * 128 + wt * 32 + (mi & 1) * 16 + g, n = n0 + wn * WN + j * 8 + 2 * t;
      if (tok < m) rs_red2(dst + (size_t)tok * N + n, acc[mi][j][0], acc[mi][j][1]);
      if (tok + 8 < m) rs_red2(dst + (size_t)(tok + 8) * N + n, acc[mi][j][2], acc[mi][j][3]);
    }
  if constexpr (RED == 2) {  // the CTA completing this (token block, n tile) copies it to out and re-zeroes it
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (tid == 0) last = atomicAdd(cnt + tb * NTL + nt, 1) == KS - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int rows = min(TOKS, m - t0);
    for (int i = tid; i < rows * (BN / 4); i += 256) {
      const int r = i / (BN / 4), c4 = i - r * (BN / 4);
      float4* src = (float4*)(ws + (size_t)(t0 + r) * N + n0) + c4;
      const float4 v = __ldcg(src);
      __stcg(src, make_float4(0.f, 0.f, 0.f, 0.f));
      ((float4*)(out + (size_t)(t0 + r) * N + n0))[c4] = v;
    }
    if (tid == 0) cnt[tb * NTL + nt] = 0;
  }
}
template <int BN, int KR, int NSTG, int TOKS, int RED, int MINB>
static int rs2_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float* ws, long long wsn, int* cnt, int cntn,
                  int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (TOKS + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + TOKS - 1) / TOKS, KS = K / KR, NTL = N / BN;
  if (RED == 2 && ((long long)TB * TOKS * N > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk2<BN, KR, NSTG, TOKS, RED, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk2<BN, KR, NSTG, TOKS, RED, MINB><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
// variant: 0 = pw22's rs64k32 (in-run reference); 5-11 = rsk2 (see RSP in the Python part)
template <int BN, int KR, int RED>
static int rs_go(const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
                 int m, int N, int K, cudaStream_t st) {
  constexpr int SM = (128 + 2 * BN) * KR * 2;
  if (N % BN || K % KR || m < 1) return -4;
  const int TB = (m + 127) / 128, KS = K / KR, NTL = N / BN;
  if (RED == 1 && ((long long)TB * NTL * KS * (BN / 16) * 2 * 32 * 8 > wsn || TB * NTL > cntn)) return -5;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(rsk<BN, KR, RED>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  rsk<BN, KR, RED><<<dim3(NTL, KS, TB), 256, SM, st>>>(x, hi, lo, out, ws, cnt, m, N, K);
  return (int)cudaGetLastError();
}
int rs_launch(int v, const __nv_bfloat16* x, const __nv_bfloat16* hi, const __nv_bfloat16* lo, float* out, float4* ws, long long wsn, int* cnt, int cntn,
              int m, int N, int K, cudaStream_t st) {
  switch (v) {
    case 0: return rs_go<64, 256, 0>(x, hi, lo, out, ws, wsn, cnt, cntn, m, N, K, st);
    case 5: return rs2_go<64, 256, 4, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 6: return rs2_go<64, 256, 2, 128, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 7: return rs2_go<64, 128, 2, 128, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 8: return rs2_go<64, 256, 4, 128, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 9: return rs2_go<64, 256, 4, 256, 0, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 10: return rs2_go<64, 128, 2, 256, 0, 2>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
    case 11: return rs2_go<64, 256, 4, 256, 2, 1>(x, hi, lo, out, (float*)ws, wsn * 4, cnt, cntn, m, N, K, st);
  }
  return -2;
}
// ---------------------------------------------------------------- pw27 PROBE ladder for the 6-8 MB regime (no paths)
// pr_ldg: G CTAs x 256 threads, every 16-B load of the CTA's share issued at launch (NL per thread, ld.global.nc
// .L1::no_allocate), then RED 0: nothing; 1: 64 fp32 atomicAdds per CTA into a zeroed 8 KB output; 2: cluster-8 DSMEM
// sum of 64 floats per CTA, rank 0 writes. pr_blk: G CTAs, one lane issues the CTA's share as CHK-byte cp.async.bulk
// copies on one mbarrier at launch, then waits.
template <int NL, int RED>
__global__ void __launch_bounds__(256) pr_ldg(const uint4* buf, long long n16, float* out, uint32_t* sink) {
  const long long T = (long long)gridDim.x * 256, t = (long long)blockIdx.x * 256 + threadIdx.x;
  uint4 v[NL];
#pragma unroll
  for (int i = 0; i < NL; i++) { const long long j = t + (long long)i * T; v[i] = j < n16 ? gnc(buf + j) : make_uint4(0u, 0u, 0u, 0u); }
  uint32_t a = 0;
#pragma unroll
  for (int i = 0; i < NL; i++) a ^= v[i].x ^ v[i].y ^ v[i].z ^ v[i].w;
  if constexpr (RED == 1) {
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) atomicAdd(out + (blockIdx.x * 64 + threadIdx.x) % 2048, (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u));
  } else if constexpr (RED == 2) {
    __shared__ float red[64];
    __shared__ uint32_t sa[256];
    sa[threadIdx.x] = a;
    __syncthreads();
    if (threadIdx.x < 64) red[threadIdx.x] = (float)((sa[threadIdx.x] ^ sa[threadIdx.x + 64]) & 1u);
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    uint32_t rk; asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rk));
    if (rk == 0 && threadIdx.x < 64) {
      float s = 0.f;
      const uint32_t la = (uint32_t)__cvta_generic_to_shared(red + threadIdx.x);
#pragma unroll
      for (int q = 0; q < 8; q++) {
        uint32_t ra; float f;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(la), "r"(q));
        asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(f) : "r"(ra) : "memory");
        s += f;
      }
      out[(blockIdx.x / 8 * 64 + threadIdx.x) % 2048] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  }
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
template <int CHK>
__global__ void __launch_bounds__(32) pr_blk(const uint8_t* buf, long long per, uint32_t* sink) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem), bar = sb + (uint32_t)per;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(bar));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(bar), "r"((uint32_t)per) : "memory");
    for (long long o = 0; o < per; o += CHK)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   :: "r"(sb + (uint32_t)o), "l"(buf + blockIdx.x * per + o), "r"(CHK), "r"(bar) : "memory");
  }
  __syncwarp();
  asm volatile("{\n.reg .pred p;\nPRB_W:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n@!p bra PRB_W;\n}\n" :: "r"(bar) : "memory");
  uint32_t a; asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(a) : "r"(sb + threadIdx.x * 4));
  if (a == 0x9e3779b9u) sink[blockIdx.x] = a;
}
// kind: 0-3 ldg NL 16 / 8 / 4 / 2 (no reduction), 4 ldg NL 8 + red, 5 ldg NL 8 + cluster-8 DSMEM, 6 bulk 8 KB, 7 bulk 16 KB
int pr_launch(int kind, int G, const void* buf, long long bytes, float* out, uint32_t* sink, cudaStream_t st) {
  const long long n16 = bytes / 16;
  switch (kind) {
    case 0: pr_ldg<16, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 1: pr_ldg<8, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 2: pr_ldg<4, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 3: pr_ldg<2, 0><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 4: pr_ldg<8, 1><<<G, 256, 0, st>>>((const uint4*)buf, n16, out, sink); break;
    case 5: {
      cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(G); cfg.blockDim = dim3(256); cfg.stream = st;
      cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = 8; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
      if (cudaLaunchKernelEx(&cfg, pr_ldg<8, 2>, (const uint4*)buf, n16, out, sink) != cudaSuccess) { cudaGetLastError(); return -3; }
      break;
    }
    case 6: case 7: {
      const long long per = bytes / G;
      const int chk = kind == 6 ? 8192 : 16384;
      if (per % chk || per > 200 * 1024) return -4;
      auto k = kind == 6 ? pr_blk<8192> : pr_blk<16384>;
      if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)per + 64) != cudaSuccess) { cudaGetLastError(); return -6; }
      k<<<G, 32, (int)per + 64, st>>>((const uint8_t*)buf, per, sink);
      break;
    }
    default: return -2;
  }
  return (int)cudaGetLastError();
}
int64_t rrrun(int64_t inst, torch::Tensor W, torch::Tensor x, torch::Tensor out) {
  const int m = x.size(0), K = x.size(1), N = out.size(1);
  auto st = at::cuda::getCurrentCUDAStream();
  const float4* w = (const float4*)W.data_ptr(); const __nv_bfloat16* xp = (const __nv_bfloat16*)x.data_ptr(); float* o = out.data_ptr<float>();
  if (K != 8192) return -1;
  switch (inst) {
    case 0: if (m > 1) return -1; rr<1, 1><<<N, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 1: if (m > 1) return -1; rr<1, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 2: if (m > 8) return -1; rr<8, 2><<<N / 2, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 3: if (m > 8) return -1; rr<8, 4><<<N / 4, 512, 0, st>>>(w, xp, o, m, N, K); break;
    case 4: if (m > 1) return -1; rr<1, 1, 256><<<N, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 5: if (m > 1) return -1; rr<1, 2, 256><<<N / 2, 256, 0, st>>>(w, xp, o, m, N, K); break;
    case 6: if (m > 1) return -1; rr<1, 1, 1024><<<N, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    case 7: if (m > 1) return -1; rr<1, 2, 1024><<<N / 2, 1024, 0, st>>>(w, xp, o, m, N, K); break;
    default: return -2;
  }
  const cudaError_t r = cudaGetLastError();
  return (int64_t)r;
}

static int64_t gvgo(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U, int O) {
  G p;
  p.W = (const uint4*)W.data_ptr(); p.W2 = (const uint4*)W2.data_ptr(); p.x = (const __nv_bfloat16*)x.data_ptr(); p.out = out.data_ptr();
  p.esc = (const uint32_t*)esc.data_ptr(); p.ws = (float4*)ws.data_ptr(); p.cnt = cnt.data_ptr<int>();
  p.m = x.size(0); p.K = x.size(1); p.N = out.size(1); p.KC = p.K / 64;
  p.T0 = (uint32_t)(T & 0xffffffff); p.T1 = (uint32_t)((uint64_t)T >> 32); p.U0 = (uint32_t)(U & 0xffffffff); p.U1 = (uint32_t)((uint64_t)U >> 32);
  auto st = at::cuda::getCurrentCUDAStream();
  const int occ = O;
  switch (inst) {
@@CASES@@
  }
  return -2;
}
int64_t gvrun(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 0);
}
int64_t gvocc(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 1);
}
int64_t prrun(int64_t kind, int64_t G, torch::Tensor buf, torch::Tensor out, torch::Tensor sink) {
  return (int64_t)pr_launch((int)kind, (int)G, buf.data_ptr(), (long long)buf.numel() * buf.element_size(), out.data_ptr<float>(),
                            (uint32_t*)sink.data_ptr(), at::cuda::getCurrentCUDAStream());
}
int64_t rsrun(int64_t v, torch::Tensor x, torch::Tensor hi, torch::Tensor lo, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt) {
  if (x.size(1) != hi.size(1) || out.size(0) != x.size(0) || out.size(1) != hi.size(0)) return -10;
  return (int64_t)rs_launch((int)v, (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)hi.data_ptr(), (const __nv_bfloat16*)lo.data_ptr(),
                            out.data_ptr<float>(), (float4*)ws.data_ptr(), (long long)(ws.numel() / 4), cnt.data_ptr<int>(), (int)cnt.numel(),
                            (int)x.size(0), (int)out.size(1), (int)x.size(1), at::cuda::getCurrentCUDAStream());
}
int64_t gvregs(int64_t inst, torch::Tensor W, torch::Tensor W2, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor ws, torch::Tensor cnt, int64_t T, int64_t U) {
  return gvgo(inst, W, W2, x, out, esc, ws, cnt, T, U, 2);
}
'''
    # gv<FMT, MT, ST, WR, WK, KU, S, RT, XS, V>: v25's only served gv instance, PROVEN_GV ('qkv_gate', 8) =
    # '1.1.1.2.16.8.1.1.2.1' (12-bit fragments, 8-token tile, geometry (2, 16, 8, 1, 1, 2), v1). v25 timed it against five
    # other gv geometries and its v12 path and kept it (>= 1.29x ahead in every logged run); GX's q144k16 (11% slower) and
    # the rest of GVI (101 instances) never served a route.
    GV_GEO = (2, 16, 8, 1, 1, 2)
    GVI = [(1, 1, 1) + GV_GEO + (1,)]
    def _gvst(geo):
        if len(geo) == 7:
            return geo[6]
        wr, wk, ku, s, rt, xs = geo
        return 1 if wr * wk >= 32 else 3 if xs == 1 and rt == 1 else 2
    _gvmod = None

    def _gvext():
        nonlocal _gvmod
        if _gvmod is None:
            cases = '\n'.join(('    case %d: return gvl<%s>(p, st, occ);' % (i, ','.join(map(str, c))) for i, c in enumerate(GVI)))
            _gvmod = _ldd19_inline('gvp45_predxor_v2_%x' % (hash(tuple(GVI)) & 4294967295), cpp_sources=r'''int64_t gvrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvocc(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t gvregs(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t); int64_t rrrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor); int64_t rsrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor); int64_t prrun(int64_t, int64_t, torch::Tensor, torch::Tensor, torch::Tensor);''', cuda_sources=GVCU.replace('@@CASES@@', cases), functions=['gvrun', 'gvocc', 'gvregs', 'rrrun', 'rsrun', 'prrun'], extra_cuda_cflags=['-O3', '-lineinfo'], verbose=False)
        return _gvmod
    def _gvesc(esc, T, geo):
        """per-warp escape lists for one launch geometry (32 entries per warp, sorted by (unit, tile)); None if a list overflows.
    esc: [E, 7] (tile, chunk, step, lane, value, xor, matrix)"""
        wr, wk, ku, s, rt, xs = geo
        gx, nw = (T // (wr * rt), wr * wk)
        out = torch.full((gx * s * nw * 32,), -1, dtype=torch.int64, device=esc.device)
        e = esc[esc[:, 0] < T]
        if len(e):
            t, c, j, l, v, x, h = e.t()
            gw = (c // (wk * ku) * gx + t // (wr * rt)) * nw + c // ku % wk * wr + t // rt % wr
            seq = c % ku * rt + t % rt
            o = torch.argsort(gw * 4096 + seq)
            gw, ent = (gw[o], (seq << 20 | l << 15 | j * 4 + v // 2 << 11 | v % 2 << 10 | h << 9 | x)[o])
            rank = torch.arange(len(gw), device=gw.device) - torch.searchsorted(gw, gw)
            if int(rank.max()) >= 32:
                return None
            out[gw * 32 + rank] = ent
        return torch.where(out >= 2 ** 31, out - 2 ** 32, out).int()
    def _bk(m):
        return 1 if m == 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m == 128 else 256 if m == 256 else 0
    def _gvok(st, plan, exact, w, mr):
        """exact on every one-hot column, close on random rows"""
        ext, n, K = (_gvext(), exact.shape[0], exact.shape[1])
        mm = (8 if plan[0] in (1002, 1003) else 1) if plan[0] >= 1000 else 8 * GVI[plan[0]][1]
        dt = exact.dtype
        eye = torch.eye(K, dtype=torch.bfloat16, device=w.device).view(K // mm, mm, K)
        got = torch.empty((K // mm, mm, n), dtype=dt, device=w.device)
        for i in range(K // mm):
            if _gvcall(st, plan, eye[i], got[i]):
                return False
        if not torch.equal(got.view(K, n), exact.t()):
            return False
        x = torch.randn((max(mm, mr), K), device=w.device).to(torch.bfloat16)
        out = torch.empty((max(mm, mr), n), dtype=dt, device=w.device)
        _gvcall(st, plan, x, out)
        ref = x.double() @ w.double().T
        return bool(torch.isfinite(out).all()) and float((out.double() - ref).norm() / ref.norm()) < (2e-05 if dt == torch.float32 else 0.004)
    def _gvcall(st, p, x, out):
        """the gv launch of the published _gvcall (the P1 zero / router / rs branches serve no kept plan)"""
        return _gvext().gvrun(p[0], p[1], p[5], x, out, p[2], st['gvws'], st['gvcnt'], p[3], p[6])

    def _gvprep(st):
        """r6's _gvprep for its one kept plan: qkv_gate m=8 on GVI[0], exact on every one-hot column, close on random rows"""
        dev = st['qkv_gate_raw'].device
        st['gvws'] = torch.zeros((1 << 24,), dtype=torch.float32, device=dev)
        st['gvcnt'] = torch.zeros((1 << 16,), dtype=torch.int32, device=dev)
        layer, bm, mt = 'qkv_gate', 8, 1
        frs = [st[layer][0]]
        css = [[_c12(f, dv) for f in frs] for dv in (0, 1)]
        if not all((c is not None for cs in css for c in cs)):
            print('gv %s m%d: 12-bit pack not exact, route not served' % (layer, bm), flush=True)
            return
        esc = torch.cat([torch.cat((c[2], torch.full_like(c[2][:, :1], h)), 1) for h, c in enumerate(css[1])])
        Tf, KC = frs[0].shape[:2]
        w = st[layer + '_raw']
        n = Tf * 16
        T, wn = (n // 16, w[:n])
        geo, v = GV_GEO, 1
        if geo[1] * geo[2] * geo[3] != KC or T % (geo[0] * geo[4]):
            return
        e = _gvesc(esc, T, geo)
        if e is None:
            print('gv %s m%d: escape list overflow, route not served' % (layer, bm), flush=True)
            return
        i = GVI.index((1, mt, _gvst(geo)) + geo + (v,))
        cs = css[v & 1]
        src = [c[0] for c in cs]
        plan = (i, src[0], e, cs[0][1], n, src[-1], cs[-1][1])
        ok = _gvok(st, plan, wn, wn, bm)
        print('gv %s m%d %s: %s' % (layer, bm, '.'.join(map(str, GVI[i])), 'ok' if ok else 'FAILED, route not served'), flush=True)
        if ok:
            st['gv'][layer, bm] = plan
    def _pack(w):
        """bf16 [N, K] -> A fragments [N/16, K/64, 32 lanes, 4 steps, 8] (K permuted so each lane's 16 k are contiguous)."""
        N, K = w.shape
        W = w.view(N // 16, 16, K // 64, 4, 4, 4)
        lo, hi = (W[:, :8], W[:, 8:])
        fr = torch.stack((lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1], lo[..., 2], lo[..., 3], hi[..., 2], hi[..., 3]), -1)
        return fr.permute(0, 2, 4, 1, 3, 5).contiguous()
    def prepare(weights):
        st = {'gv': {}, 'rq': {}}
        w = weights['qkv_gate']
        st['qkv_gate'] = (_pack(w), None)
        for k in RQLAYERS:
            st[k + '_raw'] = weights[k]
        _rqext(RQV)
        _gvext()
        _gvprep(st)
        try:
            _rqprep(st)
        except Exception as error:
            st['rq'] = {}
            print('PREP rq failed: %s' % str(error).splitlines()[-1][:200], flush=True)
        finally:
            torch.cuda.synchronize()
        return st

    def kernel_fn(data, state):
        cfg = data['config']
        layer, n = (cfg['layer'], cfg['n'])
        x = data['x']
        rp = state['rq'].get((layer, x.shape[0]))
        if rp is not None and rp['n'] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
            if _rqcall(state, rp, x.contiguous(), out) == 0:
                return out
        p = state['gv'].get((layer, _bk(x.shape[0])))
        if p is not None and p[4] == n:
            out = torch.empty((x.shape[0], n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
            _gvcall(state, p, x.contiguous(), out)
            return out
        return None
    # RQV: the published variant list cut to the variant v25 forces at m=128 (RQPICK128, "cursor + short apply") on all
    # three layers; it is chosen whenever it validates, so the bake's timings (r6 reference, five other variants) never
    # changed a pick.
    RQV = [('d2s5hzwP', 128, 5, 1, 0, 0, 0, 42999971 if LDD19_RQ128E else 42999939)]   # LDD19 (integrator): EARLY flag
    if LDD19_RQ32:   # LDD19 (sweeper): + rq<32,5> with FL 42999939 | 32 (EARLY) for qkv_gate m=32, same module
        RQV = RQV + [('s32cur5e', 32, 5, 1, 0, 0, 0, 42999971)]
    RQPICK = 'd2s5hzwP'
    RQMS = {}  # m-split variants (template M = 2); the kept variant is M = 1
    RQTOL = {1: 0.004, 2: 0.005, 3: 0.006, 4: 0.006, 5: 0.007, 6: 0.007, 7: 0.008, 8: 0.008, 9: 0.008}
    RQKS = {'pooler': 4, 'unpooler': 2, 'o': 2, 'qkv_gate': 7}
    RQLAYERS = ('pooler', 'qkv_gate', 'unpooler')
    RQCU = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;                  // packed [NB][KC][8 tiles][1536]
  const uint32_t* esc;               // escape entries, [KS][NB][8][ecap] sentinel-terminated lists (+32 sentinels)
  const uint32_t* esc2;              // FL bit 20: per-(warp,lane) sub-lists [KS][NB][8][32][4], sentinel-terminated
  int ecap;                          // entries per (ks, nb, warp) list slot: fixed stride, so no offset load
  __nv_bfloat16* out;                // [m][N] (in-kernel zeroing only; the reduce-adds go through the out tensor map)
  int* cnt;                          // [2][NB] zero / poll counters (FL bit 1), all zero between launches
  unsigned long long* prof;          // INS stamps [KS][NB][2 WG][16]
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
};
__device__ __forceinline__ void rq_wait(uint32_t b, uint32_t ph) {
  asm volatile("{\n.reg .pred p;\nRQ_W_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra RQ_W_%=;\n}\n" :: "r"(b), "r"(ph) : "memory");
}
__device__ __forceinline__ void rq_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
// wf_rq128b: relaxed (non-release) arrive. The x/W slots are only ever *read* by the consumer (LDS + wgmma, the latter
// retired by wait_group), so the release ordering of the default arrive -- which makes the arrive wait for the LDS just
// issued (P4: 95 clk) -- is not needed.
__device__ __forceinline__ void rq_arrive_r(uint32_t b) { asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory"); }
__device__ __forceinline__ void rq_expect(uint32_t b, uint32_t tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(tx) : "memory"); }
__device__ __forceinline__ void rq_bulk(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_bulkh(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n" :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void rq_tile(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar) : "memory");
}
__device__ __forceinline__ void rq_tileh(uint32_t dst, const RqTmap* tm, int k0, int r0, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;\n"
               :: "r"(dst), "l"((unsigned long long)tm), "r"(k0), "r"(r0), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint4 rq_lds(uint32_t a) {
  uint4 v; asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a) : "memory"); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
// r6 c12 decode (variant 1), pw10 pw_dec / pw_unit unchanged
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
// wf_dilp DILP (FL bit 12): the same 36 ALU ops, written as four independent 3-deep chains: every (hl, hh) pair is
// computed first (8 words, 20 ops, all independent), then the 16 PRMTs (all independent). rq_unit's per-dec form makes the
// compiler walk the four sub-blocks in order (each PRMT waits on its own LDS->LOP3->IADD chain); here the four chains'
// latencies overlap. Same values, same order of A[] writes.
__device__ __forceinline__ void rq_unit_ilp(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  const uint32_t hl0 = (q2.x & 0x87878787u) + Tb, hh0 = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  const uint32_t hl1 = (q2.y & 0x87878787u) + Tb, hh1 = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  const uint32_t hl2 = (q2.z & 0x87878787u) + Tb, hh2 = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  const uint32_t hl3 = (q2.w & 0x87878787u) + Tb, hh3 = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
  A[0] = rq_prmt(q0.x, hl0, 0x5140); A[1] = rq_prmt(q0.x, hl0, 0x7362);
  A[2] = rq_prmt(q0.y, hh0, 0x5140); A[3] = rq_prmt(q0.y, hh0, 0x7362);
  A[4] = rq_prmt(q0.z, hl1, 0x5140); A[5] = rq_prmt(q0.z, hl1, 0x7362);
  A[6] = rq_prmt(q0.w, hh1, 0x5140); A[7] = rq_prmt(q0.w, hh1, 0x7362);
  A[8] = rq_prmt(q1.x, hl2, 0x5140); A[9] = rq_prmt(q1.x, hl2, 0x7362);
  A[10] = rq_prmt(q1.y, hh2, 0x5140); A[11] = rq_prmt(q1.y, hh2, 0x7362);
  A[12] = rq_prmt(q1.z, hl3, 0x5140); A[13] = rq_prmt(q1.z, hl3, 0x7362);
  A[14] = rq_prmt(q1.w, hh3, 0x5140); A[15] = rq_prmt(q1.w, hh3, 0x7362);
}
// wf_rq128b WK: apply one patch-table word (bit 15 valid, 11:8 reg, 7 half, 6:0 xor7) to a decoded A fragment. The
// predicate is per-lane but the *instruction count* is warp-uniform, so the wgmma.fence no longer waits for a walker.
__device__ __forceinline__ void rq_patch(uint32_t (&A)[16], uint32_t w) {
  if (w & 0x8000u) {
    const int r = (int)((w >> 8) & 15);
    const uint32_t xr = (w & 0x7fu) << (8 + ((w >> 7) & 1) * 16);
#pragma unroll
    for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
  }
}
// wf_rq128b FX2: the same walk with a fixed iteration count (warp-uniform instruction count, so the wgmma.fence of the
// warpgroup no longer waits for the warp that happens to walk the longest list). K must be >= the max hits per stage.
__device__ __forceinline__ void rq_fix2(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
#pragma unroll
  for (int it = 0; it < 2; it++) {
    if (nxs == (uint32_t)seq) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
      const bool me = lane == (int)((nx >> 12) & 31);
#pragma unroll
      for (int i = 0; i < 16; i++) if (me && i == r) A[i] ^= xr;
      ++ei;
      if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
      nx = __shfl_sync(0xffffffffu, ec, ei & 31);
      nxs = nx >> 17;
    }
  }
}
// escape fix (pw10 pw_fix3 entry format): entry = seq << 17 | lane << 12 | reg << 8 | half << 7 | xor7, seq = stage within
// the CTA's range. The warp's list is cached in registers, 32 entries per lane-spread window (entry i in lane i % 32), read
// by shfl: no dependent global load per hit (pw10 reloaded the next entry from L2 after every hit, ~600 clk per escape).
__device__ __forceinline__ void rq_fix(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) ec = __ldg(el + ei + lane);
    nx = __shfl_sync(0xffffffffu, ec, ei & 31);
    nxs = nx >> 17;
  }
}
// wf_dilp WPF (FL bit 13): the same walk with the next entry's shfl issued before the current entry's 16 predicated XORs,
// so the shfl latency (the only dependent chain in the walk) hides under the apply instead of stalling the wgmma.fence.
__device__ __forceinline__ void rq_fixp(uint32_t (&A)[16], int seq, uint32_t& nx, uint32_t& nxs, int& ei, uint32_t& ec, const uint32_t* el, int lane) {
  while (nxs == (uint32_t)seq) {
    uint32_t nxn = __shfl_sync(0xffffffffu, ec, (ei + 1) & 31);
    if (lane == (int)((nx >> 12) & 31)) {
      const int r = (int)((nx >> 8) & 15);
      const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
      for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
    }
    ++ei;
    if ((ei & 31) == 0) { ec = __ldg(el + ei + lane); nxn = __shfl_sync(0xffffffffu, ec, 0); }
    nx = nxn;
    nxs = nx >> 17;
  }
}
// FL bit 20: per-thread staged cursor.  E holds this thread's own escape entries sorted by stage (head = E.x),
// hs = E.x >> 17 kept pre-shifted so the common path is one compare + one branch; no shfl, no cross-lane dependency.
// The apply needs a runtime register index, so it is either 16 predicated XORs (if-converted -> 32 instrs EVERY stage,
// measured: loop body 2120 vs 1120 no-escape) or a real branch around them.  asm volatile cannot be if-converted, so
// rq_xor16 keeps the branch: common path = ISETP + BRA.  FL bit 22 = the __any_sync (warp-uniform branch) flavour.
__device__ __forceinline__ void rq_xor16(uint32_t (&A)[16], uint32_t nx) {
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16), r = (nx >> 8) & 15;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %16, 0;\n\t@p xor.b32 %0, %0, %17;\n\t"
      "setp.eq.u32 p, %16, 1;\n\t@p xor.b32 %1, %1, %17;\n\t"
      "setp.eq.u32 p, %16, 2;\n\t@p xor.b32 %2, %2, %17;\n\t"
      "setp.eq.u32 p, %16, 3;\n\t@p xor.b32 %3, %3, %17;\n\t"
      "setp.eq.u32 p, %16, 4;\n\t@p xor.b32 %4, %4, %17;\n\t"
      "setp.eq.u32 p, %16, 5;\n\t@p xor.b32 %5, %5, %17;\n\t"
      "setp.eq.u32 p, %16, 6;\n\t@p xor.b32 %6, %6, %17;\n\t"
      "setp.eq.u32 p, %16, 7;\n\t@p xor.b32 %7, %7, %17;\n\t"
      "setp.eq.u32 p, %16, 8;\n\t@p xor.b32 %8, %8, %17;\n\t"
      "setp.eq.u32 p, %16, 9;\n\t@p xor.b32 %9, %9, %17;\n\t"
      "setp.eq.u32 p, %16, 10;\n\t@p xor.b32 %10, %10, %17;\n\t"
      "setp.eq.u32 p, %16, 11;\n\t@p xor.b32 %11, %11, %17;\n\t"
      "setp.eq.u32 p, %16, 12;\n\t@p xor.b32 %12, %12, %17;\n\t"
      "setp.eq.u32 p, %16, 13;\n\t@p xor.b32 %13, %13, %17;\n\t"
      "setp.eq.u32 p, %16, 14;\n\t@p xor.b32 %14, %14, %17;\n\t"
      "setp.eq.u32 p, %16, 15;\n\t@p xor.b32 %15, %15, %17;\n\t}"
      : "+r"(A[0]), "+r"(A[1]), "+r"(A[2]), "+r"(A[3]), "+r"(A[4]), "+r"(A[5]), "+r"(A[6]), "+r"(A[7]),
        "+r"(A[8]), "+r"(A[9]), "+r"(A[10]), "+r"(A[11]), "+r"(A[12]), "+r"(A[13]), "+r"(A[14]), "+r"(A[15])
      : "r"(r), "r"(xr));
}
__device__ __forceinline__ void rq_adv(uint4& E, uint32_t& hs) { E.x = E.y; E.y = E.z; E.z = E.w; E.w = ~0u; hs = E.x >> 17; }
__device__ __forceinline__ void rq_xor16s(uint32_t (&A)[16], uint32_t nx) {  // if-convertible 16-way select
  const int r = (int)((nx >> 8) & 15);
  const uint32_t xr = (nx & 0x7fu) << (8 + ((nx >> 7) & 1) * 16);
#pragma unroll
  for (int i = 0; i < 16; i++) if (i == r) A[i] ^= xr;
}
__device__ __forceinline__ void rq_fixc(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 20: real branch
  while (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
}
__device__ __forceinline__ void rq_fixv(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {  // bit 22: warp-uniform
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16s(A, E.x); rq_adv(E, hs); }
  }
}
// bit 23: warp-uniform branch (__any_sync) around the asm apply -> the apply is neither if-converted nor hoisted out of
// the loop (micro: loop body 107 vs 101 no-escape, 115 vote+select, 199 asm alone), so wgmma.fence has nothing to wait for
__device__ __forceinline__ void rq_fixw(uint32_t (&A)[16], int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor16(A, E.x); rq_adv(E, hs); }
  }
}
// FL bit 25 (wf_escur3 H2b, ported): patch the 8 hi-byte registers instead of the 16 A registers.
// rq_unit_ilp builds A[4g+0]=PRMT(q0,hl_g,0x5140) A[4g+1]=PRMT(q0,hl_g,0x7362) A[4g+2]=PRMT(q0,hh_g,0x5140)
// A[4g+3]=PRMT(q0,hh_g,0x7362), so entry (reg, half) -> H[reg>>1], byte (reg&1)*2 + half => 8 predicated XORs
// instead of 16 (half the apply block, so the taken branch over it is cheap).
__device__ __forceinline__ void rq_hi8(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* H) {
  H[0] = (q2.x & 0x87878787u) + Tb; H[1] = (__funnelshift_r(q2.x, q2.x, 4) & 0x87878787u) + Tb;
  H[2] = (q2.y & 0x87878787u) + Tb; H[3] = (__funnelshift_r(q2.y, q2.y, 4) & 0x87878787u) + Tb;
  H[4] = (q2.z & 0x87878787u) + Tb; H[5] = (__funnelshift_r(q2.z, q2.z, 4) & 0x87878787u) + Tb;
  H[6] = (q2.w & 0x87878787u) + Tb; H[7] = (__funnelshift_r(q2.w, q2.w, 4) & 0x87878787u) + Tb;
}
__device__ __forceinline__ void rq_prmtA(const uint4& q0, const uint4& q1, const uint32_t* H, uint32_t* A) {
  A[0] = rq_prmt(q0.x, H[0], 0x5140); A[1] = rq_prmt(q0.x, H[0], 0x7362);
  A[2] = rq_prmt(q0.y, H[1], 0x5140); A[3] = rq_prmt(q0.y, H[1], 0x7362);
  A[4] = rq_prmt(q0.z, H[2], 0x5140); A[5] = rq_prmt(q0.z, H[2], 0x7362);
  A[6] = rq_prmt(q0.w, H[3], 0x5140); A[7] = rq_prmt(q0.w, H[3], 0x7362);
  A[8] = rq_prmt(q1.x, H[4], 0x5140); A[9] = rq_prmt(q1.x, H[4], 0x7362);
  A[10] = rq_prmt(q1.y, H[5], 0x5140); A[11] = rq_prmt(q1.y, H[5], 0x7362);
  A[12] = rq_prmt(q1.z, H[6], 0x5140); A[13] = rq_prmt(q1.z, H[6], 0x7362);
  A[14] = rq_prmt(q1.w, H[7], 0x5140); A[15] = rq_prmt(q1.w, H[7], 0x7362);
}
__device__ __forceinline__ void rq_xor8(uint32_t* H, uint32_t nx) {
  const uint32_t r = (nx >> 8) & 15u;
  const uint32_t xr = (nx & 0x7fu) << (16 * (r & 1u) + 8 * ((nx >> 7) & 1u));
  const uint32_t ri = r >> 1;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.eq.u32 p, %8, 0;\n\t@p xor.b32 %0, %0, %9;\n\t"
      "setp.eq.u32 p, %8, 1;\n\t@p xor.b32 %1, %1, %9;\n\t"
      "setp.eq.u32 p, %8, 2;\n\t@p xor.b32 %2, %2, %9;\n\t"
      "setp.eq.u32 p, %8, 3;\n\t@p xor.b32 %3, %3, %9;\n\t"
      "setp.eq.u32 p, %8, 4;\n\t@p xor.b32 %4, %4, %9;\n\t"
      "setp.eq.u32 p, %8, 5;\n\t@p xor.b32 %5, %5, %9;\n\t"
      "setp.eq.u32 p, %8, 6;\n\t@p xor.b32 %6, %6, %9;\n\t"
      "setp.eq.u32 p, %8, 7;\n\t@p xor.b32 %7, %7, %9;\n\t}"
      : "+r"(H[0]), "+r"(H[1]), "+r"(H[2]), "+r"(H[3]), "+r"(H[4]), "+r"(H[5]), "+r"(H[6]), "+r"(H[7])
      : "r"(ri), "r"(xr));
}
__device__ __forceinline__ void rq_fixp8(uint32_t* H, int seq, uint4& E, uint32_t& hs) {
  while (__any_sync(0xffffffffu, hs == (uint32_t)seq)) {
    if (hs == (uint32_t)seq) { rq_xor8(H, E.x); rq_adv(E, hs); }
  }
}
__device__ __forceinline__ unsigned long long rq_gt() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {  // K-major SW128: LBO 1, SBO 1024 B, base offset 0
  return (uint64_t)((saddr & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int ND> __device__ __forceinline__ void rq_facc(float (&D)[ND]) {
#pragma unroll
  for (int i = 0; i < ND; i++) asm volatile("" : "+f"(D[i]) :: "memory");
}
__device__ __forceinline__ void rq_fa(uint32_t (&A)[16]) {
#pragma unroll
  for (int i = 0; i < 16; i++) asm volatile("" : "+r"(A[i]) :: "memory");
}
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {  // inline cvt: __float2bfloat16 serialises the RS pipeline (P4)
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@

template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;  // FL bit 4 (rs32): HALF, 64 W rows per CTA
  static constexpr int XBS = MS * XB, WBS = MS * WB;  // MS k64 stages per ring slot (MS = 2: merged k128 slot)
  static constexpr int PTW = 68;  // u32 per (slot, warp): hdr, u0 mask, u1 mask, 2 patch words per lane
  static constexpr int SMEM = 1024 + ST * (XBS + WBS) + 24 * ST + 16 + ((FL & 32768) ? ST * 8 * PTW * 4 : 0);  // WK patch table
};

// Ring: ST k64 slots, slot i = x tile (XB, 1024-aligned) in the x ring + 12 KB W chunk in the W ring. ONE full barrier
// per slot (count 2: the W lane and the x lane each arrive.expect_tx their own bytes; pw24h P4 lesson), a W empty barrier
// (released by every math warp right after its LDS: W is dead once in registers) and an x empty barrier (released after
// the wait_group that retires the slot's wgmmas). The W lane may refill a slot before the x lane: its next arrive on the
// slot's full barrier can only happen after the consumers saw the current phase complete (they LDS after the full wait),
// so phases never mix. Consumers run GS k64 stages per round: GS full waits, one wgmma.fence, 4*GS wgmma, one commit,
// one wait_group 1 (P4: k128 rounds amortise the per-round bookkeeping).
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
__global__ void __launch_bounds__((FL & 1024) ? 256 : 384, (FL & 3072) ? 2 : 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND, XBS = C::XBS, WBS = C::WBS;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
  constexpr bool MRG = (FL & 64) != 0;    // one empty barrier per slot: the x release is folded into the W release
  constexpr bool RLAX = (FL & 128) != 0;  // relaxed (non-release) empty arrive
  constexpr bool NOE = (FL & 512) != 0;   // PROBE: no escape walk (wrong output, times the floor)
  constexpr bool FX2 = (FL & 16384) != 0;  // fixed-iteration uniform escape loop (bit 14: bit 10 is pw47's H2)
  constexpr bool WK = (FL & 32768) != 0;   // patch-table walker (warp PW+3) instead of the per-warp escape list walk (bit 15: bit 11 is pw47's F2)
  constexpr bool DILP = (FL & 4096) != 0; // wf_dilp: 4 independent decode chains (rq_unit_ilp)
  constexpr bool WPF = (FL & 8192) != 0;  // wf_dilp: escape walk with the next-entry shfl prefetched
  static_assert(!WK || (PRB == 0 && MS == 1), "WK: full kernel, MS == 1");  // ZI: in-kernel zeroing of out (no memset node)
  constexpr bool RES = PRB == 8 || PRB == 10;                           // resident: no TMA, no barriers
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;       // x streamed
  constexpr bool WON = PRB != 7 && !RES;                               // W streamed
  constexpr bool SELF = PRB == 5 || PRB == 6;                          // producers self-paced (no consumer)
  // rs32 HALF (FL bit 4): one math warpgroup, 64 W rows per CTA (= half h of the 128-row pack block: tiles 4h..4h+3, a
  // contiguous 6,144 B within each 12,288 B chunk); grid x = 2 NB; math WG1 exits at once; NW = math warps
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && GS == MS && (ST >= 5 && ST <= 8) && PRB == 0 && WM <= 1 && INS == 0 && (FL == 65538 || FL == 65539 || FL == 65570 || FL == 65571))
                || (NT == 256 && GS == MS && ST == 5 && PRB == 0 && WM == 1 && INS == 0 && FL == 65603), "EH backport approved M128/ST5-8/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  // wf_rq32b: 2 CTAs per SM. FL bit 10 (H2): 256-thread CTA = 4 math warps + W lane + x lane + 2 zeroing warps, HALF
  // (64 W rows) only, launch cap 128 regs, math 216 / producer 40. FL bit 11 (F2): 384 threads, full 128 rows, launch
  // cap 84 regs, math 104 / producer 40 (KS must double so the grid still fills 2 x 132 SMs).
  constexpr bool H2 = (FL & 1024) != 0, F2 = (FL & 2048) != 0;
  static_assert(!H2 || HALF, "H2 needs HALF (64 W rows, one math warpgroup)");
  constexpr int PW = H2 ? 4 : 8;                       // first producer warp (W lane = PW, x lane = PW+1, zero = PW+2..)
  constexpr int MREG = H2 ? 128 : (F2 ? 96 : 232);     // math-warp setmaxnreg target (H2 pool = 256*128, F2 = 384*80)
  // rs32 EARLY (FL bit 5): the W lane thread itself inits the barriers and issues the first min(ST, KCS) W chunks and x
  // tiles before the CTA-wide __syncthreads (consumers touch the barriers only after it); the lanes resume at stage ST
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  // MS (k64 stages per ring slot): one full/empty barrier pair, one MS*12 KB W bulk and MS x tiles per slot; the consumer
  // round must cover a whole slot (GS == MS) so a slot is released exactly once per round
  static_assert(MS == 1 || (GS == MS && !HALF), "MS > 1: GS == MS and no HALF");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XBS, bars = wring + ST * WBS;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const uint32_t ptab = bars + 24 * ST + 16;  // WK patch table: [slot][warp][hdr + 2 words per lane]
  uint32_t* const ptabp = (uint32_t*)(smem + (sbase - sraw) + ptab);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;  // first out column of this CTA
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  constexpr bool CUR = (FL & 1048576) != 0, NOESC = (FL & 2097152) != 0, VOTE = (FL & 4194304) != 0,
                 VASM = (FL & 8388608) != 0;
  constexpr bool CURP = (FL & 33554432) != 0;  // wf_escur3 H2b: cursor + 8-way hi-byte apply (rq_hi8/rq_fixp8/rq_prmtA)
  // escape-list window issued before the producers start the W flood (a load issued after it waits behind ~28 MB)
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  // wf_rq32 FL bit 8 (ESPLIT): the entry x tiles are issued by the x lane (warp 9) in parallel with the W lane's bulks
  // (one bar.sync 6,64 after the init fence); bit 9 (EFENCE): the epilogue's fence.proxy.async.global is dropped (the
  // zeroing warp already fenced its own stores before publishing zflag).  Both measured within noise (batch 3):
  // ESPLIT -0.06..+0.11 us, EFENCE -0.22..+0.05 us on qkv m=32/16 -> no RQV entry uses them (dead code, kept for the
  // record; FL 35 is unchanged).
  constexpr bool ESPLIT = (FL & 256) != 0, EFENCE = (FL & 512) != 0;
  uint32_t ec = (warp < NW && !CUR) ? __ldg(el + lane) : 0u;
  uint4 E4 = make_uint4(~0u, ~0u, ~0u, ~0u);
  if (CUR && warp < NW)  // 16 B per thread, 512 B per warp, coalesced; latency hidden behind the first TMA wait
    E4 = *(const uint4*)(p.esc2 + (size_t)(((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * 32 + lane) * 4);
  uint32_t hs = E4.x >> 17;
  if (ESPLIT && EARLY) {
    if (warp == PW && lane == 0) {
      for (int i = 0; i < ST; i++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"(XON ? 2 : 1));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
      }
      asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (warp == PW || (warp == PW + 1 && XON)) asm volatile("bar.sync 6, 64;\n" ::: "memory");
    if (warp == PW && lane == 0) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, WB);
        if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
        else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
      }
    }
    if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t px = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      for (int s = 0; s < ST && s < KCS; s++) {
        rq_expect(full + 8 * s, XB);
        if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
        else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
      }
    }
  } else if (tid == (EARLY ? PW * 32 : 0)) {
    for (int i = 0; i < ST; i++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(full + 8 * i), "r"((XON ? 2 : 1) + (WK ? 1 : 0)));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(wempty + 8 * i), "r"(NW));
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(xempty + 8 * i), "r"(NW));
    }
    asm volatile("st.shared.u32 [%0], 0;\n" :: "r"(zflag) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    if (EARLY) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pw = 0, px = 0;
      if (HINT) {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pw));
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(px));
      }
      for (int s = 0; s < MS * ST && s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS, slot = (s / MS) % ST;
        rq_expect(full + 8 * slot, n * WB);  // two arrivals (count 2), as the W and x lanes would make
        rq_expect(full + 8 * slot, n * XB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pw);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, px);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
      }
    }
  }
  if (tid == (PW + 2) * 32 && XON) asm volatile("prefetch.tensormap [%0];\n" :: "l"((unsigned long long)&tx) : "memory");
  __syncthreads();
  const unsigned long long cta = (unsigned long long)blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t ec0 = 0xffffffffu;
  if (WK && warp == PW + 3 && lane < 8) ec0 = __ldg(p.esc + (size_t)((ks * p.NB + nb) * 8 + (lane & 7)) * p.ecap);
  auto walk = [&]() {  // WK patch-table walker (warp PW+3): lane l walks math warp l's escape list and publishes
    // (hdr, 2 words per lane) into the slot's table before arriving on the slot's full barrier
    const int lw = lane & 7;
    const uint32_t* myel = p.esc + (size_t)((ks * p.NB + nb) * 8 + lw) * p.ecap;
    uint32_t* const tb0 = ptabp + lw * C::PTW;
    int ei = 0, slot = 0, r = 0;
    uint32_t ec = ec0, u0 = 0, u1 = 0;
    for (int s = 0; s < KCS; s++) {
      if (s >= ST) rq_wait((MRG ? wempty : xempty) + 8 * slot, (r & 1) ^ 1);
      uint32_t* const tb = tb0 + slot * 8 * C::PTW;
      if (lane < 8) {  // clear the words this slot's previous fill set (the consumer is done with them: xempty)
        uint32_t q0 = tb[1], q1 = tb[2];
        while (q0) { const int l = __ffs((int)q0) - 1; tb[3 + l * 2] = 0u; q0 &= q0 - 1; }
        while (q1) { const int l = __ffs((int)q1) - 1; tb[4 + l * 2] = 0u; q1 &= q1 - 1; }
        tb[0] = 0u; tb[1] = 0u; tb[2] = 0u; u0 = u1 = 0u;
      }
      while (lane < 8 && (ec >> 17) == (uint32_t)s) {
        const int lf = (int)((ec >> 12) & 31), idx = (int)((u0 >> lf) & 1);
        tb[3 + lf * 2 + idx] = 0x8000u | (((ec >> 8) & 15u) << 8) | (((ec >> 7) & 1u) << 7) | (ec & 0x7fu);
        if (idx) u1 |= 1u << lf; else u0 |= 1u << lf;
        ++ei;
        ec = __ldg(myel + ei);
      }
      if (lane < 8) { tb[1] = u0; tb[2] = u1; tb[0] = (u0 | u1) ? 1u : 0u; }
      __syncwarp();
      if (lane == 0) rq_arrive(full + 8 * slot);
      if (++slot == ST) { slot = 0; r++; }
    }
  };
  if (warp >= PW) {  // ---------------- producer warpgroup: W lane (warp PW), x lane (warp PW+1)
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == PW && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * WB);
        if (HINT) rq_bulkh(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot, pol);
        else rq_bulk(wring + slot * WBS, src + (size_t)s * 12288, n * WB, full + 8 * slot);
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF && !XON) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) { p.prof[cta * 32 + 9] = tw; p.prof[cta * 32 + 15] = rq_gt(); }
    } else if (warp == PW + 1 && lane == 0 && XON) {
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < MS * ST ? KCS : MS * ST) : 0;  // EARLY: stages 0 .. MS*ST-1 already issued
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s += MS) {
        const int n = (KCS - s < MS) ? KCS - s : MS;
        if (s >= MS * ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : (MRG ? wempty : xempty)) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        rq_expect(full + 8 * slot, n * XB);
        for (int j = 0; j < n; j++) {
          if (HINT) rq_tileh(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot, pol);
          else rq_tile(xring + slot * XBS + j * XB, &tx, (s0 + s + j) * 64, 0, full + 8 * slot);
        }
        if (++slot == ST) { slot = 0; r++; }
      }
      if (SELF) for (int s = (KCS > MS * ST ? KCS - MS * ST : 0); s < KCS; s += MS) rq_wait(full + 8 * ((s / MS) % ST), ((s / MS) / ST) & 1);
      if (INS) p.prof[cta * 32 + 10] = tw;
    } else if (ZI && warp >= PW + 2) {  // in-kernel zeroing: this CTA's 1/KS token slice of its row block's 128 out columns,
      // then a release increment of the block's counter; every reduce-add into the block first acquires counter == KS
      const int t = tid - (PW + 2) * 32, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");  // generic zero stores -> the TMA (async proxy) adds
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        // wait (off the critical path, while the mainloop runs) until every CTA of the block zeroed its slice; publish it
        // to the epilogue through a shared flag. Bounded: a non-co-resident grid traps instead of hanging.
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");  // not co-resident: fail loudly rather than reduce into unzeroed memory
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        // the last of the block's KS pollers resets both counters for the next launch (the kernel boundary orders it)
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
      if (WK && warp == PW + 3) walk();
    } else if (WK && warp == PW + 3) {
      walk();
    }
    return;
  }
  // ---------------- math warpgroups: warp w of WG g decodes W rows 64g + 16w .. +15 of the block (= tile 4g + w = warp)
  if constexpr (MREG == 232) asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  else if constexpr (MREG == 128) asm volatile("setmaxnreg.inc.sync.aligned.u32 128;\n");
  else asm volatile("setmaxnreg.inc.sync.aligned.u32 96;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];  // never zero-filled: scale-d 0 on the CTA's first k16 (a generic write would serialise the pipe, C7515)
  uint32_t A0[GS][16], A1[GS][16];
#pragma unroll
  for (int j = 0; j < GS; j++)
#pragma unroll
    for (int i = 0; i < 16; i++) A0[j][i] = A1[j][i] = 0u;
  uint4 Q[GS][3];
#pragma unroll
  for (int j = 0; j < GS; j++) Q[j][0] = Q[j][1] = Q[j][2] = make_uint4(0, 0, 0, 0);
  uint32_t sink = 0;
  unsigned long long pc[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, tq = 0, tq0 = 0;
#define RQ_T0() do { if (INS) tq = clock64(); } while (0)
#define RQ_T1(i) do { if (INS) { const unsigned long long t_ = clock64(); pc[i] += t_ - tq; tq = t_; } } while (0)
  if (INS) { tq0 = clock64(); tq = tq0; }
  const unsigned long long gtl0 = INS ? rq_gt() : 0;
  unsigned long long gtw0 = 0;
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;  // full rounds, all rounds
  // round G = stages GS*G .. GS*G + GS - 1 (the last round may be partial)
  auto ldr = [&](int G) {  // full waits, LDS of this warp's 48 B per lane per stage, W slots released
#pragma unroll
    for (int j = 0; j < GS; j++) {
      const int s = GS * G + j;
      if (s < KCS) {
        const int slot = RES ? 0 : (s / MS) % ST;
        if (!RES && (j % MS) == 0) rq_wait(full + 8 * slot, ((s / MS) / ST) & 1);
        if (PRB != 1 && PRB != 2) {
          const uint32_t a = wl + slot * WBS + (s % MS) * WB;
          Q[j][0] = rq_lds(a); Q[j][1] = rq_lds(a + 512); Q[j][2] = rq_lds(a + 1024);
        }
      }
    }
    RQ_T1(1);
    if (!RES && !MRG) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) rq_arrive(wempty + 8 * (((GS * G + j) / MS) % ST));
      }
    }
    RQ_T1(2);
  };
  auto xrel = [&](int G) {
    if (!RES) {
      __syncwarp();
      if (lane == 0) {
#pragma unroll
        for (int j = 0; j < GS; j += MS) if (GS * G + j < KCS) {
          const uint32_t b = (MRG ? wempty : xempty) + 8 * (((GS * G + j) / MS) % ST);
          if (RLAX) rq_arrive_r(b); else rq_arrive(b);
        }
      }
    }
    RQ_T1(7);
  };
  auto dec = [&](int G, int ns, uint32_t (&A)[GS][16]) {
    if (INS) {  // force the LDS results in: time the data wait separately from the decode ALU work
      uint32_t t = 0;
#pragma unroll
      for (int j = 0; j < GS; j++) t ^= Q[j][0].x ^ Q[j][0].w ^ Q[j][1].x ^ Q[j][1].w ^ Q[j][2].x ^ Q[j][2].w;
      asm volatile("" : "+r"(t));
      sink += t & 1;
      RQ_T1(4);
    }
#pragma unroll
    for (int j = 0; j < GS; j++) {
      if (j < ns) {
        if (PRB == 10) {
#pragma unroll
          for (int i = 0; i < 16; i++) A[j][i] = Tb + i * lane;
        } else {
          if constexpr (CURP) {
            uint32_t H[8];
            rq_hi8(Q[j][0], Q[j][1], Q[j][2], Tb, H);
            rq_fixp8(H, GS * G + j, E4, hs);
            rq_prmtA(Q[j][0], Q[j][1], H, A[j]);
          } else {
          const uint32_t* const tb = ptabp + ((((GS * G + j) / MS) % ST) * 8 + (warp & 7)) * C::PTW;
          const uint32_t ph = WK ? tb[0] : 0u, pw0 = WK ? tb[3 + lane * 2] : 0u, pw1 = WK ? tb[4 + lane * 2] : 0u;
          if (DILP) rq_unit_ilp(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          else rq_unit(Q[j][0], Q[j][1], Q[j][2], Tb, A[j]);
          if (WK) {
            if (__builtin_expect(ph != 0u, 0)) { rq_patch(A[j], pw0); rq_patch(A[j], pw1); }
          } else if constexpr (CUR && VASM) rq_fixw(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR && VOTE) rq_fixv(A[j], GS * G + j, E4, hs);
          else if constexpr (CUR) rq_fixc(A[j], GS * G + j, E4, hs);
          else if (NOE) { } else if (FX2) rq_fix2(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else if (WPF) rq_fixp(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          else rq_fix(A[j], GS * G + j, nx, nxs, ei, ec, el, lane);
          }
        }
      }
    }
    RQ_T1(3);
  };
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {  // NSC: compile-time stage count of this round
    constexpr int NS = decltype(NSC)::value;
    rq_facc<ND>(D);
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#pragma unroll
    for (int j = 0; j < NS; j++) {
      const int s = GS * G + j;
      const uint64_t d = xd0 + (uint64_t)(((RES ? 0 : ((s / MS) % ST) * XBS + (s % MS) * XB)) >> 4);
#pragma unroll
      for (int q = 0; q < 4; q++) rq_wgmma<NT>(D, A[j] + 4 * q, d + 2 * q, (s | q) != 0);
    }
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    rq_facc<ND>(D);
    if (INS && G == 0) gtw0 = rq_gt();
    RQ_T1(5);
  };
  auto fa = [&](uint32_t (&A)[GS][16]) {
#pragma unroll
    for (int j = 0; j < GS; j++) rq_fa(A[j]);
  };
  if (PRB == 1 || PRB == 2 || PRB == 3) {
#pragma unroll 1
    for (int G = 0; G < NGT; G++) {
      ldr(G);
      if (PRB == 3) {
        dec(G, KCS - GS * G, A0);
#pragma unroll
        for (int j = 0; j < GS; j++)
#pragma unroll
          for (int i = 0; i < 16; i++) sink ^= A0[j][i];
      }
      if (PRB != 1) xrel(G);
    }
  } else if (PRB == 0 || PRB == 4 || RES) {
    // WM 0: dec(G) -> issue(G) -> LDS(G+1) -> wait_group 1 -> release x(G-1)  (decode on the wait->issue chain)
    // WM 1: issue(G) -> LDS(G+1) + dec(G+1) under wgmma(G) -> wait_group 0 -> release x(G) at once
    // WM 2: issue(G) -> LDS(G+1) + dec(G+1) -> wait_group 1 -> release x(G-1): the decode of the next round runs under
    //       the two queued groups; A triple-buffered (G-1 still in flight, G queued, G+1 being written)
    uint32_t A2[GS][16];
#pragma unroll
    for (int j = 0; j < GS; j++)
#pragma unroll
      for (int i = 0; i < 16; i++) A2[j][i] = 0u;
    auto round = [&](int G, auto NSC, uint32_t (&A)[GS][16], uint32_t (&An)[GS][16], uint32_t (&Ap)[GS][16]) {
      if (WM == 0) {
        dec(G, decltype(NSC)::value, A);
        mma(G, NSC, A);
        if (G + 1 < NGT) ldr(G + 1);
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      } else if (WM == 1) {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        fa(A);
        RQ_T1(6);
        xrel(G);
      } else {
        mma(G, NSC, A);
        if (G + 1 < NGT) { ldr(G + 1); dec(G + 1, KCS - GS * (G + 1), An); }
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
        fa(Ap);
        RQ_T1(6);
        if (G > 0) xrel(G - 1);
      }
    };
    using FULLR = std::integral_constant<int, GS>;
    using ONER = std::integral_constant<int, 1>;
    ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {  // buffer of round G = G % 3; host guarantees KCS >= GS
      round(0, FULLR{}, A0, A1, A2);
      G = 1;
#pragma unroll 1
      for (; G + 2 < NGF; G += 3) { round(G, FULLR{}, A1, A2, A0); round(G + 1, FULLR{}, A2, A0, A1); round(G + 2, FULLR{}, A0, A1, A2); }
      if (G < NGF) { round(G, FULLR{}, A1, A2, A0); G++; }
      if (G < NGF) { round(G, FULLR{}, A2, A0, A1); G++; }
      if (GS > 1 && NGT > NGF) {
        const int b = G % 3;
        if (b == 0) round(G, ONER{}, A0, A1, A2);
        else if (b == 1) round(G, ONER{}, A1, A2, A0);
        else round(G, ONER{}, A2, A0, A1);
        G++;
      }
    } else {
      round(0, FULLR{}, A0, A1, A1);
      G = 1;
#pragma unroll 1
      for (; G + 1 < NGF; G += 2) { round(G, FULLR{}, A1, A0, A0); round(G + 1, FULLR{}, A0, A1, A1); }
      if (G < NGF) { round(G, FULLR{}, A1, A0, A0); G++; }
      if (GS > 1 && NGT > NGF) {  // partial last round (KCS % GS stages; GS = 2: one stage), buffer parity by G
        if (G & 1) round(G, ONER{}, A1, A0, A0);
        else round(G, ONER{}, A0, A1, A1);
        G++;
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    rq_facc<ND>(D);
    fa(A0); fa(A1); fa(A2);
    if (WM != 1) xrel(NGT - 1);
  }
  if (INS) pc[0] = clock64() - tq0;
  const unsigned long long gtl1 = INS ? rq_gt() : 0;
  RQ_T0();
  if (PRB == 0 || PRB == 7) {  // ---------------- epilogue: stmatrix.trans -> SW128 slab [NT tok][64 n] -> TMA reduce-add
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    // rs128b FL bit 16 (EH): two token halves per warpgroup slab, the first half's reduce-add issued before the second
    // half's stmatrix (the TMA read of half 0 overlaps the stmatrix of half 1); the out map's box is NT / 2 tokens.
    // FL bit 17 (EP): the issuing thread polls the zero flag before its stmatrix instead of after the slab barrier.
    constexpr bool EH = (FL & 65536) != 0, EP = false;  // EH only: historical EP is deliberately unported
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
    };
    if (EP && issuer) zpoll();
    auto stsm = [&](int pp) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    };
    auto issue = [&](int tok0, uint32_t src) {
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(tok0), "r"(src) : "memory");
    };
    if (EH) {
#pragma unroll
      for (int pp = 0; pp < NT / 32; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (issuer) {
        if (!EP) zpoll();
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      }
#pragma unroll
      for (int pp = NT / 32; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (INS) ce[2] = clock64() - ce0;
        issue(NT / 2, slab + (NT / 2) * 128);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    } else {
#pragma unroll
      for (int pp = 0; pp < NT / 16; pp++) stsm(pp);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
      if (INS) ce[1] = clock64() - ce0;
      if (issuer) {
        if (!EP) zpoll();
        if (INS) ce[2] = clock64() - ce0;
        issue(0, slab);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        if (INS) ce[3] = clock64() - ce0;
        asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
        if (INS) ce[4] = clock64() - ce0;
      }
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    } else {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");  // both WGs retired every wgmma: the x ring is free
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int pp = 0; pp < NT / 16; pp++) {
      const int i0 = 2 * pp, i = i0 + (j >> 1), ch = 2 * w + (j & 1), tok = 8 * i + rr;
      const uint32_t a = slab + tok * 128 + ((ch ^ rr) << 4);
      const uint32_t r0 = rq_pk(D[4 * i0], D[4 * i0 + 1]), r1 = rq_pk(D[4 * i0 + 2], D[4 * i0 + 3]);
      const uint32_t r2 = rq_pk(D[4 * i0 + 4], D[4 * i0 + 5]), r3 = rq_pk(D[4 * i0 + 6], D[4 * i0 + 7]);
      asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n" :: "r"(a), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    asm volatile("bar.sync %0, 128;\n" :: "r"(2 + g) : "memory");
    if (INS) ce[1] = clock64() - ce0;
    if (w == 0 && lane == 0) {
      if (ZI) {  // the block's zeroing is complete (flag set by warp 10 long before; bounded spin, trap on timeout)
        uint32_t f = 0;
        for (int it = 0; it < (1 << 24) && !f; it++) asm volatile("ld.acquire.cta.shared.u32 %0, [%1];\n" : "=r"(f) : "r"(zflag) : "memory");
        if (!f) asm volatile("trap;");
        if (!EFENCE) asm volatile("fence.proxy.async.global;\n" ::: "memory");
      }
      if (INS) ce[2] = clock64() - ce0;
      if (FL & 4) asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
                               :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      else if (!(FL & 8)) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
                                       :: "l"((unsigned long long)&to), "r"(cb + 64 * g), "r"(0), "r"(slab) : "memory");
      asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
      if (INS) ce[3] = clock64() - ce0;
      asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
      if (INS) ce[4] = clock64() - ce0;
    }
    RQ_T1(8);
    if (INS && g == 0 && w == 0 && lane == 0) {
#pragma unroll
      for (int i = 0; i < 5; i++) p.prof[cta * 32 + 26 + i] = ce[i];
    }
    }
  } else {
    if (sink == 0x9e3779b9u || D[0] == 1.2345f) p.sink[0] = sink;
  }
  if (INS && w == 0 && lane == 0) {
#pragma unroll
    for (int i = 0; i < 9; i++) p.prof[cta * 32 + 16 * g + i] = pc[i];
    if (g == 0) { p.prof[cta * 32 + 11] = gt0; p.prof[cta * 32 + 12] = gtl0; p.prof[cta * 32 + 13] = gtl1; p.prof[cta * 32 + 14] = rq_gt(); p.prof[cta * 32 + 25] = gtw0; }
  }
#undef RQ_T0
#undef RQ_T1
#endif
}

typedef CUresult (*RqEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static_assert(sizeof(CUtensorMap) == sizeof(RqTmap), "CUtensorMap size");
// 2D bf16 map: dims {inner, outer}, box {64, box_outer}, 128B swizzle; out-of-range rows read as 0 and are not written
static int rq_map(RqTmap* tm, const void* ptr, int64_t inner, int64_t outer, int box_outer) {
  static RqEnc enc = nullptr;
  if (!enc) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) { cudaGetLastError(); return -20; }
    enc = (RqEnc)fn;
  }
  if (((uintptr_t)ptr) % 16 || (inner * 2) % 16 || box_outer < 1 || box_outer > 256) return -21;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)inner * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_outer};
  cuuint32_t estr[2] = {1, 1};
  return enc((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
             CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -22;
}
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL, int MS>
static int rq_go(RqP& p, torch::Tensor x, torch::Tensor out, int KS, int info) {
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL, MS>;
  auto k = rq<NT, ST, GS, PRB, WM, INS, FL, MS>;
  if (info) {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, k) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (info == 4) {  // wf_rq32b: CTAs per SM at this launch geometry (regs + smem)
      int nb = 0;
      cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
      if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, (FL & 1024) ? 256 : 384, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -1; }
      return nb;
    }
    return info == 1 ? a.numRegs : info == 2 ? (int)a.localSizeBytes : C::SMEM;
  }
  if (p.m < 1 || p.m > NT || KS < 1 || p.KC / KS < GS || KS > 64) return -4;
  if constexpr (NT == 32 && (ST == 4 || ST == 5) && FL == 52) {
    if (p.m != 32 || p.N != 8192 || p.K != 1024 || p.KC != 16 || KS != 1 || p.walias != 0) return -45;
  }
  if (p.ncnt < 2 * p.NB * ((FL & 16) ? 2 : 1)) return -5;
  RqTmap tx, to;
  int r = rq_map(&tx, x.data_ptr(), p.K, p.m, NT);
  if (r) return r;
  r = rq_map(&to, out.data_ptr(), p.N, p.m, (FL & 65536) ? NT / 2 : NT);
  if (r) return r;
  static bool once = false;
  if (!once) {
    if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM) != cudaSuccess) { cudaGetLastError(); return -6; }
    once = true;
  }
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3((FL & 1024) ? 256 : 384), C::SMEM, at::cuda::getCurrentCUDAStream()>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t rqrun(int64_t v, torch::Tensor W, torch::Tensor x, torch::Tensor out, torch::Tensor esc, torch::Tensor esc2, torch::Tensor cnt, int64_t KS,
              int64_t Tb, int64_t walias, torch::Tensor prof, torch::Tensor sink, int64_t info) {
  RqP p = {};
  if (!info) {
    if (W.dim() != 4 || W.size(2) != 8 || W.size(3) != 1536 || x.dim() != 2 || out.dim() != 2 || x.size(0) != out.size(0)) return -10;
    if (x.size(1) != W.size(1) * 64 || out.size(1) != W.size(0) * 128 || x.scalar_type() != torch::kBFloat16 || out.scalar_type() != torch::kBFloat16) return -11;
    const int64_t nl = KS * W.size(0) * 8;
    if (esc.numel() < nl * 32 + 32 || (esc.numel() - 32) % nl || sink.numel() < 1) return -12;
    if (cnt.numel() < 2 * W.size(0) || cnt.scalar_type() != torch::kInt32) return -13;
    p.ncnt = (int)cnt.numel();
    if (esc2.numel() && esc2.numel() < nl * 128) return -14;
    p.W = (const uint8_t*)W.data_ptr(); p.esc = (const uint32_t*)esc.data_ptr(); p.ecap = (int)((esc.numel() - 32) / nl);
    p.esc2 = esc2.numel() ? (const uint32_t*)esc2.data_ptr() : nullptr;
    p.out = (__nv_bfloat16*)out.data_ptr(); p.cnt = cnt.data_ptr<int>();
    p.prof = prof.numel() >= 2 * KS * W.size(0) * 32 ? (unsigned long long*)prof.data_ptr() : nullptr;
    p.sink = (uint32_t*)sink.data_ptr();
    p.m = (int)x.size(0); p.K = (int)x.size(1); p.N = (int)out.size(1); p.KC = (int)W.size(1); p.NB = (int)W.size(0);
    p.walias = (int)walias; p.Tb = (uint32_t)(Tb & 0xffffffff);
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
'''
    def _rqss(n):
        nd = n // 2
        d = ', '.join(('%%%d' % i for i in range(nd)))
        outs = ', '.join(('"+f"(D[%d])' % i for i in range(nd)))
        return 'template <> __device__ __forceinline__ void rq_wgss<%d>(float (&D)[%d], uint64_t da, uint64_t db, int scd) {\n  asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, %%%d, %%%d, p, 1, 1, 0, 0;\\n}\\n"\n    : %s\n    : "l"(da), "l"(db), "r"(scd));\n}\n' % (n, nd, nd + 2, n, d, nd, nd + 1, outs)
    def _rqsrc(vs):
        cases = '\n'.join(('    case %d: return rq_go<%s>(p, x, out, (int)KS, (int)info);' % (i, ', '.join((str(a) for a in v[1:8] + (RQMS.get(v[0], 1),)))) for i, v in enumerate(vs)))
        return RQCU.replace('@@WGMMA@@', _rqwg(16) + _rqwg(32) + _rqwg(128) + _rqwg(256) + _rqss(128) + _rqss(256)).replace('@@CASES@@', cases)
    RQFLAGS = ['-gencode=arch=compute_90a,code=sm_90a', '-O3', '-DNDEBUG', '--expt-relaxed-constexpr']
    _rqmod = [None, None]
    def _rqext(vs=None, verbose=False, slot=None):
        vs = RQV if vs is None else vs
        src = _rqsrc(vs)
        slot = _rqmod if slot is None else slot
        if slot[0] is None or slot[1] != src:
            tag = ('rq128_%x' if slot is _rqmod else 'rq32_%x') % (hash(src) & 4294967295)
            flags = RQFLAGS + (['-Xptxas=-v'] if verbose else [])
            slot[0] = _ldd19_inline(tag + ('v' if verbose else ''), cpp_sources='int64_t rqrun(int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t);', cuda_sources=src, functions=['rqrun'], extra_cuda_cflags=flags, verbose=verbose)
            slot[1] = src
        return slot[0]
    def _rqesc2(esc6, NB, KC, KS, dev, CAPT=4):
        """FL bit 20 lists: one fixed slot of CAPT entries per (ks, nb, warp, lane), sorted by stage, sentinel-terminated
    -> int32 [KS*NB*8*32*CAPT].  Returns (arr, max entries in any slot); arr is None if a slot overflows CAPT."""
        G = KS * NB * 8
        if not len(esc6):
            return (torch.full((G * 32 * CAPT,), 4294967295, dtype=torch.int32, device=dev), 0)
        tile, ch, q, ln, v, xr = [t.long() for t in esc6.t()]
        nb, wt = (tile // 8, tile % 8)
        s0 = torch.arange(KS + 1, device=dev, dtype=torch.long) * KC // KS
        ks = torch.searchsorted(s0, ch.contiguous(), right=True) - 1
        u = ch - s0[ks]
        ent = u << 17 | ln << 12 | q * 4 + v // 2 << 8 | v % 2 << 7 | xr
        slot = ((ks * NB + nb) * 8 + wt) * 32 + ln
        o = torch.argsort(slot * 65536 + u)
        slot, ent = (slot[o], ent[o])
        rank = torch.arange(len(slot), device=dev) - torch.searchsorted(slot, slot)
        mx = int(rank.max()) + 1
        if mx > CAPT:
            return (None, mx)
        arr = torch.full((G * 32 * CAPT,), 4294967295, dtype=torch.long, device=dev)
        arr[slot * CAPT + rank] = ent
        return (torch.where(arr >= 2 ** 31, arr - 2 ** 32, arr).int().contiguous(), mx)
    def _rqcall(st, p, x, out):
        e2 = p.get('esc2')
        if e2 is None:
            e2 = st.get('rq_e2e')
            if e2 is None:
                e2 = st['rq_e2e'] = torch.empty(0, dtype=torch.int32, device=x.device)
        return p.get('mod', _rqmod[0]).rqrun(p['i'], p['lay'], x, out, p['esc'], e2, p['cnt'], p['KS'], p['tab'], 0, st['rq_e'], st['rq_sink'], 0)
    def _rqval(fn, w, m, n, gen, tol):
        """one-hot x8 rows bit-exact against W over every K column (the split-K bf16 reduce-add must keep them exact),
    random rows within the KS-dependent tolerance (judge: 1e-2, rows 5x)"""
        K = w.shape[1]
        out = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
        exact = w[:n].t()
        if K not in _EYE:
            _EYE[K] = torch.eye(K, dtype=torch.bfloat16, device=w.device) * 8.0
        eye = _EYE[K]
        for start in range(0, K, m):
            x = eye[start:start + m]
            if x.shape[0] < m:
                x = torch.cat((x, torch.zeros((m - x.shape[0], K), dtype=x.dtype, device=x.device)))
            if fn(x.contiguous(), out) != 0:
                return (False, 'launch error')
            got = out[:min(m, K - start)]
            want = exact[start:start + got.shape[0]] * 8.0
            if not torch.isfinite(got).all() or not torch.equal(got.float(), want.float()):
                return (False, 'one-hot mismatch at column %d' % start)
        for dist in (1.0, 30.0):
            x = torch.randn((m, K), generator=gen, device=w.device)
            if dist != 1.0:
                x[:, torch.randperm(K, generator=gen, device=w.device)[:K // 100]] *= dist
            x = x.to(torch.bfloat16)
            o = torch.empty((m, n), dtype=torch.bfloat16, device=w.device)
            fn(x, o)
            ref = x.double() @ w[:n].double().t()
            err = float((o.double() - ref).norm() / ref.norm())
            row = float(((o.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            if not (err <= tol and row <= 5 * tol and torch.isfinite(o).all()):
                return (False, 'random rel L2 %.2e row %.2e > %.0e' % (err, row, tol))
        return (True, 'ok %.1e' % err)
    def _rqprep(st):
        """the RS128 bake cut to the forced pick: per layer the exact 12-bit pack, its escape lists, then d2s5hzwP at m=128
        is validated (one-hot x8 rows bit-exact over every K column, random rows within the KS tolerance) and served"""
        dev = st['pooler_raw'].device
        st['rq_e'] = torch.empty(0, dtype=torch.int64, device=dev)
        st['rq_sink'] = torch.zeros(4, dtype=torch.int32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(12828)
        m = 128
        for layer in RQLAYERS:
            w = st[layer + '_raw']
            n, K = w.shape
            x = torch.randn((m, K), generator=gen, device=dev).to(torch.bfloat16)
            pk = _rqpack(w)
            if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 65535):
                print('BAKE rq %s pack not exact' % layer, flush=True)
                continue
            lay, tab, esc6 = pk
            KS = RQKS[layer]
            esc = _rqesc(esc6, lay.shape[0], lay.shape[1], KS, dev)[0]
            esc2, mx2 = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
            print('BAKE rq %s esc2 slots max=%d %s' % (layer, mx2, 'ok' if esc2 is not None else 'OVERFLOW'), flush=True)
            if esc2 is None:
                continue
            cnt = torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev)
            out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            p = {'i': 0, 'lay': lay, 'esc': esc, 'esc2': esc2, 'cnt': cnt, 'KS': KS, 'tab': tab, 'n': n}
            fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
            try:
                r = fn(x, out)
                torch.cuda.synchronize()
                if r != 0:
                    print('BAKE %s m=%d rq:%s ERROR launch %d' % (layer, m, RQPICK, r), flush=True)
                    continue
                ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                torch.cuda.synchronize()
                if int(cnt.abs().sum()):
                    cnt.zero_()
                    ok, msg = (False, 'counters not reset')
            except Exception as error:
                print('BAKE %s m=%d rq:%s ERROR %s' % (layer, m, RQPICK, str(error).splitlines()[-1][:160]), flush=True)
                continue
            print('PICK %s m=%d rq:%s %s' % (layer, m, RQPICK, msg if ok else 'INVALID ' + msg), flush=True)
            if ok:
                st['rq'][layer, m] = p
        if LDD19_RQ32 and len(RQV) > 1 and RQV[1][1] == 32:   # LDD19 (sweeper, iteration 6): qkv_gate m=32 on RQV[1]
            layer, m, KS = 'qkv_gate', 32, 7
            w = st[layer + '_raw']
            n, K = w.shape
            try:
                pk = _rqpack(w)
                ok, msg = pk is not None and torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 65535), 'pack not exact'
                if ok:
                    lay, tab, esc6 = pk
                    esc = _rqesc(esc6, lay.shape[0], lay.shape[1], KS, dev)[0]
                    esc2, mx2 = _rqesc2(esc6, lay.shape[0], lay.shape[1], KS, dev)
                    ok, msg = esc2 is not None, 'esc2 overflow'
                if ok:
                    cnt = torch.zeros(8 * lay.shape[0], dtype=torch.int32, device=dev)
                    p = {'i': 1, 'lay': lay, 'esc': esc, 'esc2': esc2, 'cnt': cnt, 'KS': KS, 'tab': tab, 'n': n}
                    fn = lambda xx, oo, p=p: _rqcall(st, p, xx, oo)
                    out = torch.full((m, n), float('nan'), dtype=torch.bfloat16, device=dev)   # zero input over NaN: in-kernel zeroing
                    ok, msg = fn(torch.zeros((m, K), dtype=torch.bfloat16, device=dev), out) == 0 and bool((out == 0).all()), 'zero input'
                if ok:
                    ok, msg = _rqval(fn, w, m, n, gen, RQTOL[KS])
                if ok:   # spike row (v19_rq_verify's), FP64
                    x = torch.randn((m, K), generator=gen, device=dev)
                    x[m - 1].zero_()
                    x[m - 1, (11 * 131 + m) % K] = 16.0
                    x = x.to(torch.bfloat16)
                    o = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                    ref = x.double() @ w.double().t()
                    ok = fn(x, o) == 0
                    d = o.double() - ref
                    rows = d.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
                    err = float(d.norm() / ref.norm())
                    ok = ok and bool(torch.isfinite(o).all()) and err <= RQTOL[KS] and float(rows.max()) <= 0.01 and float(rows[m - 1]) <= 0.00025
                    msg = 'random rel L2 %.1e row max %.1e spike %.1e' % (err, float(rows.max()), float(rows[m - 1]))
                torch.cuda.synchronize()
                if ok and int(cnt.abs().sum()):
                    ok, msg = False, 'counters not reset'
            except Exception as error:   # noqa: BLE001 - the v19 route keeps qkv_gate m=32
                ok, msg = False, str(error).splitlines()[-1][:160]
            print('LDD19 rq32 qkv_gate m=32 rq:%s %s' % (RQV[1][0], msg if ok else 'INVALID ' + msg + ', v19 route kept'), flush=True)
            if ok:
                st['rq'][layer, m] = p
        torch.cuda.synchronize()
    _EYE = {}
    return {'prepare': prepare, 'kernel_fn': kernel_fn}

def _ldd19_make_ag():
    import ctypes as C
    import threading
    import torch

    class _Ptr(C.c_void_p):

        def __deepcopy__(self, memo):
            return self
    P, I, SZ = (_Ptr, C.c_int, C.c_size_t)
    TLS = threading.local()
    CUDA = []

    def ctx_bind(x):
        if getattr(TLS, 'device', None) == x.device.index:
            return
        if not CUDA:
            cu = C.CDLL('libcuda.so.1')
            for name, args in (('cuCtxGetCurrent', [C.POINTER(C.c_void_p)]), ('cuDeviceGet', [C.POINTER(C.c_int), C.c_int]), ('cuDevicePrimaryCtxRetain', [C.POINTER(C.c_void_p), C.c_int]), ('cuCtxSetCurrent', [C.c_void_p])):
                fn = getattr(cu, name)
                fn.argtypes, fn.restype = (args, C.c_int)
            CUDA.append(cu)
        cu = CUDA[0]
        ctx = C.c_void_p()
        cu.cuCtxGetCurrent(C.byref(ctx))
        if not ctx.value:
            dev, primary = (C.c_int(), C.c_void_p())
            if cu.cuDeviceGet(C.byref(dev), x.device.index or 0) == 0 and cu.cuDevicePrimaryCtxRetain(C.byref(primary), dev) == 0:
                cu.cuCtxSetCurrent(primary)
        TLS.device = x.device.index

    class Algo(C.Structure):
        _fields_ = [('data', C.c_uint64 * 8)]

    class Heur(C.Structure):
        _fields_ = [('algo', Algo), ('workspaceSize', C.c_size_t), ('state', C.c_int), ('wavesCount', C.c_float), ('reserved', C.c_int * 4)]

    def load_lib():
        for name in ('libcublasLt.so.13', 'libcublasLt.so.12', 'libcublasLt.so'):
            try:
                return C.CDLL(name)
            except OSError:
                pass
        raise RuntimeError('no libcublasLt')

    class Lt:

        def __init__(self):
            self.lib = load_lib()
            sigs = {'cublasLtCreate': [C.POINTER(P)], 'cublasLtMatmulDescCreate': [C.POINTER(P), I, I], 'cublasLtMatmulDescSetAttribute': [P, I, P, SZ], 'cublasLtMatrixLayoutCreate': [C.POINTER(P), I, C.c_uint64, C.c_uint64, C.c_int64], 'cublasLtMatrixLayoutSetAttribute': [P, I, P, SZ], 'cublasLtMatmul': [P] * 12 + [C.POINTER(Algo), P, SZ, P], 'cublasLtMatmulAlgoInit': [P, I, I, I, I, I, I, I, C.POINTER(Algo)], 'cublasLtMatmulAlgoConfigSetAttribute': [C.POINTER(Algo), I, P, SZ], 'cublasLtMatmulAlgoCheck': [P, P, P, P, P, P, C.POINTER(Algo), C.POINTER(Heur)]}
            for name, argtypes in sigs.items():
                fn = getattr(self.lib, name)
                fn.argtypes, fn.restype = (argtypes, I)
            self.handle = P()
            self.call('cublasLtCreate', C.byref(self.handle))
            self.desc = {}
            for mode, transpose in (('tn', 1), ('nn', 0)):
                desc = P()
                self.call('cublasLtMatmulDescCreate', C.byref(desc), 68, 0)
                op = I(transpose)
                self.call('cublasLtMatmulDescSetAttribute', desc, 3, C.byref(op), C.sizeof(op))
                self.desc[mode] = desc
            self.workspace = torch.empty(0, device='cuda', dtype=torch.uint8)
            self.required_workspace = 0
            self.alpha, self.beta = (C.c_float(1), C.c_float(0))

        def __deepcopy__(self, memo):
            clone = self.__class__.__new__(self.__class__)
            clone.lib, clone.handle, clone.desc = (self.lib, self.handle, dict(self.desc))
            clone.workspace = self.workspace.clone()
            clone.required_workspace = self.required_workspace
            clone.alpha, clone.beta = (self.alpha, self.beta)
            return clone

        def call(self, name, *args):
            status = getattr(self.lib, name)(*args)
            if status:
                raise RuntimeError(f'{name} status {status}')

        def layouts(self, mode, m, n, k):
            specs = ((k, n, k, k * n), (k, m, k, 0), (n, m, n, n))
            if mode == 'nn':
                specs = ((n, k, n, n),) + specs[1:]
            elif mode == 'rownn':
                specs = ((m, k, k, 0), (k, n, n, 0), (m, n, n, 0))
            out = []
            for rows, cols, ld, stride in specs:
                desc = P()
                self.call('cublasLtMatrixLayoutCreate', C.byref(desc), 14, rows, cols, ld)
                if mode == 'rownn':
                    order = I(1)
                    self.call('cublasLtMatrixLayoutSetAttribute', desc, 1, C.byref(order), C.sizeof(order))
                count, offset = (I(1), C.c_int64(stride))
                self.call('cublasLtMatrixLayoutSetAttribute', desc, 5, C.byref(count), C.sizeof(count))
                self.call('cublasLtMatrixLayoutSetAttribute', desc, 6, C.byref(offset), C.sizeof(offset))
                out.append(desc)
            return out

        def plan(self, mode, m, n, k, cfg):
            layouts = self.layouts(mode, m, n, k)
            algo = Algo()
            self.call('cublasLtMatmulAlgoInit', self.handle, 68, 0, 14, 14, 14, 14, 66, C.byref(algo))
            for name, attr, ctype in (('tile', 1, C.c_uint32), ('splitk', 2, C.c_int32), ('red', 3, C.c_uint32), ('swz', 4, C.c_uint32), ('custom', 5, C.c_uint32), ('stages', 6, C.c_uint32), ('cluster', 8, C.c_uint16)):
                value = ctype(cfg[name])
                status = self.lib.cublasLtMatmulAlgoConfigSetAttribute(C.byref(algo), attr, C.byref(value), C.sizeof(value))
                if status:
                    raise RuntimeError(f'ag config {name} status {status}')
            result = Heur()
            status = self.lib.cublasLtMatmulAlgoCheck(self.handle, self.desc['nn' if mode == 'rownn' else mode], *layouts, layouts[2], C.byref(algo), C.byref(result))
            if status:
                raise RuntimeError(f'ag AlgoCheck status {status}')
            self.required_workspace = max(self.required_workspace, int(result.workspaceSize))
            return (mode, layouts, algo)

        def run(self, x, weight, n, plan):
            ctx_bind(x)
            mode, (a, b, c), algo = plan
            out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
            ap = P(x.data_ptr() if mode == 'rownn' else weight.data_ptr())
            bp = P(weight.data_ptr() if mode == 'rownn' else x.data_ptr())
            self.call('cublasLtMatmul', self.handle, self.desc['nn' if mode == 'rownn' else mode], C.byref(self.alpha), ap, a, bp, b, C.byref(self.beta), P(out.data_ptr()), c, P(out.data_ptr()), c, C.byref(algo), P(self.workspace.data_ptr()), self.workspace.numel(), P(torch.cuda.current_stream().cuda_stream))
            return out

    def cfg_of(tile, custom, cluster):
        return {'tile': tile, 'stages': 35, 'splitk': 1, 'red': 0, 'swz': 0, 'custom': custom, 'cluster': cluster}
    PINS = {('o', 256): ('tn', 20, 6, 2)}
    TRANSPOSED = {('unpooler', 256), ('pooler', 256), ('pooler', 128)}

    def prepare(weights):
        lt = Lt()
        st = {'lt': lt, 'plans': {}, 'w': {}}
        wt = {}
        for key, (mode, tile, custom, cluster) in PINS.items():
            layer, m = key
            w = weights[layer]
            n, k = w.shape
            try:
                plan = lt.plan(mode, m, n, k, cfg_of(tile, custom, cluster))
                if key in TRANSPOSED:
                    if layer not in wt:
                        wt[layer] = w.t().contiguous()
                    weight = wt[layer]
                else:
                    weight = w
                lt.workspace = torch.empty(max(lt.required_workspace, lt.workspace.numel()), device='cuda', dtype=torch.uint8)
                g = torch.Generator(device=w.device).manual_seed(7 + m)
                x = torch.randn((m, k), generator=g, device=w.device).to(torch.bfloat16)
                got = lt.run(x, weight, n, plan)
                torch.cuda.synchronize()
                ref = x.float() @ w.float().T
                if bool(torch.isfinite(got).all()) and float((got.float() - ref).norm() / ref.norm()) < 0.01:
                    st['plans'][key] = plan
                    st['w'][key] = weight
                else:
                    print(f'AG pin {key} failed the self-test', flush=True)
            except Exception as error:
                print(f'AG pin {key} unavailable: {str(error)[:120]}', flush=True)
        return st

    def kernel_fn(data, state):
        x = data['x']
        key = (data['config']['layer'], x.shape[0])
        return state['lt'].run(x, state['w'][key], data['config']['n'], state['plans'][key])

    return {'prepare': prepare, 'kernel_fn': kernel_fn, 'has': lambda state, key: key in state['plans']}
import ctypes as _CtxC
import threading as _CtxThreading

_CTX_TLS = _CtxThreading.local()
_CTX_API = None


def _ctx_api():
    global _CTX_API
    if _CTX_API is None:
        api = _CtxC.CDLL("libcuda.so.1")
        for name, args in (
            ("cuCtxGetCurrent", [_CtxC.POINTER(_CtxC.c_void_p)]),
            ("cuCtxGetDevice", [_CtxC.POINTER(_CtxC.c_int)]),
            ("cuDeviceGet", [_CtxC.POINTER(_CtxC.c_int), _CtxC.c_int]),
            ("cuDevicePrimaryCtxRetain", [_CtxC.POINTER(_CtxC.c_void_p), _CtxC.c_int]),
            ("cuCtxSetCurrent", [_CtxC.c_void_p]),
        ):
            f = getattr(api, name)
            f.argtypes, f.restype = args, _CtxC.c_int
        _CTX_API = api
    return _CTX_API


def _ctx_check(status, op):
    if status:
        raise RuntimeError(f"{op} failed with CUDA driver status {status}")


def _ctx_bind(x):
    index = x.device.index if x.device.index is not None else 0
    if getattr(_CTX_TLS, "device", None) == index:
        return
    api = _ctx_api()
    ctx = _CtxC.c_void_p()
    _ctx_check(api.cuCtxGetCurrent(_CtxC.byref(ctx)), "cuCtxGetCurrent")
    cur = _CtxC.c_int(-1)
    if ctx.value:
        _ctx_check(api.cuCtxGetDevice(_CtxC.byref(cur)), "cuCtxGetDevice")
    if not ctx.value or cur.value != index:
        dev, primary = _CtxC.c_int(), _CtxC.c_void_p()
        _ctx_check(api.cuDeviceGet(_CtxC.byref(dev), index), "cuDeviceGet")
        _ctx_check(api.cuDevicePrimaryCtxRetain(_CtxC.byref(primary), dev), "cuDevicePrimaryCtxRetain")
        _ctx_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
    _CTX_TLS.device = index


LDD19_KEELL_ROUTES = True
_ldd19_before_routes_prepare = prepare
_ldd19_before_routes_kernel = kernel_fn
_ldd19_ext_backends = {'chenwainuo':_ldd19_make_chenwainuo(), 'ag':_ldd19_make_ag()}

def prepare(weights):
    st = _ldd19_before_routes_prepare(weights)
    st['keell_routes'] = {}
    if LDD19_KEELL_ROUTES:
        for name, backend in _ldd19_ext_backends.items():
            st['keell_routes'][name] = backend['prepare'](weights)
    return st

def kernel_fn(data, state):
    _ctx_bind(data['x'])
    cfg = data['config']
    key = (cfg['layer'], cfg['m'])
    name = _ldd19_ext_table.get('%s:%d' % key)
    if LDD19_KEELL_ROUTES and name in state.get('keell_routes', {}):
        backend = _ldd19_ext_backends[name]
        st = state['keell_routes'][name]
        if name != 'ag' or backend['has'](st, key):
            out = backend['kernel_fn'](data, st)
            if out is not None:  # readable edition: a route that failed its prepare check falls back to the chain
                return out
    return _ldd19_before_routes_kernel(data, state)

# bf16_calib re-roll copy (final_rr), code identical to the shipped file; 2026-11-03
