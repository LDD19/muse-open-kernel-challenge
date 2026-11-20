# okc: problem=muse_decode_layer
# LDD19 v22: validated SC6/MFIN port; retain v20 dense/attention and clustered M32 scheduling.

from torch.utils.cpp_extension import load_inline as _ldd19_raw_inline
def _ldd19_inline(*a, **kw):
  kw['extra_cflags'] = [*kw.get('extra_cflags', []), '-fno-gnu-unique']
  kw['extra_cuda_cflags'] = [*kw.get('extra_cuda_cflags', []), '-Xcompiler=-fno-gnu-unique']
  return _ldd19_raw_inline(*a, **kw)
# LDD19: SC6 lossless scale storage, per-expert completion and M32 unit costs
# adapted from team keell0renz (Bohdan Agarkov), copyright 2026, Apache-2.0,
# commit 8e30e1be994db4310794b8d04e90d18d895deb84. Keep LICENSE/NOTICE.keell0renz.txt.
# Based on muse_decode_layer.py v157e by team vjeux, https://github.com/vjeux/okc-2026
# (commit 3fa2f6b8c529a1107581dda54cfbb0b818223437, Apache License 2.0; see that repository's LICENSE).
# The docstring below is team vjeux's; "our" there means them. Modifications by LDD19 are marked "LDD19:".
# LDD19: also uses (a) vjeux's m = 1 cgemv kernel from muse_bf16_proj.py (same repository and commit, Apache-2.0),
# and (b) the OKC_WSH0 MoE helper-warp layout from Madhavan113/kerneloptimizer (https://github.com/Madhavan113/
# kerneloptimizer, a derived work of vjeux/okc-2026, Apache License 2.0; see that repository's NOTICE).
# LDD19: and, as carried by our muse_bf16_proj sub_v22 (lt128 section): (c) the pinned cuBLASLt path (tt_lt_*) of
# NathenFernandes' muse-bf16-proj v18 submission (https://github.com/NathenFernandes/muse-bf16-proj, v18 submission.py,
# commit 86aff75), MIT License:
#   Copyright (c) 2026 Nathen Fernandes
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the "Software"), to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#   the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#   THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
# ; and (d) chenwainuo's rq split-K register-A wgmma kernel, its
# cuBLASLt descriptor choice, from https://github.com/chenwainuo/okc (commit 785c2c01dc21,
# results/proj-rsi/releases/pw45-v1-835acb90b47b/submission.py), Apache License 2.0 (http://www.apache.org/licenses/
# LICENSE-2.0); modified by LDD19 (binding with raw pointers + stream, the RG bit, only two instances built).
# LDD19: (e) the KPERM3 idea (K permuted inside each 256-K stage of the MAIN expert copies so that a lane's k16 steps
# share one scale block; one scale word per stage) is team alygg77's, https://github.com/alygg77/okc-muse-spark-kernels
# (Apache License 2.0); reimplemented by LDD19 from the permutation formula (marked "LDD19: kperm").
# LDD19: (f) the B4FIN idea (per-tile sum-of-squares partials from the expert epilogue, column-parallel final pass at
# M <= 32) is also team alygg77's (same repository, Apache License
# 2.0); reimplemented by LDD19 (marked "LDD19: b4fin"). (u32_r96 was measured in-layer and not adopted.)
# LDD19: (g) the idea of batched (strided N-split) cuBLASLt output layouts for the T = 128 / 256 projections is team
# alygg77's (same repository, Apache License 2.0); layouts, plan enumeration and picks are LDD19's ("LDD19: ltscan").
# LDD19: (h) v20: the 8-instruction NVFP4 E2M1 dequant bit-layout (LDD19_DQ3: 2 SHF + 1 IMAD + 5 LOP3), adapted from
# ZhimaoL/sglang (https://github.com/ZhimaoL/sglang, okc_patch pg, Apache License 2.0); (i) NathenFernandes'
# ol_projection<128,128,4,3,1,1,1> and pack_olayer from muse-bf16-proj v18 (MIT License, see item (c) above),
# extended by LDD19 with Hopper PDL edges 16 and 32 for T = 128 o; and (j) our muse_attention_decode entry's
# v298 + LDD19 kernels (lxf4m2, etx2, b1split, d16s n_splits=8).
"""Muse Spark decode layer (team vjeux): one CUDA-graph-capturable kernel_fn built from four kernel sources.

Provenance: every kernel here is our own team's work, copied (not imported) from our entries on this
challenge's component boards:
  * attention: the muse_attention_decode entry's fused kernel (QK-norm, RoPE, FP8 cache write, split-KV
    attention, merge, per-head RMSNorm * sigmoid(gate)), driven by that entry's own launcher, plan and
    workspace allocator, lifted verbatim (_launch7, _plan7, _workspace7, _supported7, _inv_freq7);
  * experts: the muse_moe_w4a16 entry's NVFP4 MoE kernel in its three builds (M <= 8, 9..63, 64+), with that
    entry's own scratch allocator (_ma_make_scratch) and build check (_ma_scales_prmt_exact: the PRMT scale
    decode only when every local block-scale byte is a normal e4m3, else the LUT build) lifted verbatim;
  * dense projections: the muse_bf16_proj entry's module (every name prefixed bfp_), incl. its fused
    pooler + router launch;
  * this file's glue: residual/norm boundaries, routing (top-8 of biased sigmoid scores; router weights as
    BF16 hi + lo with FP32 accumulation; pre-expert RMSNorm), Triton split-K GEMMs, and SGLang's own FA3,
    Marlin and cuBLAS paths from the judge image.

Selection: prepare() times each backend per (token count, matrix), and the expert kernels per token count,
on the judge's own GPU with synthetic activations (it reads only weights, never data), then fixes the
choices; untuned token counts take the next tuned bucket's. Attention backend by shape (q_len, window, batch).

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding, exact escape correction);
NVFP4 experts stay NVFP4; norms apply (1 + w) formed in FP32 at use; FP8 queries and FP8 KV cache as
SGLang; softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too; FP32 accumulation throughout;
activations BF16 wherever SGLang keeps BF16.

Launch order: at T <= 32 tokens every kernel of the layer is launched with programmatic dependent launch
(PDL). Each one waits (griddepcontrol.wait) before its first read of data produced in this call; only
weights, and inputs that no kernel in this call writes, are loaded before the wait. Above 32 tokens all
launches are plain stream-ordered.

Safety: every scratch/workspace buffer this file allocates or caches for a component is followed by
sentinel words; prepare() checks them after the tuner and an attention self-test have run every path.
"""

import torch
import triton
import triton.language as tl

HIDDEN = 8192
D_EXPERT = 4096
LOCAL_EXPERTS = 32
N_EXPERTS = 256
TOP_K = 8
LOG2E = 1.4426950408889634

_TUNE_LOG = False
FORCE_PBMAX = None
PDL = True
PDL_MAX_T = 32
# LDD19: big3: above 32 tokens, PDL only on the T = 128 edges whose two kernels are ours (LDD19_PDL_EDGES, see _ldd19_edge)


def _pdl(T):
  return PDL and T <= PDL_MAX_T
DEFAULT_MOE = "agent"
TUNE_MARGIN = 0.95
PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
# LDD19: the picks every prepare we logged made (Modal and judge kernel lists), locked so a noisy isolated timing can't
LDD19_LOCK = True
# LDD19: iter2 integrator: prepare-time dead work removed (every result it produced was overwritten or never read).
# v24r builds no SGLang Marlin bank: under LDD19_LOCK every T runs the "agent" MoE (decode_slim: Marlin code removed).
# LDD19_SLIM_TUNE: v24r's tuner skips T <= 32 (all mm picks locked by LDD19_LOCK_MM) and every MoE sweep (locked).
# LDD19_SLIM_KL: the Keell pipeline only runs T in {1, 8, 16, 32} (HY_ROUTES): no main Keell MoE build, no
#   T = 128/256 policy timing or cuBLASLt plans; its tuner is removed (all its picks were overwritten).
LDD19_SLIM_TUNE = True
LDD19_SLIM_KL = True
LDD19_LOCK_MM = {(T, m): "bfp" for T in (1, 2, 4, 8, 16, 32) for m in ("qkv", "o", "un")}
LDD19_LOCK_MM.update({(T, "pr"): "bfpf" for T in (1, 2, 4, 8, 16)})
LDD19_LOCK_MM.update({(32, "pr"): "bfp", (2, "un"): False, (4, "un"): False})


def _ldd19_lock(st):
  """LDD19: apply the locked picks (only where the backend they name is available)."""
  if st.get("bfp") is not None:
    for k, v in LDD19_LOCK_MM.items():
      st["mm_pick"][k] = v
  for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    st.setdefault("gluev2", {})[T] = False
    if st.get("ma_ok"):
      st.setdefault("moe_pick", {})[T] = "agent"
FORCE_GLUEV2 = None

# LDD19 hybrid: shared base CUDA sources. These are team keell0renz's sources (Apache-2.0, see the attribution
# (v24 = the same vjeux-derived kernels plus the LDD19 additions), via _ldd19_derive. The derived strings are
def _ldd19_derive(base, edits):
  for e in edits:
    old, new = e if len(e) == 2 else (e[1], e[1] + e[2] if e[0] == 'after' else e[2] + e[1])
    assert base.count(old) == 1, old[:120]
    base = base.replace(old, new)
  return base
kl_MOE_HELPER0_CUDA_SRC = r"""#include <cuda_bf16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <stdint.h>
#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1
#define OKC_F1EARLY 1
#define OKC_WGSFINSS 1
#ifndef OKC_FINCOL
#define OKC_FINCOL 1
#endif
#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1
#ifndef OKC_FAST1
#define OKC_FAST1 3
#endif
#define FAST1_ON (OKC_FAST1 > 0 && OKC_TINY && FAST1_ARCH && OKC_USE_TMA)
#ifndef OKC_CPFF
#define OKC_CPFF 1
#endif
#ifndef OKC_FCOSTBB
#define OKC_FCOSTBB OKC_FCOSTB
#endif
#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1
#ifndef OKC_DYNK
#define OKC_DYNK 4
#endif
#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64
#ifndef OKC_SPLIT
#define OKC_SPLIT 0
#endif
#ifndef OKC_TINY
#define OKC_TINY 0
#endif
#ifndef OKC_NW
#define OKC_NW 8
#endif
#ifndef OKC_EARLYN
#define OKC_EARLYN 1
#endif
#define OKC_BIGM 64
#define OKC_XPF 0
#ifndef OKC_CPF
#define OKC_CPF 1
#endif
#define OKC_FATSTATIC 1
#define OKC_FCOST 40
#ifndef OKC_FCOSTB
#define OKC_FCOSTB 40
#endif
#ifndef OKC_WS
#define OKC_WS 0
#endif
#define OKC_WGSX 0
#ifndef OKC_WGS
#define OKC_WGS 0
#endif
#define SILU_DIV(a, b) __fdividef((a), (b))
#ifndef OKC_UWARP
#define OKC_UWARP 1
#endif
#if OKC_UWARP
#define WARP_IDX(t) __shfl_sync(0xffffffffu, (t) >> 5, 0)
#else
#define WARP_IDX(t) ((t) >> 5)
#endif
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#ifndef OKC_FINPF
#define OKC_FINPF 0
#endif
#define OKC_WSNR 4
#define OKC_NOLOAD 0
namespace okc {
#ifndef OKC_L2H
#define OKC_L2H 1
#endif
#if OKC_L2H
#define L2H_SUFFIX ".L2::cache_hint"
#define L2H_OPERAND ", %4"
#define L2H_ARG , "l"(0x12F0000000000000ull)
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) {
  asm volatile("st.global.L2::cache_hint.b16 [%0], %1, %2;\n" ::"l"(a), "h"(*reinterpret_cast<unsigned short*>(&v)), "l"(0x14F0000000000000ull) : "memory");
}
__device__ __forceinline__ void st_keep_f32(float* a, float v) {
  asm volatile("st.global.L2::cache_hint.b32 [%0], %1, %2;\n" ::"l"(a), "f"(v), "l"(0x14F0000000000000ull) : "memory");
}
#else
#define L2H_SUFFIX ""
#define L2H_OPERAND ""
#define L2H_ARG
__device__ __forceinline__ void st_keep_bf16(__nv_bfloat16* a, __nv_bfloat16 v) { *a = v; }
__device__ __forceinline__ void st_keep_f32(float* a, float v) { *a = v; }
#endif
constexpr int HD = 4096, ID = 4096;
constexpr int NW = OKC_NW, NT = NW * 32;
constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);
#define TIDX ((int)((threadIdx.x + NT) % NTL))
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;
#if OKC_WS || OKC_WGS
#define CTA_SYNC() asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory")
#define WS_JOIN() asm volatile("bar.sync 2, %0;\n" ::"n"(NTL) : "memory")
#else
#define CTA_SYNC() __syncthreads()
#endif
constexpr int NCH = 64 / NW;
#define OKC_SKT 8
constexpr int SK_T = OKC_SKT;
constexpr int RED_W = OKC_SKT > 8 ? 8 : 4;
static_assert(!OKC_WS || OKC_SKT == 8, "OKC_WS assumes single-tile skinny jobs");
constexpr int FT_T = OKC_FT_T;
constexpr int RB_A = 2 * ID / 16, RB_B = HD / 16;
constexpr int UB_A = RB_A / 16, UB_B = RB_B / 16;
constexpr int RB_A_LOG = 9, RB_B_LOG = 8;
static_assert((1 << RB_B_LOG) == RB_B, "RB_B must be 256");
constexpr int BFLAG = 1 << 30;
__device__ __forceinline__ int dyn_pool(int nS, int G) { const int d = G * OKC_DYNK; return SKDYN ? (d < nS ? d : nS) : 0; }
static_assert((1 << RB_A_LOG) == RB_A, "RB_A must be 512");
static_assert(OKC_NSF <= 7 && OKC_NS <= 5, "barrier slots");
constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;
constexpr int XP_SK = HD + 8;
constexpr int KS = 256;
constexpr int XP_FT = KS + 8;
constexpr int NST = 3;
constexpr float TWO62 = 4.611686018427388e18f;
struct Params {
  const uint4* wA; const uint4* sA; const float* gsA;
  const uint4* wB; const uint4* sB; const float* gsB;
  const uint32_t* lut; const __nv_bfloat16* normw;
  const unsigned char* wAf; const unsigned char* wBf;
  const unsigned char* wAu; const unsigned char* wBu;
  const unsigned char* wAn; const unsigned char* wBn;
  const unsigned char* wAs; const unsigned char* wBs;
  __nv_bfloat16* xA; __nv_bfloat16* xB;
  int* xready;
  int* doneAs;
  uint32_t zero;
  const __nv_bfloat16* x; const int* topk_ids; const float* topk_w;
  int M, lo, nloc;
  int* ctr;
  int* doneA;
  __nv_bfloat16* inter;
  float* ybuf;
  float* ss;
  float* ss1;
  __nv_bfloat16* out;
  unsigned long long* dbg;
};
struct Smem {
  uint32_t lut[256];
#if !OKC_WGS
  float red[NRB][NW][32][RED_W];
#else
  float red[1][1][1][RED_W];
#endif
#if OKC_WS
  uint64_t wse[8], wsrf[8], wsre[8];
  int wsr[4];
#endif
#if OKC_WGS
  uint64_t bars[24];
  int wglast[4];
  int wcp[JMAX + 1];
  int wcpB[JMAX + 1];
  int jrow[JMAX + 1];
  uint32_t wtot;
  uint32_t wtotB;
#else
  uint64_t bars[12];
#endif
#if OKC_WGS
  int wgr[4];
#endif
  int seq, fseq, preissued, preissuedB;
  int fcnt[32];
  void* xbuf_ptr;
  int cnt[64], off[64];
  int wc[NW][32];
  int jE[JMAX], jS[JMAX], jC[JMAX], forder[JMAX];
  int njF, njS, unit;
  int gen;
  float* ssp;
  int uslot[8];
  int dync[JMAX];
  int unit_next, unit_next2;
  signed char eid[EMAX];
  short ent[EMAX];
  short pos[EMAX];
};
__device__ __forceinline__ int muldiv_g(int a, int c, int G, float rG) {
  const int p = a * c;
  int q = __float2int_rz((float)p * rG);
  if (q * G > p) --q;
  else if ((q + 1) * G <= p) ++q;
  return q;
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  int sz = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
extern __shared__ __align__(16) unsigned char smem_raw[];
__device__ __forceinline__ float* ss_cur();
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void dq4(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_r(q, q, 6) & M;
  r2 = __funnelshift_r(q, q, 10) & M;
  r1 = (q * 16u) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q * 64u) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
  return;
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
}
__device__ void early_issue(const Params& p, Smem& S);
__device__ void route(const Params& p, Smem& S) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const int E = p.M * 8;
  const int CH = ((E + NW * 32 - 1) / (NW * 32)) * 32;
  const int i0 = warp * CH, i1 = min(E, i0 + CH);
  if (tid < NW * 32) S.wc[tid >> 5][tid & 31] = 0;
#if OKC_RLOAD && OKC_WGS
  {
    constexpr int LPT = (EMAX / 4 + NT - 1) / NT;
    int4 ev[LPT];
    const int E4 = E >> 2;
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      ev[k] = i4 < E4 ? __ldg(reinterpret_cast<const int4*>(p.topk_ids) + i4) : make_int4(0, 0, 0, 0);
    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {
      const int i4 = tid + k * NT;
      if (i4 < E4) {
        const int v[4] = {ev[k].x - p.lo, ev[k].y - p.lo, ev[k].z - p.lo, ev[k].w - p.lo};
        char4 c;
        c.x = (signed char)((v[0] >= 0 && v[0] < p.nloc) ? v[0] : -1);
        c.y = (signed char)((v[1] >= 0 && v[1] < p.nloc) ? v[1] : -1);
        c.z = (signed char)((v[2] >= 0 && v[2] < p.nloc) ? v[2] : -1);
        c.w = (signed char)((v[3] >= 0 && v[3] < p.nloc) ? v[3] : -1);
        *reinterpret_cast<char4*>(&S.eid[i4 * 4]) = c;
      }
    }
  }
#else
#pragma unroll 8
  for (int i = tid; i < E; i += NT) {
    int e = __ldg(p.topk_ids + i) - p.lo;
    S.eid[i] = (signed char)((e >= 0 && e < p.nloc) ? e : -1);
  }
#endif
  CTA_SYNC();
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }
  CTA_SYNC();
  if (warp == 0) {
    int c[NW];
    int tot = 0;
#pragma unroll
    for (int w = 0; w < NW; ++w) { c[w] = (lane < p.nloc) ? S.wc[w][lane] : 0; tot += c[w]; }
    int incl = tot;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, d); if (lane >= d) incl += v; }
    int off = incl - tot;
    S.cnt[lane] = tot;
    S.off[lane] = off;
    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
#if OKC_SPLIT
    static_assert(OKC_TINY, "OKC_SPLIT: skinny-only (tiny-style) build");
    const int nfj = 0;
    const int nsj = (tot + SK_T - 1) / SK_T;
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
    const int nsj = (tot > 0 && tot <= SK_T) ? 1 : 0;
#endif
    int fi = nfj, si = nsj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      int v1 = __shfl_up_sync(0xffffffffu, fi, d), v2 = __shfl_up_sync(0xffffffffu, si, d);
      if (lane >= d) { fi += v1; si += v2; }
    }
    const int nfTot = __shfl_sync(0xffffffffu, fi, 31), nsTot = __shfl_sync(0xffffffffu, si, 31);
    int fj = fi - nfj, sj = nfTot + si - nsj;
    for (int q = 0; q < nfj; ++q) {
      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);
    }
#if OKC_SPLIT
    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }
#else
    if (nsj) { S.jE[sj] = lane; S.jS[sj] = off; S.jC[sj] = tot; }
#endif
    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }
    __syncwarp();
#if !OKC_WGS
    int pos = 0;
    for (int key = FT_T / 8; key >= 1; --key) {
      for (int f0 = 0; f0 < nfTot; f0 += 32) {
        const int f = f0 + lane;
        const bool m = f < nfTot && ((S.jC[f] + 7) >> 3) == key;
        const unsigned bal = __ballot_sync(0xffffffffu, m);
        if (m) S.forder[pos + __popc(bal & ((1u << lane) - 1u))] = f;
        pos += __popc(bal);
      }
    }
#endif
  }
  CTA_SYNC();
  early_issue(p, S);
  for (int base = i0; base < i1; base += 32) {
    const int i = base + lane;
    const int e = (i < i1) ? (int)S.eid[i] : -1;
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    if (e >= 0) {
      const int q = S.wc[warp][e] + __popc(peers & ((1u << lane) - 1u));
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else if (i < i1) {
      S.pos[i] = -1;
    }
    __syncwarp();
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
    __syncwarp();
  }
  CTA_SYNC();
}
__device__ __forceinline__ float probe_q(float v) {
  return v;
}
__device__ __forceinline__ void epi_A(const Params& p, int pstart, int ntok, int tok0, int col, float gsc, const float* r) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    if (tk < ntok) {
      float gt = r[h] * gsc, up = r[2 + h] * gsc;
      float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
      st_keep_bf16(p.inter + ((size_t)(pstart + tk) * ID + col), __float2bfloat16(probe_q(v)));
    }
  }
}
__device__ __forceinline__ void epi_B(const Params& p, int pstart, int ntok, int tok0, int n0, float gsc, const float* r,
                                      float* sq) {
  const float w0 = 1.f + __bfloat162float(p.normw[n0]), w1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    int tk = tok0 + h;
    float v0 = r[h] * gsc * w0, v1 = r[2 + h] * gsc * w1;
    sq[h] = v0 * v0 + v1 * v1;
    if (tk < ntok) {
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0), v0);
      st_keep_f32(p.ybuf + ((size_t)(pstart + tk) * HD + n0 + 8), v1);
    }
  }
}
__device__ __forceinline__ void ss_reduce_add(const Params& p, int pstart, int ntok, int tok0, float* sq, int g) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 4);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 8);
    sq[h] += __shfl_xor_sync(0xffffffffu, sq[h], 16);
  }
  if (g == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}
constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
#if OKC_PRMTSC
#define SCD(word, b) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * 16u + 0x5C005C00u)
#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive1(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
template <bool PB>
__device__ __forceinline__ void ring_issue(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
  const size_t blk = (size_t)S.jE[j] * rbs + rb;
  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {
  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
}
template <bool PB>
__device__ __forceinline__ void load_x_regs(const Params& p, const Smem& S, int j, int lane, int warp,
                                            uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  {
    const int r = g;
    if (r < cnt) {
      const __nv_bfloat16* row;
      if (PB) row = p.inter + (size_t)(st + r) * ID;
      else row = p.x + (size_t)(S.ent[st + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * t);
#pragma unroll
      for (int c = 0; c < NCH; ++c) {
        x[2 * c] = PB ? __ldcg(src + c * 8) : __ldg(src + c * 8);
        x[2 * c + 1] = PB ? __ldcg(src + c * 8 + 1) : __ldg(src + c * 8 + 1);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
    }
  }
}
constexpr int X1P = 512 + 8;
template <bool PB>
__device__ __forceinline__ void load_x1_smem(const Params& p, const Smem& S, int j, int lane, int warp,
                                             __nv_bfloat16* x1s) {
  const int cnt = S.jC[j], st = S.jS[j];
  for (int i = lane; i < 8 * 64; i += 32) {
    const int r = i >> 6, c = i & 63;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (8 + r < cnt) {
      const __nv_bfloat16* row = PB ? p.inter + (size_t)(st + 8 + r) * ID : p.x + (size_t)(S.ent[st + 8 + r] >> 3) * HD;
      const uint4* src = reinterpret_cast<const uint4*>(row + warp * 512) + c;
      v = PB ? __ldcg(src) : __ldg(src);
    }
    *reinterpret_cast<uint4*>(x1s + r * X1P + c * 8) = v;
  }
  __syncwarp();
}
#if SKDYN
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int rsh = PB ? RB_B_LOG : RB_A_LOG;
  const int seq0 = seq;
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  int snext = v0 + pre, pend = -2, jdmin = 1 << 30, jdmax = -1;
  bool exh = poolBase >= nSall;
  int* pctr = p.ctr + (PB ? 1 : 0);
  auto take = [&]() -> int {
    if (snext < v1) return snext++;
    if (exh) return -1;
    const int cl = pend >= 0 ? pend : atomicAdd(pctr, 1);
    const int v = poolBase + cl;
    if (v >= nSall) { exh = true; pend = -2; return -1; }
    pend = atomicAdd(pctr, 1);
    if (!PB) {
      const int jj = njF + (v >> rsh);
      S.dync[jj] += 1;
      jdmin = min(jdmin, jj); jdmax = max(jdmax, jj);
    }
    return v;
  };
  auto refill = [&](int sl) {
    const int v = take();
    if (v >= 0) { ring_issue<PB>(p, S, v, ring + sl * UNIT_BYTES, bars + sl); S.uslot[sl] = v; }
    else if (cpf && preB < nB) {
      ring_issue<true>(p, S, v0B + preB, ring + sl * UNIT_BYTES, bars + sl);
      S.uslot[sl] = BFLAG | (v0B + preB);
      ++preB;
    } else {
      S.uslot[sl] = -1;
    }
  };
  if (tid == 0)
    for (int i = pre; i < OKC_NS; ++i) {
      const int sl = (seq0 + i) % OKC_NS;
      if (i < nU) { ring_issue<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
  if (nU < OKC_NS) CTA_SYNC();
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {
    const int u0 = nU > 0 ? v0 : (S.uslot[seq0 % OKC_NS] & ~BFLAG);
    const int rbq = (u0 < 0 ? 0 : u0) & (rbs - 1);
    nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
  }
  int nproc = 0;
  for (int i = 0;; ++i) {
    int v;
    if (i < nU) {
      v = v0 + i;
    } else {
      v = S.uslot[(seq0 + i) % OKC_NS];
      if (v < 0 || (!PB && (v & BFLAG))) break;
      v &= ~BFLAG;
    }
    const int j = njF + (v >> rsh), rb = v & (rbs - 1);
    float nwn0 = 0.f, nwn1 = 0.f;
    int vn = -1;
    if (PB) vn = i + 1 < nU ? v + 1 : S.uslot[(seq0 + i + 1) % OKC_NS];
    if (PB && tid < 128 && vn >= 0) {
      const int rbq = (vn & ~BFLAG) & (rbs - 1);
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }
      else refill(sl);
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;
  }
  seq = seq0 + nproc;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      if (nU > 0) {
        const int jf = njF + (v0 >> rsh), jl = njF + ((v1 - 1) >> rsh);
        for (int jj = jf; jj <= jl; ++jj) {
          const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
          atomicAdd(p.doneA + jj, b - a);
        }
      }
      for (int jj = jdmin; jj <= jdmax; ++jj) {
        const int k = S.dync[jj];
        if (k) { atomicAdd(p.doneA + jj, k); S.dync[jj] = 0; }
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#else
template <bool PB>
__device__ void skinny_ring(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq,
                            __nv_bfloat16* x1base, bool cpf_ok, int cB0, int cBn, int nSall, int poolBase) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;
  const int seq0 = seq;
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int preB = 0;
  const bool cpf = OKC_CPF && !PB && cpf_ok;
  const int v0B = cB0, nB = cBn;
  const int pre = PB ? S.preissuedB : S.preissued;
  for (int i = pre; i < OKC_NS; ++i) {
    const int sl = (seq0 + i) % OKC_NS;
    if (i < nU) ring_issue_all<PB>(p, S, v0 + i, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i - nU < nB) { ring_issue_all<true>(p, S, v0B + (i - nU), ring + sl * UNIT_BYTES, bars + sl); ++preB; }
  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) { const int rbq = v0 % rbs; nwc0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwc1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8)); }
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    float nwn0 = 0.f, nwn1 = 0.f;
    if (PB && tid < 128 && i + 1 < nU) {
      const int rbq = (v + 1) % rbs;
      nwn0 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg)); nwn1 = 1.f + __bfloat162float(__ldg(p.normw + rbq * 16 + egg + 8));
    }
    if (j != curj) {
      curj = j;
      two = (OKC_SKT > 8) && S.jC[j] > 8;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
      if (two) load_x1_smem<PB>(p, S, j, lane, warp, x1s);
    }
    const int sl = (seq0 + i) % OKC_NS;
    const uint32_t par = (uint32_t)(((seq0 + i) / OKC_NS) & 1);
    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
      if (!two) {
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }
      } else {
        const uint4 c0 = *reinterpret_cast<const uint4*>(x1r + c * 64);
        const uint4 c1 = *reinterpret_cast<const uint4*>(x1r + c * 64 + 8);
        const uint32_t bb[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int st = 0; st < 4; ++st) {
          uint32_t r0, r1, r2, r3;
          dq4(ww[st], r0, r1, r2, r3);
          const uint32_t a0 = MULG(r0), a1 = MULG8(r1), a2 = MULG(r2), a3 = MULG8(r3);
          mma_bf16(acc, a0, a1, a2, a3, b[2 * st], b[2 * st + 1]);
          mma_bf16(acc2, a0, a1, a2, a3, bb[2 * st], bb[2 * st + 1]);
        }
      }
    }
#undef MULG
#undef MULG8
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
    float* rd = &S.red[i & 1][warp][lane][0];
    *reinterpret_cast<float4*>(rd) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);
    else if (cpf && i + OKC_NS - nU < nB) {
      ring_issue_all<true>(p, S, v0B + (i + OKC_NS - nU), ring + sl * UNIT_BYTES, bars + sl);
      ++preB;
    }
    if (tid < (two ? 128 : 64)) {
      const int nt = tid >> 6, h = (tid >> 5) & 1, ln = tid & 31;
      const int gg = ln >> 2, tt = ln & 3;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int ww = 0; ww < NW; ++ww) {
        const float* q = &S.red[i & 1][ww][ln][nt * 4];
        s0 += q[h];
        s1 += q[2 + h];
      }
      const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
      const int tk = nt * 8 + 2 * tt + h;
      if (!PB) {
        if (tk < cnt) {
          const float gsc = p.gsA[ce] * TWO62;
          const float gt = s0 * gsc, up = s1 * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      } else {
        const float gsc = p.gsB[ce] * TWO62;
        const int n0 = rb * 16 + gg;
        const float v0f = s0 * gsc * nwc0, v1f = s1 * gsc * nwc1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
    nwc0 = nwn0; nwc1 = nwn1;
  }
  seq = seq0 + nU;
  if (!PB) {
    __threadfence();
    CTA_SYNC();
    if (tid == 0) {
      const int jf = njF + v0 / rbs, jl = njF + (v1 - 1) / rbs;
      for (int jj = jf; jj <= jl; ++jj) {
        const int a = max(v0, (jj - njF) * rbs), b = min(v1, (jj - njF + 1) * rbs);
        atomicAdd(p.doneA + jj, b - a);
      }
    }
  }
  if (!PB && tid == 0) S.preissuedB = preB;
  if (PB && tid == 0) S.preissuedB = 0;
  CTA_SYNC();
}
#endif
constexpr int NSF = OKC_NSF;
template <bool PB>
__device__ __forceinline__ void fat_issue(const Params& p, int e, int ub, int ks, unsigned char* slot, uint64_t* bar, bool nat = false) {
  const int nub = PB ? UB_B : UB_A;
  const unsigned char* src = (nat ? (PB ? p.wBn : p.wAn) : (PB ? p.wBf : p.wAf)) + (((size_t)e * nub + ub) * 16 + ks) * (size_t)UNIT_BYTES;
  if (TIDX == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                 ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
  }
}
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 62);
}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((saddr >> 4) & 0x3FFF) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int NTT> struct WG;
template <> struct WG<1> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<2> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<3> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<4> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<5> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<6> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<7> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <> struct WG<8> {
  static __device__ __forceinline__ void mma(float (&acc)[FT_T / 8][4], uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint64_t desc) {
    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
        : "+f"(acc[0][0]), "+f"(acc[0][1]), "+f"(acc[0][2]), "+f"(acc[0][3]), "+f"(acc[1][0]), "+f"(acc[1][1]), "+f"(acc[1][2]), "+f"(acc[1][3]), "+f"(acc[2][0]), "+f"(acc[2][1]), "+f"(acc[2][2]), "+f"(acc[2][3]), "+f"(acc[3][0]), "+f"(acc[3][1]), "+f"(acc[3][2]), "+f"(acc[3][3]), "+f"(acc[4][0]), "+f"(acc[4][1]), "+f"(acc[4][2]), "+f"(acc[4][3]), "+f"(acc[5][0]), "+f"(acc[5][1]), "+f"(acc[5][2]), "+f"(acc[5][3]), "+f"(acc[6][0]), "+f"(acc[6][1]), "+f"(acc[6][2]), "+f"(acc[6][3]), "+f"(acc[7][0]), "+f"(acc[7][1]), "+f"(acc[7][2]), "+f"(acc[7][3])
        , "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3)
        : "l"(desc));
  }
};
template <int NTT>
__device__ __forceinline__ void fat_compute_wg(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                               int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4], uint32_t zero) {
  const int g = lane >> 2;
  const uint32_t xb = smem_u32(xs);
  uint32_t A[2][8][4];
  uint32_t keep = 0;
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const int sb = cc & 1;
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
    const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp * 4 + cc) * 8 + g) * 8);
    const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (((warp + 8) * 4 + cc) * 8 + g) * 8);
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint32_t K92 = 0x6D806D80u;
      const uint32_t p0 = bmul2(bmul2((__byte_perm(sc0.x, sc0.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t p1 = bmul2(bmul2((__byte_perm(sc1.x, sc1.y, 0x0400u | st | ((st + 4) << 8)) & 0x00FF00FFu) << 4, K92), K92);
      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);
      const uint32_t sg1 = __byte_perm(p1, p1, 0x1010u), sg81 = __byte_perm(p1, p1, 0x3232u);
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
      uint32_t r0, r1, r2, r3, q0, q1, q2, q3;
      dq4((&w0.x)[st], r0, r1, r2, r3);
      dq4((&w1.x)[st], q0, q1, q2, q3);
      A[sb][2 * st][0] = FMG0(r0); A[sb][2 * st][1] = FMG80(r1);
      A[sb][2 * st][2] = FMG0(r2); A[sb][2 * st][3] = FMG80(r3);
      A[sb][2 * st + 1][0] = FMG1(q0); A[sb][2 * st + 1][1] = FMG81(q1);
      A[sb][2 * st + 1][2] = FMG1(q2); A[sb][2 * st + 1][3] = FMG81(q3);
#undef FMG0
#undef FMG80
#undef FMG1
#undef FMG81
    }
    wg_fence();
#pragma unroll
    for (int st = 0; st < 4; ++st) {
      const uint64_t dk = wg_desc_sw128(xb + cc * 8192 + st * 32);
      WG<NTT>::mma(acc0, A[sb][2 * st][0], A[sb][2 * st][1], A[sb][2 * st][2], A[sb][2 * st][3], dk);
      WG<NTT>::mma(acc1, A[sb][2 * st + 1][0], A[sb][2 * st + 1][1], A[sb][2 * st + 1][2], A[sb][2 * st + 1][3], dk);
    }
    wg_commit();
    if (cc >= 1) {
      wg_wait<1>();
#pragma unroll
      for (int k = 0; k < 8; ++k) keep ^= A[sb ^ 1][k][0] ^ A[sb ^ 1][k][1] ^ A[sb ^ 1][k][2] ^ A[sb ^ 1][k][3];
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int k = 0; k < 8; ++k) keep ^= A[1][k][0] ^ A[1][k][1] ^ A[1][k][2] ^ A[1][k][3];
  acc0[0][0] += __uint_as_float(keep & zero);
}
template <int NTT>
__device__ __forceinline__ void fat_compute(const Smem& S, const unsigned char* slot, const __nv_bfloat16* xs, int warp,
                                            int lane, float (&acc0)[FT_T / 8][4], float (&acc1)[FT_T / 8][4]) {
  const int g = lane >> 2, t = lane & 3;
  const uint2 sc0 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp * 32 + lane) * 8);
#if OKC_NW == 8
  const uint2 sc1 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + ((warp + 8) * 32 + lane) * 8);
#endif
#pragma unroll
  for (int cc = 0; cc < 4; ++cc) {
    const uint4 w0 = *reinterpret_cast<const uint4*>(slot + ((warp * 4 + cc) * 32 + lane) * 16);
#if OKC_NW == 8
    const uint4 w1 = *reinterpret_cast<const uint4*>(slot + (((warp + 8) * 4 + cc) * 32 + lane) * 16);
#endif
#if OKC_PRMTSC
    const uint32_t sg0 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2), sg80 = SCD(cc < 2 ? sc0.x : sc0.y, (cc & 1) * 2 + 1);
#if OKC_NW == 8
    const uint32_t sg1 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2), sg81 = SCD(cc < 2 ? sc1.x : sc1.y, (cc & 1) * 2 + 1);
#endif
#else
    const int sh = (cc & 1) * 16;
    const uint32_t sw0 = (cc < 2 ? sc0.x : sc0.y) >> sh;
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif
    const __nv_bfloat16* xr = xs + g * XP_FT + cc * 64 + 16 * t;
#pragma unroll
    for (int sp = 0; sp < 2; ++sp) {
      uint32_t a0[2][4], a1[2][4];
      (void)a1;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int s2 = 2 * sp + h;
        uint32_t r0, r1, r2, r3;
        dq4((&w0.x)[s2], r0, r1, r2, r3);
        a0[h][0] = bmul2(r0, sg0); a0[h][1] = bmul2(r1, sg80); a0[h][2] = bmul2(r2, sg0); a0[h][3] = bmul2(r3, sg80);
#if OKC_NW == 8
        dq4((&w1.x)[s2], r0, r1, r2, r3);
        a1[h][0] = bmul2(r0, sg1); a1[h][1] = bmul2(r1, sg81); a1[h][2] = bmul2(r2, sg1); a1[h][3] = bmul2(r3, sg81);
#endif
      }
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const uint4 bv = *reinterpret_cast<const uint4*>(xr + nt * 8 * XP_FT + 8 * sp);
        mma_bf16(acc0[nt], a0[0][0], a0[0][1], a0[0][2], a0[0][3], bv.x, bv.y);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[0][0], a1[0][1], a1[0][2], a1[0][3], bv.x, bv.y);
#endif
        mma_bf16(acc0[nt], a0[1][0], a0[1][1], a0[1][2], a0[1][3], bv.z, bv.w);
#if OKC_NW == 8
        mma_bf16(acc1[nt], a1[1][0], a1[1][1], a1[1][2], a1[1][3], bv.z, bv.w);
#endif
      }
    }
  }
}
template <bool PB, bool BIG>
__device__ __forceinline__ void fat_stream(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;
  int* ctr = p.ctr + (PB ? 1 : 0);
  unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
  uint64_t* fb = S.bars + 5;
  __nv_bfloat16* xs0 = reinterpret_cast<__nv_bfloat16*>(ring + NSF * UNIT_BYTES);
  constexpr int NKS = HD / KS;
  int u = blockIdx.x;
  (void)ctr;
  if (u >= nF) return;
  constexpr int UBS = PB ? UB_B : UB_A;
  (void)ubs;
  auto job_of = [&](int uu, int& j, int& ub) { const int jq = uu / UBS; ub = uu - jq * UBS; j = S.forder[jq]; };
  auto wait_A = [&](int j) {
    if (PB) {
      if (tid == 0) { while (ld_acquire(p.doneA + j) < UB_A) __nanosleep(64); }
      CTA_SYNC();
    }
  };
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
  auto stage_x = [&](int j, int ks, int xsl) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int ntl = (cnt + 7) >> 3;
    const int nrow = ntl * 8;
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
    const int per = nrow * (KS / 8);
    if (is_nat(j)) {
      unsigned char* xsb = reinterpret_cast<unsigned char*>(xs);
      for (int i = tid; i < per; i += NT) {
        const int r = i >> 5, ch = i & 31;
        const bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + ch * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + ch * 8;
        cp_async16(reinterpret_cast<__nv_bfloat16*>(xsb + (ch >> 3) * 8192 + r * 128 + (((ch & 7) ^ (r & 7)) << 4)), src, v);
      }
    } else {
      for (int i = tid; i < per; i += NT) {
        int r = i / (KS / 8), c = i - r * (KS / 8);
        bool v = r < cnt;
        const __nv_bfloat16* src;
        if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + ks * KS + c * 8;
        else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + ks * KS + c * 8;
        cp_async16(xs + r * XP_FT + c * 8, src, v);
      }
    }
    cp_commit();
  };
  const __nv_bfloat16* xsrc[8];
  uint32_t xdst[8];
  uint32_t xval = 0;
  int xn = 0;
  auto prep_x = [&](int j) {
    const int cnt = S.jC[j], st = S.jS[j];
    const int per = ((cnt + 7) >> 3) * 8 * (KS / 8);
    xval = 0;
    xn = per > tid ? min(8, (per - tid + NT - 1) / NT) : 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int i = tid + k * NT;
      const int r = i >> 5, c = i & 31;
      const bool v = i < per && r < cnt;
      if (PB) xsrc[k] = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
      else xsrc[k] = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
      xdst[k] = r * XP_FT + c * 8;
      if (v) xval |= 1u << k;
    }
  };
  auto stage_xp = [&](int ks, int xsl) {
    __nv_bfloat16* xs = xs0 + xsl * (FT_T * XP_FT);
#pragma unroll
    for (int k = 0; k < 8; ++k)
      if (k < xn) cp_async16(xs + xdst[k], xsrc[k] + ks * KS, (xval >> k) & 1u);
    cp_commit();
  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int f = S.fseq;
  int xq = 0;
  int j, ub;
  job_of(u, j, ub);
#pragma unroll
  for (int k = 0; k < NSF; ++k) fat_issue<PB>(p, S.jE[j], ub, k, ring + ((f + k) % NSF) * UNIT_BYTES, fb + (f + k) % NSF, is_nat(j));
  wait_A(j);
  if (OKC_XPRE) { prep_x(j); stage_xp(0, 0); } else stage_x(j, 0, 0);
  while (true) {
    const int un = u + gridDim.x;
    int jn = 0, ubn = 0;
    if (un < nF) job_of(un, jn, ubn);
    const int e = S.jE[j], cnt = S.jC[j], st = S.jS[j];
    const int ntile = (cnt + 7) >> 3;
    const int ntc = ntile;
    const bool nat = is_nat(j);
    const int rb0 = ub * 16 + warp, rb1 = rb0 + 8;
    float acc0[FT_T / 8][4], acc1[FT_T / 8][4];
#pragma unroll
    for (int i = 0; i < FT_T / 8; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) { acc0[i][k] = 0.f; acc1[i][k] = 0.f; }
    for (int ks = 0; ks < NKS; ++ks) {
      if (OKC_XPRE) {
        if (ks + 1 < NKS) stage_xp(ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); prep_x(jn); stage_xp(0, (xq + 1) & 1); }
        else cp_commit();
      } else {
        if (ks + 1 < NKS) stage_x(j, ks + 1, (xq + 1) & 1);
        else if (un < nF) { wait_A(jn); stage_x(jn, 0, (xq + 1) & 1); }
        else cp_commit();
      }
      cp_wait<1>();
      const int q = f + ks, sl = q % NSF;
      mbar_wait(fb + sl, (uint32_t)((q / NSF) & 1));
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      CTA_SYNC();
      const unsigned char* slot = ring + sl * UNIT_BYTES;
      const __nv_bfloat16* xs = xs0 + (xq & 1) * (FT_T * XP_FT);
      if (nat) {
        switch (ntc) {
          case 1: fat_compute_wg<1>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 2: fat_compute_wg<2>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 3: fat_compute_wg<3>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 4: fat_compute_wg<4>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 5: fat_compute_wg<5>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 6: fat_compute_wg<6>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          case 7: fat_compute_wg<7>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
          default: fat_compute_wg<8>(S, slot, xs, warp, lane, acc0, acc1, p.zero); break;
        }
      } else {
        switch (ntc) {
          case 1: fat_compute<1>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 2: fat_compute<2>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 3: fat_compute<3>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 4: fat_compute<4>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 5: fat_compute<5>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 6: fat_compute<6>(S, slot, xs, warp, lane, acc0, acc1); break;
          case 7: fat_compute<7>(S, slot, xs, warp, lane, acc0, acc1); break;
          default: fat_compute<8>(S, slot, xs, warp, lane, acc0, acc1); break;
        }
      }
      CTA_SYNC();
      ++xq;
      const int kn = ks + NSF;
      if (kn < NKS) fat_issue<PB>(p, e, ub, kn, ring + sl * UNIT_BYTES, fb + sl, nat);
      else if (un < nF) fat_issue<PB>(p, S.jE[jn], ubn, kn - NKS, ring + sl * UNIT_BYTES, fb + sl, is_nat(jn));
    }
    const float gsc = (PB ? p.gsB[e] : p.gsA[e]) * TWO62;
    if (!PB) {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb0 * 8 + g, gsc, acc0[nt]);
#if OKC_NW == 8
          epi_A(p, st, cnt, nt * 8 + 2 * t, rb1 * 8 + g, gsc, acc1[nt]);
#endif
        }
      __threadfence();
      CTA_SYNC();
      if (tid == 0) atomicAdd(p.doneA + j, 1);
    } else {
#pragma unroll
      for (int nt = 0; nt < FT_T / 8; ++nt)
        if (nt < ntile) {
          float sq0[2], sq1[2];
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb0 * 16 + g, gsc, acc0[nt], sq0);
#if OKC_NW == 8
          epi_B(p, st, cnt, nt * 8 + 2 * t, rb1 * 16 + g, gsc, acc1[nt], sq1);
          sq0[0] += sq1[0]; sq0[1] += sq1[1];
#else
          (void)sq1;
#endif
          ss_reduce_add(p, st, cnt, nt * 8 + 2 * t, sq0, g);
        }
    }
    f += NKS;
    if (un >= nF) break;
    u = un; j = jn; ub = ubn;
  }
  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  if (tid == 0) S.fseq = f;
  CTA_SYNC();
}
template <bool PB>
__device__ __noinline__ void fat_stream_ni(const Params& p, Smem& S, __nv_bfloat16* xbuf, int nF, int ubs) {
  fat_stream<PB, false>(p, S, xbuf, nF, ubs);
}
__device__ void early_issue(const Params& p, Smem& S) {
#if FAST1_ON || OKC_WGS
  return;
#endif
  if (OKC_FATSTATIC && S.njF > 0) {
    if (TIDX == 0) S.preissued = 0;
    return;
  }
  const int nS = S.njS * RB_A - dyn_pool(S.njS * RB_A, gridDim.x);
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
  unsigned char* ring = reinterpret_cast<unsigned char*>(S.xbuf_ptr);
  const int n = min(OKC_EARLYN, v1 - v0);
  if (TIDX == NT - 32) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    for (int i = 0; i < n; ++i) { ring_issue<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i); S.uslot[i] = v0 + i; }
    S.preissued = n > 0 ? n : 0;
  }
}
#if OKC_WS
static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring);
template <bool PB>
__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  const int seq0 = seq;
  uint4 x0[2 * NCH];
  int curj = -1;
  float lgs = 0.f;
  uint32_t lnw0 = 0u, lnw1 = 0u;
  for (int i = 0; i < nU; ++i) {
    const int v = v0 + i;
    const int j = njF + v / rbs;
    if (PB && warp < OKC_WSBLK && i == nU - 1 - warp) {
      const int rbl = v - (v / RB_B) * RB_B, n0l = rbl * 16 + (lane >> 2);
      lgs = __ldg(p.gsB + S.jE[j]); lnw0 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l);
      lnw1 = (uint32_t)__ldg(reinterpret_cast<const unsigned short*>(p.normw) + n0l + 8);
    }
    if (j != curj) {
      curj = j;
      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);
    }
    const int q = seq0 + i;
    const int sl = q % OKC_NS;
    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#else
      const uint32_t sw = (c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w)) >> ((c & 1) * 16);
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
      const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
      const uint32_t b[8] = {x0[2 * c].x, x0[2 * c].y, x0[2 * c].z, x0[2 * c].w,
                             x0[2 * c + 1].x, x0[2 * c + 1].y, x0[2 * c + 1].z, x0[2 * c + 1].w};
#pragma unroll
      for (int st = 0; st < 4; ++st) {
        uint32_t r0, r1, r2, r3;
        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    const int bb = q % NRB;
    mbar_wait(S.wsre + bb, (uint32_t)(((q / NRB) & 1) ^ 1));
    *reinterpret_cast<float4*>(&S.red[bb][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
    CTA_SYNC();
    if (tid == 0) {
      const int nA = S.wsr[1] - S.wsr[0], N = nA + (S.wsr[3] - S.wsr[2]);
      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);
      if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);
    }
  }
  if (PB && warp < OKC_WSBLK && warp < nU) {
    const int ul = nU - 1 - warp;
    const int q = seq0 + ul, bb = q % NRB;
    const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;
    const int gg = lane >> 2, tt = lane & 3;
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    const int n0 = rb * 16 + gg;
    const float gsc = lgs * TWO62;
    const float nw0 = 1.f + __uint_as_float(lnw0 << 16), nw1 = 1.f + __uint_as_float(lnw1 << 16);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;
      const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
      float sq = v0f * v0f + v1f * v1f;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
        st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
    }
  }
  seq = seq0 + nU;
}
__device__ __forceinline__ void ws_issue(const Params& p, Smem& S, int q, int nA, int a0, int b0, unsigned char* ring) {
  const int sl = q % OKC_NS;
  if (q < nA) ring_issue<false>(p, S, a0 + q, ring + sl * UNIT_BYTES, S.bars + sl);
  else ring_issue<true>(p, S, b0 + (q - nA), ring + sl * UNIT_BYTES, S.bars + sl);
}
__device__ void ws_helper(const Params& p, Smem& S, unsigned char* ring) {
  const int lane = TIDX & 31;
  WS_JOIN();
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  const int njF = S.njF;
  if (lane == 0)
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
  const int gg = lane >> 2, tt = lane & 3;
  const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;
  for (int q = 0; q < Nh; ++q) {
    if (!OKC_WSRF && !OKC_WSB && lane == 0 && q + OKC_NS < N) {
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
      ws_issue(p, S, q + OKC_NS, nA, a0, b0, ring);
    }
    const bool PB = q >= nA;
    const int v = PB ? b0 + (q - nA) : a0 + q;
    const int rbs = PB ? RB_B : RB_A;
    const int j = njF + v / rbs, rb = v - (v / rbs) * rbs;
    const int bb = q % NRB;
    mbar_wait(S.wsrf + bb, (uint32_t)((q / NRB) & 1));
    float4 sm = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 t = *reinterpret_cast<const float4*>(&S.red[bb][w][lane][0]);
      sm.x += t.x; sm.y += t.y; sm.z += t.z; sm.w += t.w;
    }
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wsre + bb);
    const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
    if (!PB) {
      const float gsc = p.gsA[ce] * TWO62;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        if (tk < cnt) {
          const float gt = (h ? sm.y : sm.x) * gsc, up = (h ? sm.w : sm.z) * gsc;
          st_keep_bf16(p.inter + ((size_t)(st + tk) * ID + rb * 8 + gg), __float2bfloat16(probe_q(SILU_DIV(gt, 1.f + __expf(-gt)) * up)));
        }
      }
      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters
        __threadfence();
        __syncwarp();
        if (lane == 0) {
          const int jf = njF + a0 / RB_A, jl = njF + (a1 - 1) / RB_A;
          for (int jj = jf; jj <= jl; ++jj) {
            const int a = max(a0, (jj - njF) * RB_A), b = min(a1, (jj - njF + 1) * RB_A);
            atomicAdd(p.doneA + jj, b - a);
          }
        }
      }
    } else {
      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;
      const float nw0 = 1.f + __bfloat162float(p.normw[n0]), nw1 = 1.f + __bfloat162float(p.normw[n0 + 8]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = 2 * tt + h;
        const float v0f = (h ? sm.y : sm.x) * gsc * nw0, v1f = (h ? sm.w : sm.z) * gsc * nw1;
        float sq = v0f * v0f + v1f * v1f;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0), v0f);
          st_keep_f32(p.ybuf + ((size_t)(st + tk) * HD + n0 + 8), v1f);
        }
        sq += __shfl_xor_sync(0xffffffffu, sq, 4);
        sq += __shfl_xor_sync(0xffffffffu, sq, 8);
        sq += __shfl_xor_sync(0xffffffffu, sq, 16);
        if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);
      }
    }
  }
  __threadfence();
  WS_JOIN();
}
#endif
template <bool BIG, bool PB>
__device__ __forceinline__ void skinny_split(int nS, int nF, int G, int c, int& v0, int& v1) {
  const float rG = __fdividef(1.f, (float)G);
  const int q = muldiv_g(nF, 1, G, rG), r = nF - q * G;
  auto fcost = [](long long k) -> long long {
    const int fc = BIG ? (PB ? OKC_FCOSTBB : OKC_FCOSTB) : (PB ? OKC_FCOSTSB : OKC_FCOST);
    return k > 0 ? (long long)fc * k : 0;
  };
  const long long T = (long long)nS + (long long)r * fcost(q + 1) + (long long)(G - r) * fcost(q);
  long long wA = T - (long long)G * fcost(q + 1); if (wA < 0) wA = 0;
  long long wB = T - (long long)G * fcost(q); if (wB < 0) wB = 0;
  auto W = [&](int cc) -> long long { return cc <= r ? (long long)cc * wA : (long long)r * wA + (long long)(cc - r) * wB; };
  const long long WG = W(G);
  if (nF > 0 && WG > 0) {
    const float sc = __fdividef((float)nS, (float)WG);
    auto split = [&](int x) -> int {
      if (x >= G) return nS;
      const int v = __float2int_rz((float)W(x) * sc);
      return v < nS ? v : nS;
    };
    v0 = split(c); v1 = split(c + 1);
  } else {
    v0 = muldiv_g(nS, c, G, rG); v1 = muldiv_g(nS, c + 1, G, rG);
  }
}
#if OKC_WGS
#define OKC_WGSNX 3
constexpr int WGS_NS = 3;
constexpr int WGS_NX = OKC_WGSNX;
constexpr int WGS_NT = FT_T / 8;
constexpr int WGS_PIECE = 9216;
constexpr int WGS_XS = WGS_NT * 4096;
constexpr int BW_F = 0, BW_E = 3, BX_F = 6, BX_E = 6 + WGS_NX;
static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_SLOTS = 160;
constexpr int WGS_FIXW = WGS_NT * 512;
constexpr size_t WGS_FIXF = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW + 2 * WGS_SLOTS * 4 + 136 * 16;
__device__ __forceinline__ float* wgs_fix(const Params& p, int pb) {
  return p.ybuf + (size_t)EMAX * HD + (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW;
}
__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {
  return reinterpret_cast<int*>(p.ybuf + (size_t)EMAX * HD + (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW) + pb * WGS_SLOTS * 4;
}
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
constexpr unsigned long long wgs_pack8(const int (&c)[8]) {
  unsigned long long r = 0;
  for (int i = 0; i < 8; ++i) r |= (unsigned long long)(c[i] & 255) << (8 * i);
  return r;
}
constexpr int WGS_CA[8] = {OKC_WGSA};
constexpr int WGS_CB[8] = {OKC_WGSB};
constexpr unsigned long long WGS_PA = wgs_pack8(WGS_CA), WGS_PB = wgs_pack8(WGS_CB);
static_assert(WGS_CA[0] > 0 && WGS_CA[7] < 256 && WGS_CB[0] > 0 && WGS_CB[7] < 256, "WGS cost tables: 1..255");
__device__ __forceinline__ int wgs_cost(int pb, int cnt) {
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
}
#define OKC_WGSREORD 1
#define OKC_GATHB 8
#define OKC_WGSPR 1
#define OKC_WGSDFIN 2
#define OKC_WGSFCI 3
#define OKC_WGSGS 2
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
  (void)tag;
  mbar_wait(bar, parity);
}
template <int NTT> struct WGSMMA;
template <> struct WGSMMA<1> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<2> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<3> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n"
                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]), "=f"(d[6][0]), "=f"(d[6][1]), "=f"(d[6][2]), "=f"(d[6][3]), "=f"(d[7][0]), "=f"(d[7][1]), "=f"(d[7][2]), "=f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"
                 : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]), "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]), "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]), "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]), "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }
};
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {
  const int ue = ((g1 - 1) >> 4) << 4;
  return (OKC_WGSREORD && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}
__device__ void wgs_ranges(const Params& p, Smem& S);
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
  WS_JOIN();
  if ((TIDX & 31) != 0) return;
  int ws = 0, n = 0;
  uint32_t wph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const unsigned char* base = pb ? p.wBs : p.wAs;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      uint64_t* bar = S.bars + BW_F + ws;
      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
      if (++ws == WGS_NS) { ws = 0; wph ^= 1; }
    }
  }
}
constexpr int WGS_XROWS = EMAX + 8 * JMAX;
__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  if ((TIDX & 31) == 0) wgs_ranges(p, S);
  __syncwarp();
  WS_JOIN();
  if ((TIDX & 31) != 0) return;
  unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  int xs = 0, n = 0;
  uint32_t xph = 0;
#pragma unroll 1
  for (int pb = 0; pb < 2; ++pb) {
    const int g0 = S.wgr[2 * pb], g1 = S.wgr[2 * pb + 1], sp = wgs_split(pb, g0, g1);
    const int ubl = pb ? 4 : 5;
    const __nv_bfloat16* xsrc = pb ? p.xB : p.xA;
    int curj = -1, npad = 8;
    const __nv_bfloat16* jb = xsrc;
#pragma unroll 1
    for (int i = 0, gs = sp; i < g1 - g0; ++i, ++gs, ++n) {
      if (gs == g1) gs = g0;
      const int u = gs >> 4, st = gs & 15;
      const int j = u >> ubl;
      if (n >= WGS_NX) wgs_wait(S.bars + BX_E + xs, xph ^ 1, 2000000 + gs);
      if (j != curj) {
        curj = j;
        npad = ((S.jC[j] + 7) >> 3) << 3;
        jb = xsrc + (size_t)S.jrow[j] * HD;
        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }
      }
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }
      uint64_t* bar = S.bars + BX_F + xs;
      const uint32_t bytes = (uint32_t)npad * 512u;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                   ::"r"(smem_u32(xring + xs * WGS_XS)), "l"(jb + (size_t)st * npad * 256), "r"(bytes), "r"(smem_u32(bar)) : "memory");
      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }
    }
  }
}
__device__ void wgs_gather(const Params& p, Smem& S) {
  const int tid = TIDX, nj = S.njF + S.njS;
  const int R = S.jrow[nj];
  const float rG = __fdividef(1.f, (float)gridDim.x);
  const int q0 = muldiv_g(R, blockIdx.x, gridDim.x, rG), q1 = muldiv_g(R, blockIdx.x + 1, gridDim.x, rG);
  int j = 0;
  static_assert(NT * 8 == HD, "one 16 B chunk per thread per row");
  constexpr int GB = OKC_GATHB;
  for (int qb = q0; qb < q1; qb += GB) {
    uint4 v[GB];
    uint4* d[GB];
#pragma unroll
    for (int k = 0; k < GB; ++k) {
      const int q = qb + k;
      d[k] = nullptr;
      if (q < q1) {
        while (S.jrow[j + 1] <= q) ++j;
        const int r = q - S.jrow[j], cnt = S.jC[j];
        if (r < cnt) {
          const int ntt = (cnt + 7) >> 3;
          d[k] = reinterpret_cast<uint4*>(p.xA + (size_t)S.jrow[j] * HD) + ((((tid >> 5) * ntt + (r >> 3)) * 32 + (tid & 31)) * 8 + (r & 7));
          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < GB; ++k)
      if (d[k]) *d[k] = v[k];
  }
  CTA_SYNC();
  if (tid == 0) {
    int jj = 0;
    for (int q = q0; q < q1;) {
      while (S.jrow[jj + 1] <= q) ++jj;
      const int e = min(q1, S.jrow[jj + 1]);
      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");
      q = e;
    }
  }
}
struct WgsPend { int kind, u, slot, nst; };
template <bool PB>
__device__ __forceinline__ void wgs_epi_tile(const Params& p, const Smem& S, int j, int r16, int nt, const float* d4) {
  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], jst = S.jS[j], e = S.jE[j];
  if (!PB) {
    const int col = r16 * 8 + g, ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    __nv_bfloat16* xb = p.xB + (size_t)S.jrow[j] * HD + (col & 7);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float gt = d4[h] * gsc, up = d4[2 + h] * gsc;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        st_keep_bf16(xb + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8),
                     __float2bfloat16(probe_q(v)));
      }
    }
  } else {
    const int n0 = r16 * 16 + g, tok0 = nt * 8 + 2 * t;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = tok0 + h;
      if (tk < cnt) {
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0), d4[h] * w0);
        st_keep_f32(p.ybuf + ((size_t)(jst + tk) * HD + n0 + 8), d4[2 + h] * w1);
      }
    }
  }
}
template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
  if (pd.kind == 0) return;
  const int tid = TIDX, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const int j = pd.u >> UBL, ub = pd.u & ((1 << UBL) - 1);
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  if (pd.kind == 1) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
  int* fc = wgs_cnt(p, PB) + pd.slot * 4 + wg;
  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;
  if constexpr (DF) {
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {
  if (wt == 0) {
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(fc), "r"(pd.nst) : "memory");
    S.wglast[wg] = old + pd.nst == 16;
  }
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
  const bool last = S.wglast[wg];
  pd.kind = 0;
  if (!last) return;
  }
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  for (int nt0 = 0; nt0 < ntt; nt0 += 2) {
    float4 v[2];
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) v[k] = __ldcg(reinterpret_cast<const float4*>(fb + (nt0 + k) * 512));
#pragma unroll
    for (int k = 0; k < 2; ++k)
      if (nt0 + k < ntt) {
        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        const float d4[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
        wgs_epi_tile<PB>(p, S, j, r16, nt0 + k, d4);
      }
  }
  if (wt == 0) *fc = 0;
  if (!PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }
}
template <bool PB, int NTT>
__device__ __forceinline__ void wgs_seg(const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1, int slot,
                                        int& seq, WgsPend& pd) {
  const int tid = TIDX, lane = tid & 31;
  const int warp = WARP_IDX(tid);
  const int wg = warp >> 2, wp = warp & 3, g = lane >> 2, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;
  const uint32_t xring_s = smem_u32(ring + WGS_NS * UNIT_BYTES);
  float d[WGS_NT][4];
  constexpr int GS = OKC_WGSGS;
  uint32_t af[2][GS][4];
  int prev = -1;
  const uint64_t xd0 = wg_desc(xring_s, 128, 4096);
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  auto stage = [&](auto first) {
    constexpr bool FIRST = decltype(first)::value;
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {
        mbar_arrive1(S.bars + BW_E + ws);
      }
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          const int js = h * GS + jj;
          const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
          uint32_t r0, r1, r2, r3;
          dq4(q, r0, r1, r2, r3);
          const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
          const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
        }
        wg_fence();
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          if (FIRST && ci == 0 && jj == 0)
            WGSMMA<NTT>::mma0(d, af[0][0][0], af[0][0][1], af[0][0][2], af[0][0][3], xd);
          else
            WGSMMA<NTT>::mma(d, af[ci & 1][jj][0], af[ci & 1][jj][1], af[ci & 1][jj][2], af[ci & 1][jj][3],
                             xd + (uint64_t)((lg * 4 + h * GS + jj) * 16));
        }
        wg_commit();
        wg_wait<1>();
        if (ci == 0) {
          if (prev >= 0) {
            if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
        if (FIRST && ci == OKC_WGSFCI && pd.kind) wgs_flush<PB>(p, S, pd);
      }
    }
    prev = xs;
    ++seq;
  };
  stage(std::true_type{});
#pragma unroll 1
  for (int st = s0 + 1; st < s1; ++st) stage(std::false_type{});
  wg_wait<0>();
  if (lane == 0) mbar_arrive1(S.bars + BX_E + prev);
  if (OKC_WGSX & 8) { if (d[0][0] == 1234.5f) S.wglast[0] = 7; return; }
  if (slot >= 0) {
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
  if (!PB) pd = WgsPend{1, u, -1, 0};
}
template <bool PB>
__device__ __forceinline__ void wgs_phase(const Params& p, Smem& S, unsigned char* ring, int g0, int g1, int& seq) {
  constexpr int UBL = PB ? 4 : 5;
  WgsPend pd{0, 0, 0, 0};
#pragma unroll 1
  for (int gs = wgs_split(PB, g0, g1), ge = g1;;) {
    if (gs >= ge) {
      const int a0 = S.wgr[2 * PB], a1 = S.wgr[2 * PB + 1], sp = wgs_split(PB, a0, a1);
      if (ge == a1 && sp > a0) { gs = a0; ge = sp; } else break;
    }
    const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));
    const int ntt = (S.jC[u >> UBL] + 7) >> 3;
    switch (ntt) {
      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 3: wgs_seg<PB, 3>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 4: wgs_seg<PB, 4>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 6: wgs_seg<PB, 6>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      case 7: wgs_seg<PB, 7>(p, S, ring, u, s0, s1, slot, seq, pd); break;
      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;
    }
    gs += s1 - s0;
  }
  wgs_flush<PB>(p, S, pd);
}
__device__ __forceinline__ void wgs_prefix(Smem& S, int nj) {
  uint32_t acc = 0, accB = 0;
  for (int j = 0; j < nj; ++j) {
    S.wcp[j] = (int)acc; S.wcpB[j] = (int)accB;
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
  }
  S.wcp[nj] = (int)acc; S.wcpB[nj] = (int)accB;
  S.wtot = acc; S.wtotB = accB;
}
__device__ __forceinline__ int wgs_stage_at(const Smem& S, int nj, int pb, uint32_t x) {
  const int* pre = pb ? S.wcpB : S.wcp;
  int lo = 0, hi = nj;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if ((uint32_t)pre[mid] <= x) lo = mid; else hi = mid; }
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
}
__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  g0 = wgs_stage_at(S, nj, pb, tot * (uint32_t)c / (uint32_t)G);
  g1 = c == G - 1 ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c + 1) / (uint32_t)G);
}
__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {
  const int nj = S.njF + S.njS;
  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
}
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G) {
  const uint32_t tot = pb ? S.wtotB : S.wtot;
  const int j = s >> (9 - pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);
  if (c > G - 1) c = G - 1;
  while (c + 1 < G && tot * (uint32_t)(c + 1) / (uint32_t)G < xe) ++c;
  while (c > 0 && tot * (uint32_t)c / (uint32_t)G >= xe) --c;
  return c;
}
#endif
template <bool PB, bool BIG>
__device__ __forceinline__ void run_phase(const Params& p, Smem& S, __nv_bfloat16* xbuf) {
  const int njF = S.njF, njS = S.njS;
  const int ubs = PB ? UB_B : UB_A, rbs = PB ? RB_B : RB_A;
  const int nS = njS * rbs, nF = njF * ubs;
#if OKC_TINY
  if (nF > 0) __trap();
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
  const int G = gridDim.x, c = blockIdx.x;
  int v0, v1;
  const int nSs = nS - dyn_pool(nS, G);
  skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);
  int cB0 = 0, cBn = 0;
  bool cpf = false;
  if (!PB) {
    const int nFB = njF * UB_B;
    if (c >= nFB) {
      int b1;
      skinny_split<BIG, true>(njS * RB_B - dyn_pool(njS * RB_B, G), nFB, G, c, cB0, b1);
      cBn = b1 - cB0;
      cpf = OKC_CPFF || njF == 0;
    }
  }
#if OKC_WS
  (void)cpf; (void)cBn;
  skinny_ring_ws<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq);
#else
  skinny_ring<PB>(p, S, v0, v1, reinterpret_cast<unsigned char*>(xbuf), S.bars, S.seq,
                  reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<unsigned char*>(xbuf) + OKC_NS * UNIT_BYTES), cpf, cB0, cBn, nS, nSs);
#endif
}
__device__ __forceinline__ float* ss_cur() {
#if OKC_WGS
  unsigned long long v;
  asm volatile("ld.volatile.shared.u64 %0, [%1];\n" : "=l"(v) : "r"(smem_u32(&reinterpret_cast<const Smem*>(smem_raw)->ssp)));
  return reinterpret_cast<float*>(v);
#else
  return reinterpret_cast<const Smem*>(smem_raw)->ssp;
#endif
}
template <bool BIG>
__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {
  Smem& S = *reinterpret_cast<Smem*>(smem_raw);
  __nv_bfloat16* xbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + ((sizeof(Smem) + 127) / 128) * 128);
  const int tid = TIDX;
  DBG(0);
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WS
  if (tid >= NT) { ws_helper(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif
#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif
#if FAST1_ON
  if (tid >= NT - 32) {
    const int ln = tid & 31;
#if OKC_SPLIT
    S.fcnt[ln] = 0;
    __syncwarp();
    for (int i0 = 0; i0 < p.M * 8; i0 += 32) {
      const int i = i0 + ln;
      int e = -1;
      if (i < p.M * 8) { e = __ldg(p.topk_ids + i) - p.lo; if (e < 0 || e >= p.nloc) e = -1; }
      const unsigned peers = __match_any_sync(0xffffffffu, e);
      if (e >= 0 && ln == __ffs(peers) - 1) atomicAdd(&S.fcnt[e], __popc(peers));
    }
    __syncwarp();
    int jincl = (S.fcnt[ln] + SK_T - 1) / SK_T;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int v = __shfl_up_sync(0xffffffffu, jincl, d); if (ln >= d) jincl += v; }
    const int njob = __shfl_sync(0xffffffffu, jincl, 31);
    const int nS1 = njob * RB_A - dyn_pool(njob * RB_A, gridDim.x);
    const float rG1 = __fdividef(1.f, (float)gridDim.x);
    const int v01 = muldiv_g(nS1, blockIdx.x, gridDim.x, rG1), v11 = muldiv_g(nS1, blockIdx.x + 1, gridDim.x, rG1);
    const int n1 = min(OKC_FAST1, v11 - v01);
    int fex[OKC_FAST1 > 0 ? OKC_FAST1 : 1];
#pragma unroll
    for (int i = 0; i < OKC_FAST1; ++i) {
      const int jj = (v01 + i) >> RB_A_LOG;
      fex[i] = i < n1 ? __ffs(__ballot_sync(0xffffffffu, jincl > jj)) - 1 : 0;
    }
#elif OKC_F1EARLY
    uint32_t m = 0;
    const int E8 = p.M * 8;
    const int ev0 = ln < E8 ? __ldg(p.topk_ids + ln) : 0, ev1 = ln + 32 < E8 ? __ldg(p.topk_ids + ln + 32) : 0;
    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    if (ln < E8) { const int e = ev0 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    if (ln + 32 < E8) { const int e = ev1 - p.lo; if (e >= 0 && e < p.nloc) m |= 1u << e; }
    for (int i = ln + 64; i < E8; i += 32) {
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
#endif
    if (ln == 0) {
#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
      for (int k = 0; k < OKC_NS; ++k) mbar_init(S.wse + k, NW);
      for (int k = 0; k < NRB; ++k) { mbar_init(S.wsrf + k, OKC_WSB ? 1 : NW); mbar_init(S.wsre + k, 1); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
#if OKC_SPLIT
      const int v0 = v01, n = n1;
#else
      const int nS = __popc(m) * RB_A - dyn_pool(__popc(m) * RB_A, gridDim.x);
      const float rG = __fdividef(1.f, (float)gridDim.x);
      const int v0 = muldiv_g(nS, blockIdx.x, gridDim.x, rG), v1 = muldiv_g(nS, blockIdx.x + 1, gridDim.x, rG);
      const int n = min(OKC_FAST1, v1 - v0);
#endif
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#if OKC_SPLIT
#pragma unroll
      for (int i = 0; i < OKC_FAST1; ++i) {
        if (i >= n) break;
        const int v = v0 + i;
        const int e = fex[i];
#else
      for (int i = 0; i < n; ++i) {
        const int v = v0 + i;
        const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);
#endif
        const unsigned char* src = p.wAu + ((size_t)e * RB_A + (v & (RB_A - 1))) * (size_t)UNIT_BYTES;
        unsigned char* slot = reinterpret_cast<unsigned char*>(xbuf) + i * UNIT_BYTES;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(S.bars + i)), "r"(UNIT_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
      }
      S.preissued = n > 0 ? n : 0;
    }
  }
#endif
  for (int li = blockIdx.x * NT + tid; li < p.M * (HD * 2 / 128); li += gridDim.x * NT)
    asm volatile("prefetch.global.L2 [%0];\n" ::"l"(reinterpret_cast<const char*>(p.x) + (size_t)li * 128));
  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {
    const int sgn = i >> 7, ex = (i >> 3) & 15, mt = i & 7;
    uint32_t b;
    if (ex == 15 && mt == 7) {
      b = 0xFFFFu;
    } else {
      float v = ex == 0 ? (float)mt * 0x1p-9f : (1.f + (float)mt * 0.125f) * __int_as_float((ex - 7 + 127) << 23);
      v *= 0x1p64f;
      b = __float_as_uint(sgn ? -v : v) >> 16;
    }
    S.lut[i] = b | (b << 16);
  }
  if (tid == 0) {
#if !FAST1_ON
#if OKC_WGS
    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);
#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    S.preissued = 0;
#endif
    S.seq = 0;
    S.fseq = 0;
    S.preissuedB = 0;
    S.xbuf_ptr = xbuf;
    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
    S.gen = gen;
    S.ssp = (gen & 1) ? p.ss1 : p.ss;
  }
#if SKDYN
  for (int i = tid; i < JMAX; i += NT) S.dync[i] = 0;
#endif
  route(p, S);
#if OKC_WGS && OKC_WGSPR
  __threadfence_block();
  asm volatile("bar.arrive 7, %0;\n" ::"n"(NT + 32) : "memory");
#endif
  DBG(1);
  {
    float* other = (S.gen & 1) ? p.ss : p.ss1;
    for (int i = blockIdx.x * NT + tid; i < EMAX; i += gridDim.x * NT) other[i] = 0.f;
    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;
  }
#if OKC_WS
  if (tid == 0) {
    const int G = gridDim.x, c = blockIdx.x;
    const float rG = __fdividef(1.f, (float)G);
    const int nA = S.njS * RB_A, nB = S.njS * RB_B;
    S.wsr[0] = muldiv_g(nA, c, G, rG); S.wsr[1] = muldiv_g(nA, c + 1, G, rG);
    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);
  }
  WS_JOIN();
#endif
#if OKC_WGS
  if (tid < 32) {
    const int nj = S.njF + S.njS;
    int carry = 0;
    for (int j0 = 0; j0 < nj; j0 += 32) {
      const int j = j0 + tid;
      const int v = j < nj ? (((S.jC[j] + 7) >> 3) << 3) : 0;
      int incl = v;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int t = __shfl_up_sync(0xffffffffu, incl, d); if (tid >= d) incl += t; }
      if (j < nj) S.jrow[j] = carry + incl - v;
      carry += __shfl_sync(0xffffffffu, incl, 31);
    }
    if (tid == 0) S.jrow[nj] = carry;
  }
  CTA_SYNC();
  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);
  {
    unsigned char* ring = reinterpret_cast<unsigned char*>(xbuf);
    int seq = 0;
    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);
    DBG(3);
  }
#else
  run_phase<false, BIG>(p, S, xbuf);
  DBG(2);
  run_phase<true, BIG>(p, S, xbuf);
  DBG(3);
#endif
#if OKC_WS
  WS_JOIN();
#endif
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#if OKC_FINPF && OKC_FINCOL && !OKC_WGS
  constexpr int FSEGP = HD / (NT * 4);
  const int itp = blockIdx.x;
  const bool hasp = itp < p.M * FSEGP;
  const int tkp = hasp ? itp / FSEGP : 0;
  int qvp[8];
  float wvp[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    qvp[s] = hasp ? (int)S.pos[tkp * 8 + s] : -1;
    wvp[s] = (qvp[s] >= 0) ? __ldg(p.topk_w + tkp * 8 + s) : 0.f;
  }
#endif
  CTA_SYNC();
  if (tid == 0) {
    int* bc = p.ctr + 4 + (S.gen & 1);
    __threadfence();
    const int old = atomicAdd(bc, 1);
    if (old == (int)gridDim.x - 1) atomicAdd(p.ctr + 3, 1);
    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(64);
  }
  CTA_SYNC();
  DBG(4);
  {
    const int G = gridDim.x, i0 = blockIdx.x * NT + tid;
    if (i0 < 2) p.ctr[i0] = 0;
    for (int i = i0; i < JMAX; i += G * NT) p.doneA[i] = 0;
#if OKC_WGS
    for (int i = i0; i < JMAX; i += G * NT) p.xready[i] = 0;
    for (int i = i0; i < JMAX * 16; i += G * NT) p.doneAs[i] = 0;
#endif
  }
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();
    }
  }
#elif OKC_FINCOL
  constexpr int FSEG = HD / (NT * 4);
  int it0 = blockIdx.x;
#if OKC_FINPF
  if (hasp) {
    const int col = (itp % FSEG) * (NT * 4) + tid * 4;
    float ssv[8];
    float4 yv[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      ssv[s] = qvp[s] >= 0 ? __ldcg(ss_cur() + qvp[s]) : 0.f;
      yv[s] = qvp[s] >= 0 ? __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qvp[s] * HD + col)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (qvp[s] >= 0) {
        const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
        acc.x += r * yv[s].x; acc.y += r * yv[s].y; acc.z += r * yv[s].z; acc.w += r * yv[s].w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;
  }
  it0 += gridDim.x;
#endif
  for (int it = it0; it < p.M * FSEG; it += gridDim.x) {
    const int tk = it / FSEG, col = (it % FSEG) * (NT * 4) + tid * 4;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD + col));
        acc.x += r * v.x; acc.y += r * v.y; acc.z += r * v.z; acc.w += r * v.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(acc.x, acc.y), o1 = __floats2bfloat162_rn(acc.z, acc.w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&o0);
    o.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = o;
  }
#else
  for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int q = S.pos[tk * 8 + s];
      if (q >= 0) {
        const float r = p.topk_w[tk * 8 + s] * rsqrtf(__ldcg(ss_cur() + q) * (1.f / HD) + 1e-8f);
        const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float4 v = __ldcg(src + i);
          acc[4 * i] += r * v.x; acc[4 * i + 1] += r * v.y; acc[4 * i + 2] += r * v.z; acc[4 * i + 3] += r * v.w;
        }
      }
    }
    uint4 o[2];
    uint32_t* ow = reinterpret_cast<uint32_t*>(o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
      ow[i] = *reinterpret_cast<uint32_t*>(&v);
    }
    uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
    dst[0] = o[0]; dst[1] = o[1];
  }
#endif
  DBG(5);
}
}
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
static size_t okc_smem_bytes() {
  size_t base = ((sizeof(okc::Smem) + 127) / 128) * 128;
#if OKC_WGS
  return base + (size_t)okc::WGS_NS * okc::UNIT_BYTES + (size_t)okc::WGS_NX * okc::WGS_XS;
#endif
  size_t xa = (size_t)OKC_NS * okc::UNIT_BYTES + (OKC_SKT > 8 ? (size_t)okc::NW * 8 * okc::X1P * 2 : 0);
  size_t xb = (size_t)okc::NSF * okc::UNIT_BYTES + 2 * (size_t)okc::FT_T * okc::XP_FT * 2;
  return base + (xa > xb ? xa : xb);
}
static int g_okc_pdl = 0;
void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {
  okc::Params p;
  p.wA = (const uint4*)w[0].data_ptr(); p.sA = (const uint4*)w[1].data_ptr(); p.gsA = (const float*)w[2].data_ptr();
  p.wB = (const uint4*)w[3].data_ptr(); p.sB = (const uint4*)w[4].data_ptr(); p.gsB = (const float*)w[5].data_ptr();
  p.lut = (const uint32_t*)w[6].data_ptr(); p.normw = (const __nv_bfloat16*)w[7].data_ptr();
  p.wAf = (const unsigned char*)w[8].data_ptr(); p.wBf = (const unsigned char*)w[9].data_ptr();
  p.wAu = (const unsigned char*)w[10].data_ptr(); p.wBu = (const unsigned char*)w[11].data_ptr();
  if (w.size() > 13) { p.wAn = (const unsigned char*)w[12].data_ptr(); p.wBn = (const unsigned char*)w[13].data_ptr(); }
  else { p.wAn = p.wAf; p.wBn = p.wBf; }
#if OKC_WGS
  TORCH_CHECK(w.size() > 13, "OKC_WGS needs the WGS weight copies (w[12], w[13])");
  TORCH_CHECK((size_t)scratch[1].numel() >= (size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,
              "OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])");
  p.wAs = (const unsigned char*)w[12].data_ptr(); p.wBs = (const unsigned char*)w[13].data_ptr();
#else
  p.wAs = nullptr; p.wBs = nullptr;
#endif
  p.zero = 0;
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = (const int*)ids.data_ptr(); p.topk_w = (const float*)tw.data_ptr();
  p.M = (int)x.size(0); p.lo = (int)lo; p.nloc = (int)nloc;
#if OKC_WGS && OKC_RLOAD
  TORCH_CHECK(((uintptr_t)ids.data_ptr() & 15) == 0 && ids.is_contiguous(), "topk_ids must be 16 B aligned and contiguous");
#endif
  int* ib = (int*)scratch[0].data_ptr();
  p.ctr = ib; p.doneA = ib + 8;
  p.ss = (float*)scratch[1].data_ptr();
  TORCH_CHECK(scratch[0].numel() >= 8 + 1024 + okc::EMAX, "scratch[0] needs the second ss copy");
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_WGS
  TORCH_CHECK((size_t)scratch[2].numel() >= ((size_t)okc::EMAX + 2 * (size_t)okc::WGS_XROWS) * okc::HD,
              "OKC_WGS needs xA / xB after inter (scratch[2])");
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.xready = ib + 8 + 512;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
#else
  p.xA = p.xB = nullptr;
  p.xready = nullptr;
  p.doneAs = nullptr;
#endif
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;
  static bool attr = false;
  size_t sm = okc_smem_bytes();
  if (!attr) {
    cudaError_t e0 = cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t e1 = cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    (void)e0; (void)e1;
    attr = true;
  }
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (g_okc_pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)grid);
    cfg.blockDim = dim3(okc::NTL);
    cfg.dynamicSmemBytes = sm;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    if (p.M >= OKC_BIGM) cudaLaunchKernelEx(&cfg, okc::moe_fused<true>, p);
    else cudaLaunchKernelEx(&cfg, okc::moe_fused<false>, p);
    return;
  }
  if (p.M >= OKC_BIGM) okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);
  else okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);
}
"""
kl_ATT7_CUDA_SRC = r"""#include <torch/extension.h>
// LDD19 solo_optimizer r5: eat_attn items, set per build by -D (0 = off, the previous behaviour):
// LDD_PSORT 7: q_len 1 flat, each segment's full old tiles after the first LDD_PSORT_F stream in physical-row bucket order (d16 f, d32 f)
// LDD_RBFQ: BF16x8 merge reds + BF16 fin_nc1 read at nc 1: bit 0 q_len 1 per-request plain (<1,0,1>: d16 s, d32 s), bit 1 flat q_len 1 (d16 f, d32 f), bit 2 + <1,0,13> etc
// LDD_RBF: 1 flat q_len 8 coop (v16 f, v32 f), 2 + <8,0,11> (v1 f; 8 B stack in kl), 3 + <1,0,11>, 4 = 1 + <1,0,11> without <8,0,11>; LDD_RBFS: bit 0 q_len 8 per-request coop (v16 s, v32 s)
#ifndef LDD_PSORT
#define LDD_PSORT 0
#endif
#ifndef LDD_PSORT_CAP
#define LDD_PSORT_CAP 7936
#endif
#ifndef LDD_PSORT_F
#define LDD_PSORT_F 24
#endif
#ifndef LDD_PSORT_NH
#define LDD_PSORT_NH 6
#endif
#ifndef LDD_RBFQ
#define LDD_RBFQ 0
#endif
#ifndef LDD_RBFS
#define LDD_RBFS 0
#endif
#ifndef LDD_RBF
#define LDD_RBF 0
#endif
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <stdint.h>
namespace {
#define OKC_NP 128
#define OKC_WS 1
#define OKC_WS8 1
#define OKC_NP8 128
#define OKC_WSMINB 2
#define OKC_SETMAXNREG 1
#define OKC_PREGS 0
#define OKC_TMAPG 1
#define OKC_XHALFPV 0
#define OKC_XNOWAIT 0
#define OKC_MRGDBG 0
#define OKC_MERGE2 1
#define OKC_MERGE3 1
#define OKC_IDXFIRST 1
#define OKC_MAGICDIV 1
#define OKC_SC1 1
#define OKC_QP16 1
#define OKC_ONENEW 1
#define OKC_SC8 1
#define OKC_PINDIV 1
#define OKC_ONEPRED 1
#define OKC_LATETID 1
#define OKC_PGCONT 0
#define OKC_TMAPG8 0
#define OKC_ST8F 4
#define OKC_COOP 1
#define OKC_PVW 0
#define OKC_WGDEC 1
#define OKC_SEGDBG 0
#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)
#define OKC_PV8 1
#define OKC_PROF 0
#define OKC_PROFPV8 0
#define PV8ON (OKC_PV8 && (!OKC_PROF || OKC_PROFPV8) && !OKC_PVW)
#define OKC_PV8FLAT 1
#define OKC_QSS 1
#define OKC_PV8SYNC 0
#define OKC_Q8RAW 1
#define OKC_FLATCOOP1C 0
#define OKC_RAWTHR 0
#define OKC_PCHK 1
#define OKC_TURN 0
#define OKC_CLSM 0
#define OKC_PDLKV 0
#define OKC_RARE 1
#define OKC_PCHKTEST 0
#define OKC_UNEV 0
#define OKC_W0L1 0
#define OKC_SEGK8 600
#define OKC_SEGK1 400
#define OKC_W0SPREAD 0
#define OKC_W0EARLY 1
#define OKC_W0FORCEBAD 0
#define OKC_W0BID 1
#define OKC_DBGON 0
#define OKC_BALPART 0
#define OKC_CONT 0
#define OKC_DYN 0
#define OKC_FINWARM 0
#define OKC_PVW1 1
#define OKC_NOXCH1 0
#define OKC_PVSPLIT 1
#define OKC_QKZ 0
#define OKC_INV2 1
#define OKC_OZF 1
#define OKC_XP_ARRALL 1
#define OKC_XP_NOMASK 0
#define OKC_IRANGE 0
#define OKC_PATCHNI 1
#define OKC_PARITY 1
#define OKC_SENSE2 0
#define OKC_UNRN 1
#define OKC_ADDRN 1
#define OKC_OFF32 1
#define OKC_UNR8 2
#define OKC_RED4 1
#define OKC_SWAPHALF 0
#define OKC_LATESYNC 1
#define OKC_LATESYNCN 1
#define OKC_QKFIRST 1
#define OKC_FPART 1
#define OKC_EARLYBAR 1
#define OKC_PTPF 1
#define OKC_FPRO 0
#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)
#define OKC_COOPGUARD 1
__device__ __noinline__ void spin_fail() { __trap(); }
#define SPIN_T0 unsigned _spin_n = 0
#define SPIN_CHECK do { if (OKC_COOPGUARD && ++_spin_n > (1u << 24)) spin_fail(); } while (0)
#define OKC_ST1F 3
#define OKC_STN 3
#define OKC_UWARP 1
#define OKC_INEW 0
#define OKC_ADDR 1
#define OKC_STSM 0
#define OKC_STSMPR 1
#define OKC_TCHUNK 1
#define OKC_EXPPOLY 0
#define OKC_EXH8 0
#define OKC_NOSYNC0 1
#define OKC_LAZYMAX 1
#define OKC_SENSE 0
#define OKC_WSPLIT 500
#define OKC_F2 0
#define OKC_S2 0
#define OKC_PTR 0
#define OKC_PTRD 2
#define OKC_XP_NOTR 0
#define OKC_XP_NOFENCE 0
#define OKC_XP_NOEXP 0
#define OKC_CLDBG 0
#define OKC_SPINDBG 0
template <int L, bool F>
struct Cfg {
  static constexpr bool S2 = (L == 8) && !F && OKC_S2 && OKC_SM90 && PV8ON;
  static constexpr bool F2 = (L == 8) && F && OKC_F2 && OKC_SM90 && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT && !OKC_PTR;
  static constexpr int NT = (L == 1) ? 128 : ((S2 || F2) ? 512 : 256);
  static constexpr bool WS = F && OKC_SM90 && OKC_WS && (L == 1 || OKC_WS8);
  static constexpr int NP = WS ? ((L == 1) ? OKC_NP : OKC_NP8) : 0;
  static constexpr int NTX = NT + NP;
  static constexpr int MINB = WS ? (F2 ? 1 : OKC_WSMINB) : (S2 ? 1 : 2);
  static constexpr int BASEREGS = (65536 / (MINB * NTX)) / 8 * 8 > 255 ? 248 : (65536 / (MINB * NTX)) / 8 * 8;
  static constexpr int PREGS = OKC_PREGS > 0 ? OKC_PREGS : ((L == 1 && NP <= 128) ? 56 : 32);
  static constexpr int CREGS = ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8 > 248
                                   ? 248 : ((BASEREGS * NTX - PREGS * NP) / (NT > 0 ? NT : 1)) / 8 * 8;
  static constexpr int STAGES = F ? ((L == 1) ? OKC_ST1F : (F2 ? 8 : OKC_ST8F)) : (S2 ? 6 : OKC_STN);
  static_assert(!WS || (NP % 8 == 0 && NP >= 8 && 64 % (NP >= 8 ? NP / 8 : 1) == 0),
                "producer threads must cover a tile in whole passes");
};
constexpr int TILE = 64;
constexpr int STAGES = 4;
constexpr int ROWB = 128;
constexpr int STAGE_BYTES = TILE * ROWB;
constexpr int IDX_SLOTS = 2 * STAGES;
constexpr int MAXB = 512;
constexpr int MAXSEG = 64;
struct Params {
  const __nv_bfloat16* qkv;
  const int* seq_lens;
  const int64_t* cache_loc;
  uint8_t* k_cache;
  uint8_t* v_cache;
  const int* page_table;
  const float* k_scale;
  const float* v_scale;
  const float* inv_freq;
  __nv_bfloat16* out;
  float* ws_acc;
  float* ws_l;
  int* ws_cnt;
  int B, Hq, Hkv, P, max_pages, window, rope, n_splits, flat, nc;
  long long* dbg;
  int cs, ncl;
  float* scratch;
  int* bar;
  int ld_qkv, row_bytes;
  float scale_log2;
  int tma;
  int coop;
  int lp;
  int lrb;
  int off32;
  int w0;
  int resident;
  unsigned ns_magic;
  int unev;
  int pdlf;
  CUtensorMap tk, tv;
  CUtensorMap tpt;
  const uint8_t* pf;
  unsigned pf_chunk;
};
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async16_pred(uint32_t dst, const void* src, int src_bytes, bool pred) {
  asm volatile("{\n .reg .pred q;\n setp.ne.b32 q, %3, 0;\n @q cp.async.cg.shared.global [%0], [%1], 16, %2;\n}\n"
               ::"r"(dst), "l"(src), "r"(src_bytes), "r"((int)pred));
}
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
#define OKC_TMA 0
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_init_n(uint64_t* bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {
  asm volatile("{\n.reg .pred q;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra WAIT_%=;\n}\n"
               ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(dst), "l"(any_src));
}
__device__ __forceinline__ void bar_sync_n(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive_n(int id, int n) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void wgsync(int wg) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
}
template <bool WS, int NT>
__device__ __forceinline__ void csync() {
  if constexpr (WS) asm volatile("bar.sync 1, %0;\n" ::"n"(NT) : "memory");
  else __syncthreads();
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
__device__ __forceinline__ uint32_t cvt2h(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {_, t}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t cvt2(uint32_t x) {
  uint32_t d;
  asm("{\n\t.reg .b16 t;\n\tmov.b32 {t, _}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, t;\n\t}" : "=r"(d) : "r"(x));
  return d;
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t ex2h2(float lo, float hi) {
  return pack_h2(ex2(lo), ex2(hi));
}
__device__ __forceinline__ float pr8(float x) {
  return x;
}
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ void mma_f16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint4 lds128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint2 lds64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ void load_v_frags(const uint8_t* V, int kb, int hk, int g, int tig, uint32_t* b0,
                                             uint32_t* b1) {
  const int ch = hk * 4 + (g >> 1), off = (g & 1) * 8;
  const int k0 = kb + 2 * tig;
  const uint2 v0 = lds64(V + k0 * ROWB + ((ch ^ (k0 & 7)) << 4) + off);
  const uint2 v1 = lds64(V + (k0 + 1) * ROWB + ((ch ^ ((k0 + 1) & 7)) << 4) + off);
  const uint2 v2 = lds64(V + (k0 + 8) * ROWB + ((ch ^ ((k0 + 8) & 7)) << 4) + off);
  const uint2 v3 = lds64(V + (k0 + 9) * ROWB + ((ch ^ ((k0 + 9) & 7)) << 4) + off);
  uint32_t x;
  x = prmt(v0.x, v1.x, 0x5140); b0[0] = cvt2(x); b0[1] = cvt2h(x);
  x = prmt(v0.x, v1.x, 0x7362); b0[2] = cvt2(x); b0[3] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x5140); b0[4] = cvt2(x); b0[5] = cvt2h(x);
  x = prmt(v0.y, v1.y, 0x7362); b0[6] = cvt2(x); b0[7] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x5140); b1[0] = cvt2(x); b1[1] = cvt2h(x);
  x = prmt(v2.x, v3.x, 0x7362); b1[2] = cvt2(x); b1[3] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x5140); b1[4] = cvt2(x); b1[5] = cvt2h(x);
  x = prmt(v2.y, v3.y, 0x7362); b1[6] = cvt2(x); b1[7] = cvt2h(x);
}
__device__ __forceinline__ void gpu_fence() {
  __threadfence();
}
__device__ __forceinline__ void red_f32(float* a, float x) {
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory");
}
// LDD19 LDD_RBF: 8 BF16 adds in one 16-byte reduction (REDG.E.ADD.BF16x8 on sm_90a)
__device__ __forceinline__ void red_bf8(void* a, const float (&v)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    __nv_bfloat162 h = __floats2bfloat162_rn(v[2 * e], v[2 * e + 1]);
    w[e] = *reinterpret_cast<uint32_t*>(&h);
  }
  asm volatile("red.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};" ::"l"(a), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
}
__device__ __forceinline__ void red_v4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
#define DBG(k)                                                                                      \
  if (OKC_DBGON && p.dbg && threadIdx.x == 0)                                                       \
    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k)] = gtime();
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)
               : "memory");
}
#define PROF_INIT
#define PROF_MARK(k)
#define PROF_TILE
#define PROF_DUMP
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void reg_fence(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
__device__ __forceinline__ void wgmma_m64n64k32_e4m3(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                     uint64_t desc, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "%32, %33, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]),
        "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
        "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]),
        "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
      : "l"(da), "l"(db), "r"(0));
}
__device__ __forceinline__ uint64_t desc_sw64(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(512 >> 4) << 32) | ((uint64_t)2 << 62);
}
__device__ __forceinline__ void wgmma_m64n72k32_e4m3(float* d, float* e, const uint32_t* a, uint64_t desc) {
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, "
      "{%36, %37, %38, %39}, %40, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(e[0]), "+f"(e[1]), "+f"(e[2]), "+f"(e[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
}
__device__ __forceinline__ int fdiv_i(int x, int d) {
  const float rd = __fdividef(1.f, __int2float_rn(d));
  int q = __float2int_rz(__int2float_rn(x) * rd);
  const int r = x - q * d;
  q += (r >= d) - (r < 0);
  return q;
}
__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));
}
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
  return base + row * p.row_bytes + col_bytes;
}
template <int N, int NTH, class F>
__device__ __forceinline__ void tfor(int tid, F&& f) {
#pragma unroll
  for (int e0 = 0; e0 < N; e0 += NTH) {
    const int e = e0 + tid;
    if ((N % NTH) == 0 || e < N) f(e);
  }
}
__device__ __forceinline__ uint32_t e4m3x4f(float x0, float x1, float x2, float x3) {
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(x0), "f"(x1));
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(x2), "f"(x3));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ uint32_t pack_e4m3x4(uint32_t h01, uint32_t h23) {
  __half2_raw a, b;
  a.x = (unsigned short)(h01 & 0xffffu); a.y = (unsigned short)(h01 >> 16);
  b.x = (unsigned short)(h23 & 0xffffu); b.y = (unsigned short)(h23 >> 16);
  const uint32_t lo = __nv_cvt_halfraw2_to_fp8x2(a, __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_halfraw2_to_fp8x2(b, __NV_SATFINITE, __NV_E4M3);
  return (lo & 0xffffu) | (hi << 16);
}
__device__ __forceinline__ void red_v2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return v;
}
__device__ __forceinline__ float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  return v;
}
__device__ __forceinline__ float oct_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  return v;
}
__device__ __forceinline__ int cl_rank(const Params& p) {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return (int)r;
}
__device__ __forceinline__ void cl_sync(const Params& p, int phase) {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cl_done(const Params& p) {
}
__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(r));
  return v;
}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {
  uint32_t a = smem_u32(local + off_floats), r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r));
  return v;
}
template <int L, bool F>
struct Smem {
  uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint8_t v[Cfg<L, F>::STAGES * STAGE_BYTES];
  uint64_t mbar[8];
  uint64_t pbar[4];
  int pgc;
  int dbgn;
  int vidx;
  alignas(128) int pidx[4][80];
  uint64_t full[8];
  uint64_t empty[8];
  uint64_t vtf[2];
  uint64_t vte[2];
  uint64_t qdone;
  int claim[2];
  int nxt[4][2];
  int idx[2 * Cfg<L, F>::STAGES * TILE];
  alignas(1024) uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : (OKC_PVW ? 32768 : (PV8ON ? ((Cfg<L, F>::S2 || Cfg<L, F>::F2) ? 36864 : 18432) : 16384))];
  alignas(1024) uint8_t q8s[(L == 8 && PV8ON && OKC_QSS && (!F || OKC_PV8FLAT)) ? 8192 : 16];
  __device__ __forceinline__ __nv_bfloat16* qst() { return reinterpret_cast<__nv_bfloat16*>(kvq); }
  __nv_bfloat16 kvst[L * 4 * 64];
  __nv_bfloat16 gst[L * 4 * 64];
  uint8_t kn[L * ROWB];
  uint8_t vn[L * ROWB];
  float cs[L * 32];
  float sn[L * 32];
  float ifreq[32];
  float scl[4];
  long long cloc[L];
  int flag;
  int sense;
  int cum[F ? MAXB + 1 : 1];
  int seg[F ? MAXSEG : 1][3];
  int nseg;
  int fcs, fce;
  int pseg[F ? MAXSEG : 1][3];
  int pnseg, pchunk;
  int dq[F ? 16 : 1][4];
  uint64_t dqf[F ? 16 : 1];
  uint64_t dqe[F ? 16 : 1];
#if LDD_PSORT
  alignas(16) int plist[(LDD_PSORT && L == 1 && F) ? LDD_PSORT_CAP + 4 : 4];   // LDD19 LDD_PSORT: page indices of tiles [F, nst) (TMA bulk, 16-B window)
  uint16_t pperm[(LDD_PSORT && L == 1 && F) ? LDD_PSORT_CAP : 2];   // LDD19 LDD_PSORT: plist slots in bucket order
  uint32_t phist[(LDD_PSORT && L == 1 && F) ? 1024 : 1];           // LDD19 LDD_PSORT: 2048 u16 bucket counts -> starts
  uint64_t pbarl;                                                   // LDD19 LDD_PSORT: plist landed
#endif
};
template <int L, typename SM>
__device__ __noinline__ bool cluster_multi(const Params& p, SM& S, const int CS, const int rank, const int RPC,
                                           const int row0, const int bp, float* RB, float* RL) {
  constexpr int NT = Cfg<L, false>::NT;
  const int tid = threadIdx.x;
  bool fin = true;
    const int clu = fdiv_i(blockIdx.x, CS);
    const int copy = clu - fdiv_i(clu, p.nc) * p.nc;
    for (int e = tid; e < RPC * 16; e += NT) {
      const int row = row0 + (e >> 4);
      float* acc = p.ws_acc + ((ix_t)(bp * p.nc + copy) * L * 16 + row) * 64 + (e & 15) * 4;
      const float4 a = *reinterpret_cast<const float4*>(&RB[e * 4]);
      red_v4(acc, a.x, a.y, a.z, a.w);
    }
    for (int e = tid; e < RPC; e += NT) red_f32(p.ws_l + (bp * p.nc + copy) * L * 16 + row0 + e, RL[e]);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int old = atomicAdd(p.ws_cnt + bp * CS + rank, 1);
      S.flag = (old == p.ncl - 1);
    }
    __syncthreads();
    fin = S.flag;
    if (fin) {
      __threadfence();
      for (int e = tid; e < RPC * 16; e += NT) {
        const int row = row0 + (e >> 4);
        constexpr int NCM = 4;
        float4 v[NCM];
#pragma unroll
        for (int c = 0; c < NCM; ++c)
          if (c < p.nc) v[c] = __ldcg(reinterpret_cast<const float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4));
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int c = 0; c < NCM; ++c) {
          if (c < p.nc) {
            a.x += v[c].x; a.y += v[c].y; a.z += v[c].z; a.w += v[c].w;
            *reinterpret_cast<float4*>(p.ws_acc + ((ix_t)(bp * p.nc + c) * L * 16 + row) * 64 + (e & 15) * 4) = make_float4(0.f, 0.f, 0.f, 0.f);
          }
        }
        *reinterpret_cast<float4*>(&RB[e * 4]) = a;
      }
      for (int e = tid; e < RPC; e += NT) {
        float a = 0.f;
        for (int c = 0; c < p.nc; ++c) {
          float* lw = p.ws_l + (bp * p.nc + c) * L * 16 + row0 + e;
          a += __ldcg(lw);
          *lw = 0.f;
        }
        RL[e] = a;
      }
      if (tid == 0) p.ws_cnt[bp * CS + rank] = 0;
      __syncthreads();
    }
  return fin;
}
template <int L, typename SM>
__device__ __forceinline__ void cluster_finish(const Params& p, SM& S, const int b, const int pair,
                                               const int n_keys, const int chunk) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int tid = threadIdx.x;
  const int CS = p.cs, rank = cl_rank(p);
  const int RPC = fdiv_i(L * 16, CS);
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  float* PB = reinterpret_cast<float*>(S.k);
  float* PL = PB + L * 16 * 64;
  float* RB = PL + 1024;
  float* RL = RB + RPC * 64;
  constexpr int REGION = L * 16 * 64 + 1024;
  if (OKC_CLDBG) DBG(7);
  cl_sync(p, 1);
  DBG(4);
  long long dbgc0 = 0;
  if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
  const int row0 = rank * RPC;
  if constexpr (L == 1) {
    const int TOT0 = RPC * 17 * CS;
    const int lcs = 31 - __clz(CS);
#pragma unroll 1
    for (int rep2 = (OKC_CLDBG == 2 ? 0 : 1); rep2 < 2; ++rep2) {
    int TOT = TOT0;
    if (rep2 == 0) asm volatile("mov.u32 %0, 0;" : "=r"(TOT));
    if (OKC_CLDBG == 2 && rep2 == 1) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc0));
    for (int k0 = 0; k0 < (rep2 == 0 ? 2 * NT : TOT); k0 += 2 * NT) {
      float4 v[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        v[u] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (k < TOT) {
          if (slot < 16) v[u] = cl_ld4(p, PB, q, (row0 + rl) * 64 + slot * 4, REGION);
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
        }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        for (int off = 1; off < CS; off <<= 1) {
          v[u].x += __shfl_xor_sync(0xffffffffu, v[u].x, off);
          v[u].y += __shfl_xor_sync(0xffffffffu, v[u].y, off);
          v[u].z += __shfl_xor_sync(0xffffffffu, v[u].z, off);
          v[u].w += __shfl_xor_sync(0xffffffffu, v[u].w, off);
        }
        const int k = k0 + u * NT + tid;
        const int item = k >> lcs, q = k & (CS - 1);
        const int rl = item / 17, slot = item - rl * 17;
        if (k < TOT && q == 0) {
          if (slot < 16) *reinterpret_cast<float4*>(&RB[(rl * 16 + slot) * 4]) = v[u];
          else RL[rl] = v[u].x;
        }
      }
    }
    }
  } else
  {
  constexpr int CSMAX = 16;
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll 1
    for (int q0 = 0; q0 < CS; q0 += 4) {
      float4 v[4];
      float lv[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        v[u] = cl_ld4(p, PB, q0 + u, off, REGION);
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
      }
#pragma unroll
      for (int u = 0; u < 4; ++u) { a.x += v[u].x; a.y += v[u].y; a.z += v[u].z; a.w += v[u].w; la += lv[u]; }
    }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  if (false)
  for (int e = tid; e < RPC * 16; e += NT) {
    const int off = (row0 + (e >> 4)) * 64 + (e & 15) * 4;
    const bool lrow = (e & 15) == 0;
    float4 v[CSMAX];
    float lv[CSMAX];
#pragma unroll
    for (int q = 0; q < CSMAX; ++q) {
      if (q < CS) {
        v[q] = cl_ld4(p, PB, q, off, REGION);
        lv[q] = lrow ? cl_ld1(p, PL, q, row0 + (e >> 4), REGION) : 0.f;
      }
    }
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    float la = 0.f;
#pragma unroll
    for (int q = 0; q < CSMAX; ++q)
      if (q < CS) { a.x += v[q].x; a.y += v[q].y; a.z += v[q].z; a.w += v[q].w; la += lv[q]; }
    *reinterpret_cast<float4*>(&RB[e * 4]) = a;
    if (lrow) RL[e >> 4] = la;
  }
  }
  if (OKC_CLDBG) {
    long long c1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(c1));
    if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 11] = c1 - dbgc0;
  }
  if (OKC_CLDBG) DBG(8);
  DBG(14);
  cl_sync(p, 2);
  cl_done(p);
  DBG(6);
  bool fin = true;
  if (OKC_CLDBG) DBG(9);
  if (p.ncl > 1) {
    fin = cluster_multi<L>(p, S, CS, rank, RPC, row0, bp, RB, RL);
  }
  if (fin) {
    if (OKC_CLDBG) DBG(10);
    long long dbgc2 = 0;
    if (OKC_CLDBG) asm volatile("mov.u64 %0, %%clock64;" : "=l"(dbgc2));
    const float vscale = S.scl[1];
    for (int eb = tid - (tid & 31); eb < RPC * 16; eb += NT) {
      const int e = eb + (tid & 31);
      const bool act = e < RPC * 16;
      const int rl = act ? (e >> 4) : 0, c4 = e & 15;
      const int row = row0 + rl;
      const int j = row >> 4, hh = row & 15;
      const float lsum = act ? RL[rl] : 0.f;
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      const float4 a = act ? *reinterpret_cast<const float4*>(&RB[e * 4]) : make_float4(0.f, 0.f, 0.f, 0.f);
      float y[4] = {pr8(a.x * inv), pr8(a.y * inv), pr8(a.z * inv), pr8(a.w * inv)};
      float ss = y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      ss += __shfl_xor_sync(0xffffffffu, ss, 8);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const uint2 graw = *reinterpret_cast<const uint2*>(&S.gst[rl * 64 + c4 * 4]);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint2 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const float2 gt = __bfloat1622float2(g2[k]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * k] * rn * s0, y[2 * k + 1] * rn * s1);
        ow[k] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint2*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + c4 * 4) = ov;
    }
    if (OKC_CLDBG) {
      long long c3;
      asm volatile("mov.u64 %0, %%clock64;" : "=l"(c3));
      if (p.dbg && threadIdx.x == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 15] = c3 - dbgc2;
    }
    DBG(5);
  }
}
template <int L, typename SM>
__device__ __forceinline__ void cluster_stage_gate(const Params& p, SM& S, const int b, const int pair) {
  constexpr int NT = Cfg<L, false>::NT;
  constexpr int STAGES = Cfg<L, false>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  const int RPC = fdiv_i(L * 16, p.cs), row0 = cl_rank(p) * RPC;
  for (int e = threadIdx.x; e < RPC * 8; e += NT) {
    const int rl = e >> 3, ch = e & 7;
    const int row = row0 + rl, j = row >> 4, hh = row & 15;
    cp_async16(smem_u32(&S.gst[rl * 64 + ch * 8]),
               p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + ch * 8, 16);
  }
}
// LDD19 LDD_RBF: sum_copies for BF16 merge rows (128 B per row; written by red_bf8); every copy's loads in one round trip for nc 8, re-zeroes what it read
template <int L>
__device__ __forceinline__ void sum_copies_bf(const Params& p, const int bp, const int j, const int hh, const int part,
                                              const bool act, float (&y)[8], float& lsum) {
  if (!act) return;
  constexpr int NCU = 8;
  const int nc = p.nc;
#pragma unroll 1
  for (int c0 = 0; c0 < nc; c0 += NCU) {
    uint4 raw[NCU];
    float lc[NCU];
#pragma unroll
    for (int c = 0; c < NCU; ++c) {
      if (c0 + c < nc) {
        const __nv_bfloat16* rowb = reinterpret_cast<const __nv_bfloat16*>(p.ws_acc) + ((ix_t)((bp * nc + c0 + c) * L + j) * 16 + hh) * 64 + part * 8;
        raw[c] = __ldcg(reinterpret_cast<const uint4*>(rowb));
        lc[c] = __ldcg(p.ws_l + ((bp * nc + c0 + c) * L + j) * 16 + hh);
      }
    }
#pragma unroll
    for (int c = 0; c < NCU; ++c) {
      if (c0 + c < nc) {
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw[c]);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float2 fv = __bfloat1622float2(h2[e]);
          y[2 * e] += fv.x;
          y[2 * e + 1] += fv.y;
        }
        lsum += lc[c];
        __nv_bfloat16* rowb = reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + ((ix_t)((bp * nc + c0 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<uint4*>(rowb) = make_uint4(0u, 0u, 0u, 0u);
        if (part == 0) p.ws_l[((bp * nc + c0 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
  }
}
template <int L>
__device__ __forceinline__ void sum_copies(const Params& p, const int bp, const int j, const int hh, const int part,
                                           const bool act, float (&y)[8], float& lsum) {
  if (OKC_SC1 && p.nc == 1) {
    if (act) {
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      const float4 xa = __ldcg(reinterpret_cast<const float4*>(acc));
      const float4 xb = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      const float lc = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      y[0] += xa.x; y[1] += xa.y; y[2] += xa.z; y[3] += xa.w;
      y[4] += xb.x; y[5] += xb.y; y[6] += xb.z; y[7] += xb.w;
      lsum += lc;
      *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
      *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
      if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
    }
    return;
  }
  if (OKC_SC8 && p.nc == 8) {
    if (act) {
      float4 xa[8], xb[8];
      float lc[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[c] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[c] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[c] = __ldcg(p.ws_l + ((bp * 8 + c) * L + j) * 16 + hh);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        y[0] += xa[c].x; y[1] += xa[c].y; y[2] += xa[c].z; y[3] += xa[c].w;
        y[4] += xb[c].x; y[5] += xb[c].y; y[6] += xb[c].z; y[7] += xb[c].w;
        lsum += lc[c];
        float* acc = p.ws_acc + ((ix_t)((bp * 8 + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * 8 + c) * L + j) * 16 + hh] = 0.f;
      }
    }
    return;
  }
#pragma unroll 1
  for (int c0 = 0; c0 < p.nc; c0 += 4) {
    float4 xa[4], xb[4];
    float lc[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      xa[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      xb[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      lc[u] = 0.f;
      if (act && c < p.nc) {
        const float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
        xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
        lc[u] = __ldcg(p.ws_l + ((bp * p.nc + c) * L + j) * 16 + hh);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c = c0 + u;
      if (act && c < p.nc) {
        y[0] += xa[u].x; y[1] += xa[u].y; y[2] += xa[u].z; y[3] += xa[u].w;
        y[4] += xb[u].x; y[5] += xb[u].y; y[6] += xb[u].z; y[7] += xb[u].w;
        lsum += lc[u];
        float* acc = p.ws_acc + ((ix_t)((bp * p.nc + c) * L + j) * 16 + hh) * 64 + part * 8;
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[((bp * p.nc + c) * L + j) * 16 + hh] = 0.f;
      }
    }
  }
}
template <int L, int NT, bool BF = false>   // LDD19 LDD_RBFQ: BF = the merge wrote BF16 rows (128 B per row)
__device__ __forceinline__ void fin_nc1(const Params& p, const int b, const int pair, const int bp, const float vscale,
                                     const int dry) {
  const int tid = threadIdx.x;
  constexpr int PASSES = L * 16 / (NT / 8);
  constexpr int GP = PASSES >= 2 ? 2 : 1;
  const int part = tid & 7;
#pragma unroll 1
  for (int p0 = 0; p0 < PASSES; p0 += GP) {
    float4 xa[GP], xb[GP];
    uint4 xr[GP];   // (LDD19 LDD_RBFQ: BF rows)
    float lc[GP];
    uint4 graw[GP];
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      const float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      if constexpr (BF) {   // LDD19 LDD_RBFQ: one 16-B load of 8 BF16 dims
        xr[u] = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ws_acc) + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8));
      } else {
      xa[u] = __ldcg(reinterpret_cast<const float4*>(acc));
      xb[u] = __ldcg(reinterpret_cast<const float4*>(acc) + 1);
      }
      lc[u] = __ldcg(p.ws_l + (bp * L + j) * 16 + hh);
      graw[u] = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
    }
#pragma unroll
    for (int u = 0; u < GP; ++u) {
      const int rr = (p0 + u) * (NT / 8) + (tid >> 3);
      const int j = rr >> 4, hh = rr & 15;
      float* acc = p.ws_acc + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8;
      if constexpr (BF) {   // LDD19 LDD_RBFQ
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&xr[u]);
        const float2 f0 = __bfloat1622float2(h2[0]), f1 = __bfloat1622float2(h2[1]);
        const float2 f2 = __bfloat1622float2(h2[2]), f3 = __bfloat1622float2(h2[3]);
        xa[u] = make_float4(f0.x, f0.y, f1.x, f1.y);
        xb[u] = make_float4(f2.x, f2.y, f3.x, f3.y);
        if (!dry) {
          *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + ((ix_t)(bp * L + j) * 16 + hh) * 64 + part * 8) = make_uint4(0u, 0u, 0u, 0u);
          if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
        }
      } else
      if (!dry) {
        *reinterpret_cast<float4*>(acc) = make_float4(0.f, 0.f, 0.f, 0.f);
        *(reinterpret_cast<float4*>(acc) + 1) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (part == 0) p.ws_l[(bp * L + j) * 16 + hh] = 0.f;
      }
      const float inv = lc[u] > 0.f ? __fdividef(vscale, lc[u]) : 0.f;
      float y[8] = {pr8(xa[u].x * inv), pr8(xa[u].y * inv), pr8(xa[u].z * inv), pr8(xa[u].w * inv),
                    pr8(xb[u].x * inv), pr8(xb[u].y * inv), pr8(xb[u].z * inv), pr8(xb[u].w * inv)};
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += y[e] * y[e];
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (!dry) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
}
template <int L, int NT, bool PTR, class SM>
__device__ __noinline__ bool ws_patch_ni(SM* Sp, int i, int st, int start, int seq, int end, int tid) {
  SM& S = *Sp;
  const int t0 = start + i * TILE;
  const int lo = max(t0, seq), hi = min(t0 + TILE, end);
  if (lo < hi) {
    for (int e = tid; e < (hi - lo) * 8; e += NT) {
      const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
      const int sw = ((c ^ (r & 7)) << 4);
      *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
      if constexpr (!PTR)
        *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
    }
    if (L == 8) fence_proxy_async();
  }
  return lo < hi;
}
template <int L, bool FLAT, int ROLE, int MODE = 0>
__device__ __forceinline__ bool segment(const Params& p, Smem<L, FLAT>& S, const int b_, const int pair_, const bool first,
                                        const int start_, const int end_, const int seq_, const int n_keys_,
                                        const int copy_, const int chunk_, const int tb_ = 0, const int seq_real_ = 0,
                                        const bool skip_init_ = false) {
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
  const int b = UNI(b_), pair = UNI(pair_), start = UNI(start_), end = UNI(end_), seq = UNI(seq_);
  const int n_keys = UNI(n_keys_), copy = UNI(copy_), chunk = UNI(chunk_), tb = UNI(tb_);
#undef UNI
  constexpr bool ONE = !FLAT && (MODE & 4) != 0;
  const bool M_CL = !FLAT && ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2);
  const bool M_COOP = (MODE & 3) == 0 ? p.coop != 0 : (MODE & 3) == 3;
  // LDD19 LDD_RBF / LDD_RBFS / LDD_RBFQ (eat_attn): which instances merge in BF16 rows
  constexpr bool RBF_B1 = !FLAT && ((((L == 8 && (LDD_RBF == 2 || LDD_RBF == 3)) || (L == 1 && LDD_RBF >= 3)) && (MODE & 8) != 0 && (MODE & 4) == 0 && (MODE & 16) == 0 && (MODE & 32) == 0)
                                                || ((LDD_RBFS & 1) && L == 8 && (MODE & 3) == 3 && (MODE & 4) == 0 && (MODE & 32) == 0));
  constexpr bool RBF_Q = L == 1 && ((!FLAT && (MODE & 3) == 1 && (((LDD_RBFQ & 1) && (MODE & 12) == 0) || (LDD_RBFQ & 4))) || (FLAT && (LDD_RBFQ & 2)));
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr bool CONT = FLAT && Cfg<L, FLAT>::WS && OKC_CONT;
  constexpr bool PTR = (L == 8) && FLAT && Cfg<L, FLAT>::WS && OKC_PTR && PV8ON && OKC_PV8FLAT && OKC_QSS && !OKC_CONT;
  constexpr int IDX_SLOTS = 2 * STAGES;
  constexpr bool WS = Cfg<L, FLAT>::WS;
  const int tid = threadIdx.x, lane = tid & 31, warp = OKC_UWARP ? __shfl_sync(0xffffffffu, tid >> 5, 0) : tid >> 5;
  const int g = lane >> 2, tig = lane & 3;
  const int split = first ? 0 : 1;
  const int pairs = p.Hkv >> 1;
  const int bp = b * pairs + pair;
  const int ntiles = ONE ? 1 : (end - start + TILE - 1) / TILE;
  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;
    constexpr int PR = NP / 8;
    constexpr int PK = TILE / PR;
    if (!CONT && !skip_init_) {
      if (ROLE == 2 && tid == 0) {
        for (int st = 0; st < STAGES; ++st) {
          mbar_init_n(&S.full[st], NP);
          mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? NT : NT / 32));
        }
        if constexpr (PTR) {
          for (int k = 0; k < 2; ++k) {
            mbar_init_n(&S.vtf[k], NP);
            mbar_init_n(&S.vte[k], NT / 32);
          }
          mbar_init_n(&S.qdone, NT / 32);
        }
        mbar_fence_init();
      }
      __syncthreads();
    }
    if constexpr (ROLE == 1) {
      const int q = tid - NT;
      if (q == 0) FPRO_STAMP(42);
      const int ch = q & 7, r0 = q >> 3;
      const int* ptr = p.page_table + (ix_t)b * p.max_pages;
      const int cache_end = min(end, seq);
      int pg[PK];
      static_assert(PR % 8 == 0, "PFAST: the swizzle chunk is the same for all of a thread's rows");
      const bool pfast = OKC_OFF32 && p.off32 && p.lp == 0;
      const uint32_t pcol = (uint32_t)(pair * 128 + ch * 16);
      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);
      constexpr bool TMAPG_ON = OKC_TMAPG && (L == 1 || OKC_TMAPG8);
      constexpr int PS = 4, PD = 3;
      int pgb = 0;
      if constexpr (TMAPG_ON) {
        if constexpr (OKC_PGCONT) {
          pgb = S.pgc;
        } else {
          if (q == 0) {
            for (int st = 0; st < PS; ++st) mbar_init_n(&S.pbar[st], 1);
            mbar_fence_init();
          }
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        }
      }
#if LDD_PSORT
      // LDD19 LDD_PSORT: tiles [PF, nst) of this segment (full old tiles only) take their page indices from S.plist[S.pperm[.]],
      // in 2048-bucket physical-row order, built in smem-only steps during tiles PH0 .. PF - 1; pskip = tiles without a pidx TMA
      constexpr bool PSORT_ON = LDD_PSORT == 7 && L == 1 && FLAT && TMAPG_ON;
      constexpr int PF = LDD_PSORT_F, PNH = LDD_PSORT_NH, PH0 = 3, PNS = PF - PH0 - 1 - PNH;
      static_assert(!PSORT_ON || (NP == 128 && PNS >= 1), "LDD_PSORT: 16 buckets per producer thread; scatter steps");
      int nst = 0, pskip = 0, psh = 0, pq = 0, plo = 0, phi = 0;
      long long pa0 = 0;
      if constexpr (PSORT_ON) {
        if (p.lp == 0 && p.window == 0 && p.B > 1) nst = min((cache_end - start) / TILE, PF + LDD_PSORT_CAP / TILE);
        if (nst < PF + 2) nst = 0;
        if (nst > 0) {   // (the same values in every producer thread)
          const int tot = p.B * p.max_pages;   // the table's extent (rows beyond it only land in the last bucket)
          psh = max(0, 32 - __clz(max(tot - 1, 1)) - 11);
          const long long g0 = (long long)b * p.max_pages + start + PF * TILE;   // first ordered entry (page size 1)
          pa0 = g0 & ~3LL;
          plo = (int)(g0 - pa0);
          phi = plo + (nst - PF) * TILE;
          pq = (phi + 3) >> 2;   // 16-byte quads from pa0
          for (int e = q; e < 1024; e += NP) S.phist[e] = 0;
          if (q == 0) {   // the list in one bulk copy (lands while the first tiles stream; no thread waits before tile PH0)
            mbar_init_n(&S.pbarl, 1);
            mbar_fence_init();
            mbar_expect_tx(&S.pbarl, 16 * pq);
            bulk_load(S.plist, p.page_table + pa0, 16 * pq, &S.pbarl);
          }
          pskip = nst - PF;
        }
      }
#endif
      const long long pt_total = (long long)p.B * p.max_pages;
      const long long pt_lim = (pt_total + 3) & ~3LL;
#if LDD_PSORT
      auto fetch_pages = [&](int v) {   // LDD19 LDD_PSORT: v = fill number; tile x skips the ordered tiles
        const int x = (v < PF) ? v : v + pskip;
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + v) % PS;
#else
      auto fetch_pages = [&](int x) {
        if (q == 0 && x < ntiles) {
          const int slot = (pgb + x) % PS;
#endif
          const long long e0 = (long long)b * p.max_pages + ((start + x * TILE) >> p.lp);
          const long long w0 = e0 & ~3LL;
          const long long w1 = min(pt_lim, (e0 + 64 + 3) & ~3LL);
          const int bytes = (int)(w1 - w0) * 4;
          mbar_expect_tx(&S.pbar[slot], bytes);
          bulk_load(&S.pidx[slot][0], p.page_table + w0, bytes, &S.pbar[slot]);
        }
      };
      if constexpr (TMAPG_ON) {
        for (int x = 0; x < PD; ++x) fetch_pages(x);
      }
      auto load_pages_t = [&](int x, int* dst) {
#if LDD_PSORT
        const int v = (x < PF) ? x : x - pskip;   // LDD19 LDD_PSORT: fill number of tile x
#else
        const int v = x;
#endif
        const int slot = (pgb + v) % PS;
        mbar_wait_full(&S.pbar[slot], ((pgb + v) / PS) & 1);
        const int fp = (start + x * TILE) >> p.lp;
        const int dl = (int)(((long long)b * p.max_pages + fp) & 3);
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (pos < cache_end) ? S.pidx[slot][(pos >> p.lp) - fp + dl] : 0;
        }
      };
      auto load_pages = [&](int x, int* dst) {
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int pos = start + x * TILE + r0 + PR * k;
          dst[k] = (x < ntiles && pos < cache_end) ? __ldg(ptr + (pos >> p.lp)) : 0;
        }
      };
      if constexpr (!TMAPG_ON) load_pages(0, pg);
      if (OKC_FPRO >= 2 && q == 0) {
        asm volatile("" :: "r"(pg[0]), "r"(pg[PK - 1]));
        volatile int sink = pg[0] + pg[PK - 1];
        (void)sink;
        FPRO_STAMP(45);
      }
      auto ptr_transpose = [&](int y) {
        if constexpr (PTR) {
          const int sty = (tb + y) % STAGES;
          if (y == 0) {
            mbar_wait_full(&S.qdone, 0);
            for (int e = q; e < 4 * 32; e += NP)
              *reinterpret_cast<uint4*>(S.kvq + (e >> 5) * 4608 + 4096 + (e & 31) * 16) =
                  make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
          }
          mbar_wait_full(&S.full[sty], ((tb + y) / STAGES) & 1);
          const int t0 = start + y * TILE;
          const int nlo = max(t0, seq), nhi = min(t0 + TILE, end);
          if (nlo < nhi) {
            for (int e = q; e < (nhi - nlo) * 8; e += NP) {
              const int pos = nlo + (e >> 3), c = e & 7, r = pos - t0;
              *reinterpret_cast<uint4*>(S.v + sty * STAGE_BYTES + r * ROWB + ((c ^ (r & 7)) << 4)) =
                  *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
            }
            asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
          }
          if (y >= 2) mbar_wait_full(&S.vte[y & 1], ((y >> 1) - 1) & 1);
          const int pl = q & 31, pw = (q >> 5) & 3, pvq = pl & 3, pvw = pl >> 2;
          const uint32_t s0 = (pvw >= 4) ? 0x7531u : 0x6420u, s1 = (pvw >= 4) ? 0x6420u : 0x7531u;
          const int pr0 = (pvw >= 4) ? 1 : 0;
          const uint32_t o0 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * pr0;
          const uint32_t o1 = 128 * pvw + ((pw ^ (pvw & 3)) << 4) + 4 * pvq + 64 * (1 - pr0);
          const int pkey = 16 * pw + 8 * ((pl >> 3) & 1) + (pl & 7);
          const uint32_t vsrc = smem_u32(S.v + sty * STAGE_BYTES) + pkey * ROWB;
          uint8_t* vt = S.kvq + (y & 1) * 2 * 4608;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              uint32_t r0, r1, r2, r3;
              ldsm_x4_t(vsrc + (((4 * h + 2 * u + ((pl >> 3) >> 1)) ^ (pl & 7)) << 4), r0, r1, r2, r3);
#pragma unroll
              for (int g2 = 0; g2 < 2; ++g2) {
                const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o0 + 1024 * (2 * u + g2)) = prmt(R0, R1, s0);
                *reinterpret_cast<uint32_t*>(vt + h * 4608 + o1 + 1024 * (2 * u + g2)) = prmt(R0, R1, s1);
              }
            }
          }
          fence_proxy_async();
          mbar_arrive(&S.vtf[y & 1]);
        }
      };
#if LDD_PSORT
      // LDD19 LDD_PSORT: one step of the bucket sort (uniform in the producer threads); the loop's per-tile producer barrier separates the phases
      auto psort_range = [&](int c, int nc, auto f) {   // entries [lo, hi) of chunk c of nc, from shared memory
        const int n = phi - plo;
        const int lo = n * c / nc, hi = n * (c + 1) / nc;
#pragma unroll 4
        for (int e = plo + lo + q; e < plo + hi; e += NP) f(e, S.plist[e]);
      };
      auto psort_step = [&](int x) {
        if (x < PH0 + PNH) {   // histogram
          if (x == PH0) mbar_wait_full(&S.pbarl, 0);
          psort_range(x - PH0, PNH, [&](int e, int row) {
            const int bk = (int)min(2047u, (uint32_t)row >> psh);   // (unsigned: any entry lands in a bucket)
            asm volatile("red.shared.add.u32 [%0], %1;\n" ::"r"(smem_u32(S.phist + (bk >> 1))), "r"(1u << (16 * (bk & 1))) : "memory");
          });
        } else if (x == PH0 + PNH) {   // exclusive scan: thread q owns buckets [16q, 16q + 16) = words [8q, 8q + 8)
          int s = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j) s += (int)(S.phist[8 * q + j] & 0xffffu) + (int)(S.phist[8 * q + j] >> 16);
          const int lane_p = q & 31, wq = q >> 5;
          int incl = s;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane_p >= o) incl += t;
          }
          if (lane_p == 31) S.pperm[2 * wq] = (uint16_t)incl, S.pperm[2 * wq + 1] = (uint16_t)(incl >> 16);
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
          int run = incl - s;
          for (int w2 = 0; w2 < wq; ++w2) run += (int)S.pperm[2 * w2] | ((int)S.pperm[2 * w2 + 1] << 16);
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const uint32_t w = S.phist[8 * q + j];
            const uint32_t lo0 = (uint32_t)run;
            run += (int)(w & 0xffffu);
            S.phist[8 * q + j] = lo0 | ((uint32_t)run << 16);
            run += (int)(w >> 16);
          }   // (LDD19 LDD_PSORT)
        } else {   // scatter: pperm[start(bucket) + rank] = plist slot
          psort_range(x - PH0 - PNH - 1, PNS, [&](int e, int row) {
            const int bk = (int)min(2047u, (uint32_t)row >> psh);   // (unsigned: any entry lands in a bucket)
            uint32_t old = 0;
            asm volatile("atom.shared.add.u32 %0, [%1], %2;\n" : "=r"(old) : "r"(smem_u32(S.phist + (bk >> 1))), "r"(1u << (16 * (bk & 1))) : "memory");
            S.pperm[(old >> (16 * (bk & 1))) & 0xffffu] = (uint16_t)e;
          });
        }
      };
#endif
      PROF_INIT
      for (int x = 0; x < ntiles; ++x) {
        const int st = (tb + x) % STAGES;
        int pgn[PK];
        if constexpr (TMAPG_ON) {
#if LDD_PSORT
          if (PSORT_ON && x >= PF && x < PF + pskip) {   // LDD19 LDD_PSORT: an ordered tile (every row cached)
#pragma unroll
            for (int k = 0; k < PK; ++k) pg[k] = S.plist[S.pperm[(x - PF) * TILE + r0 + PR * k]];
          } else {
            fetch_pages(((x < PF) ? x : x - pskip) + PD);
            load_pages_t(x, pg);
          }   // (LDD19 LDD_PSORT)
#else
          fetch_pages(x + PD);
          load_pages_t(x, pg);
#endif
        } else {
          load_pages(x + 1, pgn);
        }
        PROF_MARK(2);
        if (tb + x >= STAGES) mbar_wait_full(&S.empty[st], (((tb + x) / STAGES) - 1) & 1);
        PROF_MARK(0);
        if (x == 0 && q == 0) FPRO_STAMP(46);
        const uint32_t kb = smem_u32(S.k + st * STAGE_BYTES), vb = smem_u32(S.v + st * STAGE_BYTES);
        if (pfast && start + x * TILE + TILE <= cache_end) {
#pragma unroll
          for (int k = 0; k < PK; ++k) {
            const uint32_t off = ((uint32_t)pg[k] << p.lrb) + pcol;
            cp_async16(kb + pso + k * (PR * ROWB), p.k_cache + off, 16);
            cp_async16(vb + pso + k * (PR * ROWB), p.v_cache + off, 16);
          }
        } else
#pragma unroll
        for (int k = 0; k < PK; ++k) {
          const int r = r0 + PR * k;
          const int pos = start + x * TILE + r;
          const uint32_t so = r * ROWB + ((ch ^ (r & 7)) << 4);
          if (pos < cache_end) {
            if (p.off32) {
              const uint32_t off = ((((uint32_t)pg[k] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + ch * 16);
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            } else
            {
              const int64_t off = (((int64_t)pg[k] << p.lp) + (pos & (p.P - 1))) * p.row_bytes + pair * 128 + ch * 16;
              cp_async16(kb + so, p.k_cache + off, 16);
              cp_async16(vb + so, p.v_cache + off, 16);
            }
          } else if (pos >= end) {
            cp_async16_zero(kb + so, p.k_cache);
            cp_async16_zero(vb + so, p.v_cache);
          }
        }
        cp_async_mbar_arrive(&S.full[st]);
#if LDD_PSORT
        if constexpr (PSORT_ON) {
          if (nst > 0 && x >= PH0 && x < PF) psort_step(x);   // LDD19 LDD_PSORT: sort step while tile x streams
        }
#endif
        if (x == 0 && q == 0) FPRO_STAMP(43);
        if constexpr (PTR) {
          if (x >= OKC_PTRD) ptr_transpose(x - OKC_PTRD);
        }
        PROF_MARK(1);
        PROF_TILE;
        if constexpr (TMAPG_ON) {
          asm volatile("bar.sync 2, %0;\n" ::"n"(NP) : "memory");
        } else {
#pragma unroll
          for (int k = 0; k < PK; ++k) pg[k] = pgn[k];
        }
      }
      if constexpr (PTR) {
        for (int y = max(0, ntiles - OKC_PTRD); y < ntiles; ++y) ptr_transpose(y);
      }
      if constexpr (TMAPG_ON && OKC_PGCONT) {
#if LDD_PSORT
        if (q == 0) S.pgc = pgb + ntiles - pskip;   // LDD19 LDD_PSORT: fills issued
#else
        if (q == 0) S.pgc = pgb + ntiles;
#endif
      }
      PROF_DUMP;
      if constexpr (!CONT) {
        cp_wait<0>();
        __syncthreads();   // matches the consumers' end-of-segment barrier
      }
      return true;
    }
  }
  if constexpr (ROLE != 1) {
  const int prow = tid >> 1, phalf = tid & 1;
  const int* pt_row = p.page_table + (ix_t)b * p.max_pages;
  const int part = tid & 7;
  const int cchunk = tid & 7, crow0 = tid >> 3;
  constexpr int RPP = NT / 8;
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % STAGES;
    const int* idxs = &S.idx[(x % IDX_SLOTS) * TILE];
    bool generic = false;
    if (with_new && p.off32 && p.lp == 0 && start + x * TILE + TILE <= min(end, seq)) {
      const uint32_t kst = smem_u32(S.k + st * STAGE_BYTES), vst = smem_u32(S.v + st * STAGE_BYTES);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        const uint32_t off = ((uint32_t)idxs[r] << p.lrb) + col;
        cp_async16(kst + so, p.k_cache + off, 16);
        cp_async16(vst + so, p.v_cache + off, 16);
      }
      return;
    }
#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {
      const int r = crow0 + RPP * k;
      const int pos = start + x * TILE + r;
      const int sw = ((cchunk ^ (r & 7)) << 4);
      uint8_t* dk = S.k + st * STAGE_BYTES + r * ROWB + sw;
      uint8_t* dv = S.v + st * STAGE_BYTES + r * ROWB + sw;
      if (pos < end && pos < seq) {
        if (p.off32) {
          const uint32_t off = ((((uint32_t)idxs[r] << p.lp) + (uint32_t)(pos & (p.P - 1))) << p.lrb) + (uint32_t)(pair * 128 + cchunk * 16);
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        } else
        {
          const int64_t row = ((int64_t)idxs[r] << p.lp) + (pos & (p.P - 1));
          const int64_t off = row * p.row_bytes + pair * 128 + cchunk * 16;
          cp_async16(smem_u32(dk), p.k_cache + off, 16);
          cp_async16(smem_u32(dv), p.v_cache + off, 16);
        }
      } else if (pos < end) {
        if (with_new) {
          *reinterpret_cast<uint4*>(dk) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(dv) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
          generic = true;
        }
      } else {
        *reinterpret_cast<uint4*>(dk) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(dv) = make_uint4(0, 0, 0, 0);
      }
    }
    if (L == 8 && generic) fence_proxy_async();
  };
  auto issue_idx = [&](int y) {
    if (phalf == 0 && prow < TILE && y < ntiles) {
      const int pos = start + y * TILE + prow;
      if (pos < end && pos < seq) cp_async4(smem_u32(&S.idx[(y % IDX_SLOTS) * TILE + prow]), pt_row + (pos >> p.lp));
    }
  };
  auto ws_patch = [&](int i, int st) {
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  auto ws_acquire = [&](int i, int st) {
    mbar_wait_full(&S.full[st], ((tb + i) / STAGES) & 1);
    const int t0 = start + i * TILE;
    const int lo = max(t0, seq), hi = min(t0 + TILE, end);
    if (lo < hi) {
      for (int e = tid; e < (hi - lo) * 8; e += NT) {
        const int pos = lo + (e >> 3), c = e & 7, r = pos - t0;
        const int sw = ((c ^ (r & 7)) << 4);
        *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
        if constexpr (!PTR)
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
      }
      if (L == 8) fence_proxy_async();
    }
    return lo < hi;
  };
  int tidL = tid, partL = part;
  constexpr int QPASS = L * 16 / RPP;
  const bool has_new = end > seq;
  constexpr int KPASS = (L * 4 + RPP - 1) / RPP;
  constexpr bool QDIR = OKC_QDIRECT && L == 8 && !WS && !Cfg<L, FLAT>::S2 && PV8ON && OKC_QSS && OKC_SM90 && OKC_Q8RAW &&
                        (!FLAT || OKC_PV8FLAT);
  constexpr bool IDXF = OKC_IDXFIRST && !WS && !Cfg<L, FLAT>::S2 && !OKC_PDLKV;
  {
  if constexpr (!WS) {
#pragma unroll
    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);
  }
  cp_commit();
  if (OKC_LATETID && !WS && (MODE & 8)) {
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tidL));
    partL = tidL & 7;
  }
  pdl_wait();
  pdl_launch_dependents();
  if (tid == 0) FPRO_STAMP(44);
  tfor<L * 16 * 8, NT>(tidL, [&](int e) {
    const int rr = e >> 3, ch = e & 7;
    const int j = rr >> 4, hh = rr & 15;
    cp_async16(smem_u32(&S.qst()[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (16 * pair + hh) * 64 + ch * 8, 16);
  });
  if (has_new) {
    tfor<L * 4 * 8, NT>(tidL, [&](int e) {
      const int rr = e >> 3, ch = e & 7;
      const int kind = rr / (2 * L), jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      const int col = (kind == 0 ? p.Hq : p.Hq + p.Hkv) * 64 + (2 * pair + hh) * 64 + ch * 8;
      cp_async16(smem_u32(&S.kvst[rr * 64 + ch * 8]), p.qkv + (ix_t)(b * L + jj) * p.ld_qkv + col, 16);
    });
  }
  if (M_CL) cluster_stage_gate<L>(p, S, b, pair);
  if (tidL < 8) cp_async16(smem_u32(&S.ifreq[tidL * 4]), p.inv_freq + tidL * 4, 16);
  else if (tidL == 8) cp_async4(smem_u32(&S.scl[0]), p.k_scale);
  else if (tidL == 9) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
  else if ((OKC_SENSE2 || OKC_PARITY) && !FLAT && M_COOP && tidL == 10)
    cp_async4(smem_u32(&S.sense), p.ws_cnt + 24576 + bp);
  else if (tidL >= 16 && tidL < 16 + L) {
    const int jj = tidL - 16;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(&S.cloc[jj])), "l"(p.cache_loc + b * L + jj));
  }
  cp_commit();
  if (split == 0 && (FLAT || !M_CL)) {
    tfor<L * 16, NT>(tidL, [&](int e) {
      const __nv_bfloat16* gp = p.qkv + (ix_t)(b * L + (e >> 4)) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + (e & 15)) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(gp));
    });
  }
  cp_wait<IDXF ? 1 : 0>();
  csync<WS, NT>();
  if constexpr (!FLAT && (MODE & 8) != 0) {
    if (seq_real_ != seq) {
      cp_wait<0>();
      __syncthreads();
      return false;
    }
  }
  DBG(1);
  if constexpr (!WS) {
    constexpr int PRE = Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1;
    bool one = false;
    if (ONE && !Cfg<L, FLAT>::S2 && p.off32 && p.lp == 0 && (OKC_ONENEW || start + TILE <= min(end, seq))) {
      one = true;
      const uint32_t kst = smem_u32(S.k), vst = smem_u32(S.v);
      const uint32_t col = (uint32_t)(pair * 128 + cchunk * 16);
      const int cend = min(end, seq);
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const uint32_t so = r * ROWB + ((cchunk ^ (r & 7)) << 4);
        if constexpr (OKC_ONEPRED) {
          const int pr = start + r;
          const bool cached = !OKC_ONENEW || pr < cend;
          const bool doit = cached || pr >= end;
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;
          cp_async16_pred(kst + so, p.k_cache + off, cached ? 16 : 0, doit);
          cp_async16_pred(vst + so, p.v_cache + off, cached ? 16 : 0, doit);
        } else if (!OKC_ONENEW || start + r < cend) {
          const uint32_t off = ((uint32_t)S.idx[r] << p.lrb) + col;
          cp_async16(kst + so, p.k_cache + off, 16);
          cp_async16(vst + so, p.v_cache + off, 16);
        } else if (start + r >= end) {
          cp_async16_zero(kst + so, p.k_cache);
          cp_async16_zero(vst + so, p.v_cache);
        }
      }
#pragma unroll
      for (int k = 0; k < STAGES - 1; ++k) cp_commit();
    }
    if (!one) {
#pragma unroll
    for (int k = 0; k < STAGES - 1; ++k) {
      if (k < PRE && k < ntiles) issue_kv(k, false);
      issue_idx(k + STAGES - 1);
      if (k < PRE - 1 || k == STAGES - 2) cp_commit();
    }
    }
  }
  DBG(8);
  }
  if constexpr (IDXF) {
    cp_wait<STAGES - 1>();
    csync<WS, NT>();
  }
  if (p.rope) {
    tfor<L * 32, NT>(tidL, [&](int e) {
      const int j = e >> 5, i = e & 31;
      const float ang = (float)(seq + j) * S.ifreq[i];
      float sv, cv;
      const float kq = rintf(ang * 0.15915493667125702f);
      float rr = fmaf(-kq, 6.25f, ang);
      rr = fmaf(-kq, 0.03318530693650246f, rr);
      rr = fmaf(-kq, 2.430837753308879e-10f, rr);
      __sincosf(rr, &sv, &cv);
      S.cs[e] = cv;
      S.sn[e] = sv;
    });
  }
  csync<WS, NT>();
  DBG(9);
  const float kscale = S.scl[0], vscale = S.scl[1];
  const float sl2 = p.scale_log2 * kscale;
  if constexpr (QDIR && OKC_QP16) {
    constexpr int RP4 = NT / 4;
    constexpr int QP4 = L * 16 / RP4;
    const int q4 = tidL & 3;
#pragma unroll
    for (int ps = 0; ps < QP4; ++ps) {
      const int rr = ps * RP4 + (tidL >> 2);
      const int j = rr >> 4, hh = rr & 15;
      const uint4 qa = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16]);
      const uint4 qb = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + q4 * 16 + 8]);
      const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&qa);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&qb);
      float x[16];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 fa = __bfloat1622float2(ha[e]);
        const float2 fb = __bfloat1622float2(hb[e]);
        x[2 * e] = fa.x; x[2 * e + 1] = fa.y;
        x[8 + 2 * e] = fb.x; x[8 + 2 * e + 1] = fb.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) ss += x[e] * x[e];
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[e] *= r;
      if (p.rope) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int i = q4 * 8 + e;
          const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
          const float a0 = x[2 * e], a1 = x[2 * e + 1];
          x[2 * e] = a0 * c - a1 * sn;
          x[2 * e + 1] = a0 * sn + a1 * c;
        }
      }
      const int hk4 = hh >> 3, gg4 = hh & 7;
      const int r4 = j * 8 + gg4, kb = q4 * 16;
      *reinterpret_cast<uint4*>(S.q8s + hk4 * 4096 + r4 * 64 + (((kb >> 4) ^ ((r4 >> 1) & 3)) << 4)) =
          make_uint4(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]),
                     e4m3x4f(x[8], x[9], x[10], x[11]), e4m3x4f(x[12], x[13], x[14], x[15]));
    }
  } else
#pragma unroll
  for (int ps = 0; ps < QPASS; ++ps) {
    const int rr = ps * RPP + (tidL >> 3);
    const int j = rr >> 4, hh = rr & 15;
    const uint4 qv = *reinterpret_cast<const uint4*>(&S.qst()[rr * 64 + partL * 8]);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
    float x[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(h2[e]);
      x[2 * e] = f.x;
      x[2 * e + 1] = f.y;
    }
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
    ss = oct_sum(ss);
    const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] *= r;
    if (p.rope) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = partL * 4 + e;
        const float c = S.cs[j * 32 + i], sn = S.sn[j * 32 + i];
        const float a0 = x[2 * e], a1 = x[2 * e + 1];
        x[2 * e] = a0 * c - a1 * sn;
        x[2 * e + 1] = a0 * sn + a1 * c;
      }
    }
    uint4 qw;
    uint32_t* qwp = reinterpret_cast<uint32_t*>(&qw);
    const float qsc = (L == 8 && OKC_SM90 && OKC_Q8RAW) ? 1.f : sl2;
    const int hk = hh >> 3, gg = hh & 7;
    if constexpr (QDIR) {
      const int r = j * 8 + gg, kb = partL * 8;
      *reinterpret_cast<uint2*>(S.q8s + hk * 4096 + r * 64 + (((kb >> 4) ^ ((r >> 1) & 3)) << 4) + (kb & 15)) =
          make_uint2(e4m3x4f(x[0], x[1], x[2], x[3]), e4m3x4f(x[4], x[5], x[6], x[7]));
      continue;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) qwp[e] = pack_h2(x[2 * e] * qsc, x[2 * e + 1] * qsc);
    if constexpr (L == 1) { const uint32_t t1 = qwp[1]; qwp[1] = qwp[2]; qwp[2] = t1; }
    if constexpr (WS) {
      *reinterpret_cast<uint4*>(S.kvq + rr * 128 + partL * 16) = qw;
    } else {
      uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
      *reinterpret_cast<uint4*>(qs + (j * 8 + gg) * 128 + partL * 16) = qw;
    }
  }
  if constexpr (QDIR) fence_proxy_async();
  DBG(10);
  if (has_new) {
#pragma unroll
    for (int ps = 0; ps < KPASS; ++ps) {
      const int rr = ps * RPP + (tidL >> 3);
      const bool live = rr < L * 4;
      const int kind = rr / (2 * L);
      const int jj = (rr % (2 * L)) >> 1, hh = rr & 1;
      float x[8];
      const uint4 kvv = live ? *reinterpret_cast<const uint4*>(&S.kvst[rr * 64 + partL * 8]) : make_uint4(0, 0, 0, 0);
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&kvv);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(h2[e]);
        x[2 * e] = f.x;
        x[2 * e + 1] = f.y;
      }
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) ss += x[e] * x[e];
      ss = oct_sum(ss);
      float sc;
      if (kind == 0) {
        const float r = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
#pragma unroll
        for (int e = 0; e < 8; ++e) x[e] *= r;
        if (p.rope && live) {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = partL * 4 + e;
            const float c = S.cs[jj * 32 + i], sn = S.sn[jj * 32 + i];
            const float a0 = x[2 * e], a1 = x[2 * e + 1];
            x[2 * e] = a0 * c - a1 * sn;
            x[2 * e + 1] = a0 * sn + a1 * c;
          }
        }
        sc = __fdividef(1.f, kscale);
      } else {
        sc = __fdividef(1.f, vscale);
      }
      if (live) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          w0 |= (uint32_t)to_e4m3(x[e] * sc) << (8 * e);
          w1 |= (uint32_t)to_e4m3(x[4 + e] * sc) << (8 * e);
        }
        uint8_t* dst = (kind == 0 ? S.kn : S.vn) + jj * ROWB + hh * 64 + partL * 8;
        *reinterpret_cast<uint2*>(dst) = make_uint2(w0, w1);
        const int pos = seq + jj;
        if (pos >= start && pos < end) {
          const int64_t row = S.cloc[jj];
          uint8_t* g8 = const_cast<uint8_t*>(cache_row(kind == 0 ? p.k_cache : p.v_cache, p, row, (2 * pair + hh) * 64 + partL * 8));
          *reinterpret_cast<uint2*>(g8) = make_uint2(w0, w1);
        }
      }
    }
  }
  csync<WS, NT>();
  DBG(11);
  if (!WS && has_new) {
    for (int x = 0; x < (Cfg<L, FLAT>::S2 ? STAGES - 2 : STAGES - 1) && x < ntiles; ++x) {
      const int st = x % STAGES;
#pragma unroll
      for (int k = 0; k < TILE / RPP; ++k) {
        const int r = crow0 + RPP * k;
        const int pos = start + x * TILE + r;
        if (pos >= seq && pos < end) {
          const int sw = ((cchunk ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + cchunk * 16);
          *reinterpret_cast<uint4*>(S.v + st * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + cchunk * 16);
        }
      }
    }
    if (L == 8) fence_proxy_async();
  }
#ifndef LDD19_FINWARM  // LDD19: integrator_scout it3: OKC_FINWARM warm-up, switched on only in the kl_ATT7 build (-DLDD19_FINWARM=1); 0 = previous behaviour
#define LDD19_FINWARM 0
#endif
  if constexpr (!FLAT && (OKC_FINWARM || LDD19_FINWARM)) {
    if (p.cs <= 1 && !p.coop && p.nc == 1) fin_nc1<L, NT, RBF_Q>(p, b, pair, bp, 0.f, 1);   // (LDD19 LDD_RBFQ)
  }
  DBG(2);
  constexpr int MT = 1;
  constexpr int NROW = (L == 1) ? 1 : 2;
  constexpr bool S2 = Cfg<L, FLAT>::S2;
  constexpr bool F2 = Cfg<L, FLAT>::F2;
  const int sid = (S2 || F2) ? (warp >> 3) : 0;
  const int hk = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp & 1) : ((warp & 7) >> 2), 0)
                          : ((L == 1) ? (warp & 1) : ((warp & 7) >> 2));
  const int wsub = OKC_INV2 ? __shfl_sync(0xffffffffu, (L == 1) ? (warp >> 1) : (warp & 3), 0)
                            : ((L == 1) ? (warp >> 1) : (warp & 3));
  const int wgi = (S2 || F2) ? 2 * sid + hk : hk;
  float o[MT][8][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][t][e] = 0.f;
  float m_r[NROW], l_r[NROW];
#pragma unroll
  for (int r = 0; r < NROW; ++r) { m_r[r] = -INFINITY; l_r[r] = 0.f; }
  const int lo_all = (L == 1 || !p.window) ? start : max(start, seq + (L - 1) - p.window + 1);
  const int hi_all = OKC_INV2 ? __shfl_sync(0xffffffffu, min(end - 1, seq), 0) : min(end - 1, seq);
  if constexpr (L == 1) {
  uint32_t qa[4][2];
  {
    const uint8_t* qs = WS ? S.kvq + hk * 8 * 128 : (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const uint4 l0 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig);
    const uint4 l1 = *reinterpret_cast<const uint4*>(qs + g * 128 + 32 * tig + 16);
    qa[0][0] = l0.x; qa[0][1] = l0.y; qa[1][0] = l0.z; qa[1][1] = l0.w;
    qa[2][0] = l1.x; qa[2][1] = l1.y; qa[3][0] = l1.z; qa[3][1] = l1.w;
  }
  PROF_INIT
  for (int i = 0; i < ntiles; ++i) {
    if constexpr (WS) {
      if (ws_acquire(i, (tb + i) % STAGES)) csync<WS, NT>();
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
      PROF_MARK(1);
    }
    const uint8_t* K = S.k + ((tb + i) % STAGES) * STAGE_BYTES;
    const uint8_t* V = S.v + ((tb + i) % STAGES) * STAGE_BYTES;
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
    float s[4][4];
    uint32_t kw[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int key = 32 * wsub + 8 * nt + g;
      const uint4 kb = lds128(K + key * ROWB + (((hk * 4 + tig) ^ (key & 7)) << 4));
      kw[nt][0] = kb.x; kw[nt][1] = kb.y; kw[nt][2] = kb.z; kw[nt][3] = kb.w;
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      float se[4] = {0.f, 0.f, 0.f, 0.f}, so[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        mma_f16(se, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h]), cvt2h(kw[nt][2 * h]));
        mma_f16(so, qa[2 * h][0], qa[2 * h][1], qa[2 * h + 1][0], qa[2 * h + 1][1], cvt2(kw[nt][2 * h + 1]), cvt2h(kw[nt][2 * h + 1]));
      }
      s[nt][0] = se[0] + so[2]; s[nt][1] = se[1] + so[3]; s[nt][2] = 0.f; s[nt][3] = 0.f;
    }
    PROF_MARK(2);
    if (!interior) {
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int pos = t0 + 32 * wsub + 8 * nt + 2 * tig + e;
          if (pos >= end) s[nt][e] = -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) mx = fmaxf(mx, fmaxf(s[nt][0], s[nt][1]));
    mx = quad_max(mx);
    const float m_new = fmaxf(m_r[0], mx);
    if (m_new != -INFINITY) {
      const float alpha = ex2(m_r[0] - m_new);
      m_r[0] = m_new;
      float ls = 0.f;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float pv = ex2(s[nt][e] - m_new);
          s[nt][e] = pv;
          ls += pv;
        }
      if constexpr (ONE) {
        l_r[0] = ls;
      } else {
        l_r[0] = l_r[0] * alpha + ls;
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alpha; o[0][t][1] *= alpha; }
      }
#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {
        uint32_t b0[8], b1[8];
        load_v_frags(V, 32 * wsub + 16 * ks, hk, g, tig, b0, b1);
        const uint32_t a0 = pack_h2(s[2 * ks][0], s[2 * ks][1]);
        const uint32_t a2 = pack_h2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
#pragma unroll
        for (int t = 0; t < 8; ++t) mma_f16(o[0][t], a0, 0u, a2, 0u, b0[t], b1[t]);
      }
    }
    if constexpr (WS) {
      __syncwarp();
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]);
    }
    PROF_MARK(3);
    PROF_TILE;
    if (i == 0) DBG(13);
  }
  PROF_DUMP;
  } else {
  uint8_t* V16 = S.kvq;
  const uint32_t v16_base = smem_u32(V16) + hk * 8192;
  uint32_t qa8[2][4];
  if constexpr (!QDIR) {
    const uint8_t* qs = (hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES;
    const int r0 = 16 * wsub + g, r1 = r0 + 8;
    const uint8_t* q0 = WS ? S.kvq + ((r0 >> 3) * 16 + hk * 8 + (r0 & 7)) * 128 : qs + r0 * 128;
    const uint8_t* q1 = WS ? S.kvq + ((r1 >> 3) * 16 + hk * 8 + (r1 & 7)) * 128 : qs + r1 * 128;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int dofs = (32 * ks + 16 * hf + 4 * tig) * 2;
        const uint2 lo = *reinterpret_cast<const uint2*>(q0 + dofs);
        const uint2 hi = *reinterpret_cast<const uint2*>(q1 + dofs);
        qa8[ks][2 * hf] = pack_e4m3x4(lo.x, lo.y);
        qa8[ks][2 * hf + 1] = pack_e4m3x4(hi.x, hi.y);
      }
    }
    if constexpr (PV8ON && OKC_QSS && (!FLAT || OKC_PV8FLAT)) {
#pragma unroll
      for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
          const int kb = 32 * ks + 16 * hf + 4 * tig;
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r0 * 64 + (((kb >> 4) ^ ((r0 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf];
          *reinterpret_cast<uint32_t*>(S.q8s + hk * 4096 + r1 * 64 + (((kb >> 4) ^ ((r1 >> 1) & 3)) << 4) + (kb & 15)) =
              qa8[ks][2 * hf + 1];
        }
      fence_proxy_async();
    }
    if constexpr (PTR) {
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.qdone);
    }
  }
  float cr[2] = {0.f, 0.f};
  const float ssc = OKC_Q8RAW ? sl2 : 1.f;
  const float rssc = OKC_RAWTHR ? __fdividef(1.f, ssc) : 1.f;
  float thr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
  if constexpr (OKC_OZF) {
    float* oz = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(oz[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  const uint32_t ONES = 0x3C003C00u;
  const int ckey = tid >> 2, cq = tid & 3;
  constexpr bool PV8 = PV8ON && (!FLAT || OKC_PV8FLAT);
  const int vwq = warp & 3, vq = lane & 3, vw = lane >> 2;
  const uint32_t vsel0 = (vw >= 4) ? 0x7531u : 0x6420u, vsel1 = (vw >= 4) ? 0x6420u : 0x7531u;
  const int vr0 = (vw >= 4) ? 1 : 0;
  const uint32_t vt_off0 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * vr0;
  const uint32_t vt_off1 = 128 * vw + ((vwq ^ (vw & 3)) << 4) + 4 * vq + 64 * (1 - vr0);
  const int vkey_t = 16 * vwq + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld_off0 = vkey_t * ROWB + (((4 * hk + 0 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  const uint32_t vld_off1 = vkey_t * ROWB + (((4 * hk + 2 + ((lane >> 3) >> 1)) ^ (lane & 7)) << 4);
  const int stj = lane & 7, stm = lane >> 3;
  const uint32_t vst_off = 1024 * (stm >> 1) + 128 * stj + 64 * ((stm & 1) ^ (stj >> 2)) + ((vwq ^ (stj & 3)) << 4);
  constexpr bool STSM_ON = OKC_STSM || (OKC_STSMPR && !FLAT);
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  constexpr bool ADDR = OKC_ADDR && !S2 && !F2 && (WS || OKC_ADDRN);
  constexpr bool PVSPLIT = OKC_PVSPLIT && !S2 && !F2 && PV8 && !PTR && !OKC_PROF;
  constexpr bool LSYNCN = OKC_LATESYNCN && !WS && !FLAT && PV8 && !PTR && PVSPLIT && !OKC_XP_NOFENCE && !S2 && !F2;
  constexpr bool LSYNC = OKC_LATESYNC && OKC_PK2 && (OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF &&
                         !PTR && WS) && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                         !(OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF) && !OKC_XP_NOFENCE;
  int st_c = tb % STAGES;
  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;
  const uint32_t qlo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.q8s) + hk * 4096) >> 4) | 0x10000u, 0);
  const uint32_t klo0 = __shfl_sync(0xffffffffu, (uint32_t)((smem_u32(S.k) + hk * 64) >> 4) | 0x10000u, 0);
  const uint64_t qdesc0 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | qlo0;
  const uint64_t kdesc0 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | klo0;
  const uint32_t vsrc0 = smem_u32(S.v);
  uint8_t* vtb_c = S.kvq + hk * 4608;
  const uintptr_t vtb_x = (uintptr_t)(S.kvq + hk * 4608) ^ (uintptr_t)(S.kvq + (2 + hk) * 4608);
  uint64_t vdesc_c = desc_sw64(smem_u32(S.kvq + hk * 4608));
  const uint64_t vdesc_x = vdesc_c ^ desc_sw64(smem_u32(S.kvq + (2 + hk) * 4608));
  const int i_new = seq > start ? (seq - start) / TILE : 0;
  uint32_t i_lo = 0, i_span = 0;
  if constexpr (OKC_IRANGE) {
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi_all - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    i_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    i_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  uint32_t r_lo = 0x7fffffffu, r_span = 0u;
  if constexpr (OKC_RARE && WS && ADDR) {
    const int hi2 = min(end - 1, seq - 1);
    const int a = (lo_all - start + TILE - 1) / TILE;
    const int z = hi2 - (TILE - 1) - start;
    const int bz = z >= 0 ? z / TILE : -1;
    r_lo = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)a : 0x7fffffffu, 0);
    r_span = __shfl_sync(0xffffffffu, bz >= a ? (uint32_t)(bz - a) : 0u, 0);
  }
  PROF_INIT
  constexpr bool TURN = OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF;
  if constexpr (TURN && OKC_TURN == 1) {
    if (hk == 1 && ntiles > 0) bar_arrive_n(5, 256);
  }
  int i_cur = 0, cnt2 = 0;
  if constexpr (F2) {
    if (tid == 0) { S.claim[0] = 0; S.claim[1] = 0; }
    csync<WS, NT>();
    if ((tid & 127) == 0) S.nxt[wgi][0] = atomicAdd(&S.claim[hk], 1);
    wgsync(wgi);
    i_cur = S.nxt[wgi][0];
  }
  const int nit = S2 ? ((ntiles + 1) >> 1) : (F2 ? 0x7fffffff : ntiles);
  constexpr int UNR8 = WS ? OKC_UNR8 : ((MODE & 3) == 1 ? OKC_UNRN : 1);
#pragma unroll (UNR8)
  for (int jj = 0; jj < nit; ++jj) {
    const int i = S2 ? 2 * jj + sid : (F2 ? i_cur : jj);
    if (F2 && i >= ntiles) break;
    if constexpr (F2) {
      if ((tid & 127) == 0) S.nxt[wgi][(cnt2 + 1) & 1] = atomicAdd(&S.claim[hk], 1);
      const int st0 = (tb + i) % STAGES;
      mbar_wait_full(&S.full[st0], ((tb + i) / STAGES) & 1);
      const int t0n = start + i * TILE;
      const int nlo = max(t0n, seq), nhi = min(t0n + TILE, end);
      if (nlo < nhi) {
        for (int e = (tid & 127); e < (nhi - nlo) * 4; e += 128) {
          const int pos = nlo + (e >> 2), c = 4 * hk + (e & 3), r = pos - t0n;
          const int sw = ((c ^ (r & 7)) << 4);
          *reinterpret_cast<uint4*>(S.k + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.kn + (pos - seq) * ROWB + c * 16);
          *reinterpret_cast<uint4*>(S.v + st0 * STAGE_BYTES + r * ROWB + sw) = *reinterpret_cast<const uint4*>(S.vn + (pos - seq) * ROWB + c * 16);
        }
        fence_proxy_async();
        wgsync(wgi);
      }
      PROF_MARK(0);
    } else if constexpr (WS) {
      bool patched;
      if constexpr (ADDR) {
        mbar_wait_full(&S.full[st_c], ph_c);
        if constexpr (OKC_RARE) patched = ((uint32_t)i - r_lo > r_span) && (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else if constexpr (OKC_PATCHNI && WS) patched = (i >= i_new) && ws_patch_ni<L, NT, PTR>(&S, i, st_c, start, seq, end, tid);
        else patched = (i >= i_new) && ws_patch(i, st_c);
      } else if constexpr (OKC_INEW) {
        mbar_wait_full(&S.full[(tb + i) % STAGES], ((tb + i) / STAGES) & 1);
        patched = (i >= i_new) && ws_patch(i, (tb + i) % STAGES);
      } else {
        patched = ws_acquire(i, (tb + i) % STAGES);
      }
      if (patched || i == 0 || !OKC_WGDEC) csync<WS, NT>();
      else if (!(PV8ON && OKC_PV8FLAT && OKC_NOSYNC0)) wgsync(hk);
      PROF_MARK(0);
      if (i == 0) DBG(12);
    } else if constexpr (S2) {
      cp_wait<2>();
      __syncthreads();       // and both streams' previous PVs read their V^T (the stages of tiles 2jj-2, 2jj-1 are free)
      PROF_MARK(0);
      if (jj == 0) DBG(12);
      if (2 * jj + 4 < ntiles) issue_kv(2 * jj + 4, true);
      issue_idx(2 * jj + 2 * STAGES - 2);
      cp_commit();
      if (2 * jj + 5 < ntiles) issue_kv(2 * jj + 5, true);
      issue_idx(2 * jj + 2 * STAGES - 1);
      cp_commit();
      PROF_MARK(1);
      if (i >= ntiles) continue;
    } else {
      if constexpr (!OKC_XNOWAIT) cp_wait<STAGES - 2>();
      __syncthreads();       // tile i landed; previous tile's PV done (V16 free)
      PROF_MARK(0);
      if (i == 0) DBG(12);
      if constexpr (!OKC_QKFIRST) {
        if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
        issue_idx(i + 2 * STAGES - 2);
        cp_commit();
      }
      PROF_MARK(1);
    }
    const int st = ADDR ? st_c : (tb + i) % STAGES;
    const uint32_t kaddr = smem_u32(S.k + st * STAGE_BYTES) + hk * 64;
    if constexpr (PVSPLIT && OKC_PVSPLIT == 2) {
      wgmma_wait<0>();
      float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    float s[32];
    wgmma_fence();
    if constexpr (PV8 && OKC_QSS && ADDR) {
      const uint32_t kl = klo0 + (uint32_t)(st * (STAGE_BYTES >> 4));
      const uint64_t kd = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | kl;
      const uint64_t kd2 = ((uint64_t)((1024 >> 4) | (1u << 30)) << 32) | (kl + 2u);
      const uint64_t qd2 = ((uint64_t)((512 >> 4) | (2u << 30)) << 32) | (qlo0 + 2u);
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, qdesc0, kd);
      else wgmma_m64n64k32_e4m3_ss(s, qdesc0, kd, 0);
      wgmma_m64n64k32_e4m3_ss(s, qd2, kd2, 1);
    } else if constexpr (PV8 && OKC_QSS) {
      const uint32_t qaddr = smem_u32(S.q8s) + hk * 4096;
      if constexpr (OKC_QKZ) wgmma_m64n64k32_e4m3_ss_z(s, desc_sw64(qaddr), desc_sw128(kaddr));
      else wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr), desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3_ss(s, desc_sw64(qaddr + 32), desc_sw128(kaddr + 32), 1);
    } else {
      wgmma_m64n64k32_e4m3(s, qa8[0][0], qa8[0][1], qa8[0][2], qa8[0][3], desc_sw128(kaddr), 0);
      wgmma_m64n64k32_e4m3(s, qa8[1][0], qa8[1][1], qa8[1][2], qa8[1][3], desc_sw128(kaddr + 32), 1);
    }
    wgmma_commit();
    if constexpr (OKC_QKFIRST && !WS && !Cfg<L, FLAT>::S2) {
      if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
      issue_idx(i + 2 * STAGES - 2);
      cp_commit();
    }
    if constexpr (PV8 && FLAT && OKC_PVW1) {
      wgmma_wait<1>();
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      float* od0 = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od0[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
    }
    uint8_t* V16t = V16 + (OKC_PVW ? (i & 1) * 16384 : 0);
    const uint32_t v16b = smem_u32(V16t) + hk * 8192;
    uint8_t* vtb = ADDR ? vtb_c : S.kvq + ((S2 ? 2 * sid + (jj & 1) : (F2 ? 2 * sid + (cnt2 & 1) : (i & 1))) * 2 + hk) * 4608;
    if constexpr (PV8 && !PTR) {
      if (F2 ? cnt2 == 0 : jj == 0) {
        const int wt = tid & 127;
        *reinterpret_cast<uint2*>(S.kvq + ((((S2 || F2) ? 2 * sid : 0) + (wt >> 6)) * 2 + hk) * 4608 + 4096 + (wt & 63) * 8) =
            make_uint2(0x38383838u, 0x38383838u);
      }
    }
    if constexpr (PTR) {
    } else if constexpr (PV8) {
      const uint32_t vsrc = ADDR ? vsrc0 + st * STAGE_BYTES : smem_u32(S.v + st * STAGE_BYTES);
#pragma unroll
      for (int u = 0; u < (OKC_XP_NOTR ? 0 : 2); ++u) {
        uint32_t r0, r1, r2, r3;
        ldsm_x4_t(vsrc + (u ? (OKC_INV2 ? (vld_off0 ^ 32u) : vld_off1) : vld_off0), r0, r1, r2, r3);
        if constexpr (STSM_ON) {
          stsm_x4(smem_u32(vtb) + vst_off + 2048 * u, prmt(r0, r1, vsel0), prmt(r0, r1, vsel1), prmt(r2, r3, vsel0),
                  prmt(r2, r3, vsel1));
        } else {
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2) {
          const uint32_t R0 = g2 ? r2 : r0, R1 = g2 ? r3 : r1;
          const uint32_t w0 = prmt(R0, R1, vsel0), w1 = prmt(R0, R1, vsel1);
          *reinterpret_cast<uint32_t*>(vtb + vt_off0 + 1024 * (2 * u + g2)) = w0;
          *reinterpret_cast<uint32_t*>(vtb + vt_off1 + 1024 * (2 * u + g2)) = w1;
        }
        }
      }
      if (!OKC_XP_NOFENCE && !LSYNC && !LSYNCN) fence_proxy_async();
    } else {
      const uint8_t* V = S.v + st * STAGE_BYTES;
      const bool dec = WS && OKC_WGDEC;
      const int wt = tid & 127;
      const int vkey = dec ? (wt >> 1) : ckey;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = dec ? (hk * 4 + (wt & 1) * 2 + u) : (2 * cq + u);
        const int hd = c >> 2, fc = (c & 3) * 2;
        const int ckey = vkey;
        const uint4 vb = lds128(V + ckey * ROWB + ((c ^ (ckey & 7)) << 4));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + ((fc ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.x), cvt2(vb.x >> 16), cvt2(vb.y), cvt2(vb.y >> 16));
        *reinterpret_cast<uint4*>(V16t + hd * 8192 + ckey * 128 + (((fc + 1) ^ (ckey & 7)) << 4)) =
            make_uint4(cvt2(vb.z), cvt2(vb.z >> 16), cvt2(vb.w), cvt2(vb.w >> 16));
      }
      if (OKC_PVW) fence_proxy_async();
    }
    if constexpr (PTR) {
    } else if (LSYNC || LSYNCN) {
    } else if ((WS && OKC_WGDEC) || PV8) wgsync(wgi);
    else csync<WS, NT>();
    PROF_MARK(2);
    wgmma_wait<0>();
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(s[e]);
    if constexpr (PV8) {
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
    }
    if constexpr (WS) {
      if (OKC_XP_ARRALL || lane == 0) mbar_arrive(&S.empty[st]);
    }
    if constexpr (PTR) {
      if (i >= 1 && lane == 0) mbar_arrive(&S.vte[(i - 1) & 1]);
    }
    PROF_MARK(3);
    const bool common_t = OKC_RARE && WS && ADDR && ((uint32_t)i - r_lo <= r_span);
    const int t0 = start + i * TILE;
    const bool interior = OKC_XP_NOMASK || common_t ||
                          (OKC_IRANGE ? ((uint32_t)i - i_lo <= i_span) : ((t0 >= lo_all) && (t0 + TILE - 1 <= hi_all)));
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 8 * nt + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
          }
      }
    }
    constexpr bool PCHK = OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR && PV8 && !OKC_EXH8 && !OKC_PROF && !PTR && WS;
    const bool chk = PCHK && !first;
    float mlo = -INFINITY, mhi = -INFINITY;
    if (!chk) {
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mlo = fmaxf(mlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
      mhi = fmaxf(mhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
    }
    }
    if (!OKC_RAWTHR && !chk) {
      mlo *= ssc;
      mhi *= ssc;
    }
    float dlo, dhi;
    if (first) {
      if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }
      if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
      dlo = (mlo == -INFINITY) ? 0.f : mlo;
      dhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = dlo; cr[1] = dhi;
      if (OKC_RAWTHR) { thr[0] = (dlo + 8.f) * rssc; thr[1] = (dhi + 8.f) * rssc; }
      first = false;
    } else {
      dlo = cr[0]; dhi = cr[1];
      if (!PCHK && __any_sync(0xffffffffu, (OKC_LAZYMAX && OKC_RAWTHR) ? (mlo > thr[0] || mhi > thr[1])
                                                                       : (mlo > cr[0] + 8.f || mhi > cr[1] + 8.f))) {
        if (OKC_LAZYMAX) { mlo = quad_max(mlo); mhi = quad_max(mhi); }
        if (OKC_LAZYMAX && OKC_RAWTHR) { mlo *= ssc; mhi *= ssc; }
        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;
        if (OKC_RAWTHR) { thr[0] = (nlo + 8.f) * rssc; thr[1] = (nhi + 8.f) * rssc; }
      }
    }
    PROF_MARK(4);
    if constexpr (TURN && OKC_TURN == 1) bar_sync_n(5 + hk, 256);
    if constexpr (TURN && OKC_TURN == 2) {
      if (hk == 1 && jj == 0) bar_sync_n(6, 256);
    }
    if constexpr (PV8) {
      constexpr bool PK2 = OKC_PK2 && PCHK && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_EXH8 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                           !TURN && !PTR;
      if constexpr (PK2) {
        auto pack2 = [&]() {
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = s + 16 * kk;
            pa[kk][0] = e4m3x4f(ex2(fmaf(sv[0], ssc, -dlo)), ex2(fmaf(sv[1], ssc, -dlo)), ex2(fmaf(sv[4], ssc, -dlo)), ex2(fmaf(sv[5], ssc, -dlo)));
            pa[kk][1] = e4m3x4f(ex2(fmaf(sv[2], ssc, -dhi)), ex2(fmaf(sv[3], ssc, -dhi)), ex2(fmaf(sv[6], ssc, -dhi)), ex2(fmaf(sv[7], ssc, -dhi)));
            pa[kk][2] = e4m3x4f(ex2(fmaf(sv[8], ssc, -dlo)), ex2(fmaf(sv[9], ssc, -dlo)), ex2(fmaf(sv[12], ssc, -dlo)), ex2(fmaf(sv[13], ssc, -dlo)));
            pa[kk][3] = e4m3x4f(ex2(fmaf(sv[10], ssc, -dhi)), ex2(fmaf(sv[11], ssc, -dhi)), ex2(fmaf(sv[14], ssc, -dhi)), ex2(fmaf(sv[15], ssc, -dhi)));
          }
        };
        pack2();
        if (chk) {
          constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
          const uint32_t ov = ((pa[0][0] + PB) | (pa[0][1] + PB) | (pa[0][2] + PB) | (pa[0][3] + PB) |
                               (pa[1][0] + PB) | (pa[1][1] + PB) | (pa[1][2] + PB) | (pa[1][3] + PB)) & 0x80808080u;
          if (__any_sync(0xffffffffu, ov != 0u)) {
            float hlo = -INFINITY, hhi = -INFINITY;
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
              hlo = fmaxf(hlo, fmaxf(s[4 * nt], s[4 * nt + 1]));
              hhi = fmaxf(hhi, fmaxf(s[4 * nt + 2], s[4 * nt + 3]));
            }
            hlo = quad_max(hlo) * ssc;
            hhi = quad_max(hhi) * ssc;
            const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
            const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
#pragma unroll
            for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
            lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
            cr[0] = nlo; cr[1] = nhi;
            dlo = nlo; dhi = nhi;
            pack2();
          }
        }
        float* odk = &o[0][0][0];
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(odk[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        if constexpr (LSYNC) {
          fence_proxy_async();
          wgsync(wgi);
        }
        wgmma_fence();
        if constexpr (ADDR) {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], vdesc_c);
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], vdesc_c + 2);
        } else {
          wgmma_m64n72k32_e4m3(odk, lacc, pa[0], desc_sw64(smem_u32(vtb)));
          wgmma_m64n72k32_e4m3(odk, lacc, pa[1], desc_sw64(smem_u32(vtb) + 32));
        }
        wgmma_commit();
      }
#pragma unroll
      for (int kk = 0; kk < (PK2 ? 0 : 2); ++kk) {
        const float* sv = s + 16 * kk;
#define EXQ(x, q) ex2(x)
        bool rchk = chk;
#pragma unroll 1
        for (;;) {
        pa[kk][0] = e4m3x4f(EXQ(fmaf(sv[0], ssc, -dlo), 0), EXQ(fmaf(sv[1], ssc, -dlo), 1), EXQ(fmaf(sv[4], ssc, -dlo), 2), EXQ(fmaf(sv[5], ssc, -dlo), 3));
        pa[kk][1] = e4m3x4f(EXQ(fmaf(sv[2], ssc, -dhi), 0), EXQ(fmaf(sv[3], ssc, -dhi), 1), EXQ(fmaf(sv[6], ssc, -dhi), 2), EXQ(fmaf(sv[7], ssc, -dhi), 3));
        pa[kk][2] = e4m3x4f(EXQ(fmaf(sv[8], ssc, -dlo), 0), EXQ(fmaf(sv[9], ssc, -dlo), 1), EXQ(fmaf(sv[12], ssc, -dlo), 2), EXQ(fmaf(sv[13], ssc, -dlo), 3));
        pa[kk][3] = e4m3x4f(EXQ(fmaf(sv[10], ssc, -dhi), 0), EXQ(fmaf(sv[11], ssc, -dhi), 1), EXQ(fmaf(sv[14], ssc, -dhi), 2), EXQ(fmaf(sv[15], ssc, -dhi), 3));
        if constexpr (!PCHK) break;
        else {
          if (!rchk) break;
          {
            constexpr uint32_t PB = OKC_PCHKTEST ? 0x47474747u : 0x07070707u;
            const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
            if (!__any_sync(0xffffffffu, ov != 0u)) break;
            {
              float hlo = fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[4], sv[5])), fmaxf(fmaxf(sv[8], sv[9]), fmaxf(sv[12], sv[13])));
              float hhi = fmaxf(fmaxf(fmaxf(sv[2], sv[3]), fmaxf(sv[6], sv[7])), fmaxf(fmaxf(sv[10], sv[11]), fmaxf(sv[14], sv[15])));
              hlo = quad_max(hlo) * ssc;
              hhi = quad_max(hhi) * ssc;
              const float nlo = fmaxf(hlo, cr[0]), nhi = fmaxf(hhi, cr[1]);
              const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);
              if (PVSPLIT && (kk == 1 || OKC_PVSPLIT == 2)) {
                wgmma_wait<0>();
                float* odw = &o[0][0][0];
#pragma unroll
                for (int e = 0; e < 32; ++e) reg_fence(odw[e]);
#pragma unroll
                for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
              }
#pragma unroll
              for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
              lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
              cr[0] = nlo; cr[1] = nhi;
              dlo = nlo; dhi = nhi;
              rchk = false;
            }
          }
        }
        }
#undef EXQ
        if constexpr (PVSPLIT) {
          float* od = &o[0][0][0];
          if constexpr (LSYNCN) {
            if (kk == 0) {
              fence_proxy_async();
              wgsync(wgi);
            }
          }
          if (kk == 0) {
#pragma unroll
            for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
            for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(pa[kk][e]));
          wgmma_fence();
          if constexpr (ADDR) wgmma_m64n72k32_e4m3(od, lacc, pa[kk], vdesc_c + 2 * kk);
          else wgmma_m64n72k32_e4m3(od, lacc, pa[kk], desc_sw64(smem_u32(vtb) + 32 * kk));
          wgmma_commit();
        }
      }
      if constexpr (TURN && OKC_TURN == 1) {
        if (!(hk == 1 && i == ntiles - 1)) bar_arrive_n(6 - hk, 256);
      }
      if constexpr (TURN && OKC_TURN == 2) {
        if (hk == 0 && jj == 0) bar_arrive_n(6, 256);
      }
      float* od = &o[0][0][0];
      if constexpr (PVSPLIT) {
        if constexpr (OKC_PVSPLIT == 1) {
          wgmma_wait<0>();
#pragma unroll
          for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
        }
      } else {
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
      if constexpr (PTR) mbar_wait_full(&S.vtf[i & 1], (i >> 1) & 1);
      wgmma_fence();
      if constexpr (ADDR) {
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], vdesc_c);
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], vdesc_c + 2);
      } else {
        const uint32_t vtaddr = smem_u32(vtb);
        wgmma_m64n72k32_e4m3(od, lacc, pa[0], desc_sw64(vtaddr));
        wgmma_m64n72k32_e4m3(od, lacc, pa[1], desc_sw64(vtaddr + 32));
      }
      wgmma_commit();
      if (OKC_PV8SYNC) {
        wgmma_wait<0>();
#pragma unroll
        for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
        for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
      }
      }
    } else {
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t a0 = ex2h2(fmaf(s[8 * ks], ssc, -dlo), fmaf(s[8 * ks + 1], ssc, -dlo));
      const uint32_t a1 = ex2h2(fmaf(s[8 * ks + 2], ssc, -dhi), fmaf(s[8 * ks + 3], ssc, -dhi));
      const uint32_t a2 = ex2h2(fmaf(s[8 * ks + 4], ssc, -dlo), fmaf(s[8 * ks + 5], ssc, -dlo));
      const uint32_t a3 = ex2h2(fmaf(s[8 * ks + 6], ssc, -dhi), fmaf(s[8 * ks + 7], ssc, -dhi));
      const int mi = lane >> 3;
      const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4_t(v16b + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(o[0][2 * dp], a0, a1, a2, a3, b0a, b1a);
        mma_f16(o[0][2 * dp + 1], a0, a1, a2, a3, b0b, b1b);
      }
      mma_f16(lacc, a0, a1, a2, a3, ONES, ONES);
    }
    }
    PROF_MARK(5);
    PROF_TILE;
    if (i == 0) DBG(13);
    if constexpr (F2) {
      i_cur = S.nxt[wgi][(cnt2 + 1) & 1];
      ++cnt2;
    }
    if constexpr (ADDR) {
      if (++st_c == STAGES) { st_c = 0; ph_c ^= 1u; }
      vtb_c = reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(vtb_c) ^ vtb_x);
      vdesc_c ^= vdesc_x;
    }
  }
  PROF_DUMP;
  if constexpr ((OKC_PVW && !OKC_PROF) || PV8) {
  wgmma_wait<0>();
  if constexpr (PV8) {
#pragma unroll
    for (int e = 0; e < 8; ++e) asm volatile("" : "+r"(pa[e >> 2][e & 3]));
  }
  {
    float* od = &o[0][0][0];
#pragma unroll
    for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  }
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];
  l_r[1] = 0.25f * lacc[2];
  if constexpr (F2) {
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + (hk * 128 + (tid & 127)) * 37;
    if (sid == 1) {
      xs[0] = m_r[0]; xs[1] = m_r[1]; xs[2] = l_r[0]; xs[3] = l_r[1];
#pragma unroll
      for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int e = 0; e < 4; ++e) xs[4 + 4 * t + e] = o[0][t][e];
    }
    csync<WS, NT>();
    if (sid == 0) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float m1 = xs[r], mm = fmaxf(m_r[r], m1);
        const float f0 = (m_r[r] == -INFINITY) ? 0.f : ex2(m_r[r] - mm);
        const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
        m_r[r] = mm;
        l_r[r] = l_r[r] * f0 + xs[2 + r] * f1;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          o[0][t][2 * r] = o[0][t][2 * r] * f0 + xs[4 + 4 * t + 2 * r] * f1;
          o[0][t][2 * r + 1] = o[0][t][2 * r + 1] * f0 + xs[4 + 4 * t + 2 * r + 1] * f1;
        }
      }
    }
  }
  }
  DBG(3);
  bool do_red = true;
  if constexpr (Cfg<L, FLAT>::F2) do_red = (warp >> 3) == 0;
  if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL)) {
    cp_wait<0>();
    csync<WS, NT>();
    float* xs = reinterpret_cast<float*>(S.k) + ((warp & 1) * 32 + lane) * 18;
    if (wsub == 1) {
      xs[0] = m_r[0];
      xs[1] = l_r[0];
#pragma unroll
      for (int t = 0; t < 8; ++t) { xs[2 + 2 * t] = o[0][t][0]; xs[3 + 2 * t] = o[0][t][1]; }
    }
    csync<WS, NT>();
    if (wsub == 0) {
      const float m1 = xs[0], l1 = xs[1];
      const float mm = fmaxf(m_r[0], m1);
      const float f0 = (m_r[0] == -INFINITY) ? 0.f : ex2(m_r[0] - mm);
      const float f1 = (m1 == -INFINITY) ? 0.f : ex2(m1 - mm);
      m_r[0] = mm;
      l_r[0] = l_r[0] * f0 + l1 * f1;
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        o[0][t][0] = o[0][t][0] * f0 + xs[2 + 2 * t] * f1;
        o[0][t][1] = o[0][t][1] * f0 + xs[3 + 2 * t] * f1;
      }
    } else {
      do_red = false;
    }
  }
  if (M_CL) {
    cp_wait<0>();
    csync<WS, NT>();
    float* PB = reinterpret_cast<float*>(S.k);
    float* PL = PB + L * 16 * 64;
#pragma unroll 1
    for (int ph = 0; ph < (Cfg<L, FLAT>::S2 ? 2 : 1); ++ph) {
    if (ph == 1) csync<WS, NT>();
    const bool mine = !Cfg<L, FLAT>::S2 || (warp >> 3) == ph;
#pragma unroll
    for (int r = 0; r < NROW; ++r) {
      const int mt = r >> 1, hf = r & 1;
      const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
      const int row = j * 16 + hk * 8 + g;
      const float lsum = quad_sum(l_r[r]);
      if (do_red && mine) {
        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
        if constexpr (L == 1) {
#pragma unroll
          for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int t4 = 0; t4 < 8; t4 += 4)
              *reinterpret_cast<float4*>(&PB[row * 64 + 16 * tig + 8 * c + t4]) =
                  make_float4(o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                              o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
        } else {
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            float2* d2 = reinterpret_cast<float2*>(&PB[row * 64 + 8 * t + 2 * tig]);
            const float2 pv = ph ? *d2 : make_float2(0.f, 0.f);
            *d2 = make_float2(pv.x + o[mt][t][2 * hf] * f, pv.y + o[mt][t][2 * hf + 1] * f);
          }
        }
        if (tig == 0) PL[row] = (ph ? PL[row] : 0.f) + lsum * f;
      }
    }
    }
    csync<WS, NT>();
    cluster_finish<L>(p, S, b, pair, n_keys, chunk);
    return true;
  }
  if (do_red) {
#pragma unroll
  for (int r = 0; r < NROW; ++r) {
    const int mt = r >> 1, hf = r & 1;
    const int j = (L == 1) ? 0 : 2 * (wsub + mt) + hf;
    const int hh = hk * 8 + g;
    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));
    float* acc = p.ws_acc + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
    if constexpr (L == 1 && (RBF_B1 || RBF_Q)) {   // LDD19 LDD_RBF 3 (coop, <1,0,11>) / LDD_RBFQ (nc 1): each lane already holds 8 consecutive dims per c; BF16 rows
      if (RBF_B1 ? M_COOP : p.nc == 1) {
        __nv_bfloat16* accb = reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          float x8[8];
#pragma unroll
          for (int t = 0; t < 8; ++t) x8[t] = o[mt][t][2 * hf + c] * f;
          red_bf8(accb + 16 * tig + 8 * c, x8);
        }
      } else {
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
          for (int t4 = 0; t4 < 8; t4 += 4)
            red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                   o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
      }
    } else if constexpr (L == 1) {
#pragma unroll
      for (int c = 0; c < 2; ++c)
#pragma unroll
        for (int t4 = 0; t4 < 8; t4 += 4)
          red_v4(acc + 16 * tig + 8 * c + t4, o[mt][t4][2 * hf + c] * f, o[mt][t4 + 1][2 * hf + c] * f,
                 o[mt][t4 + 2][2 * hf + c] * f, o[mt][t4 + 3][2 * hf + c] * f);
    } else if ((LDD_RBF && FLAT && L == 8 && M_COOP) || (RBF_B1 && M_COOP)) {   // LDD19 LDD_RBF (coop grids only: the coop finalize reads BF16 rows)
      // as OKC_RED4, then lanes tig and tig ^ 2 trade halves so each lane holds 8 consecutive dims:
      // blocks t and t + 2 at once; hi = tig >> 1 keeps block t + odd (hi 0) or t + 2 + odd (hi 1)
      const bool odd = tig & 1, hi = (tig >> 1) & 1;
      __nv_bfloat16* accb = reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + ((ix_t)((bp * p.nc + copy) * L + j) * 16 + hh) * 64;
#pragma unroll
      for (int t = 0; t < 8; t += 4) {
        float v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
          const int tt = t + 2 * u;
          const float a0 = o[mt][tt][2 * hf] * f, a1 = o[mt][tt][2 * hf + 1] * f;
          const float b0 = o[mt][tt + 1][2 * hf] * f, b1 = o[mt][tt + 1][2 * hf + 1] * f;
          const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
          const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
          v[u][0] = odd ? r0 : a0; v[u][1] = odd ? r1 : a1; v[u][2] = odd ? b0 : r0; v[u][3] = odd ? b1 : r1;
        }
        float x8[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float snd = hi ? v[0][e] : v[1][e];
          const float rcv = __shfl_xor_sync(0xffffffffu, snd, 2);
          x8[e] = hi ? rcv : v[0][e];
          x8[4 + e] = hi ? v[1][e] : rcv;
        }
        red_bf8(accb + 8 * (t + (odd ? 1 : 0) + (hi ? 2 : 0)), x8);
      }
    } else if constexpr (OKC_RED4) {
      const bool odd = tig & 1;
#pragma unroll
      for (int t = 0; t < 8; t += 2) {
        const float a0 = o[mt][t][2 * hf] * f, a1 = o[mt][t][2 * hf + 1] * f;
        const float b0 = o[mt][t + 1][2 * hf] * f, b1 = o[mt][t + 1][2 * hf + 1] * f;
        const float s0 = odd ? a0 : b0, s1 = odd ? a1 : b1;
        const float r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r1 = __shfl_xor_sync(0xffffffffu, s1, 1);
        red_v4(acc + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? r0 : a0, odd ? r1 : a1, odd ? b0 : r0, odd ? b1 : r1);
      }
    } else {
#pragma unroll
      for (int t = 0; t < 8; ++t) red_v2(acc + 8 * t + 2 * tig, o[mt][t][2 * hf] * f, o[mt][t][2 * hf + 1] * f);
    }
    if (tig == 0) red_f32(p.ws_l + ((bp * p.nc + copy) * L + j) * 16 + hh, lsum * f);
  }
  }
  if (OKC_MRGDBG) DBG(7);
  constexpr bool MRG2 = OKC_MERGE2;
  const bool mode1 = !((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) && !(!FLAT && M_COOP);
  if (!((MRG2 && mode1) || (OKC_MERGE3 && !mode1))) gpu_fence();
  csync<WS, NT>();
  if (OKC_MRGDBG) DBG(6);
  if ((FLAT && (L == 8 || OKC_FLATCOOP1C)) && M_COOP) {
    DBG(4);
  } else if (!FLAT && M_COOP) {
    const int split_idx = (int)blockIdx.x;
    int nact = (OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && ((n_keys + 63) >> 6) >= 2 * p.n_splits)
                   ? p.n_splits : fdiv_i(n_keys + chunk - 1, chunk);
    int R = fdiv_i(L * 16 + nact - 1, nact);
    if (OKC_PINDIV) asm volatile("" : "+r"(nact), "+r"(R));
    if (tid == 0 && (OKC_SENSE || OKC_SENSE2)) {
      int* flag = p.ws_cnt + 24576 + bp;
      int sense;
      if (OKC_SENSE2) sense = S.sense;
      else asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(sense) : "l"(flag) : "memory");
      const int mine = end - start;
      int old;
      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
      if (old + mine == n_keys) {
        p.ws_cnt[bp] = 0;
        asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(flag), "r"(sense ^ 1) : "memory");
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(flag) : "memory");
          if (v != sense) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0 && OKC_PARITY) {
      const int ep = S.sense;
      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;
      int old;
      if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(c), "r"(mine) : "memory");
      else old = atomicAdd(c, mine);
      if (old + mine == n_keys) {
        p.ws_cnt[((ep & 1) ? 8192 : 12288) + bp] = 0;
        p.ws_cnt[24576 + bp] = ep + 1;
      } else {
        int v;
        SPIN_T0;
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
          if (v >= n_keys) break;
          SPIN_CHECK;
          __nanosleep(64);
        }
      }
    } else if (tid == 0) {
      atomicAdd(p.ws_cnt + bp, end - start);
      int v;
      SPIN_T0;
      for (int spin = 0;; ++spin) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + bp) : "memory");
        if (v >= n_keys) break;
        SPIN_CHECK;
        if (OKC_SPINDBG && spin > (1 << 20)) {
          printf("coop wait stuck: block (%d,%d,%d) bp %d counter %d n_keys %d start %d end %d\n", blockIdx.x,
                 blockIdx.y, blockIdx.z, bp, v, n_keys, start, end);
          break;
        }
        __nanosleep(64);
      }
    }
    csync<WS, NT>();
    DBG(4);
    const int r_lo = split_idx * R, r_hi = min(L * 16, r_lo + R);
    const int part = tid & 7;
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {
      const int rr = rbase + (tid >> 3);
      const bool act = rr < r_hi;
      const int j = act ? rr >> 4 : 0, hh = act ? rr & 15 : 0;
      const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                             : make_uint4(0, 0, 0, 0);
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      if constexpr (RBF_B1) sum_copies_bf<L>(p, bp, j, hh, part, act, y, lsum);   // LDD19 LDD_RBF: BF16 rows (M_COOP only)
      else
      sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss += __shfl_xor_sync(0xffffffffu, ss, 1);
      ss += __shfl_xor_sync(0xffffffffu, ss, 2);
      ss += __shfl_xor_sync(0xffffffffu, ss, 4);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
    if (!OKC_SENSE && !OKC_SENSE2 && !OKC_PARITY) {
      csync<WS, NT>();
      if (tid == 0) {
        const int old = atomicAdd(p.ws_cnt + 16384 + bp, 1);
        if (old + 1 == nact) {
          p.ws_cnt[bp] = 0;
          p.ws_cnt[16384 + bp] = 0;
        }
      }
    }
    DBG(5);
  } else {
  if (tid == 0) {
    const int mine = end - start;
    int old;
    if constexpr (MRG2) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(p.ws_cnt + bp), "r"(mine) : "memory");
    else old = atomicAdd(p.ws_cnt + bp, mine);
    S.flag = (old + mine == n_keys);
  }
  csync<WS, NT>();
  DBG(4);
  if (S.flag) {
  if constexpr (!MRG2) gpu_fence();
  DBG(14);
  if (p.nc == 1) {
    fin_nc1<L, NT, RBF_Q>(p, b, pair, bp, vscale, 0);   // (LDD19 LDD_RBFQ)
  } else
  {
    const int part = tid & 7;
    for (int rr = tid >> 3; rr < L * 16; rr += NT / 8) {
      const int j = rr >> 4, hh = rr & 15;
      float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float lsum = 0.f;
      const uint4 graw = *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8);
      sum_copies<L>(p, bp, j, hh, part, true, y, lsum);
      const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
      ss = oct_sum(ss);
      const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
      const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
      uint4 ov;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 gt = __bfloat1622float2(g2[e]);
        const float s0 = sigm(gt.x), s1 = sigm(gt.y);
        __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
        ow[e] = *reinterpret_cast<uint32_t*>(&r2);
      }
      *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
    }
  }
  if (tid == 0) p.ws_cnt[bp] = 0;
  DBG(5);
  if (p.dbg && tid == 0) p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 6] = 1;
  }
  }
  if constexpr (CONT) csync<WS, NT>();
  else __syncthreads();                   // shared memory is reused by the next segment
  }
  return true;
}
template <int L, bool WS, int NT>
__device__ __forceinline__ void flat_finalize(const Params& p, const int* sense_s) {
  const int tid = threadIdx.x;
  pdl_wait();
  const int G = gridDim.x * gridDim.y * gridDim.z;
  const int c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  int* cnt = p.ws_cnt + 30000;
  if constexpr (!OKC_MERGE3) gpu_fence();
  csync<WS, NT>();
  constexpr bool PAR = OKC_PARITY;
  if (tid == 0 && PAR) {
    cp_wait<0>();
    const int ep = *reinterpret_cast<const volatile int*>(sense_s);
    int* ac = p.ws_cnt + 30020 + (ep & 1);
    int old;
    if constexpr (OKC_MERGE3) asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(ac) : "memory");
    else old = atomicAdd(ac, 1);
    if (old + 1 == G) {
      p.ws_cnt[30020 + ((ep & 1) ^ 1)] = 0;
      p.ws_cnt[30022] = ep + 1;
      if (OKC_W0ENABLE) {
        cnt[2] = 0;
        cnt[3] = 0;
        for (int k = 0; k < 8; ++k) p.ws_cnt[30030 + k] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(ac) : "memory");
        if (v >= G) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0 && OKC_SENSE) {
    int sense;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(sense) : "l"(cnt + 8) : "memory");
    int old;
    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\n" : "=r"(old) : "l"(cnt) : "memory");
    if (old + 1 == G) {
      cnt[0] = 0;
      asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(cnt + 8), "r"(sense ^ 1) : "memory");
    } else {
      int v;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt + 8) : "memory");
        if (v != sense) break;
        SPIN_CHECK;
        __nanosleep(64);
      }
    }
  } else if (tid == 0) {
    atomicAdd(cnt, 1);
    int v;
    SPIN_T0;
    for (;;) {
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(cnt) : "memory");
      if (v >= G) break;
      SPIN_CHECK;
      __nanosleep(64);
    }
  }
  csync<WS, NT>();
  const int pairs = p.Hkv >> 1;
  const int rows = p.B * pairs * L * 16;
  const int R = fdiv_i(rows + G - 1, G);
  const int r_lo = c * R, r_hi = min(rows, r_lo + R);
  const float vscale = __ldg(p.v_scale);
  const int part = tid & 7;
  for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {
    const int r = rbase + (tid >> 3);
    const bool act = r < r_hi;
    const int bp = act ? r / (L * 16) : 0, rr = act ? r % (L * 16) : 0;
    const int j = rr >> 4, hh = rr & 15;
    const int b = (pairs == 1) ? bp : fdiv_i(bp, pairs), pair = bp - b * pairs;
    const uint4 graw = act ? *reinterpret_cast<const uint4*>(p.qkv + (ix_t)(b * L + j) * p.ld_qkv + (p.Hq + 2 * p.Hkv) * 64 + (16 * pair + hh) * 64 + part * 8)
                           : make_uint4(0, 0, 0, 0);
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float lsum = 0.f;
    if constexpr (LDD_RBF && L == 8) {   // LDD19 LDD_RBF: the merge wrote BF16 rows (128 B per row)
      if (act) {
#pragma unroll 1
        for (int cc = 0; cc < p.nc; ++cc) {
          __nv_bfloat16* rowb = reinterpret_cast<__nv_bfloat16*>(p.ws_acc) + ((ix_t)((bp * p.nc + cc) * L + j) * 16 + hh) * 64 + part * 8;
          const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(rowb));
          const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float2 fv = __bfloat1622float2(h2[e]);
            y[2 * e] += fv.x;
            y[2 * e + 1] += fv.y;
          }
          lsum += __ldcg(p.ws_l + ((bp * p.nc + cc) * L + j) * 16 + hh);
          *reinterpret_cast<uint4*>(rowb) = make_uint4(0u, 0u, 0u, 0u);
          if (part == 0) p.ws_l[((bp * p.nc + cc) * L + j) * 16 + hh] = 0.f;
        }
      }
    } else
    sum_copies<L>(p, bp, j, hh, part, act, y, lsum);
    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = pr8(y[e] * inv); ss += y[e] * y[e]; }
    ss += __shfl_xor_sync(0xffffffffu, ss, 1);
    ss += __shfl_xor_sync(0xffffffffu, ss, 2);
    ss += __shfl_xor_sync(0xffffffffu, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const float s0 = sigm(gt.x), s1 = sigm(gt.y);
      __nv_bfloat162 r2 = __floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1);
      ow[e] = *reinterpret_cast<uint32_t*>(&r2);
    }
    if (act) *reinterpret_cast<uint4*>(p.out + (ix_t)(b * L + j) * (p.Hq * 64) + (16 * pair + hh) * 64 + part * 8) = ov;
  }
  if (!OKC_SENSE && !PAR) {
    csync<WS, NT>();
    if (tid == 0) {
      const int old = atomicAdd(cnt + 1, 1);
      if (old + 1 == G) {
        cnt[0] = 0;
        cnt[1] = 0;
        cnt[2] = 0;
        cnt[3] = 0;
        for (int i = 0; i < G / 2 && i < 1700; ++i) p.ws_cnt[31000 + i] = 0;
      }
    }
  }
}
template <int L, bool FLAT, int MODE = 0>
__global__ void __launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel(const __grid_constant__ Params p) {
  constexpr int NT = Cfg<L, FLAT>::NT;
  constexpr int STAGES = Cfg<L, FLAT>::STAGES;
  constexpr int IDX_SLOTS = 2 * STAGES;
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem<L, FLAT>& S = *reinterpret_cast<Smem<L, FLAT>*>(smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u));
  const int pair = blockIdx.z;
  DBG(0);
  if (p.pf_chunk && threadIdx.x == 0) {
    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
      const unsigned n = min(32768u, p.pf_chunk - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(n) : "memory");
    }
  }
  if (!FLAT) {
    const int split = blockIdx.x, b = blockIdx.y;
    const int seq_ld = __ldg(p.seq_lens + b);
    constexpr bool SPEC = (MODE & 8) != 0;
    int seq = SPEC ? p.max_pages - L : seq_ld;
    for (;;) {
    const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
    const int n_keys = seq + L - lo;
    int chunk = fdiv_i(n_keys + p.n_splits - 1, p.n_splits);
    chunk = (chunk + 15) & ~15;
    if ((MODE & 3) == 0 ? p.cs <= 1 : (MODE & 3) != 2) {
      const int ntl = ((n_keys + 63) >> 6) + p.n_splits - 1;
      if constexpr (OKC_MAGICDIV) chunk = (p.ns_magic ? (int)__umulhi((unsigned)ntl, p.ns_magic) : ntl) << 6;
      else chunk = fdiv_i(ntl, p.n_splits) << 6;
    }
    int start = lo + split * chunk;
    int end = min(seq + L, start + chunk);
    const int unev_tiles = (n_keys + 63) >> 6;
    const bool unev = OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && unev_tiles >= 2 * p.n_splits;
    if (unev) {
      const int H = p.unev;
      const long long wn = (long long)p.w0 * min(p.n_splits, H) + 1024LL * max(0, p.n_splits - H);
      const long long w_s = (long long)p.w0 * min(split, H) + 1024LL * max(0, split - H);
      const long long w_e = (long long)p.w0 * min(split + 1, H) + 1024LL * max(0, split + 1 - H);
      start = lo + (int)(unev_tiles * w_s / wn) * 64;
      end = min(seq + L, lo + (int)(unev_tiles * w_e / wn) * 64);
    }
    if (start >= end) {
      if (SPEC && seq != seq_ld) { seq = seq_ld; continue; }
      if ((MODE & 3) == 0 ? p.cs > 1 : (MODE & 3) == 2) {
        pdl_wait();
        cluster_stage_gate<L>(p, S, b, pair);
        if (threadIdx.x == 0) cp_async4(smem_u32(&S.scl[1]), p.v_scale);
        cp_commit();
        float* PB = reinterpret_cast<float*>(S.k);
        float* PL = PB + L * 16 * 64;
        for (int e = threadIdx.x; e < L * 16 * 64; e += NT) PB[e] = 0.f;
        for (int e = threadIdx.x; e < L * 16; e += NT) PL[e] = 0.f;
        cp_wait<0>();
        __syncthreads();
        cluster_finish<L>(p, S, b, pair, n_keys, chunk);
      }
      return;
    }
    if (segment<L, FLAT, 0, MODE>(p, S, b, pair, split == 0, start, end, seq, n_keys, split - fdiv_i(split, p.nc) * p.nc,
                                  chunk, 0, SPEC ? seq_ld : seq))
      return;
    seq = seq_ld;
    }
  }
  if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C) && OKC_PARITY) {
    if (p.coop && threadIdx.x == 0) {
      cp_async4(smem_u32(&S.sense), p.ws_cnt + 30022);
      cp_commit();
    }
  }
  const bool w0on = OKC_W0ENABLE && p.coop && p.resident && p.w0 != 1024 && gridDim.y * gridDim.z == 1 && (gridDim.x & 1) == 0;
  const bool w0g = OKC_W0L1 && OKC_W0ENABLE && !p.coop && p.resident && p.w0 != 1024;
  int w0g_rank = 2, w0g_par = 0, w0g_gen = 0;
  unsigned w0_smid = 0;
  int w0_rank = 0;
  if (OKC_W0EARLY && w0on && threadIdx.x == 32) {
    asm volatile("mov.u32 %0, %%smid;" : "=r"(w0_smid));
    const int Gx = gridDim.x;
    w0_rank = ((int)w0_smid < Gx / 2) ? atomicAdd(p.ws_cnt + 31000 + w0_smid, 1) : 2;
    if (w0_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 30003, 1);
    if (OKC_W0BID && w0_rank < 2) p.ws_cnt[31200 + 2 * w0_smid + w0_rank] = (int)blockIdx.x + 1;
    gpu_fence();
    atomicAdd(p.ws_cnt + (OKC_W0SPREAD ? 30030 + (blockIdx.x & 7) : 30002), 1);
  }
  if (w0g && threadIdx.x == 32) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const int Gx = gridDim.x;
    const unsigned og = (unsigned)atomicAdd(p.ws_cnt + 32000, 1);
    w0g_gen = (int)(og / (unsigned)Gx);
    w0g_par = w0g_gen & 1;
    if (og % (unsigned)Gx == (unsigned)Gx - 1) {
      for (int i = 0; i < Gx / 2; ++i) p.ws_cnt[32010 + (w0g_par ^ 1) * 136 + i] = 0;
      p.ws_cnt[32002 + (w0g_par ^ 1)] = 0;
    }
    w0g_rank = ((int)smid < Gx / 2) ? atomicAdd(p.ws_cnt + 32010 + w0g_par * 136 + smid, 1) : 2;
    if (w0g_rank >= 2 || OKC_W0FORCEBAD) atomicExch(p.ws_cnt + 32002 + w0g_par, 1);
    if (w0g_rank < 2) p.ws_cnt[32300 + 2 * smid + w0g_rank] = (int)blockIdx.x + 1;
    __threadfence();
    atomicAdd(p.ws_cnt + 32001, 1);
    w0g_rank = (w0g_rank & 3) | ((int)smid << 2);
  }
  constexpr bool fpart = OKC_FPART && L == 1 && !OKC_W0ENABLE && !OKC_W0L1 && OKC_WSPLIT == 500 && !OKC_BALPART &&
                         !OKC_SWAPHALF && !OKC_DYN;
  if constexpr (OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR) {
    if (threadIdx.x == 32) {
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], Cfg<L, FLAT>::F2 ? 8 : (OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32));
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x < 32) {
    int run = 0;
    int incl_l = 0, n_l = 0, seq_l = 0;
    for (int b0 = 0; b0 < p.B; b0 += 32) {
      const int b = b0 + (int)threadIdx.x;
      int n = 0;
      if (b < p.B) {
        const int seq = __ldg(p.seq_lens + b);
        const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
        n = seq + L - lo;
        seq_l = seq;
      }
      int incl = n;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int t = __shfl_up_sync(0xffffffffu, incl, o);
        if ((int)threadIdx.x >= o) incl += t;
      }
      if (b < p.B) S.cum[b + 1] = run + incl;
      run += __shfl_sync(0xffffffffu, incl, 31);
      incl_l = incl;
      n_l = n;
    }
    if (threadIdx.x == 0) S.cum[0] = 0;
    if (fpart && p.B > 32) {
      __syncwarp();
      if (threadIdx.x == 0) {
        constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
        const int64_t Tg = (int64_t)S.cum[p.B] + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
        int csg = (int)((int64_t)blockIdx.x * Tg / gridDim.x), ceg = (int)((int64_t)(blockIdx.x + 1) * Tg / gridDim.x);
        if (SEGKF > 0) {
          auto vmapg = [&](int v) -> int {
            int lo_b = 0, hi_b = p.B - 1;
            while (lo_b < hi_b) {
              const int mid = (lo_b + hi_b + 1) >> 1;
              if (S.cum[mid] + mid * SEGKF <= v) lo_b = mid; else hi_b = mid - 1;
            }
            return min(v - lo_b * SEGKF, S.cum[lo_b + 1]);
          };
          csg = vmapg(csg);
          ceg = vmapg(ceg);
        }
        int b = 0, n = 0;
        while (S.cum[b + 1] <= csg) ++b;
        for (; b < p.B && S.cum[b] < ceg && n < MAXSEG; ++b) {
          S.seg[n][0] = b;
          S.seg[n][1] = max(csg, S.cum[b]) - S.cum[b];
          S.seg[n][2] = min(ceg, S.cum[b + 1]) - S.cum[b];
          ++n;
        }
        S.nseg = n;
        S.fcs = csg;
        S.fce = ceg;
      }
    } else if (fpart) {
      const unsigned FULL = 0xffffffffu;
      const int lane_f = (int)threadIdx.x;
      constexpr int SEGKF = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
      const int excl = incl_l - n_l;
      const int64_t Tf = (int64_t)__shfl_sync(FULL, incl_l, 31) + (SEGKF > 0 ? (int64_t)(p.B - 1) * SEGKF : 0);
      const int cf = blockIdx.x, Gf = gridDim.x;
      int csf = (int)((int64_t)cf * Tf / Gf), cef = (int)((int64_t)(cf + 1) * Tf / Gf);
      if (SEGKF > 0) {
        const int vstart = excl + lane_f * SEGKF;
        const unsigned ms = __ballot_sync(FULL, lane_f < p.B && vstart <= csf);
        const unsigned me = __ballot_sync(FULL, lane_f < p.B && vstart <= cef);
        const int bs = 31 - __clz(ms), be = 31 - __clz(me);
        const int is = __shfl_sync(FULL, incl_l, bs), ie = __shfl_sync(FULL, incl_l, be);
        csf = min(csf - bs * SEGKF, is);
        cef = min(cef - be * SEGKF, ie);
      }
      const bool in = lane_f < p.B && incl_l > csf && excl < cef;
      const unsigned mi = __ballot_sync(FULL, in);
      const int rank = __popc(mi & ((1u << lane_f) - 1u));
      if (in && rank < MAXSEG) {
        S.seg[rank][0] = lane_f;
        S.seg[rank][1] = max(csf, excl) - excl;
        S.seg[rank][2] = min(cef, incl_l) - excl;
      }
      if (lane_f == 0) {
        S.nseg = min(__popc(mi), MAXSEG);
        S.fcs = csf;
        S.fce = cef;
      }
      if (OKC_PTPF && mi != 0u && csf < cef) {
        const int b0 = __ffs(mi) - 1;
        const int seqb = __shfl_sync(FULL, seq_l, b0), exb = __shfl_sync(FULL, excl, b0), inb = __shfl_sync(FULL, incl_l, b0);
        const int lob = p.window ? max(0, seqb + 1 - p.window) : 0;
        const int s0 = max(csf, exb) - exb + lob, s1 = min(cef, inb) - exb + lob;
        const int cend = min(min(s1, seqb), s0 + Cfg<L, FLAT>::STAGES * TILE);
        if (cend > s0) {
          const long long e0 = (long long)b0 * p.max_pages + (s0 >> p.lp);
          const long long e1 = (long long)b0 * p.max_pages + ((cend - 1) >> p.lp);
          const long long l0 = e0 >> 5, l1 = e1 >> 5;
          if (l0 + lane_f <= l1) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.page_table + ((l0 + lane_f) << 5)));
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) FPRO_STAMP(40);
  const int64_t T0 = S.cum[p.B];
  constexpr int SEGK = (L == 8) ? OKC_SEGK8 : OKC_SEGK1;
  const int64_t T = T0 + (SEGK > 0 ? (int64_t)(p.B - 1) * SEGK : 0);
  const int c = blockIdx.x, G = gridDim.x;
  const int cpart = (OKC_SWAPHALF && (G & 1) == 0) ? (c + (G >> 1)) % G : c;
  int cs, ce;
  if constexpr (fpart) {
    cs = S.fcs;
    ce = S.fce;
  } else {
    cs = (int)((int64_t)cpart * T / G);
    ce = (int)((int64_t)(cpart + 1) * T / G);
  }
  if (OKC_WSPLIT != 500 && (G & 1) == 0 && gridDim.y * gridDim.z == 1) {
    const int64_t Th = T * OKC_WSPLIT / 1000, H = G >> 1;
    if (c < H) { cs = (int)((int64_t)c * Th / H); ce = (int)((int64_t)(c + 1) * Th / H); }
    else { cs = (int)(Th + (int64_t)(c - H) * (T - Th) / H); ce = (int)(Th + (int64_t)(c + 1 - H) * (T - Th) / H); }
  }
  int vc = c;
  if (w0on) {
    if (threadIdx.x == (OKC_W0EARLY ? 32 : 0)) {
      unsigned smid = w0_smid;
      int* reg = p.ws_cnt + 31000;
      int* arrive = p.ws_cnt + 30002;
      int* bad = p.ws_cnt + 30003;
      int rank = w0_rank;
      if (!OKC_W0EARLY) {
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        rank = ((int)smid < G / 2) ? atomicAdd(reg + smid, 1) : 2;
        if (rank >= 2 || OKC_W0FORCEBAD) atomicExch(bad, 1);
        if (OKC_W0BID && rank < 2) p.ws_cnt[31200 + 2 * smid + rank] = c + 1;
        gpu_fence();
        atomicAdd(OKC_W0SPREAD ? p.ws_cnt + 30030 + (c & 7) : arrive, 1);
      }
      int v;
      SPIN_T0;
      if (OKC_W0SPREAD) {
        for (int k = 0; k < 8; ++k) {
          const int need = (G - k + 7) >> 3;
          for (;;) {
            asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 30030 + k) : "memory");
            if (v >= need) break;
            SPIN_CHECK;
          }
        }
      } else {
        for (;;) {
          asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(arrive) : "memory");
          if (v >= G) break;
          SPIN_CHECK;
        }
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(bad) : "memory");
      if (OKC_W0BID && !v) {
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(other) : "l"(p.ws_cnt + 31200 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (w0g) {
    if (threadIdx.x == 32) {
      const unsigned smid = (unsigned)w0g_rank >> 2;
      int rank = w0g_rank & 3;
      int v;
      const unsigned need = ((unsigned)w0g_gen + 1u) * (unsigned)G;
      SPIN_T0;
      for (;;) {
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 32001) : "memory");
        if ((int)((unsigned)v - need) >= 0) break;
        SPIN_CHECK;
      }
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p.ws_cnt + 32002 + w0g_par) : "memory");
      if (!v) {
        int other;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(other) : "l"(p.ws_cnt + 32300 + 2 * smid + (rank ^ 1)) : "memory");
        rank = (other > 0 && other - 1 < c) ? 1 : 0;
      }
      S.vidx = v ? -1 : (int)(smid * 2 + rank);
    }
    __syncthreads();
    if (S.vidx >= 0) {
      vc = S.vidx;
      const int64_t sp = vc >> 1, r = vc & 1;
      cs = (int)(T * (2048 * sp + r * p.w0) / (1024LL * G));
      ce = (int)(T * (2048 * sp + (r ? 2048 : p.w0)) / (1024LL * G));
    }
  }
  if (SEGK > 0 && !fpart) {
    auto vmap = [&](int v) -> int {
      int lo_b = 0, hi_b = p.B - 1;
      while (lo_b < hi_b) {
        const int mid = (lo_b + hi_b + 1) >> 1;
        if (S.cum[mid] + mid * SEGK <= v) lo_b = mid; else hi_b = mid - 1;
      }
      return min(v - lo_b * SEGK, S.cum[lo_b + 1]);
    };
    cs = vmap(cs);
    ce = vmap(ce);
  }
  if (!(OKC_DYN && Cfg<L, FLAT>::WS) && cs >= ce) {
    if constexpr (L == 8 || OKC_FLATCOOP1C) {
      if (p.coop) {
        if constexpr (Cfg<L, FLAT>::WS) {
          if (threadIdx.x < Cfg<L, FLAT>::NT) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
        } else {
          flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
        }
      }
    }
    return;
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_TMAPG && OKC_PGCONT && (L == 1 || OKC_TMAPG8)) {
    if (threadIdx.x == Cfg<L, FLAT>::NT) {
      for (int st = 0; st < 4; ++st) mbar_init_n(&S.pbar[st], 1);
      S.pgc = 0;
      mbar_fence_init();
    }
    __syncthreads();
  }
  if constexpr (Cfg<L, FLAT>::WS && OKC_CONT) {
    if (threadIdx.x == 0) {
      for (int st = 0; st < Cfg<L, FLAT>::STAGES; ++st) {
        mbar_init_n(&S.full[st], Cfg<L, FLAT>::NP);
        mbar_init_n(&S.empty[st], OKC_XP_ARRALL ? Cfg<L, FLAT>::NT : Cfg<L, FLAT>::NT / 32);
      }
      mbar_fence_init();
    }
  }
  if (threadIdx.x == 0 && !fpart) {
    int b = 0, n = 0;
    while (S.cum[b + 1] <= cs) ++b;
    for (; b < p.B && S.cum[b] < ce && n < MAXSEG; ++b) {
      S.seg[n][0] = b;
      S.seg[n][1] = max(cs, S.cum[b]) - S.cum[b];
      S.seg[n][2] = min(ce, S.cum[b + 1]) - S.cum[b];
      ++n;
    }
    S.nseg = n;
  }
  if (!fpart) __syncthreads();   // (OKC_FPART: warp 0 wrote the list before the barrier above)
  if (threadIdx.x == 0) FPRO_STAMP(41);
  const int nseg = S.nseg;
  auto run_segments = [&](auto role) {
    constexpr int R = decltype(role)::value;
    int tb = 0;
#pragma unroll 1
    for (int k = 0; k < nseg; ++k) {
      const int b = S.seg[k][0];
      const int seq = __ldg(p.seq_lens + b);
      const int lo = p.window ? max(0, seq + 1 - p.window) : 0;
      const int n_keys = seq + L - lo;
      const int s0 = S.seg[k][1] + lo, s1 = S.seg[k][2] + lo;
      const bool lead = OKC_SEGDBG && p.dbg && threadIdx.x == 0 && k < 2;
      const int dbase = (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + 16 + 4 * k;
      if (lead) { p.dbg[dbase] = gtime(); p.dbg[dbase + 2] = s1 - s0; p.dbg[dbase + 3] = b; }
      if (s0 < s1) {
        segment<L, FLAT, R>(p, S, b, pair, s0 == lo, s0, s1, seq, n_keys, vc - fdiv_i(vc, p.nc) * p.nc, 0,
                            OKC_CONT ? tb : 0, 0,
                            OKC_EARLYBAR && L == 1 && Cfg<L, FLAT>::WS && !OKC_CONT && !OKC_PTR && k == 0);
        tb += (s1 - s0 + TILE - 1) / TILE;
      }
      if (lead) p.dbg[dbase + 1] = gtime();
    }
    if constexpr (R == 1 && OKC_CONT) cp_wait<0>();
  };
  if constexpr (Cfg<L, FLAT>::WS) {
    if (threadIdx.x >= Cfg<L, FLAT>::NT) {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());
      if constexpr (L == 8 || OKC_FLATCOOP1C) {
        if (p.coop) flat_finalize<L, true, Cfg<L, FLAT>::NT>(p, &S.sense);
      }
    }
  } else {
    run_segments(std::integral_constant<int, 0>());
    if constexpr (FLAT && (L == 8 || OKC_FLATCOOP1C)) {
      if (p.coop) flat_finalize<L, false, Cfg<L, FLAT>::NTX>(p, &S.sense);
    }
  }
}
}
template <int L>
void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool clusters, bool pdl) {
  Params p = p0;
  p.pdlf = pdl ? 1 : 0;
  static bool init = false;
  const int sm_f = (int)sizeof(Smem<L, true>) + 1024, sm_n = (int)sizeof(Smem<L, false>) + 1024;
  const int sm = p.flat ? sm_f : sm_n;
  static int occ_c = 0, occ_f = 0, nsm = 0;
  static int occ_c1 = 0, occ_sp = 0, occ_sp1 = 0;
  typedef void (*KernFn)(const Params);
  static const KernFn kn[4] = {attn_kernel<L, false, OKC_SM90 ? 2 : 0>, attn_kernel<L, false, 1>,
                               attn_kernel<L, false, 2>, attn_kernel<L, false, 3>};
  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};
  static const KernFn ksp[4] = {nullptr, attn_kernel<L, false, 9>, nullptr, attn_kernel<L, false, 11>};
  static const KernFn ksp1[4] = {nullptr, attn_kernel<L, false, 13>, nullptr, attn_kernel<L, false, 15>};
  if (!init) {
    for (int m = 1; m < 4; ++m)
      cudaFuncSetAttribute(kn[m], cudaFuncAttributeMaxDynamicSharedMemorySize, (m == 2 && OKC_CLSM > sm_n) ? OKC_CLSM : sm_n);
    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kn1[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    for (int m = 1; m < 4; m += 2) {
      cudaFuncSetAttribute(ksp[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }
    cudaFuncSetAttribute(attn_kernel<L, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_f);
    if (clusters) cudaFuncSetAttribute(kn[2], cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c, kn[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp, ksp[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_f, attn_kernel<L, true>, Cfg<L, true>::NTX, sm_f);
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    init = true;
  }
  Params q = p;
  if (q.coop && (long long)grid.x * grid.y * grid.z > (long long)(p.flat ? occ_f : occ_c) * nsm) q.coop = 0;
  q.resident = (p.flat && grid.y * grid.z == 1 && (int)grid.x == 2 * nsm && occ_f >= 2) ? 1 : 0;
  q.unev = (OKC_UNEV && !p.flat && q.coop && p.w0 != 1024 && grid.y * grid.z == 1 && (int)grid.x > nsm &&
            (int)grid.x <= 2 * nsm) ? nsm : 0;
  const bool cl = cs > 1 && clusters;
  KernFn kern = p.flat ? (KernFn)attn_kernel<L, true> : kn[cl ? 2 : (q.coop ? 3 : 1)];
  if (!p.flat && !cl && !(OKC_UNEV && q.unev)) {
    const long long kmax = p.window ? std::min<long long>((long long)p.window + L - 1, (long long)p.max_pages * p.P)
                                    : (long long)p.max_pages * p.P;
    if (OKC_TCHUNK && (long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }
  if (!p.flat && !cl && !(OKC_UNEV && q.unev) && p.B == 1 && p.P == 1 && p.max_pages >= L && (int)grid.y == 1) {
    const bool one = kern != kn[q.coop ? 3 : 1];
    kern = one ? ksp1[q.coop ? 3 : 1] : ksp[q.coop ? 3 : 1];
  }
  if (OKC_COOPGUARD && q.coop && !p.flat && !cl) {
    int occ_k = occ_c;
    KernFn plain = kn[1];
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
    if (kern == ksp[3]) { occ_k = occ_sp; plain = ksp[1]; }
    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }
    if ((long long)grid.x * grid.y * grid.z > (long long)occ_k * nsm) {
      q.coop = 0;
      kern = plain;
    }
  }
  if (!p.flat && cs > 1 && !clusters) kern = kn[0];
  if (cl || pdl) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, 1, 1);
    cfg.dynamicSmemBytes = (cl && OKC_CLSM > sm) ? OKC_CLSM : sm;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    int na = 0;
    if (cl) {
      attr[na].id = cudaLaunchAttributeClusterDimension;
      attr[na].val.clusterDim.x = cs;
      attr[na].val.clusterDim.y = 1;
      attr[na].val.clusterDim.z = 1;
      ++na;
    }
    if (pdl) {
      attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
    cudaLaunchKernelEx(&cfg, kern, q);
  } else {
    kern<<<grid, p.flat ? Cfg<L, true>::NTX : Cfg<L, false>::NTX, sm, stream>>>(q);
  }
}
typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static CUtensorMap row_map(void* base, long long rows, long long row_bytes) {
  static EncodeTiledFn encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<EncodeTiledFn>(fn);
  }
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)row_bytes, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
  cuuint32_t box[2] = {128, 1};
  cuuint32_t es[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes) {
  Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.seq_lens = seq_lens.data_ptr<int>();
  p.cache_loc = cache_loc.data_ptr<int64_t>();
  p.k_cache = reinterpret_cast<uint8_t*>(k_cache.data_ptr());
  p.v_cache = reinterpret_cast<uint8_t*>(v_cache.data_ptr());
  p.page_table = page_table.data_ptr<int>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.inv_freq = inv_freq.data_ptr<float>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws_acc = ws_acc.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.ws_cnt = ws_cnt.data_ptr<int>();
  p.B = (int)B; p.Hq = (int)Hq; p.Hkv = (int)Hkv; p.P = (int)P;
  TORCH_CHECK(P > 0 && (P & (P - 1)) == 0, "page size must be a power of two");
  TORCH_CHECK(k_cache.numel() / (Hkv * 64) < (1LL << 31), "cache rows must fit in 31 bits");
  p.lp = 0;
  while ((1 << p.lp) < P) ++p.lp;
  p.max_pages = (int)page_table.size(1);
  p.window = (int)window; p.rope = (int)rope; p.n_splits = (int)n_splits; p.flat = (int)flat; p.nc = (int)nc;
  p.ns_magic = (n_splits >= 2 && n_splits <= 1024) ? (unsigned)(((1ull << 32) + (uint64_t)n_splits - 1) / (uint64_t)n_splits) : 0u;
  TORCH_CHECK(n_splits <= 1024 || !OKC_MAGICDIV, "OKC_MAGICDIV: n_splits <= 1024");
  p.dbg = dbg.has_value() ? reinterpret_cast<long long*>(dbg->data_ptr<int64_t>()) : nullptr;
  p.cs = (int)cs; p.ncl = (int)ncl;
  p.scratch = scratch.data_ptr<float>();
  p.bar = bar.data_ptr<int>();
  p.ld_qkv = (int)qkv.size(-1);
  p.row_bytes = (int)(Hkv * 64);
  TORCH_CHECK((p.row_bytes & (p.row_bytes - 1)) == 0, "Hkv * 64 must be a power of two");
  p.lrb = 0;
  while ((1 << p.lrb) < p.row_bytes) ++p.lrb;
  p.off32 = (k_cache.numel() < (1LL << 32) && v_cache.numel() < (1LL << 32)) ? 1 : 0;
  TORCH_CHECK(!OKC_OFF32 || ws_acc.numel() < (1LL << 31), "OKC_OFF32: workspace too large");
  TORCH_CHECK(B * (Hkv / 2) <= 4096, "at most 4096 (request, kv pair) slots in the counter workspace");
  TORCH_CHECK(!OKC_OFF32 || qkv.numel() < (1LL << 31), "OKC_OFF32: qkv too large");
  TORCH_CHECK(!OKC_OFF32 || (out.numel() < (1LL << 31) && page_table.numel() < (1LL << 31)), "OKC_OFF32: out / page table too large");
  p.scale_log2 = (float)scale_log2;
  dim3 grid((unsigned)n_splits, flat ? 1u : (unsigned)B, (unsigned)(Hkv / 2));
  p.pf = reinterpret_cast<const uint8_t*>(pf_ptr);
  p.pf_chunk = pf_ptr ? (unsigned)((pf_bytes / ((int64_t)grid.x * grid.y * grid.z)) & ~15ll) : 0u;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  static int major = -1;
  if (major < 0) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  }
  p.tma = (OKC_TMA && (tma & 1) && major >= 9) ? 1 : 0;
  p.coop = OKC_COOP ? (int)((tma >> 1) & 1) : 0;
  p.w0 = (int)((tma >> 2) & 4095);
  if (p.w0 == 0) p.w0 = 1024;
  if (p.tma) {
    const long long rows = k_cache.numel() / (Hkv * 64);
    p.tk = row_map(k_cache.data_ptr(), rows, Hkv * 64);
    p.tv = row_map(v_cache.data_ptr(), rows, Hkv * 64);
  } else {
    memset(&p.tk, 0, sizeof(p.tk));
    memset(&p.tv, 0, sizeof(p.tv));
  }
  memset(&p.tpt, 0, sizeof(p.tpt));
  const bool pdl = ((tma >> 16) & 1) && major >= 9;
  if (L == 1) launch<1>(p, grid, stream, (int)cs, major >= 9, pdl);
  else launch<8>(p, grid, stream, (int)cs, major >= 9, pdl);
}
"""
kl_ATT7_CPP_SRC = r"""void run(torch::Tensor qkv, torch::Tensor seq_lens, torch::Tensor cache_loc, torch::Tensor k_cache,
         torch::Tensor v_cache, torch::Tensor page_table, torch::Tensor k_scale, torch::Tensor v_scale,
         torch::Tensor inv_freq, torch::Tensor out, torch::Tensor ws_acc, torch::Tensor ws_l,
         torch::Tensor ws_cnt, int64_t B, int64_t L, int64_t Hq, int64_t Hkv, int64_t P, int64_t window,
         int64_t rope, int64_t n_splits, int64_t flat, int64_t nc, double scale_log2,
         std::optional<torch::Tensor> dbg, int64_t cs, int64_t ncl, torch::Tensor scratch, torch::Tensor bar,
         int64_t tma, int64_t pf_ptr, int64_t pf_bytes);
"""
kl_bfp_CUDA_SRC = r"""#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
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
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xD8;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
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
@@MBAR@@
__device__ __noinline__ void okc_spin_trap() { asm volatile("trap;"); }
__device__ __forceinline__ void okc_spin_ge(const int* p, int target, int ns) {
  for (unsigned it = 0;; ++it) {
    if (*((volatile const int*)p) - target >= 0) return;
    if (ns > 0) __nanosleep(ns);
    if (it >= (1u << 24)) okc_spin_trap();
  }
}
@@BODY@@
"""
kl_bfp_MBAR_HELPERS = r"""__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
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
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p)); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(l2_pol_ef()) : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}
"""
kl_bfp_WG_HELPERS = r"""__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {
  uint64_t d = 0;
  d |= (uint64_t)((saddr & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
"""
kl_bfp_TMA_SRC = r"""template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
__global__ void __launch_bounds__((RG * WS + 1) * 32, 1)
skinny_tma_kernel(const uint4* __restrict__ Wp, const __nv_bfloat16* __restrict__ x,
                  OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
                  const int2* __restrict__ ptab, int pmax, uint32_t base4,
                  int M, int N, int Kp, int Kx, int splits, int nrg, int dist, int dg) {
  constexpr int NW = RG * WS;
  constexpr int NC = NW * 32;
  constexpr int TOK = MT * 8;
  constexpr int TOKP = TOK + 1;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  constexpr int SLOT = NW * CHUNK;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  constexpr int XP = 4;
  const uint32_t sb = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t full0 = sb, empty0 = sb + NSTAGE * 8, xbar0 = sb + 2 * NSTAGE * 8;
  const uint32_t ring = sb + 256;
  const int Kc = Kp / splits;
  const int XS = Kc + 8;
  const size_t x_off = WGM ? (size_t)((((sb + 256u + (uint32_t)(NSTAGE * SLOT)) + 1023u) & ~1023u) - sb)
                           : (size_t)256 + NSTAGE * SLOT;
  const size_t x_bytes = WGM ? (size_t)Kc * TOK * 2 : (((size_t)M * XS * 2 + 16 + 15) & ~(size_t)15);
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw + x_off);
  float* red = reinterpret_cast<float*>(smem_raw + x_off + x_bytes);
  int2* ptab_s = reinterpret_cast<int2*>(red + NW * 16 * TOKP + ((NW * 16 * TOKP) & 1));
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int rb = blockIdx.x;
  const int sp = blockIdx.y;
  const int k0 = sp * Kc;
  const int S = Kp >> 4;
  const int NS = Kc / (16 * WS);
  const int nst = NS / KS;
  const char* wsrc = reinterpret_cast<const char*>(Wp) +
      (WS == 1 ? ((size_t)blockIdx.x * (S / KS) + (size_t)blockIdx.y * nst) : (size_t)(blockIdx.y * gridDim.x + blockIdx.x) * nst) * SLOT;
  if (threadIdx.x == 0) {
    for (int i = 0; i < NSTAGE; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NW);
    }
    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int st = 0; st < NSTAGE && st < nst; ++st) {
      mbar_expect_tx(full0 + 8 * st, SLOT);
      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }
  }
  if (!WGM && threadIdx.x < 4) reinterpret_cast<uint32_t*>(xs + M * XS)[threadIdx.x] = 0u;
  __syncthreads();
  if (warp == NW) {
    const int st0 = NSTAGE < nst ? NSTAGE : nst;
    for (int st = st0; st < nst; ++st) {
      const int slot = st % NSTAGE;
      if (st >= NSTAGE) mbar_wait(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
      const uint32_t dst = ring + slot * SLOT;
      const char* src = wsrc + (size_t)st * SLOT;
      if (lane == 0) {
        mbar_expect_tx(full0 + 8 * slot, SLOT);
        bulk_g2s(dst, src, SLOT, full0 + 8 * slot);
      }
    }
    return;
  }
  const int rg = rb * RG + warp % RG;
  const int wsub = warp / RG;
  if (threadIdx.x == 0 && splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  if (COMP && dg < 2) {
    const int cta = blockIdx.y * gridDim.x + blockIdx.x;
    const char* src = reinterpret_cast<const char*>(ptab + (size_t)cta * pmax);
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(ptab_s);
    for (int i = threadIdx.x; i < pmax / 2; i += NC)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + i * 16), "l"(src + i * 16) : "memory");
  }
  if (dg < 2) {
    const int kx0 = k0 % Kx;
    const int pch = Kc / 8 / XP;
    const uint32_t xs_s = (uint32_t)__cvta_generic_to_shared(xs);
    for (int p = 0; p < XP; ++p) {
      if constexpr (WGM) {
        for (int idx = threadIdx.x; idx < TOK * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          const uint32_t dst = xs_s + (uint32_t)((((c >> 3) * TOK + t) << 7) + ((((c & 7) ^ (t & 7))) << 4));
          const __nv_bfloat16* src = x + (size_t)(t < M ? t : 0) * Kx + kx0 + c * 8;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < M ? 16 : 0) : "memory");
        }
      } else {
        for (int idx = threadIdx.x; idx < M * pch; idx += NC) {
          int t = idx / pch, c = p * pch + (idx - t * pch);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::
                       "r"(xs_s + (uint32_t)((t * XS + c * 8) * 2)), "l"(x + (size_t)t * Kx + kx0 + c * 8) : "memory");
        }
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(xbar0 + 8 * p) : "memory");
    }
  }
  int xp_ready = -1;
  const int kpart = Kc / XP;
  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  float wacc[WGM ? TOK / 2 : 1];
#pragma unroll
  for (int q = 0; q < (WGM ? TOK / 2 : 1); ++q) wacc[q] = 0.f;
  uint32_t afrag[2][4];
  const uint32_t xs_base = (uint32_t)__cvta_generic_to_shared(xs);
  const int lrow = lane & 7, lmat = lane >> 3;
  const int kw0 = wsub * NS * 16;
  const uint32_t zero_row = xs_base + (uint32_t)(M * XS * 2);
  uint32_t xs_row[MT];
  uint32_t xs_adv[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int t = j * 8 + lrow;
    xs_row[j] = t < M ? xs_base + (uint32_t)((t * XS + kw0 + lmat * 8) * 2) : zero_row;
    xs_adv[j] = t < M ? 32u : 0u;
  }
  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    {
      const int need = (kw0 + (st + 1) * KS * 16 - 1) / kpart;
      if (need > xp_ready && dg < 2) {
        mbar_wait(xbar0 + 8 * need, 0);
        xp_ready = need;
        if constexpr (WGM) fence_async_smem();
      }
    }
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    constexpr int NL = CHUNK / 512;
    uint4 wr[NL];
    {
      const uint32_t src = ring + slot * SLOT + warp * CHUNK + lane * 16;
#pragma unroll
      for (int q = 0; q < NL; ++q)
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(wr[q].x), "=r"(wr[q].y), "=r"(wr[q].z), "=r"(wr[q].w) : "r"(src + q * 512));
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    if (dg == 3) {
      if (wr[0].x == 0x9e3779b9u && wr[NL - 1].w == 0x7f4a7c15u) acc[0][0] += 1.f;
      continue;
    }
    if constexpr (!COMP && HL) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 h0 = wr[2 * j2], l0 = wr[2 * j2 + 1], h1 = wr[2 * j2 + 2], l1 = wr[2 * j2 + 3];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], h0.x, h0.y, h0.z, h0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], l0.x, l0.y, l0.z, l0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], h1.x, h1.y, h1.z, h1.w, b[j][2], b[j][3]);
          mma_bf16(acc[j], l1.x, l1.y, l1.z, l1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (!COMP) {
#pragma unroll
      for (int j2 = 0; j2 < KS; j2 += 2) {
        const uint4 a0 = wr[j2], a1 = wr[j2 + 1];
        const int s = st * KS + j2;
        uint32_t b[MT][4];
#pragma unroll
        for (int j = 0; j < MT; ++j)
          ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)s, b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          mma_bf16(acc[j], a0.x, a0.y, a0.z, a0.w, b[j][0], b[j][1]);
          mma_bf16(acc[j], a1.x, a1.y, a1.z, a1.w, b[j][2], b[j][3]);
        }
      }
    } else if constexpr (WGM) {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;
        const uint32_t xt = xs_base + (uint32_t)((kl >> 6) * TOK * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t* af = afrag[j & 1];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(wacc, af[0], af[1], af[2], af[3], make_desc_sw128(xt + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        uint32_t a[16];
        decode8(L0.x, L0.y, C.x, base4, a + 0);
        decode8(L0.z, L0.w, C.y, base4, a + 4);
        decode8(L1.x, L1.y, C.z, base4, a + 8);
        decode8(L1.z, L1.w, C.w, base4, a + 12);
        const int s = st * KS + u * 4;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t b[MT][4];
#pragma unroll
          for (int j = 0; j < MT; ++j)
            ldsm_x4(xs_row[j] + xs_adv[j] * (uint32_t)(s + 2 * h), b[j][0], b[j][1], b[j][2], b[j][3]);
#pragma unroll
          for (int j = 0; j < MT; ++j) {
            const uint32_t* a0 = a + 8 * h;
            mma_bf16(acc[j], a0[0], a0[1], a0[2], a0[3], b[j][0], b[j][1]);
            mma_bf16(acc[j], a0[4], a0[5], a0[6], a0[7], b[j][2], b[j][3]);
          }
        }
      }
    }
  }
  if constexpr (WGM) {
    wg_wait<0>();
#pragma unroll
    for (int q = 0; q < TOK / 2; ++q) asm volatile("" : "+f"(wacc[q]) :: "memory");
  }
  if constexpr (WGM) {
#pragma unroll
    for (int j = 0; j < MT; ++j) { acc[j][0] = wacc[4 * j]; acc[j][1] = wacc[4 * j + 1]; acc[j][2] = wacc[4 * j + 2]; acc[j][3] = wacc[4 * j + 3]; }
  }
  if (dg >= 1) {
    if (acc[0][0] == 1.2345e-30f) out[threadIdx.x] = cvt_out<OutT>(acc[0][1]);
    return;
  }
  {
    const int g = lane >> 2, t = lane & 3;
    float* r = red + warp * 16 * TOKP;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      r[g * TOKP + j * 8 + 2 * t] = acc[j][0];
      r[g * TOKP + j * 8 + 2 * t + 1] = acc[j][1];
      r[(g + 8) * TOKP + j * 8 + 2 * t] = acc[j][2];
      r[(g + 8) * TOKP + j * 8 + 2 * t + 1] = acc[j][3];
    }
  }
  if constexpr (COMP) {
    mbar_wait(xbar0 + 8 * (XP - 1), 0);
    named_bar(1, NC);
    const int cnt = ptab_s[0].x;
    for (int p = 1 + warp; p <= cnt; p += NW) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16, kl = e.x & 0xFFFF;
      const float d = __int_as_float(e.y);
      float* r = red + ((rl >> 4) * 16 + (rl & 15)) * TOKP;
      for (int t = lane; t < M; t += 32) {
        float xv;
        if constexpr (WGM) {
          const int c = kl >> 3;
          xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
              reinterpret_cast<const unsigned char*>(xs) + (((c >> 3) * TOK + t) << 7) + (((c & 7) ^ (t & 7)) << 4) + (kl & 7) * 2));
        } else {
          xv = __bfloat162float(xs[t * XS + kl]);
        }
        atomicAdd(r + t, xv * d);
      }
    }
  }
  named_bar(1, NC);
  const int rows = RG * 16;
  const int row0 = rb * rows;
  if (splits == 1) {
    for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
      int t = idx / rows, rr = idx - t * rows;
      int grp = rr >> 4, r16 = rr & 15;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
      out[(size_t)t * N + row0 + rr] = cvt_out<OutT>(v);
    }
    return;
  }
  for (int idx = threadIdx.x; idx < rows * M; idx += NC) {
    int t = idx / rows, rr = idx - t * rows;
    int grp = rr >> 4, r16 = rr & 15;
    float v = 0.f;
#pragma unroll
    for (int w = 0; w < WS; ++w) v += red[((w * RG + grp) * 16 + r16) * TOKP + t];
    __stcg(ws + ((size_t)sp * M + t) * N + row0 + rr, v);
  }
  __threadfence();
  named_bar(1, NC);
  __shared__ int s_last;
  if (threadIdx.x == 0) {
    const int old = atomicAdd(counters + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + rb, target, 20);
    }
  }
  named_bar(1, NC);
  if (dist || s_last) {
  __threadfence();
  {
    const int r4n = rows / 4;
    const int nch = M * r4n;
    const int c_lo = dist ? (int)(((long long)sp * nch) / splits) : 0;
    const int c_hi = dist ? (int)(((long long)(sp + 1) * nch) / splits) : nch;
    for (int c0 = c_lo + threadIdx.x; c0 < c_hi; c0 += 2 * NC) {
      float4 acc4[2];
      int tt[2], rr[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = c0 + u * NC;
        tt[u] = c / r4n; rr[u] = (c - tt[u] * r4n) * 4;
        acc4[u] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
      for (int q = 0; q < splits; q += 4) {
        float4 v[2][4];
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            v[u][qq] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c0 + u * NC < c_hi && q + qq < splits)
              v[u][qq] = __ldcg(reinterpret_cast<const float4*>(ws + ((size_t)(q + qq) * M + tt[u]) * N + row0 + rr[u]));
          }
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            acc4[u].x += v[u][qq].x; acc4[u].y += v[u][qq].y; acc4[u].z += v[u][qq].z; acc4[u].w += v[u][qq].w;
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        if (c0 + u * NC < c_hi) {
          OutT* o = out + (size_t)tt[u] * N + row0 + rr[u];
          o[0] = cvt_out<OutT>(acc4[u].x); o[1] = cvt_out<OutT>(acc4[u].y);
          o[2] = cvt_out<OutT>(acc4[u].z); o[3] = cvt_out<OutT>(acc4[u].w);
        }
      }
    }
  }
  }
}
template <int MT, int RG, int WS, int KS, int NSTAGE, bool COMP, bool HL, bool WGM, typename OutT>
static void launch_tma(const at::Tensor& Wp, const at::Tensor& x, at::Tensor& out,
                       at::Tensor& ws, at::Tensor& counters, const at::Tensor& ptab, int64_t pmax,
                       int64_t base4, int M, int N, int Kp, int Kx, int splits, int nrg, int dg) {
  constexpr int NW = RG * WS;
  constexpr int CHUNK = COMP ? (KS / 4) * 1536 : KS * 512 * (HL ? 2 : 1);
  int Kc = Kp / splits;
  size_t x_bytes = WGM ? (size_t)Kc * MT * 8 * 2 + 1024 : (((size_t)M * (Kc + 8) * 2 + 16 + 15) & ~(size_t)15);
  size_t smem = 256 + (size_t)NSTAGE * NW * CHUNK + x_bytes + (size_t)NW * 16 * (MT * 8 + 1) * 4 + 8 + (size_t)pmax * 8;
  auto kern = skinny_tma_kernel<MT, RG, WS, KS, NSTAGE, COMP, HL, WGM, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  dim3 grid(N / (RG * 16), splits);
  int dev = 0, nsm = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, (NW + 1) * 32, smem);
  const int dist = (int)(grid.x * grid.y) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, (NW + 1) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(Wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4,
      M, N, Kp, Kx, splits, nrg, dist, dg);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void @@TMA_NAME@@(at::Tensor Wp, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
                at::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@TMA_DISPATCH@@
  TORCH_CHECK(false, "skinny_tma: no config");
}
"""
kl_bfp_MAP_SRC = r"""#include <cudaTypedefs.h>
#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "l"(pol) : "memory");
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2), "l"(l2_pol_el()) : "memory");
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}
static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}
"""
kl_bfp_SK2_SRC = r"""#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
static int g_sk2_pdl = 0;
void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }
#define SK2_DIAG 0
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK, bool FUSE>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg, const uint8_t* __restrict__ Wsr, float* __restrict__ outr,
           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {
  const int dgv = SK2_DIAG ? dg : 0;
  auto gtime = []() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; };
  const unsigned other_cta = CL > 1 ? 1u : gridDim.x / 2;
  const bool trace = (dgv & 32) && dbg != nullptr && (blockIdx.x == 0 || blockIdx.x == other_cta);
  unsigned long long* tr = trace ? dbg + (blockIdx.x == 0 ? 0 : 4096) : nullptr;
  const unsigned long long t_start = (dgv & 32) ? gtime() : 0ull;
  if (trace && threadIdx.x == 0) tr[0] = t_start;
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);
  const int gs0 = (crank * tst) / CL;
  const int nst = ((crank + 1) * tst) / CL - gs0;
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;
  constexpr int NSTR = FUSE ? 4 : 0;
  constexpr int WBYR = NW * 2048;
  static_assert(!FUSE || (COMP && CL == 1 && UPW == 1 && UPS == 8 && WBYR <= WBY), "sk2 fused: pooler-like config only");
  const int rt = cta >> 2, rq = cta & 3;
  const uint8_t* wsr = FUSE ? Wsr + (size_t)rt * WBYR : nullptr;
  const int nstt = nst + NSTR;
  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;
  if (tid == 0) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&xmap) : "memory");
    mbar_init(full0, 1);
    if constexpr (CL == 1)
      for (int i = 1; i < P; ++i) mbar_init(full0 + 8 * i, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int st = 0; st < P; ++st) {
      mbar_expect_tx(full0 + 8 * st, ((dgv & 4) ? 0 : XBYr) + WBY);
      bulk_g2s(sbase + st * SST + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * st);
    }
    if constexpr (CL == 1) {
      if (COMP) {
        mbar_init(pbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) {
        if (i >= P) mbar_init(full0 + 8 * i, 1);
        mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0));
      }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    } else {
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      if (!(dgv & 4))
        for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      if (COMP) {
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  __syncthreads();
  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread: x and the outputs are ours from here
  if constexpr (CL == 1) {
    if (tid == 0 && !(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
  }
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (warp == NW) {
    if (lane == 0) {
      for (int st = P; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        if (trace) tr[16 + 3 * st] = gtime();
        const uint32_t s0 = sbase + slot * SST;
        if (!FUSE || st < nst) {
          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
        } else {   // router piece: global router stage rq * 4 + j of tile rt
          const int gsr = rq * NSTR + (st - nst);
          mbar_expect_tx(full0 + 8 * slot, XBYr + WBYR);
          bulk_g2s(s0 + XBY, wsr + (size_t)gsr * (32 * WBYR), WBYR, full0 + 8 * slot);
          tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, gsr * UPS);
        }
      }
      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  if (warp >= NW + 1) {
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      int pe = 1;
      for (int st = 0; st < nstt; ++st) {
        const int slot = st % NSTAGE;
        if (FUSE && st >= nst) {   // router stages: nothing to correct, only the slot hand-back (in order, as below)
          SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
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
            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);
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
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 1] = gtime();
    const uint32_t s0 = sbase + slot * SST;
    if (dgv & 2) { __syncwarp(); if (lane == 0) mbar_arrive(empty0 + 8 * slot); continue; }
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
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
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) a[j][mt][0] = a[j][mt][1] = a[j][mt][2] = a[j][mt][3] = 0x3f803f80u;
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      if (dgv & 8) {
        const int nh = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            if (j < nh) mma_bf16(acc[j % NACC][0][nt], a[j][0][0], a[j][0][1], a[j][0][2], a[j][0][3], b[nt][2 * j], b[nt][2 * j + 1]);
            else acc[j % NACC][0][nt][0] += __uint_as_float((b[nt][2 * j] ^ b[nt][2 * j + 1]) & a[j][0][0] & 0x007fffffu);
          }
        continue;
      }
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
    if (trace && warp == 0 && lane == 0) tr[16 + 3 * st + 2] = gtime();
  }
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if constexpr (FUSE) {
    // router piece: 8 rows (one tile) over 4 stages of 8 units; hi and lo BF16 tiles into two FP32 accumulators
    float accr[2][MT][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) accr[h][mt][0] = accr[h][mt][1] = accr[h][mt][2] = accr[h][mt][3] = 0.f;
    for (int st = nst; st < nstt; ++st) {
      const int slot = st % NSTAGE;
      mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
      const uint32_t s0 = sbase + slot * SST;
      const int ui = wk;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t a0 = s0 + XBY + warp * 2048 + lane * 32;
      uint32_t bh[8], bo[8];
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[0]), "=r"(bh[1]), "=r"(bh[2]), "=r"(bh[3]) : "r"(a0));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bh[4]), "=r"(bh[5]), "=r"(bh[6]), "=r"(bh[7]) : "r"(a0 + 16));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[0]), "=r"(bo[1]), "=r"(bo[2]), "=r"(bo[3]) : "r"(a0 + 1024));
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(bo[4]), "=r"(bo[5]), "=r"(bo[6]), "=r"(bo[7]) : "r"(a0 + 1040));
      uint32_t a[4][MT][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          ldsm_x4((PK && trow[mt] >= M) ? zrow
                      : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                  a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma_bf16(accr[0][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bh[2 * j], bh[2 * j + 1]);
          mma_bf16(accr[1][mt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bo[2 * j], bo[2 * j + 1]);
        }
    }
    if constexpr (COMP) mbar_wait(cbar, 0);
    named_bar(1, NC);
    float* redr = reinterpret_cast<float*>(gbase);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const int row = 2 * t4, tok = mt * 16 + g;
      redr[(wk * TOKS + tok) * 8 + row] = accr[0][mt][0] + accr[1][mt][0];
      redr[(wk * TOKS + tok) * 8 + row + 1] = accr[0][mt][1] + accr[1][mt][1];
      redr[(wk * TOKS + tok + 8) * 8 + row] = accr[0][mt][2] + accr[1][mt][2];
      redr[(wk * TOKS + tok + 8) * 8 + row + 1] = accr[0][mt][3] + accr[1][mt][3];
    }
    named_bar(1, NC);
    float* pp = part + ((size_t)rt * 4 + rq) * 32 * 8;
    for (int i = tid; i < TOKS * 8; i += NC) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < WSK; ++w) v += redr[(w * TOKS + (i >> 3)) * 8 + (i & 7)];
      __stcg(pp + i, v);
    }
    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
      int old;
      asm volatile("atom.add.release.gpu.global.s32 %0, [%1], 1;" : "=r"(old) : "l"(rcnt + rt) : "memory");
      s_rlast = (old & 3) == 3;
      if (s_rlast) asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
    named_bar(1, NC);
    if (s_rlast) {
      const float* pr = part + (size_t)rt * 4 * 32 * 8;
      for (int i = tid; i < M * 8; i += NC) {
        const float v = __ldcg(pr + i) + __ldcg(pr + 256 + i) + __ldcg(pr + 512 + i) + __ldcg(pr + 768 + i);
        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);
  }
  if (trace && lane == 0) tr[1 + warp] = gtime();
  if (dgv & 1) {
    if (acc[0][0][0][0] == 1.2345e-30f) out[tid] = cvt_out<OutT>(acc[NACC - 1][0][0][1]);
    return;
  }
  if constexpr (COMP) mbar_wait(cbar, 0);
  if (CL > 1 && crank > 0) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
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
    if (trace && tid == 0) tr[15] = gtime();
    return;
  }
  named_bar(1, NC);
  if (trace && tid == 0) tr[9] = gtime();
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
      if (trace && mt == 0 && nt == 0) {
        asm volatile("" :: "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
        if (tid == 0) tr[12] = gtime();
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  if (trace && tid == 0) tr[11] = gtime();
  named_bar(1, NC);
  if (trace && tid == 0) tr[10] = gtime();
  if (trace && tid == 0) tr[13] = gtime();
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
  }
  if (trace && tid == 0) tr[14] = gtime();
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
  if (trace && tid == 0) tr[15] = gtime();
  if ((dgv & 32) && dbg != nullptr && tid == 0) {
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    dbg[8192 + 3 * cta] = smid; dbg[8192 + 3 * cta + 1] = t_start; dbg[8192 + 3 * cta + 2] = gtime();
  }
}
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool FUSE = false>
static void launch_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg,
                       const uint8_t* wsr = nullptr, float* outr = nullptr, float* part = nullptr, int* rcnt = nullptr) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK, FUSE>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    const int pdl = g_sk2_pdl;
    if (CL > 1 || pdl > 0) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[2];
      int na = 0;
      if (CL > 1) {
        at_[na].id = cudaLaunchAttributeClusterDimension;
        at_[na].val.clusterDim.x = CL;
        at_[na].val.clusterDim.y = 1;
        at_[na].val.clusterDim.z = 1;
        ++na;
      }
      if (pdl > 0) {
        at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        at_[na].val.programmaticStreamSerializationAllowed = 1;
        ++na;
      }
      lc.attrs = at_;
      lc.numAttrs = na;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg,
                                        wsr, outr, part, rcnt, pdl));
    } else {
      kern<<<N / R, (NW + 1 + NESC) * 32, smem, at::cuda::getCurrentCUDAStream()>>>(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg,
                                                                               wsr, outr, part, rcnt, 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}
void sk2f(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
          int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {
  @@SK2F_DISPATCH@@
  TORCH_CHECK(false, "sk2f: no config");
}
void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  @@SK2_DISPATCH@@
  TORCH_CHECK(false, "sk2: no config");
}
"""
kl_bfp_WIDE_SRC = r"""template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
__global__ void __launch_bounds__(NWG * 128, 1)
wide_kernel(const __grid_constant__ CUtensorMap mapB, const uint8_t* __restrict__ Wst,
            const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out, float* __restrict__ ws,
            int* __restrict__ counters, const int2* __restrict__ ptab, int pmax, uint32_t base4,
            int M, int N, int K, int splits, int dist) {
  constexpr int NWARP = NWG * 4;
  constexpr int NT = NWG * 128;
  constexpr int WB = HLW ? 4096 : 1536;
  constexpr int A_BYTES = NWARP * WB;
  constexpr int B_BYTES = TOK * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = TOK / 2;
  static_assert(ST_BYTES % 1024 == 0, "stage alignment");
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  __shared__ int s_last;
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t full0 = sbase + STAGES * ST_BYTES, empty0 = full0 + 8 * STAGES;
  int2* ptab_s = reinterpret_cast<int2*>(gbase + STAGES * ST_BYTES + 16 * STAGES);
  __nv_bfloat16* xpatch = reinterpret_cast<__nv_bfloat16*>(ptab_s + pmax);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y, nrb = gridDim.x;
  const int zh = blockIdx.z;
  const int t0z = zh * TOK;
  const int Mz = min(TOK, M - t0z);
  const int U = K >> 6;
  const int u0 = (int)(((long long)sp * U) / splits);
  const int u1 = (int)(((long long)(sp + 1) * U) / splits);
  const int nst = u1 - u0;
  const int cta = sp * nrb + rb;
  const uint8_t* wsrc = Wst + ((size_t)rb * U + u0) * A_BYTES;
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, NWG);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + zh * nrb + rb));
  }
  auto issue = [&](int i) {
    const int slot = i % STAGES;
    const uint32_t st = sbase + slot * ST_BYTES;
    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
    tma_load_2d_el(st, &mapB, full0 + 8 * slot, (u0 + i) * 64, t0z);
    bulk_g2s(st + B_BYTES, wsrc + (size_t)i * A_BYTES, A_BYTES, full0 + 8 * slot);
  };
  const bool producer = tid == 0;
  if (producer) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
  }
  __syncthreads();
  {
    const int2* tab = ptab + (size_t)cta * pmax;
    const int cnt = pmax ? tab[0].x : 0;
    if (pmax) for (int p = tid; p <= cnt; p += NT) ptab_s[p] = tab[p];
    for (int idx = tid; idx < cnt * TOK; idx += NT) {
      const int p = idx / TOK + 1, t = idx - (p - 1) * TOK;
      const int kl = tab[p].x & 0xFFFF;
      xpatch[p * TOK + t] = t < Mz ? x[(size_t)(t0z + t) * K + u0 * 64 + kl] : __float2bfloat16(0.f);
    }
  }
  const int g = lane >> 2, t4 = lane & 3;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  uint32_t afrag[2][4];
  for (int i = 0; i < nst; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t st = sbase + slot * ST_BYTES;
    const uint32_t wa = st + B_BYTES + warp * WB + lane * 16;
    if constexpr (HLW) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t* af = afrag[h];
          asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                       : "r"(wa + h * 2048 + j * 512));
#pragma unroll
          for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
          wg_fence();
          wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
          wg_commit();
          wg_wait<1>();
#pragma unroll
          for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
        }
      }
    } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t lw0, lw1, cw;
      asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(lw0), "=r"(lw1) : "r"(wa + (j >> 1) * 512 + (j & 1) * 8));
      asm volatile("ld.shared.u32 %0, [%1];" : "=r"(cw) : "r"(wa + 1024 + 4 * j));
      uint32_t* af = afrag[j & 1];
      decode8(lw0, lw1, cw, base4, af);
#pragma unroll
      for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
      wg_fence();
      wgmma_bn<TOK>(acc, af[0], af[1], af[2], af[3], make_desc_sw128(st + j * 32));
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int q = 0; q < 4; ++q) asm volatile("" : "+r"(afrag[0][q]), "+r"(afrag[1][q]) :: "memory");
    }
    }
    if (i > 0) {
      if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
      if (producer && i - 1 + STAGES < nst) {
        mbar_wait(empty0 + 8 * ((i - 1) % STAGES), ((i - 1) / STAGES) & 1);
        issue(i - 1 + STAGES);
      }
    }
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  named_bar(1, NT);
  {
    const int cnt = pmax ? ptab_s[0].x : 0;
    for (int p = 1; p <= cnt; ++p) {
      const int2 e = ptab_s[p];
      const int rl = e.x >> 16;
      if ((rl >> 4) != warp) continue;
      const int r16 = rl & 15;
      if (g != (r16 & 7)) continue;
      const float d = __int_as_float(e.y);
      const __nv_bfloat16* xv = xpatch + p * TOK;
      const bool hi = r16 >= 8;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const float v0 = d * __bfloat162float(xv[8 * jj + 2 * t4]);
        const float v1 = d * __bfloat162float(xv[8 * jj + 2 * t4 + 1]);
        if (hi) { acc[4 * jj + 2] += v0; acc[4 * jj + 3] += v1; }
        else { acc[4 * jj] += v0; acc[4 * jj + 1] += v1; }
      }
    }
  }
  const int row_w = rb * (NWARP * 16) + warp * 16;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < TOK / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < Mz) {
        out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < Mz) {
        out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + ((size_t)(zh * nrb + rb) * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    const int old = atomicAdd(counters + zh * nrb + rb, 1);
    s_last = (old % splits) == splits - 1;
    if (dist) {
      const int target = (old / splits + 1) * splits;
      okc_spin_ge(counters + zh * nrb + rb, target, 32);
    }
  }
  named_bar(1, NT);
  if (!dist && !s_last) return;
  __threadfence();
  const int j0 = dist ? (int)(((long long)sp * (NACC / 4)) / splits) : 0;
  const int j1 = dist ? (int)(((long long)(sp + 1) * (NACC / 4)) / splits) : NACC / 4;
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)(zh * nrb + rb) * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s4 = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      const float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      const float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      const float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      const float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s4.x += v0.x + v1.x + v2.x + v3.x; s4.y += v0.y + v1.y + v2.y + v3.y;
      s4.z += v0.z + v1.z + v2.z + v3.z; s4.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      const float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s4.x += v.x; s4.y += v.y; s4.z += v.z; s4.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < Mz) {
      out[(size_t)(t0z + t0) * N + row_w + g] = cvt_out<OutT>(s4.x);
      out[(size_t)(t0z + t0) * N + row_w + g + 8] = cvt_out<OutT>(s4.z);
    }
    if (t0 + 1 < Mz) {
      out[(size_t)(t0z + t0 + 1) * N + row_w + g] = cvt_out<OutT>(s4.y);
      out[(size_t)(t0z + t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s4.w);
    }
  }
}
template <int NWG, int TOK, int STAGES, bool HLW, typename OutT>
static void launch_wide(const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                        at::Tensor& counters, const at::Tensor& ptab, int64_t pmax, int64_t base4,
                        int M, int N, int K, int splits) {
  constexpr int ST_BYTES = NWG * 4 * (HLW ? 4096 : 1536) + TOK * 128;
  const int nz = (M + TOK - 1) / TOK;
  const size_t smem = 1024 + (size_t)STAGES * ST_BYTES + 16 * STAGES + (size_t)pmax * 8 + (size_t)pmax * TOK * 2 + 64;
  auto kern = wide_kernel<NWG, TOK, STAGES, HLW, OutT>;
  int dev = 0;
  cudaGetDevice(&dev);
  static bool attr_done = false;
  if (!attr_done) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
    attr_done = true;
  }
  int major = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  CUtensorMap mB;
  memset(&mB, 0, sizeof(mB));
  if (major >= 9) mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, TOK);
  dim3 grid(N / (NWG * 64), splits, nz);
  int nsm = 0, per_sm = 0;
  cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, NWG * 128, smem);
  const int dist = (int)(grid.x * grid.y * grid.z) <= per_sm * nsm ? 1 : 0;
  kern<<<grid, NWG * 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      reinterpret_cast<const int2*>(ptab.data_ptr()), (int)pmax, (uint32_t)base4, M, N, K, splits, dist);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void wide(at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, at::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok) {
  const int hl = out.scalar_type() == at::kFloat ? 1 : 0;
  @@WIDE_DISPATCH@@
  TORCH_CHECK(false, "wide: no config");
}
"""
kl_bfp_GQ_SRC = r"""template <int BN, int NWG, int NSTAGE, int CL>
__global__ void __launch_bounds__(NWG * 128 + 32, 1)
gq_kernel(const __grid_constant__ CUtensorMap mapW, const __grid_constant__ CUtensorMap mapX,
          __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * ST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, qbar = bars + 16 * NSTAGE;
  float* peer = reinterpret_cast<float*>(gbase + NSTAGE * ST + 16 * NSTAGE + 16);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int U = K >> 6;
  const int u0 = (crank * U) / CL, u1 = ((crank + 1) * U) / CL;
  const int nkb = u1 - u0;
  const int xrows = min(M, BN);
  if (tid == 0) {
    for (int i = 0; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    mbar_init(qbar, 1);
    if (CL == 2) mbar_expect_tx(qbar, NWG * 64 * NACC * 4);
    else if (CL > 1 && crank == 0) mbar_expect_tx(qbar, (CL - 1) * NWG * NACC * 128 * 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapW) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&mapX) : "memory");
    for (int i = 0; i < NSTAGE && i < nkb; ++i) {
      mbar_expect_tx(full0 + 8 * i, A_BYTES + xrows * 128);
      const int kg = (u0 + i) * 64;
      tma_load_2d_ef(sbase + i * ST, &mapW, full0 + 8 * i, kg, rb * BM);
      tma_load_2d_el(sbase + i * ST + A_BYTES, &mapX, full0 + 8 * i, kg, 0);
    }
  }
  __syncthreads();
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  if (warp == NWG * 4) {
    if (lane == 0) {
      for (int i = NSTAGE; i < nkb; ++i) {
        const int slot = i % NSTAGE;
        if (i >= NSTAGE) mbar_wait_sleep(empty0 + 8 * slot, ((i / NSTAGE) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, A_BYTES + xrows * 128);
        const int kg = (u0 + i) * 64;
        tma_load_2d_ef(sbase + slot * ST, &mapW, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST + A_BYTES, &mapX, full0 + 8 * slot, kg, 0);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    return;
  }
  const int wg = warp >> 2, t = tid & 127;
  float acc[NACC];
#pragma unroll
  for (int q = 0; q < NACC; ++q) acc[q] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % NSTAGE;
    mbar_wait_sleep(full0 + 8 * slot, (i / NSTAGE) & 1);
    const uint32_t a_st = sbase + slot * ST + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j) wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && t == 0) mbar_arrive(empty0 + 8 * ((i - 1) % NSTAGE));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  if constexpr (CL <= 2) {
    constexpr int RS = NACC + 4;
    constexpr int KR = CL == 2 ? 32 : 64;
    const int q = warp & 3;
    const int keep_lo = CL == 2 ? 2 * crank : 0;
    const bool keep = CL == 1 || (q >> 1) == crank;
    if (CL == 2 && !keep) {
      const int ql = q & 1;
      const uint32_t rl = (uint32_t)__cvta_generic_to_shared(peer + ((wg * 2 + ql) * 32 + lane) * RS);
      uint32_t rr, qb;
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rr) : "r"(rl), "r"(crank ^ 1));
      asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(qb) : "r"(qbar), "r"(crank ^ 1));
#pragma unroll
      for (int i = 0; i < NACC; i += 4)
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     :: "r"(rr + i * 4), "r"(__float_as_uint(acc[i])), "r"(__float_as_uint(acc[i + 1])),
                        "r"(__float_as_uint(acc[i + 2])), "r"(__float_as_uint(acc[i + 3])), "r"(qb) : "memory");
      return;
    }
    if constexpr (CL == 2) {
      asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
      mbar_wait_cluster(qbar, 0);
      const float* rv = peer + ((wg * 2 + (q & 1)) * 32 + lane) * RS;
#pragma unroll
      for (int i = 0; i < NACC; i += 4) {
        const float4 p4 = *reinterpret_cast<const float4*>(rv + i);
        acc[i] += p4.x; acc[i + 1] += p4.y; acc[i + 2] += p4.z; acc[i + 3] += p4.w;
      }
    }
    if constexpr (NWG > 1) named_bar(1, NWG * (CL == 2 ? 64 : 128));
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(gbase);
    {
      const int g = lane >> 2, t4 = lane & 3;
      const int rk = (q - keep_lo) * 16;
#pragma unroll
      for (int j = 0; j < BN / 8; ++j) {
        const int t0 = 8 * j + 2 * t4;
        __nv_bfloat16* s0 = stg + ((size_t)wg * BN + t0) * KR + rk + g;
        s0[0] = __float2bfloat16_rn(acc[4 * j]);
        s0[8] = __float2bfloat16_rn(acc[4 * j + 2]);
        s0[KR] = __float2bfloat16_rn(acc[4 * j + 1]);
        s0[KR + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
      }
    }
    named_bar(2, NWG * (CL == 2 ? 64 : 128));
    {
      constexpr int CH = KR / 8;
      const int nkt = NWG * (CL == 2 ? 64 : 128);
      const int kt = (q - keep_lo) * 32 + lane + wg * (CL == 2 ? 64 : 128);
      for (int idx = kt; idx < NWG * BN * CH; idx += nkt) {
        const int c = idx % CH, tt = (idx / CH) % BN, w2 = idx / (CH * BN);
        if (tt < M) {
          const uint4 v = *reinterpret_cast<const uint4*>(stg + ((size_t)w2 * BN + tt) * KR + c * 8);
          *reinterpret_cast<uint4*>(out + (size_t)tt * N + rb * BM + w2 * 64 + (CL == 2 ? crank * 32 : 0) + c * 8) = v;
        }
      }
    }
    return;
  }
  if (CL > 1 && crank > 0) {
    float* mine = reinterpret_cast<float*>(gbase);
    if constexpr (NWG > 1) named_bar(1, NWG * 128);
#pragma unroll
    for (int q = 0; q < NACC; ++q) mine[(wg * NACC + q) * 128 + t] = acc[q];
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    named_bar(1, NWG * 128);
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid == 0) {
      const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * NWG * NACC * 128);
      uint32_t peer_r, qbar_r;
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(peer_r) : "r"(peer_l));
      asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(qbar_r) : "r"(qbar));
      asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   :: "r"(peer_r), "r"((uint32_t)__cvta_generic_to_shared(mine)), "r"(NWG * NACC * 128 * 4), "r"(qbar_r)
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;\n\tcp.async.bulk.wait_group.read 0;" ::: "memory");
    }
    return;
  }
  if constexpr (CL > 1) {
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    mbar_wait_cluster(qbar, 0);
#pragma unroll
    for (int r = 0; r < CL - 1; ++r)
#pragma unroll
      for (int q = 0; q < NACC; ++q) acc[q] += peer[((r * NWG + wg) * NACC + q) * 128 + t];
  }
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + wg * 64 + (warp & 3) * 16;
#pragma unroll
  for (int j = 0; j < BN / 8; ++j) {
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = __float2bfloat16_rn(acc[4 * j]);
      out[(size_t)t0 * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = __float2bfloat16_rn(acc[4 * j + 1]);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
  }
}
template <int BN, int NWG, int NSTAGE, int CL>
static void launch_gq(const at::Tensor& W, const at::Tensor& x, at::Tensor& out, int M, int N, int K) {
  constexpr int BM = NWG * 64;
  constexpr int ST = BM * 128 + BN * 128;
  const size_t smem = 1024 + (size_t)NSTAGE * ST + 16 * NSTAGE + 16 +
                      (CL == 2 ? (size_t)NWG * 64 * (BN / 2 + 4) * 4 : (size_t)(CL - 1) * NWG * (BN / 2) * 128 * 4);
  auto kern = gq_kernel<BN, NWG, NSTAGE, CL>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mW = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mX = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)K, (uint32_t)(M < BN ? M : BN));
  __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if constexpr (CL > 1) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(N / BM * CL);
    lc.blockDim = dim3(NWG * 128 + 32);
    lc.dynamicSmemBytes = smem;
    lc.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeClusterDimension;
    at_[0].val.clusterDim.x = CL;
    at_[0].val.clusterDim.y = 1;
    at_[0].val.clusterDim.z = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, mW, mX, op, M, N, K));
  } else {
    kern<<<N / BM, NWG * 128 + 32, smem, at::cuda::getCurrentCUDAStream()>>>(mW, mX, op, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
void gq(at::Tensor W, at::Tensor x, at::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage) {
  @@GQ_DISPATCH@@
  TORCH_CHECK(false, "gq: no config");
}
"""
kl__SL_GLUE_CUDA = r"""#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;
struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
  const float* alpha2;
  float* tail;
};
__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}
template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* p=cluster.map_shared_rank(sm,dst);
      p[rank*warps_per_cta+warp]=sum;
    }
  }
  cluster.sync();
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* px=cluster.map_shared_rank(sm,dst);
      volatile float* py=cluster.map_shared_rank(smy,dst);
      px[rank*warps_per_cta+warp]=sx;
      py[rank*warps_per_cta+warp]=sy;
    }
  }
  cluster.sync();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
}
template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];
  __shared__ float partials[3][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glf<V>(a.alpha2+col,a2+r*V);
    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
      }
    }
  }
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
    }
  }
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    #pragma unroll
    for(int j=0;j<V;j++) {
      int k=r*V+j;
      carry[k]=gm(a2[k],ns[k]);
      masked[k]=(col+j>=1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);
  if(rank==0 && tid==0) a.tail[row]=tail;
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}
template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0),(const float*)alpha2.data_ptr(),(float*)tail.data_ptr()};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
"""

# LDD19 v24: the attention source is Keell's (vjeux v298) with these exact LDD19 edits (readable template).
ATT7_CUDA_SRC = _ldd19_derive(kl_ATT7_CUDA_SRC, [
  (r"""  if (L == 1) launch<1>(p, grid, stream, (int)cs, major >= 9, pdl);
  else launch<8>(p, grid, stream, (int)cs, major >= 9, pdl);""", r"""  const bool etx = ((tma >> 17) & 1) != 0;
  const bool hs = ((tma >> 18) & 1) != 0;
  if (L == 1) launch<1>(p, grid, stream, (int)cs, major >= 9, pdl);
  else launch<8>(p, grid, stream, (int)cs, major >= 9, pdl, etx, hs);"""),
  (r"""  p.pf = reinterpret_cast<const uint8_t*>(pf_ptr);
  p.pf_chunk = pf_ptr ? (unsigned)((pf_bytes / ((int64_t)grid.x * grid.y * grid.z)) & ~15ll) : 0u;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();""", r"""  cudaStream_t stream = at::cuda::getCurrentCUDAStream();"""),
  (r"""         int64_t tma, int64_t pf_ptr, int64_t pf_bytes) {""", r"""         int64_t tma) {"""),
  ('before', r"""static CUtensorMap row_map(void* base, long long rows, long long row_bytes) {""", r"""static EncodeTiledFn get_encode() {
  static EncodeTiledFn encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
    TORCH_CHECK(fn != nullptr, "cuTensorMapEncodeTiled unavailable");
    encode = reinterpret_cast<EncodeTiledFn>(fn);
  }
  return encode;
}
static CUtensorMap pt_map(void* base, long long n) {
  CUtensorMap m;
  cuuint64_t dims[1] = {(cuuint64_t)n};
  cuuint64_t strides[1] = {0};
  cuuint32_t box[1] = {64};
  cuuint32_t es[1] = {1};
  const CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_INT32, 1, base, dims, strides, box, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (page table) failed: ", (int)r);
  return m;
}
"""),
  ('after', r"""    if (kern == ksp1[3]) { occ_k = occ_sp1; plain = ksp1[1]; }""", r"""
#endif
#ifdef OKC_COOPGUARD_TEST
    occ_k = 0;
#endif"""),
  (r"""    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }""", r"""    if (kern == kx[3]) { occ_k = occ_x; plain = kx[1]; }
#if OKC_SPEC && OKC_ONE
    if (hs_on) occ_k = 1 << 20;
#endif
#if OKC_ONE
    if (kern == kn1[3]) { occ_k = occ_c1; plain = kn1[1]; }
#endif
#if OKC_SPEC"""),
  ('before', r"""  if (OKC_COOPGUARD && q.coop && !p.flat && !cl) {""", r"""#endif
  if (etx && L == 8 && !p.flat && !cl && (kern == kn[1] || kern == kn[3])) kern = (kern == kn[3]) ? kx[3] : kx[1];
#if OKC_SPEC && OKC_ONE
  bool hs_on = false;
  if (hs && L == 8 && !p.flat && !cl && q.coop && kern == ksp1[3] &&
      (long long)grid.x * grid.y * grid.z * 2 <= (long long)occ_hs * nsm) {
    kern = khs;
    grid.z *= 2;
    hs_on = true;
  }
#endif
"""),
  ('after', r"""    if (OKC_TCHUNK && (long long)p.n_splits * 64 >= kmax) kern = kn1[q.coop ? 3 : 1];
  }""", r"""
#endif
#if OKC_SPEC"""),
  ('before', r"""  if (!p.flat && !cl && !(OKC_UNEV && q.unev)) {""", r"""#if OKC_ONE
"""),
  ('after', r"""    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_sp1, ksp1[3], Cfg<L, false>::NTX, sm_n);""", r"""
#endif"""),
  (r"""    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);""", r"""#if OKC_ONE
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_c1, kn1[3], Cfg<L, false>::NTX, sm_n);
#endif
#if OKC_SPEC"""),
  ('after', r"""      cudaFuncSetAttribute(ksp1[m], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    }""", r"""
#endif"""),
  ('before', r"""    for (int m = 1; m < 4; m += 2) {""", r"""#endif
#if OKC_SPEC
"""),
  ('before', r"""    cudaFuncSetAttribute(kn1[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);""", r"""#if OKC_ONE
"""),
  (r"""  if (!init) {""", r"""#endif
  static const KernFn kx[4] = {nullptr, attn_kernel<L, false, (L == 8 ? 17 : 1)>, nullptr, attn_kernel<L, false, (L == 8 ? 19 : 3)>};
  static const KernFn khs = attn_kernel<L, false, (L == 8 ? (15 | 32) : 15)>;
  if (!init) {
    cudaFuncSetAttribute(khs, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_hs, khs, Cfg<L, false>::NTX, sm_n);
    cudaFuncSetAttribute(kx[1], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaFuncSetAttribute(kx[3], cudaFuncAttributeMaxDynamicSharedMemorySize, sm_n);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ_x, kx[3], Cfg<L, false>::NTX, sm_n);"""),
  (r"""  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};""", r"""#if OKC_ONE
  static const KernFn kn1[4] = {nullptr, attn_kernel<L, false, 5>, nullptr, attn_kernel<L, false, 7>};
#endif
#if OKC_SPEC"""),
  ('before', r"""  typedef void (*KernFn)(const Params);""", r"""  static int occ_x = 0;
  static int occ_hs = 0;
"""),
  (r"""void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool clusters, bool pdl) {""", r"""void launch(const Params& p0, dim3 grid, cudaStream_t stream, int cs, bool clusters, bool pdl, bool etx = false, bool hs = false) {"""),
  (r"""      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
      run_segments(std::integral_constant<int, 1>());
    } else {
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
      run_segments(std::integral_constant<int, 2>());""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::PREGS));
#endif
#if !(OKC_DYN)
      run_segments(std::integral_constant<int, 1>());
#endif
    } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if constexpr (OKC_SETMAXNREG) asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(Cfg<L, FLAT>::CREGS));
#endif
#if !(OKC_DYN)
      run_segments(std::integral_constant<int, 2>());
#endif"""),
  ('after', r"""    ce = (int)((int64_t)(cpart + 1) * T / G);
  }""", r"""
#endif"""),
  ('after', r"""  const int c = blockIdx.x, G = gridDim.x;""", r"""
#if !(OKC_BALPART)"""),
  (r"""                                  chunk, 0, SPEC ? seq_ld : seq))""", r"""                                  chunk, 0, SPEC ? seq_ld : seq, false, xk))"""),
  (r"""    int start = lo + split * chunk;""", r"""#endif
    int xk = 0;
    if constexpr (ETX_OK(L, FLAT, MODE)) {
      if (p.window && lo > 0) {
        xk = L - 1;
        chunk = fdiv_i(((n_keys - xk + 63) >> 6) + p.n_splits - 1, p.n_splits) << 6;
      }
    }
    int start = lo + xk + split * chunk;"""),
  ('after', r"""    chunk = (chunk + 15) & ~15;""", r"""
#if OKC_TCHUNK"""),
  (r"""  const int pair = blockIdx.z;
  DBG(0);
  if (p.pf_chunk && threadIdx.x == 0) {
    const uint8_t* src = p.pf + (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * p.pf_chunk;
    for (unsigned o = 0; o < p.pf_chunk; o += 32768u) {
      const unsigned n = min(32768u, p.pf_chunk - o);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(src + o), "r"(n) : "memory");
    }
  }""", r"""  const int pair = (!FLAT && (MODE & 32)) ? (int)(blockIdx.z >> 1) : (int)blockIdx.z;
  DBG(0);"""),
  (r"""    S.flag = (old + mine == n_keys);""", r"""    S.flag = (old + mine == ntgt);"""),
  (r"""    const int mine = end - start;
    int old;""", r"""    const int mine = end - start + (xon ? xk : 0);
    int old;"""),
  (r"""    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {""", r"""#if OKC_FIN1
#pragma unroll 1
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {
#else
    for (int rbase = r_lo; rbase < r_hi; rbase += NT / 8) {
#endif"""),
  (r"""        if (v >= n_keys) break;
        SPIN_CHECK;""", r"""        if (v >= ntgt) break;
        SPIN_CHECK;"""),
  (r"""      atomicAdd(p.ws_cnt + bp, end - start);""", r"""      atomicAdd(p.ws_cnt + bp, end - start + (xon ? xk : 0));"""),
  (r"""          if (v >= n_keys) break;""", r"""#ifdef OKC_SPINTEST
          if (v >= n_keys + 1) break;
#else
          if (v >= ntgt) break;
#endif"""),
  (r"""      else old = atomicAdd(c, mine);
      if (old + mine == n_keys) {""", r"""      else old = atomicAdd(c, mine);
      if (old + mine == ntgt) {"""),
  (r"""      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start;""", r"""      int* c = p.ws_cnt + ((ep & 1) ? 12288 : 8192) + bp;
      const int mine = end - start + (xon ? xk : 0);"""),
  (r"""      if (old + mine == n_keys) {""", r"""      if (old + mine == ntgt) {"""),
  (r"""      const int mine = end - start;""", r"""      const int mine = end - start + (xon ? xk : 0);"""),
  (r"""    const int split_idx = (int)blockIdx.x;
    int nact = (OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && ((n_keys + 63) >> 6) >= 2 * p.n_splits)
                   ? p.n_splits : fdiv_i(n_keys + chunk - 1, chunk);""", r"""    const int split_idx = HS ? 2 * (int)blockIdx.x + hsel : (int)blockIdx.x;
    int nact = (OKC_UNEV && (MODE & 3) == 3 && p.unev > 0 && ((n_keys + 63) >> 6) >= 2 * p.n_splits)
                   ? p.n_splits : fdiv_i(n_keys - xk + chunk - 1, chunk);
    if constexpr (HS) nact *= 2;"""),
  (r"""    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));""", r"""    const float lsum = quad_sum(l_r[r]);
    const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f)) * lx_fs;"""),
  (r"""        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f));""", r"""        const float f = (m_r[r] == -INFINITY) ? 0.f : ex2(fminf(m_r[r], 126.f)) * lx_fs;"""),
  ('after', r"""  bool do_red = true;""", r"""
  if constexpr (HS) do_red = (hk == hsel);
  const int ntgt = HS ? 2 * n_keys : n_keys;"""),
  ('before', r"""  }
  DBG(3);""", r"""#else
  uint8_t* K16 = S.kvq;
  uint8_t* V16 = S.kvq + 2 * 64 * 128;
  const uint32_t k16_base = smem_u32(K16) + hk * 8192, v16_base = smem_u32(V16) + hk * 8192;
  uint32_t qa[4][4];
  {
    const uint32_t qs = smem_u32((hk ? S.v : S.k) + (STAGES - 1) * STAGE_BYTES);
    const int mi = lane >> 3, rr = lane & 7;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int row = 16 * wsub + (mi & 1) * 8 + rr;
      ldsm_x4(qs + row * 128 + (2 * ks + (mi >> 1)) * 16, qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3]);
    }
  }
  __syncthreads();   // q staging (last stage) is overwritten by the first issue below
  float cr[2] = {0.f, 0.f};
  float lacc[4] = {0.f, 0.f, 0.f, 0.f};
  bool first = true;
  const uint32_t ONES = 0x3C003C00u;
  const int ckey = tid >> 2, cq = tid & 3;
  for (int i = 0; i < ntiles; ++i) {
    cp_wait<STAGES - 2>();
    __syncthreads();   // tile i landed; previous tile's compute done (kv16 free)
    if (i + STAGES - 1 < ntiles) issue_kv(i + STAGES - 1, true);
    issue_idx(i + 2 * STAGES - 2);
    cp_commit();
    {
      const uint8_t* K = S.k + (i % STAGES) * STAGE_BYTES;
      const uint8_t* V = S.v + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const int c = 2 * cq + u;
        const int hd = c >> 2, fc = (c & 3) * 2;
        const int src = ckey * ROWB + ((c ^ (ckey & 7)) << 4);
        const int dst0 = hd * 8192 + ckey * 128 + ((fc ^ (ckey & 7)) << 4);
        const int dst1 = hd * 8192 + ckey * 128 + (((fc + 1) ^ (ckey & 7)) << 4);
        const uint4 kb = lds128(K + src);
        const uint4 vb = lds128(V + src);
        *reinterpret_cast<uint4*>(K16 + dst0) = make_uint4(cvt2(kb.x), cvt2(kb.x >> 16), cvt2(kb.y), cvt2(kb.y >> 16));
        *reinterpret_cast<uint4*>(K16 + dst1) = make_uint4(cvt2(kb.z), cvt2(kb.z >> 16), cvt2(kb.w), cvt2(kb.w >> 16));
        *reinterpret_cast<uint4*>(V16 + dst0) = make_uint4(cvt2(vb.x), cvt2(vb.x >> 16), cvt2(vb.y), cvt2(vb.y >> 16));
        *reinterpret_cast<uint4*>(V16 + dst1) = make_uint4(cvt2(vb.z), cvt2(vb.z >> 16), cvt2(vb.w), cvt2(vb.w >> 16));
      }
    }
    __syncthreads();   // kv16 ready
    const int t0 = start + i * TILE;
    const bool interior = (t0 >= lo_all) && (t0 + TILE - 1 <= hi_all);
#pragma unroll 1
    for (int kh = 0; kh < 2; ++kh) {
    float s[4][4];
    const float ilo = first ? 0.f : -cr[0], ihi = first ? 0.f : -cr[1];
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      const int nt = 4 * kh + nn;
      s[nn][0] = ilo; s[nn][1] = ilo; s[nn][2] = ihi; s[nn][3] = ihi;
      const int key = 8 * nt + (lane & 7);
#pragma unroll
      for (int h2 = 0; h2 < 2; ++h2) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4(k16_base + key * 128 + (((4 * h2 + (lane >> 3)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(s[nn], qa[2 * h2][0], qa[2 * h2][1], qa[2 * h2][2], qa[2 * h2][3], b0a, b1a);
        mma_f16(s[nn], qa[2 * h2 + 1][0], qa[2 * h2 + 1][1], qa[2 * h2 + 1][2], qa[2 * h2 + 1][3], b0b, b1b);
      }
    }
    if (!interior) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int pj = seq + 2 * wsub + hf;
        const int vlo = p.window ? max(start, pj - p.window + 1) : start;
        const int vhi = min(end - 1, pj);
#pragma unroll
        for (int nn = 0; nn < 4; ++nn)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int pos = t0 + 32 * kh + 8 * nn + 2 * tig + e;
            if (pos < vlo || pos > vhi) s[nn][2 * hf + e] = -INFINITY;
          }
      }
    }
    float mlo = -INFINITY, mhi = -INFINITY;
#pragma unroll
    for (int nn = 0; nn < 4; ++nn) {
      mlo = fmaxf(mlo, fmaxf(s[nn][0], s[nn][1]));
      mhi = fmaxf(mhi, fmaxf(s[nn][2], s[nn][3]));
    }
    mlo = quad_max(mlo);
    mhi = quad_max(mhi);
    if (first) {
      const float nlo = (mlo == -INFINITY) ? 0.f : mlo, nhi = (mhi == -INFINITY) ? 0.f : mhi;
      cr[0] = nlo; cr[1] = nhi;
#pragma unroll
      for (int nn = 0; nn < 4; ++nn) { s[nn][0] -= nlo; s[nn][1] -= nlo; s[nn][2] -= nhi; s[nn][3] -= nhi; }
      first = false;
    } else if (__any_sync(0xffffffffu, mlo > 8.f || mhi > 8.f)) {
      const float dlo = fmaxf(mlo, 0.f), dhi = fmaxf(mhi, 0.f);
      const float alo = ex2(-dlo), ahi = ex2(-dhi);
      cr[0] += dlo; cr[1] += dhi;
#pragma unroll
      for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
      lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
#pragma unroll
      for (int nn = 0; nn < 4; ++nn) { s[nn][0] -= dlo; s[nn][1] -= dlo; s[nn][2] -= dhi; s[nn][3] -= dhi; }
    }
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int ks = 2 * kh + kk;
      const uint32_t a0 = pack_h2(ex2(s[2 * kk][0]), ex2(s[2 * kk][1]));
      const uint32_t a1 = pack_h2(ex2(s[2 * kk][2]), ex2(s[2 * kk][3]));
      const uint32_t a2 = pack_h2(ex2(s[2 * kk + 1][0]), ex2(s[2 * kk + 1][1]));
      const uint32_t a3 = pack_h2(ex2(s[2 * kk + 1][2]), ex2(s[2 * kk + 1][3]));
      const int mi = lane >> 3;
      const int key = 16 * ks + (mi & 1) * 8 + (lane & 7);
#pragma unroll
      for (int dp = 0; dp < 4; ++dp) {
        uint32_t b0a, b1a, b0b, b1b;
        ldsm_x4_t(v16_base + key * 128 + (((2 * dp + (mi >> 1)) ^ (key & 7)) << 4), b0a, b1a, b0b, b1b);
        mma_f16(o[0][2 * dp], a0, a1, a2, a3, b0a, b1a);
        mma_f16(o[0][2 * dp + 1], a0, a1, a2, a3, b0b, b1b);
      }
      mma_f16(lacc, a0, a1, a2, a3, ONES, ONES);
    }
    }
  }
  m_r[0] = first ? -INFINITY : cr[0];
  m_r[1] = first ? -INFINITY : cr[1];
  l_r[0] = 0.25f * lacc[0];
  l_r[1] = 0.25f * lacc[2];
#endif
"""),
  ('after', r"""    for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
  }
  }""", r"""
  if constexpr (XB) {
    if (xon) {
      float sxa[4];
      const uint64_t qxd = desc_sw64(smem_u32(S.q8s) + hk * 4096);
      const uint64_t kxd = desc_sw64(smem_u32(xr) + hk * 512);
      wgmma_fence();
      wgmma_m64n8k32_e4m3_ss_z(sxa, qxd, kxd);
      wgmma_m64n8k32_e4m3_ss(sxa, qxd + 2, kxd + 2);
      wgmma_commit();
      uint8_t* const xvt = S.kvq + hk * 4608;
      {
        const int wt = tid & 127, n = wt >> 1, h = wt & 1;
        uint4 w = make_uint4(0u, 0u, 0u, 0u);
        if (h == 0) {
          const uint8_t* vc = xr + 1024 + hk * 64 + n;
          w.x = (uint32_t)vc[0] | ((uint32_t)vc[ROWB] << 8);
          w.y = (uint32_t)vc[2 * ROWB] | ((uint32_t)vc[3 * ROWB] << 8);
          w.z = (uint32_t)vc[4 * ROWB] | ((uint32_t)vc[5 * ROWB] << 8);
          w.w = (uint32_t)vc[6 * ROWB] | ((uint32_t)vc[7 * ROWB] << 8);
        }
        *reinterpret_cast<uint4*>(xvt + n * 64 + ((h ^ ((n >> 1) & 3)) << 4)) = w;
      }
      fence_proxy_async();
      wgsync(wgi);
      wgmma_wait<0>();
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(sxa[e]);
      float sx[4];
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int vlo = seq + 2 * wsub + hf - p.window + 1;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = 2 * tig + e;
          sx[2 * hf + e] = (col < xk && start - xk + col >= vlo) ? sxa[2 * hf + e] * ssc : -INFINITY;
        }
      }
      const float xlo = quad_max(fmaxf(sx[0], sx[1])), xhi = quad_max(fmaxf(sx[2], sx[3]));
      const float c0 = first ? xlo : cr[0], c1 = first ? xhi : cr[1];
      const float nlo = (xlo > c0 + 8.f) ? xlo : c0, nhi = (xhi > c1 + 8.f) ? xhi : c1;
      if (__any_sync(0xffffffffu, nlo != c0 || nhi != c1)) {
        const float alo = ex2(c0 - nlo), ahi = ex2(c1 - nhi);
#pragma unroll
        for (int t = 0; t < 8; ++t) { o[0][t][0] *= alo; o[0][t][1] *= alo; o[0][t][2] *= ahi; o[0][t][3] *= ahi; }
        lacc[0] *= alo; lacc[1] *= alo; lacc[2] *= ahi; lacc[3] *= ahi;
      }
      cr[0] = nlo; cr[1] = nhi;
      uint32_t px[4];
      px[0] = e4m3x4f(ex2(sx[0] - nlo), ex2(sx[1] - nlo), 0.f, 0.f);
      px[1] = e4m3x4f(ex2(sx[2] - nhi), ex2(sx[3] - nhi), 0.f, 0.f);
      px[2] = 0u;
      px[3] = 0u;
      float* od = &o[0][0][0];
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(px[e]));
      wgmma_fence();
      wgmma_m64n72k32_e4m3(od, lacc, px, desc_sw64(smem_u32(xvt)));
      wgmma_commit();
      wgmma_wait<0>();
#pragma unroll
      for (int e = 0; e < 32; ++e) reg_fence(od[e]);
#pragma unroll
      for (int e = 0; e < 4; ++e) reg_fence(lacc[e]);
      first = false;
    }
    pdl_launch_dependents();
  }
  if constexpr (LX) lx_fs = 1.f / (float)(1 << LDD_LX);"""),
  ('after', r"""    }
  }
  PROF_DUMP;""", r"""
  if constexpr (!XB) pdl_launch_dependents();"""),
  ('after', r"""      ++cnt2;
    }""", r"""
#if OKC_YIELD > 0
    if constexpr (FLAT && WS) {
      if (blockIdx.x * 2 < gridDim.x) __nanosleep(OKC_YIELD);
    }
#endif"""),
  ('after', r"""    PROF_MARK(5);""", r"""
#endif"""),
  ('before', r"""    }
    PROF_MARK(5);""", r"""#endif
"""),
  ('after', r"""      }
      }
    } else {""", r"""
#if !(OKC_PVW)"""),
  ('after', r"""#undef EXQ""", r"""
#if OKC_XP_NOEXP
#undef ex2
#endif"""),
  (r"""#define EXQ(x, q) ex2(x)""", r"""#if !(OKC_EXPPOLY > 0)
#define EXQ(x, q) ex2(x)
#endif"""),
  ('before', r"""            pack2();""", r"""            if constexpr (LX) { lxb[0] = LXB0 - nlo * 0.0625f; lxb[1] = LXB0 - nhi * 0.0625f; }
"""),
  ('after', r"""        auto pack2 = [&]() {""", r"""
          if constexpr (LX) {
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
              const float* sv = s + 16 * kk;
              pa[kk][0] = lx_p4(lxf_b2(sv[0], sv[1], lxa, lxb[0]), lxf_b2(sv[4], sv[5], lxa, lxb[0]));
              pa[kk][1] = lx_p4(lxf_b2(sv[2], sv[3], lxa, lxb[1]), lxf_b2(sv[6], sv[7], lxa, lxb[1]));
              pa[kk][2] = lx_p4(lxf_b2(sv[8], sv[9], lxa, lxb[0]), lxf_b2(sv[12], sv[13], lxa, lxb[0]));
              pa[kk][3] = lx_p4(lxf_b2(sv[10], sv[11], lxa, lxb[1]), lxf_b2(sv[14], sv[15], lxa, lxb[1]));
            }
            return;
          }"""),
  (r"""    }
    if constexpr (PV8) {""", r"""    }
#if !(OKC_PROF && !OKC_PROFPV8)
    if constexpr (PV8) {"""),
  ('before', r"""        cr[0] = nlo; cr[1] = nhi;
        dlo = nlo; dhi = nhi;""", r"""#endif
"""),
  ('after', r"""        const float nlo = fmaxf(mlo, cr[0]), nhi = fmaxf(mhi, cr[1]);
        const float alo = ex2(cr[0] - nlo), ahi = ex2(cr[1] - nhi);""", r"""
#if !OKC_XP_NORESC"""),
  ('after', r"""      cr[0] = dlo; cr[1] = dhi;""", r"""
      if constexpr (LX) { lxb[0] = LXB0 - dlo * 0.0625f; lxb[1] = LXB0 - dhi * 0.0625f; }"""),
  ('before', r"""    float dlo, dhi;""", r"""#else
    mlo = quad_max(mlo) * ssc;
    mhi = quad_max(mhi) * ssc;
#endif
"""),
  ('before', r"""    if (!OKC_RAWTHR && !chk) {""", r"""#if OKC_LAZYMAX
"""),
  (r"""            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;""", r"""#if !(OKC_XP_MASKRO)
            if (pos < vlo || pos > vhi) s[4 * nt + 2 * hf + e] = -INFINITY;
#endif"""),
  ('before', r"""    } else if constexpr (PV8 && OKC_QSS) {""", r"""#endif
"""),
  ('after', r"""    if constexpr (PV8 && OKC_QSS && ADDR) {""", r"""
#if OKC_DSHFL"""),
  ('after', r"""      }
      PROF_MARK(1);
    }""", r"""
    if constexpr (HS) { if (hk != hsel) continue; }"""),
  ('after', r"""      if constexpr (!OKC_XNOWAIT) cp_wait<STAGES - 2>();""", r"""
#if OKC_TMA
      if (p.tma) mbar_wait(&S.mbar[i % STAGES], (i / STAGES) & 1);
#endif"""),
  ('before', r"""  const uint32_t vsrc0 = smem_u32(S.v);""", r"""#else
  const uint64_t qdesc0 = desc_sw64(smem_u32(S.q8s) + hk * 4096);
  const uint64_t kdesc0 = desc_sw128(smem_u32(S.k) + hk * 64);
#endif
"""),
  ('after', r"""  uint32_t ph_c = (uint32_t)(tb / STAGES) & 1u;""", r"""
  constexpr bool LX = LDD_LX && PV8 && WS && FLAT && OKC_QSS && ADDR && OKC_PK2 && OKC_PCHK && OKC_LAZYMAX && !OKC_RAWTHR &&
                      !OKC_EXH8 && !OKC_PROF && !PTR && PVSPLIT && OKC_PVSPLIT == 1 && !OKC_XP_NOEXP && OKC_EXPPOLY == 0 &&
                      !(OKC_TURN && WS && PV8 && !S2 && !F2 && !OKC_PROF) && !OKC_PCHKTEST;
  constexpr float LXB0 = (56.f + 8.f * LDD_LX) / 128.f;
  const float lxa = ssc * 0.0625f;
  float lxb[2] = {0.f, 0.f};
#if OKC_DSHFL"""),
  (r"""  PROF_DUMP;
  } else {""", r"""  PROF_DUMP;
  pdl_launch_dependents();
  } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('before', r"""#pragma unroll
      for (int ks = 0; ks < (OKC_XHALFPV ? 1 : 2); ++ks) {""", r"""#endif
"""),
  ('before', r"""    mx = quad_max(mx);""", r"""#if !(OKC_LAZY1)
"""),
  ('after', r"""      cp_commit();
      PROF_MARK(1);
    }""", r"""
#if OKC_NOCOMPUTE
    if constexpr (WS) { __syncwarp(); if (lane == 0) mbar_arrive(&S.empty[(tb + i) % STAGES]); }
    continue;
#endif"""),
  ('after', r"""      cp_wait<STAGES - 2>();""", r"""
#if OKC_TMA
      if (p.tma) mbar_wait(&S.mbar[i % STAGES], (i / STAGES) & 1);
#endif"""),
  ('after', r"""  float m_r[NROW], l_r[NROW];""", r"""
  float lx_fs = 1.f;"""),
  (r"""    }
  } else
#pragma unroll""", r"""    }
  } else
#if OKC_QROLL
#pragma unroll 1
#else
#pragma unroll
#endif"""),
  ('before', r"""      S.cs[e] = cv;""", r"""#else
      sincosf(ang, &sv, &cv);
#endif
"""),
  ('after', r"""      float sv, cv;""", r"""
#if OKC_FASTSC"""),
  (r"""    if (!one) {
#pragma unroll""", r"""#endif
    if (!one) {
#if OKC_PREUNR
#pragma unroll
#else
#pragma unroll 1
#endif"""),
  ('after', r"""    bool one = false;""", r"""
#if OKC_ONETILE && OKC_OFF32 && !OKC_TMA"""),
  ('after', r"""  DBG(1);""", r"""
  if constexpr (XB) {
    if (xon && tid < 128) {
      const int r = (tid >> 3) & 7, c = tid & 7;
      const bool isv = tid >= 64;
      const int pos = start - xk + r;
      const bool ok = r < xk;
      const uint8_t* base = isv ? p.v_cache : p.k_cache;
      const int64_t off = (((int64_t)reinterpret_cast<const int*>(xr + 2048)[r] << p.lp) + (pos & (p.P - 1))) * p.row_bytes +
                          pair * 128 + c * 16;
      const uint32_t dst = isv ? smem_u32(xr + 1024 + r * ROWB + c * 16)
                               : smem_u32(xr + (c >> 2) * 512 + r * 64 + (((c & 3) ^ ((r >> 1) & 3)) << 4));
      cp_async16(dst, ok ? base + off : base, ok ? 16 : 0);
    }
  }"""),
  (r"""  pdl_wait();
  pdl_launch_dependents();""", r"""  pdl_wait();"""),
  ('after', r"""    for (int y = 0; y < STAGES - 1; ++y) issue_idx(y);""", r"""
  }
  if constexpr (XB) {
    if (xon && (tid & 1) && tid < 16) {
      const int r = tid >> 1, pos = start - xk + r;
      cp_async4z(smem_u32(xr + 2048 + 4 * r), r < xk ? pt_row + (pos >> p.lp) : pt_row, r < xk ? 4 : 0);
    }"""),
  ('before', r"""  int tidL = tid, partL = part;""", r"""#if OKC_TMA
  if (p.tma && tid == 0) {
    for (int st = 0; st < STAGES; ++st) mbar_init(&S.mbar[st]);
    mbar_fence_init();
  }
#endif
"""),
  ('before', r"""  };
  auto ws_patch = [&](int i, int st) {""", r"""#endif
"""),
  ('after', r"""  auto issue_idx = [&](int y) {""", r"""
#if !(OKC_BRFREE)"""),
  (r"""        } else
        {""", r"""        } else
#endif
        {"""),
  ('after', r"""      if (pos < end && pos < seq) {""", r"""
#if OKC_TMA
        if (p.tma) continue;
#endif
#if OKC_OFF32"""),
  ('before', r"""#pragma unroll
    for (int k = 0; k < TILE / RPP; ++k) {""", r"""#endif
"""),
  ('after', r"""    bool generic = false;""", r"""
#if OKC_PFASTPR && OKC_OFF32 && !OKC_TMA"""),
  (r"""            } else
            {""", r"""            } else
#endif
            {"""),
  ('before', r"""            if (p.off32) {""", r"""#if OKC_OFF32
"""),
  ('after', r"""          }
        } else""", r"""
#endif"""),
  ('before', r"""        if (pfast && start + x * TILE + TILE <= cache_end) {""", r"""#if OKC_PFAST
"""),
  ('before', r"""        }
        PROF_MARK(2);""", r"""          if constexpr (LDD_PFD > 0) {
            if (q < 3 && x + LDD_PFD < ntiles) {
              const int pp = min(start + (x + LDD_PFD) * TILE + (q == 0 ? 0 : (q == 1 ? 32 : 63)), cache_end - 1);
              if (pp >= start) asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr + (pp >> p.lp)));
            }
          }
"""),
  ('after', r"""      const uint32_t pso = r0 * ROWB + ((ch ^ (r0 & 7)) << 4);""", r"""
#endif"""),
  ('after', r"""      int pg[PK];""", r"""
#if OKC_PFAST"""),
  ('before', r"""  if constexpr (WS) {
    constexpr int NP = Cfg<L, FLAT>::NP;""", r"""  constexpr bool XB = ETX_OK(L, FLAT, MODE);
  constexpr bool HS = !FLAT && L == 8 && (MODE & 32) != 0;
  const int hsel = HS ? (int)(blockIdx.z & 1) : 0;
  const int xk = XB ? xk_ : 0;
  const bool xon = XB && first && xk > 0;
  uint8_t* const xr = reinterpret_cast<uint8_t*>(S.gst) + ((512u - (smem_u32(S.gst) & 511u)) & 511u);
"""),
  (r"""                                        const bool skip_init_ = false) {
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)""", r"""                                        const bool skip_init_ = false, const int xk_ = 0) {
#if OKC_UARGS
#define UNI(x) __shfl_sync(0xffffffffu, (x), 0)
#endif"""),
  (r"""        lv[q] = lrow ? cl_ld1(p, PL, q, row0 + (e >> 4), REGION) : 0.f;""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        lv[q] = lrow ? cl_ld1(p, PL, q, row0 + (e >> 4), REGION) : 0.f;
#else
        lv[q] = lrow ? cl_ld1(p, PL, q, L * 16 * 64 + row0 + (e >> 4), REGION) : 0.f;
#endif"""),
  ('after', r"""  if (false)""", r"""
#endif"""),
  (r"""        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, row0 + (e >> 4), REGION) : 0.f;
#else
        lv[u] = lrow ? cl_ld1(p, PL, q0 + u, L * 16 * 64 + row0 + (e >> 4), REGION) : 0.f;
#endif"""),
  ('after', r"""  constexpr int CSMAX = 16;""", r"""
#if OKC_CLR4"""),
  (r"""    }
  } else
  {""", r"""    }
  } else
#endif
  {"""),
  (r"""          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
          else v[u].x = cl_ld1(p, PL, q, row0 + rl, REGION);
#else
          else v[u].x = cl_ld1(p, PL, q, L * 16 * 64 + row0 + rl, REGION);
#endif"""),
  ('after', r"""  const int row0 = rank * RPC;""", r"""
#if OKC_CLR2"""),
  ('before', r"""  if (OKC_CLDBG) DBG(7);""", r"""#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  {
    float* g = p.scratch + ((int64_t)cl_gid(p) * CS + rank) * REGION;
    for (int e = tid; e < L * 16 * 64; e += NT) g[e] = PB[e];
    for (int e = tid; e < L * 16; e += NT) g[L * 16 * 64 + e] = PL[e];
    __threadfence();
  }
#endif
"""),
  ('after', r"""  alignas(1024) uint8_t q8s[(L == 8 && PV8ON && OKC_QSS && (!F || OKC_PV8FLAT)) ? 8192 : 16];""", r"""
#else
  uint8_t kvq[(L == 1) ? L * 16 * 64 * 2 : 32768];
#endif"""),
  ('after', r"""  int idx[2 * Cfg<L, F>::STAGES * TILE];""", r"""
#if OKC_SM90"""),
  ('before', r"""}
template <int L, bool F>""", r"""#else
  return __ldcg(p.scratch + ((int64_t)cl_gid(p) * p.cs + rank) * region_floats + off_floats);
#endif
"""),
  ('after', r"""__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('before', r"""}
__device__ __forceinline__ float cl_ld1(const Params& p, const float* local, int rank, int off_floats, int region_floats) {""", r"""#else
  const float* g = p.scratch + ((int64_t)cl_gid(p) * p.cs + rank) * region_floats + off_floats;
  return __ldcg(reinterpret_cast<const float4*>(g));
#endif
"""),
  ('after', r"""__device__ __forceinline__ float4 cl_ld4(const Params& p, const float* local, int rank, int off_floats, int region_floats) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('after', r"""__device__ __forceinline__ void cl_done(const Params& p) {""", r"""
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  if (threadIdx.x == 0) {
    int* bar = p.bar + cl_gid(p);
    if (atomicAdd(bar, 1) == 3 * p.cs - 1) *bar = 0;
  }
#endif"""),
  (r"""  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
#else
  __syncthreads();
  if (threadIdx.x == 0) {
    int* bar = p.bar + cl_gid(p);
    __threadfence();
    atomicAdd(bar, 1);
    const int target = phase * p.cs;
    SPIN_T0;
    while (true) {
      int v;
      asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(bar));
      if (v >= target) break;
      SPIN_CHECK;
    }
    __threadfence();
  }
  __syncthreads();
#endif"""),
  ('after', r"""  return (int)r;""", r"""
#else
  return blockIdx.x % p.cs;
#endif
}
__device__ __forceinline__ int cl_gid(const Params& p) {
  return (blockIdx.z * gridDim.y + blockIdx.y) * (gridDim.x / p.cs) + blockIdx.x / p.cs;"""),
  ('before', r"""  uint32_t r;""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  (r"""  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(a), "f"(x), "f"(y) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};" ::"l"(a), "f"(x), "f"(y) : "memory");
#else
  red_f32(a, x); red_f32(a + 1, y);
#endif"""),
  ('after', r"""  return (uint32_t)lo | ((uint32_t)hi << 16);""", r"""
#endif
}
__device__ __forceinline__ void wgmma_m64n64k16_f16_tb(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                       uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
      "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
      "{%32, %33, %34, %35}, %36, 1, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
#endif"""),
  ('before', r"""  uint16_t lo, hi;""", r"""#if !(OKC_PACK4)
"""),
  (r"""  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
typedef int ix_t;
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !OKC_PDLLATE
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
#if OKC_OFF32
typedef int ix_t;
#endif
__device__ __forceinline__ const uint8_t* cache_row(const uint8_t* base, const Params& p, int64_t row, int col_bytes) {
#if OKC_OFF32
  if (p.off32) return base + (((uint32_t)row << p.lrb) + (uint32_t)col_bytes);
#endif"""),
  (r"""  asm volatile("griddepcontrol.wait;" ::: "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif"""),
  (r"""__device__ __forceinline__ float sigm(float x) {
  return __fdividef(1.f, 1.f + __expf(-x));""", r"""__device__ __forceinline__ uint32_t ex2_e4m3x4_h(float x0, float x1, float x2, float x3) {
  uint32_t a, b, ea, eb;
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(a) : "f"(x0), "f"(x1));
  asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(b) : "f"(x2), "f"(x3));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(ea) : "r"(a));
  asm("ex2.approx.f16x2 %0, %1;" : "=r"(eb) : "r"(b));
  uint16_t lo, hi;
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(ea));
  asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(eb));
  return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ float sigm(float x) {
#if !(OKC_TANHSIG)
  return __fdividef(1.f, 1.f + __expf(-x));
#endif
}
__device__ __forceinline__ float ex2_poly(float x) {
  x = fmaxf(x, -127.f);
  const float j = floorf(x);
  const float f = x - j;
  float p = fmaf(f, 0.0790209f, 0.2249424f);
  p = fmaf(f, p, 0.6960656f);
  p = fmaf(f, p, 0.9999893f);
  return __int_as_float(__float_as_int(p) + ((int)j << 23));"""),
  ('before', r"""}
__device__ __forceinline__ int fdiv_i(int x, int d) {""", r"""#endif
"""),
  ('before', r'''  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('after', r"""      : "l"(da), "l"(db), "r"(0));""", r"""
#endif
}
#define ETX_OK(L_, F_, MODE_) ((L_) == 8 && !(F_) && (((MODE_) & 16) != 0) && PV8ON && OKC_QSS && OKC_SM90 && \
                               !OKC_S2 && OKC_PVSPLIT == 1 && !OKC_PROF)
__device__ __forceinline__ void wgmma_m64n8k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, p, 1, 1;\n}\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "l"(da), "l"(db), "r"(0));
#endif
}
__device__ __forceinline__ void wgmma_m64n8k32_e4m3_ss(float* d, uint64_t da, uint64_t db) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, %4, %5, 1, 1, 1;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "l"(da), "l"(db));
#endif"""),
  ('after', r"""__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_z(float* d, uint64_t da, uint64_t db) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('after', r"""      : "l"(da), "l"(db), "r"(scale_d));""", r"""
#endif
}
__device__ __forceinline__ uint32_t lx_fma2_sat(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.sat.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t lx_fma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}
__device__ __forceinline__ uint32_t lx_max2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ float lx_hmax(uint32_t a) {
  float x, y;
  asm("{\n.reg .b16 l, h;\nmov.b32 {l, h}, %2;\ncvt.f32.f16 %0, l;\ncvt.f32.f16 %1, h;\n}" : "=f"(x), "=f"(y) : "r"(a));
  return fmaxf(x, y);
}
__device__ __forceinline__ uint32_t lx_h2(float a) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(d) : "f"(a));
  return d;
}
__device__ __forceinline__ uint32_t lx_b2(uint32_t s2, uint32_t a2, uint32_t b2) {
  return lx_fma2(lx_fma2_sat(s2, a2, b2), 0x58005800u, 0x64006400u);
}
__device__ __forceinline__ uint32_t lx_p4(uint32_t p, uint32_t q) { return prmt(p, q, 0x6420u); }
__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss_h(uint32_t* d, uint64_t da, uint64_t db, int scale_d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
      "%16, %17, p, 1, 1;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]),
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15])
      : "l"(da), "l"(db), "r"(scale_d));
#endif
}
__device__ __forceinline__ float lxf_sat(float s, float a, float b) {
  float d;
  asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(s), "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ uint32_t lxf_b2(float s0, float s1, float a, float b) {
  uint32_t h;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(lxf_sat(s1, a, b)), "f"(lxf_sat(s0, a, b)));
  return lx_fma2(h, 0x58005800u, 0x64006400u);"""),
  ('after', r"""__device__ __forceinline__ void wgmma_m64n64k32_e4m3_ss(float* d, uint64_t da, uint64_t db, int scale_d) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('after', r"""      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));""", r"""
#endif"""),
  ('before', r'''  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  (r"""  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif"""),
  (r"""  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif"""),
  (r"""  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif"""),
  (r"""__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");""", r"""#endif
__device__ __forceinline__ void fence_proxy_async() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif"""),
  (r"""}
#define PROF_INIT""", r"""#endif
}
#if !(OKC_PROF)
#define PROF_INIT"""),
  ('after', r"""__device__ __forceinline__ void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('after', r"""    p.dbg[(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k)] = gtime();""", r"""
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}"""),
  (r"""  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
#else
  red_f32(a, x); red_f32(a + 1, y); red_f32(a + 2, z); red_f32(a + 3, w);
#endif"""),
  (r"""  __threadfence();
}""", r"""#if OKC_FENCEAR
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
  __threadfence();
#endif
}"""),
  ('after', r"""      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));""", r"""
}
__device__ __forceinline__ void mma_fp8(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  mma_f16(c, cvt2(a0), cvt2(a1), cvt2(a0 >> 16), cvt2(a1 >> 16), cvt2(b0), cvt2(b0 >> 16));
  mma_f16(c, cvt2(a2), cvt2(a3), cvt2(a2 >> 16), cvt2(a3 >> 16), cvt2(b1), cvt2(b1 >> 16));
#endif"""),
  (r"""  return x;""", r"""#if !(OKC_PROBE_O8)
  return x;
#endif"""),
  (r"""  return pack_h2(ex2(lo), ex2(hi));""", r"""#if !(OKC_EXH2)
  return pack_h2(ex2(lo), ex2(hi));
#endif"""),
  ('before', r"""}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {""", r"""#else
  const uint32_t h = prmt(x, 0u, 0x7170);
  const uint32_t bits = ((h & 0x007f007fu) << 7) | ((h & 0x00800080u) << 8);
  __half2 v = *reinterpret_cast<const __half2*>(&bits);
  v = __hmul2(v, __floats2half2_rn(256.f, 256.f));
  return *reinterpret_cast<uint32_t*>(&v);
#endif
"""),
  ('after', r"""__device__ __forceinline__ uint32_t cvt2(uint32_t x) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890"""),
  (r"""  else __syncthreads();
}""", r"""  else __syncthreads();
}
__device__ __forceinline__ void tma_row(void* dst, const CUtensorMap* map, int x, int y, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(x), "r"(y), "r"(smem_u32(bar)) : "memory");
#endif
}"""),
  ('before', r"""}
__device__ __forceinline__ void cp_async16_zero(uint32_t dst, const void* any_src) {""", r"""#endif
}
__device__ __forceinline__ void tma_load_1d(void* dst, const CUtensorMap* map, int c0, uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];\n"
               ::"r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(smem_u32(bar)) : "memory");
#endif
"""),
  ('after', r"""__device__ __forceinline__ void bulk_load(void* dst, const void* src, int bytes, uint64_t* bar) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  (r"""  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif"""),
  ('after', r"""               ::"r"(smem_u32(bar)), "r"(parity) : "memory");""", r"""
#endif"""),
  ('after', r"""__device__ __forceinline__ void mbar_wait_full(uint64_t* bar, int parity) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  (r"""  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif"""),
  (r"""  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count) : "memory");
#endif"""),
  (r"""  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
#endif
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, int parity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  for (int spin = 0; spin < (1 << 16); ++spin) {
    uint32_t ok;
    asm volatile("{\n.reg .pred q;\nmbarrier.try_wait.parity.shared::cta.b64 q, [%1], %2;\nselp.u32 %0, 1, 0, q;\n}\n"
                 : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
    if (ok) break;
  }
#endif"""),
  (r"""#define OKC_TMA 0
__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");""", r"""#ifndef OKC_TMA
#define OKC_TMA 0
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" ::"r"(smem_u32(bar)) : "memory");
#endif
}
__device__ __forceinline__ void mbar_fence_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif"""),
  ('after', r"""  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}""", r"""
__device__ __forceinline__ void cp_async4z(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}"""),
  (r"""  const uint8_t* pf;
  unsigned pf_chunk;
};""", r"""};"""),
  (r"""#define OKC_ST1F 3
#define OKC_STN 3
#define OKC_UWARP 1
#define OKC_INEW 0
#define OKC_ADDR 1
#define OKC_STSM 0
#define OKC_STSMPR 1
#define OKC_TCHUNK 1
#define OKC_EXPPOLY 0
#define OKC_EXH8 0
#define OKC_NOSYNC0 1
#define OKC_LAZYMAX 1
#define OKC_SENSE 0
#define OKC_WSPLIT 500
#define OKC_F2 0
#define OKC_S2 0
#define OKC_PTR 0
#define OKC_PTRD 2
#define OKC_XP_NOTR 0
#define OKC_XP_NOFENCE 0
#define OKC_XP_NOEXP 0
#define OKC_CLDBG 0
#define OKC_SPINDBG 0""", r"""#ifndef OKC_PK2
#define OKC_PK2 1
#endif
#ifndef OKC_YIELD
#define OKC_YIELD 0
#endif
#ifndef OKC_ST1F
#define OKC_ST1F 3
#endif
#ifndef OKC_STN
#define OKC_STN 3
#endif
#ifndef OKC_PACK4
#define OKC_PACK4 0
#endif
#ifndef OKC_UARGS
#define OKC_UARGS 1
#endif
#ifndef OKC_UWARP
#define OKC_UWARP 1
#endif
#ifndef OKC_INEW
#define OKC_INEW 0
#endif
#ifndef OKC_ADDR
#define OKC_ADDR 1
#endif
#ifndef OKC_FASTSC
#define OKC_FASTSC 1
#endif
#ifndef OKC_STSM
#define OKC_STSM 0
#endif
#ifndef OKC_STSMPR
#define OKC_STSMPR 1
#endif
#ifndef OKC_TCHUNK
#define OKC_TCHUNK 1
#endif
#ifndef OKC_EXPPOLY
#define OKC_EXPPOLY 0
#endif
#ifndef OKC_EXH8
#define OKC_EXH8 0
#endif
#ifndef OKC_LAZY1
#define OKC_LAZY1 0
#endif
#ifndef OKC_NOSYNC0
#define OKC_NOSYNC0 1
#endif
#ifndef OKC_LAZYMAX
#define OKC_LAZYMAX 1
#endif
#ifndef OKC_QROLL
#define OKC_QROLL 0
#endif
#ifndef OKC_CLR4
#define OKC_CLR4 1
#endif
#ifndef OKC_SENSE
#define OKC_SENSE 0
#endif
#ifndef OKC_WSPLIT
#define OKC_WSPLIT 500
#endif
#ifndef OKC_F2
#define OKC_F2 0
#endif
#ifndef OKC_S2
#define OKC_S2 0
#endif
#ifndef OKC_PTR
#define OKC_PTR 0
#endif
#ifndef OKC_PTRD
#define OKC_PTRD 2
#endif
#ifndef OKC_XP_NOTR
#define OKC_XP_NOTR 0
#endif
#ifndef OKC_XP_NOFENCE
#define OKC_XP_NOFENCE 0
#endif
#ifndef OKC_XP_NOEXP
#define OKC_XP_NOEXP 0
#endif
#ifndef OKC_CLDBG
#define OKC_CLDBG 0
#endif
#ifndef OKC_SPINDBG
#define OKC_SPINDBG 0
#endif"""),
  (r"""#define OKC_NP 128
#define OKC_WS 1
#define OKC_WS8 1
#define OKC_NP8 128
#define OKC_WSMINB 2
#define OKC_SETMAXNREG 1
#define OKC_PREGS 0
#define OKC_TMAPG 1
#define OKC_XHALFPV 0
#define OKC_XNOWAIT 0
#define OKC_MRGDBG 0
#define OKC_MERGE2 1
#define OKC_MERGE3 1
#define OKC_IDXFIRST 1
#define OKC_MAGICDIV 1
#define OKC_SC1 1
#define OKC_QP16 1
#define OKC_ONENEW 1
#define OKC_SC8 1
#define OKC_PINDIV 1
#define OKC_ONEPRED 1
#define OKC_LATETID 1
#define OKC_PGCONT 0
#define OKC_TMAPG8 0
#define OKC_ST8F 4
#define OKC_COOP 1
#define OKC_PVW 0
#define OKC_WGDEC 1
#define OKC_SEGDBG 0
#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)
#define OKC_PV8 1
#define OKC_PROF 0
#define OKC_PROFPV8 0
#define PV8ON (OKC_PV8 && (!OKC_PROF || OKC_PROFPV8) && !OKC_PVW)
#define OKC_PV8FLAT 1
#define OKC_QSS 1
#define OKC_PV8SYNC 0
#define OKC_Q8RAW 1
#define OKC_FLATCOOP1C 0
#define OKC_RAWTHR 0
#define OKC_PCHK 1
#define OKC_TURN 0
#define OKC_CLSM 0
#define OKC_PDLKV 0
#define OKC_RARE 1
#define OKC_PCHKTEST 0
#define OKC_UNEV 0
#define OKC_W0L1 0
#define OKC_SEGK8 600
#define OKC_SEGK1 400
#define OKC_W0SPREAD 0
#define OKC_W0EARLY 1
#define OKC_W0FORCEBAD 0
#define OKC_W0BID 1
#define OKC_DBGON 0
#define OKC_BALPART 0
#define OKC_CONT 0
#define OKC_DYN 0
#define OKC_FINWARM 0
#define OKC_PVW1 1
#define OKC_NOXCH1 0
#define OKC_PVSPLIT 1
#define OKC_QKZ 0
#define OKC_INV2 1
#define OKC_OZF 1
#define OKC_XP_ARRALL 1
#define OKC_XP_NOMASK 0
#define OKC_IRANGE 0
#define OKC_PATCHNI 1
#define OKC_PARITY 1
#define OKC_SENSE2 0
#define OKC_UNRN 1
#define OKC_ADDRN 1
#define OKC_OFF32 1
#define OKC_UNR8 2
#define OKC_RED4 1
#define OKC_SWAPHALF 0
#define OKC_LATESYNC 1
#define OKC_LATESYNCN 1
#define OKC_QKFIRST 1
#define OKC_FPART 1
#define OKC_EARLYBAR 1
#define OKC_PTPF 1
#define OKC_FPRO 0
#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)
#define OKC_COOPGUARD 1""", r"""#ifndef LDD_PFD
#define LDD_PFD 4
#endif
#ifndef LDD_LX
#define LDD_LX 4
#endif
#ifndef OKC_NP
#define OKC_NP 128
#endif
#ifndef OKC_WS
#define OKC_WS 1
#endif
#ifndef OKC_WS8
#define OKC_WS8 1
#endif
#ifndef OKC_NP8
#define OKC_NP8 128
#endif
#ifndef OKC_WSMINB
#define OKC_WSMINB 2
#endif
#ifndef OKC_SETMAXNREG
#define OKC_SETMAXNREG 1
#endif
#ifndef OKC_NOCOMPUTE
#define OKC_NOCOMPUTE 0
#endif
#ifndef OKC_PREGS
#define OKC_PREGS 0
#endif
#ifndef OKC_TMAPG
#define OKC_TMAPG 1
#endif
#ifndef OKC_XHALFPV
#define OKC_XHALFPV 0
#endif
#ifndef OKC_XNOWAIT
#define OKC_XNOWAIT 0
#endif
#ifndef OKC_FENCEAR
#define OKC_FENCEAR 0
#endif
#ifndef OKC_MRGDBG
#define OKC_MRGDBG 0
#endif
#ifndef OKC_MERGE2
#define OKC_MERGE2 1
#endif
#ifndef OKC_MERGE3
#define OKC_MERGE3 1
#endif
#ifndef OKC_IDXFIRST
#define OKC_IDXFIRST 1
#endif
#ifndef OKC_ONETILE
#define OKC_ONETILE 1
#endif
#ifndef OKC_MAGICDIV
#define OKC_MAGICDIV 1
#endif
#ifndef OKC_SC1
#define OKC_SC1 1
#endif
#ifndef OKC_ONE
#define OKC_ONE 1
#endif
#ifndef OKC_QDIRECT
#define OKC_QDIRECT 1
#endif
#ifndef OKC_SPEC
#define OKC_SPEC 1
#endif
#ifndef OKC_PDLLATE
#define OKC_PDLLATE 0
#endif
#ifndef OKC_QP16
#define OKC_QP16 1
#endif
#ifndef OKC_ONENEW
#define OKC_ONENEW 1
#endif
#ifndef OKC_SC8
#define OKC_SC8 1
#endif
#ifndef OKC_PINDIV
#define OKC_PINDIV 1
#endif
#ifndef OKC_ONEPRED
#define OKC_ONEPRED 1
#endif
#ifndef OKC_LATETID
#define OKC_LATETID 1
#endif
#ifndef OKC_PFASTPR
#define OKC_PFASTPR 1
#endif
#ifndef OKC_PFAST
#define OKC_PFAST 1
#endif
#ifndef OKC_PGCONT
#define OKC_PGCONT 0
#endif
#ifndef OKC_TMAPG8
#define OKC_TMAPG8 0
#endif
#ifndef OKC_ST8F
#define OKC_ST8F 4
#endif
#ifndef OKC_COOP
#define OKC_COOP 1
#endif
#ifndef OKC_PVW
#define OKC_PVW 0
#endif
#ifndef OKC_WGDEC
#define OKC_WGDEC 1
#endif
#ifndef OKC_SEGDBG
#define OKC_SEGDBG 0
#endif
#define DBGSTRIDE (OKC_SEGDBG ? 64 : 16)
#ifndef OKC_EXH2
#define OKC_EXH2 0
#endif
#ifndef OKC_PV8
#define OKC_PV8 1
#endif
#ifndef OKC_PROF
#define OKC_PROF 0
#endif
#ifndef OKC_PROFPV8
#define OKC_PROFPV8 0
#endif
#define PV8ON (OKC_PV8 && (!OKC_PROF || OKC_PROFPV8) && !OKC_PVW)
#ifndef OKC_PV8FLAT
#define OKC_PV8FLAT 1
#endif
#ifndef OKC_QSS
#define OKC_QSS 1
#endif
#ifndef OKC_PV8SYNC
#define OKC_PV8SYNC 0
#endif
#ifndef OKC_Q8RAW
#define OKC_Q8RAW 1
#endif
#ifndef OKC_PROBE_O8
#define OKC_PROBE_O8 0
#endif
#ifndef OKC_FLATCOOP1C
#define OKC_FLATCOOP1C 0
#endif
#ifndef OKC_RAWTHR
#define OKC_RAWTHR 0
#endif
#ifndef OKC_PCHK
#define OKC_PCHK 1
#endif
#ifndef OKC_TURN
#define OKC_TURN 0
#endif
#ifndef OKC_CLSM
#define OKC_CLSM 0
#endif
#ifndef OKC_PDLKV
#define OKC_PDLKV 0
#endif
#ifndef OKC_BRFREE
#define OKC_BRFREE 0
#endif
#ifndef OKC_RARE
#define OKC_RARE 1
#endif
#ifndef OKC_PREUNR
#define OKC_PREUNR 1
#endif
#ifndef OKC_FIN1
#define OKC_FIN1 0
#endif
#ifndef OKC_PCHKTEST
#define OKC_PCHKTEST 0
#endif
#ifndef OKC_UNEV
#define OKC_UNEV 0
#endif
#ifndef OKC_W0L1
#define OKC_W0L1 0
#endif
#ifndef OKC_SEGK8
#define OKC_SEGK8 600
#endif
#ifndef OKC_SEGK1
#define OKC_SEGK1 400
#endif
#ifndef OKC_W0SPREAD
#define OKC_W0SPREAD 0
#endif
#ifndef OKC_W0EARLY
#define OKC_W0EARLY 1
#endif
#ifndef OKC_W0FORCEBAD
#define OKC_W0FORCEBAD 0
#endif
#ifndef OKC_W0BID
#define OKC_W0BID 1
#endif
#ifndef OKC_W0ENABLE
#define OKC_W0ENABLE 0
#endif
#ifndef OKC_CLR2
#define OKC_CLR2 1
#endif
#ifndef OKC_DBGON
#define OKC_DBGON 0
#endif
#ifndef OKC_BALPART
#define OKC_BALPART 0
#endif
#ifndef OKC_CONT
#define OKC_CONT 0
#endif
#ifndef OKC_DYN
#define OKC_DYN 0
#endif
#ifndef OKC_DYNK
#define OKC_DYNK 3
#endif
#ifndef OKC_FINWARM
#define OKC_FINWARM 0
#endif
#ifndef OKC_PVW1
#define OKC_PVW1 1
#endif
#ifndef OKC_NOXCH1
#define OKC_NOXCH1 0
#endif
#ifndef OKC_TANHSIG
#define OKC_TANHSIG 0
#endif
#ifndef OKC_PVSPLIT
#define OKC_PVSPLIT 1
#endif
#ifndef OKC_QKZ
#define OKC_QKZ 0
#endif
#ifndef OKC_XP_NORESC
#define OKC_XP_NORESC 0
#endif
#ifndef OKC_INV2
#define OKC_INV2 1
#endif
#ifndef OKC_OZF
#define OKC_OZF 1
#endif
#ifndef OKC_XP_ARRALL
#define OKC_XP_ARRALL 1
#endif
#ifndef OKC_XP_MASKRO
#define OKC_XP_MASKRO 0
#endif
#ifndef OKC_XP_NOMASK
#define OKC_XP_NOMASK 0
#endif
#ifndef OKC_IRANGE
#define OKC_IRANGE 0
#endif
#ifndef OKC_PATCHNI
#define OKC_PATCHNI 1
#endif
#ifndef OKC_DSHFL
#define OKC_DSHFL 1
#endif
#ifndef OKC_PARITY
#define OKC_PARITY 1
#endif
#ifndef OKC_SENSE2
#define OKC_SENSE2 0
#endif
#ifndef OKC_UNRN
#define OKC_UNRN 1
#endif
#ifndef OKC_ADDRN
#define OKC_ADDRN 1
#endif
#ifndef OKC_OFF32
#define OKC_OFF32 1
#endif
#ifndef OKC_UNR8
#define OKC_UNR8 2
#endif
#ifndef OKC_RED4
#define OKC_RED4 1
#endif
#ifndef OKC_SWAPHALF
#define OKC_SWAPHALF 0
#endif
#ifndef OKC_LATESYNC
#define OKC_LATESYNC 1
#endif
#ifndef OKC_LATESYNCN
#define OKC_LATESYNCN 1
#endif
#ifndef OKC_QKFIRST
#define OKC_QKFIRST 1
#endif
#ifndef OKC_FPART
#define OKC_FPART 1
#endif
#ifndef OKC_EARLYBAR
#define OKC_EARLYBAR 1
#endif
#ifndef OKC_PTPF
#define OKC_PTPF 1
#endif
#ifndef OKC_FPRO
#define OKC_FPRO 0
#endif
#define FPRO_STAMP(k) do { if (OKC_FPRO && p.dbg) { long long* _dp = p.dbg + (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * DBGSTRIDE + (k); if (*_dp == 0) *_dp = gtime(); } } while (0)
#ifndef OKC_COOPGUARD
#define OKC_COOPGUARD 1
#endif"""),
])
ATT7_CPP_SRC = _ldd19_derive(kl_ATT7_CPP_SRC, [
  (r"""         int64_t tma, int64_t pf_ptr, int64_t pf_bytes);""", r"""         int64_t tma);"""),
])
_ATT7 = None


def _att7():
  global _ATT7
  if _ATT7 is None:
    import os
    from torch.utils.cpp_extension import load_inline
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
    _ATT7 = _ldd19_inline(name=f"muse_att_v298v20_{arch}", cpp_sources=ATT7_CPP_SRC, cuda_sources=ATT7_CUDA_SRC,   # LDD19: v20
                        functions=["run"], extra_cuda_cflags=["-O3", "-split-compile=0", f"-gencode=arch=compute_{arch},code=sm_{arch}", f"-DOKC_SM90={1 if major >= 9 else 0}", f"-DLDD_RBF={LDD19_D_RBF}", f"-DLDD_RBFS={LDD19_D_RBFS}", "--expt-relaxed-constexpr"],
                        verbose=False)
  return _ATT7


_WS7 = {}
_NUM_SMS7 = {}
LOG2E = 1.4426950408889634


def _workspace7(n_rows, device):
  key = str(device)
  ws = _WS7.get(key)
  if ws is None or ws[0].numel() < n_rows * 64:
    n = max(n_rows, 1 << 15)
    ws = (torch.zeros(n * 64, dtype=torch.float32, device=device),
          torch.zeros(n, dtype=torch.float32, device=device),
          torch.zeros(n, dtype=torch.int32, device=device))
    _WS7[key] = ws
  return ws


def _num_sms7(device):
  key = str(device)
  if key not in _NUM_SMS7:
    _NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
  return _NUM_SMS7[key]


_TUNED7 = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}


OKC_FLATCOOP = 1
LDD19_ATT = int(__import__("os").environ.get("LDD19_ATT", "3"))   # LDD19: v20: 1=etx16, 2=b1split (opt: 4=d16s_s8, 8=hs_nopdl, 16=etx32)
OKC_PDL = 0
OKC_FLAT1B = 0
OKC_PLAIN8 = 1
OKC_F2 = 0
OKC_S2 = 0
OKC_W0_L1 = 1024
OKC_W0_PR = 1024
OKC_W0_L8 = 1135


def _plan7(B, pairs, L, window, max_pages, P, device):
  sms = _num_sms7(device)
  target = 2 * sms
  if not window and 1 < B <= 512:
    ft = sms if (L == 8 and OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
    return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
  keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
  tiles = max(1, (keys_max + 63) // 64)
  reqs = B * pairs
  if L == 8 and OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
    target = sms
  if not window:
    if L == 1 and B == 1 and OKC_FLAT1B:
      return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
    if L == 8:
      n = max(1, min(tiles, ((target * 21) // 22) // reqs))
      return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
  if L == 1 and reqs < 24:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
  if L == 8 and reqs < 16 and OKC_PLAIN8:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
  if L == 8 and reqs >= 16 and OKC_PLAIN8:
    n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
  if reqs >= 24:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
  cs = 16
  limit = 2 * sms if L == 1 else sms
  if L == 8 and target == sms:
    limit = (sms * 15) // 16
  while cs > 4 and reqs * cs > limit:
    cs //= 2
  ncl = 1
  while cs * ncl > tiles and ncl > 1:
    ncl -= 1
  while cs * ncl > tiles and cs > 4:
    cs //= 2
  if cs * ncl > tiles or cs < 4:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
  return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))


_CL7 = {}


def _cluster_bufs7(n_ctas, L, device):
  key = str(device)
  need = n_ctas * (L * 16 * 64 + 1024)
  cur = _CL7.get(key)
  if cur is None or cur[0].numel() < need or cur[1].numel() < n_ctas:
    cur = (torch.zeros(max(need, 1 << 20), dtype=torch.float32, device=device),
           torch.zeros(max(n_ctas, 1 << 12), dtype=torch.int32, device=device))
    _CL7[key] = cur
  return cur
# LDD19 v24r: _MA_SMALL_CUDA_SRC is derived right after MOE_CUDA_SRC below.


# LDD19 v24: the MoE source is the moe_helper0 base (Keell's copy of the moe agent's kernel) with these exact LDD19 edits.
MOE_CUDA_SRC = _ldd19_derive(kl_MOE_HELPER0_CUDA_SRC, [
  (r"""    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;""", r"""    cudaLaunchAttribute at[2];
    int na = 0;
    if (g_okc_pdl) {
      at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      at[na].val.programmaticStreamSerializationAllowed = 1;
      ++na;
    }
    if (cl2) {
      at[na].id = cudaLaunchAttributeClusterDimension;
      at[na].val.clusterDim.x = 2; at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1;
      ++na;
    }
    cfg.attrs = at;
    cfg.numAttrs = na;"""),
  (r"""  if (g_okc_pdl) {
    cudaLaunchConfig_t cfg = {};""", r"""  bool cl2 = false;
#if OKC_WGS && LDD19_DSMEM
  if (p.M <= 32 && (grid & 1) == 0) {
    static int ok[2] = {-1, -1};
    const int kb = p.M >= OKC_BIGM ? 1 : 0;
    if (ok[kb] < 0) {
      cudaLaunchConfig_t oc = {};
      oc.gridDim = dim3((unsigned)grid);
      oc.blockDim = dim3(okc::NTL);
      oc.dynamicSmemBytes = sm;
      cudaLaunchAttribute oa[1];
      oa[0].id = cudaLaunchAttributeClusterDimension;
      oa[0].val.clusterDim.x = 2; oa[0].val.clusterDim.y = 1; oa[0].val.clusterDim.z = 1;
      oc.attrs = oa;
      oc.numAttrs = 1;
      int ncl = 0;
      const cudaError_t e = kb ? cudaOccupancyMaxActiveClusters(&ncl, (const void*)okc::moe_fused<true>, &oc)
                               : cudaOccupancyMaxActiveClusters(&ncl, (const void*)okc::moe_fused<false>, &oc);
      if (e != cudaSuccess) (void)cudaGetLastError();
      ok[kb] = (e == cudaSuccess && 2 * ncl >= (int)grid) ? 1 : 0;
    }
    cl2 = ok[kb] == 1;
  }
#endif
  if (g_okc_pdl || cl2) {
    cudaLaunchConfig_t cfg = {};"""),
  (r"""    (void)e0; (void)e1;""", r"""#if !(OKC_LAUNCHCHK)
    (void)e0; (void)e1;
#endif"""),
  ('before', r"""#else
  p.xA = p.xB = nullptr;""", r"""#if OKC_MFIN
  p.xready = p.mfl + okc::MF_XR;
  p.doneAs = p.mfl + okc::MF_DA;
#endif
"""),
  ('before', r"""  p.ybuf = p.ss + okc::EMAX;""", r"""#if OKC_MFIN
  p.mfl = ib + 8 + 1024 + okc::EMAX;
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), "MFIN flag region");
#endif
"""),
  ('before', r"""      int qv[8];""", r"""#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
      if (tk == blockIdx.x) DBG(4);
#endif
"""),
  (r"""  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS""", r"""#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
#if OKC_WGS && OKC_WGSFINSS && LDD19_B4FIN
  {
    constexpr int CPT = HD / (B4Q * NT);
    static_assert((CPT == 2 || CPT == 4 || CPT == 8) && HD == B4Q * CPT * NT && NW == 16, "LDD19 b4fin: 2/4/8 columns per thread, 16 warps");
    float* red = reinterpret_cast<float*>(xbuf);
    const float* part = ldd19_b4_part(p);
    const int lane = tid & 31, warp = WARP_IDX(tid), rsl = warp >> 1, rh = warp & 1;
    for (int it = blockIdx.x; it < p.M * B4Q; it += gridDim.x) {
      const int tk = it / B4Q, col = (it % B4Q) * (CPT * NT) + CPT * tid;
      float wv[8];
      float yv[8][CPT];
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int q = S.pos[tk * 8 + s];
        wv[s] = it == (int)blockIdx.x ? b4w[s] : (q >= 0 ? __ldg(p.topk_w + tk * 8 + s) : 0.f);
        const float* ys = p.ybuf + (size_t)(q >= 0 ? q : 0) * HD + col;
        if constexpr (CPT >= 4) {
#pragma unroll
          for (int h = 0; h < CPT / 4; ++h) {
            const float4 a = q >= 0 ? __ldcg(reinterpret_cast<const float4*>(ys) + h) : make_float4(0.f, 0.f, 0.f, 0.f);
            yv[s][4 * h] = a.x; yv[s][4 * h + 1] = a.y; yv[s][4 * h + 2] = a.z; yv[s][4 * h + 3] = a.w;
          }
        } else {
          const float2 a = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(ys)) : make_float2(0.f, 0.f);
          yv[s][0] = a.x; yv[s][1] = a.y;
        }
      }
      const int qr = S.pos[tk * 8 + rsl];
      float v = 0.f;
      if (qr >= 0) {
        const float4 a = __ldcg(reinterpret_cast<const float4*>(part + (size_t)qr * 256 + rh * 128) + lane);
        v = (a.x + a.y) + (a.z + a.w);
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[warp] = v;
      CTA_SYNC();
      float acc[CPT];
#pragma unroll
      for (int c = 0; c < CPT; ++c) acc[c] = 0.f;
#pragma unroll
      for (int s = 0; s < 8; ++s)
        if (S.pos[tk * 8 + s] >= 0) {
          const float r = wv[s] * rsqrtf((red[2 * s] + red[2 * s + 1]) * (1.f / HD) + 1e-8f);
#pragma unroll
          for (int c = 0; c < CPT; ++c) acc[c] += r * yv[s][c];
        }
      if constexpr (CPT >= 4) {
        uint32_t ow[CPT / 2];
#pragma unroll
        for (int i = 0; i < CPT / 2; ++i) {
          __nv_bfloat162 v2 = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
          ow[i] = *reinterpret_cast<uint32_t*>(&v2);
        }
        if constexpr (CPT == 8) *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + col) = make_uint4(ow[0], ow[1], ow[2], ow[3]);
        else *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = make_uint2(ow[0], ow[1]);
      } else {
        *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc[0], acc[1]);
      }
      CTA_SYNC();
    }
  }
#elif OKC_WGS && OKC_WGSFINSS"""),
  (r"""#endif
  CTA_SYNC();
  if (tid == 0) {""", r"""#endif
#if OKC_WGS && LDD19_B4FIN
  constexpr int B4Q = LDD19_B4FIN == 3 ? 1 : LDD19_B4FIN == 2 ? 2 : 4;
  const int b4tk = blockIdx.x / B4Q;
  float b4w[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) b4w[s] = (b4tk < p.M && S.pos[b4tk * 8 + s] >= 0) ? __ldg(p.topk_w + b4tk * 8 + s) : 0.f;
#endif
#if !OKC_MFIN
  CTA_SYNC();
  if (tid == 0) {"""),
  ('after', r"""    int seq = 0;""", r"""
#if LDD19_DSMEM
    asm volatile("barrier.cluster.wait.acquire;\n" ::: "memory");
#endif"""),
  (r"""  wgs_gather(p, S);
  DBG(7);
  WS_JOIN();
  DBG(6);""", r"""#endif
  wgs_gather(p, S);
#if !OKC_WGSFIT
  DBG(7);
#endif
  WS_JOIN();
#if !OKC_WGSFIT
  DBG(6);
#endif"""),
  ('before', r"""  if (tid < 32) {""", r"""#if !LDD19_JROW
"""),
  ('before', r"""#if OKC_WS
  if (tid == 0) {""", r"""#endif
"""),
  ('after', r"""  DBG(1);""", r"""
#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)
  {
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32) p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS) p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
  }
#else"""),
  (r"""  }
#if SKDYN""", r"""#endif
  }
#if OKC_WGS && LDD19_DSMEM
  asm volatile("barrier.cluster.arrive.release;\n" ::: "memory");
#endif
#if SKDYN"""),
  (r"""    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);""", r"""#if !(OKC_WGS && OKC_RLOAD && LDD19_GEN)
    #if OKC_MFIN
      const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
      const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
#endif"""),
  (r"""#endif
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");""", r"""#endif
#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, NT);
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif"""),
  ('before', r"""#else
    for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);""", r"""#if LDD19_DSMEM
    for (int k = 0; k < 8; ++k) mbar_init(S.dsb + k, 1);
    for (int k = 0; k < 4; ++k) S.dsc[k] = 0;
#endif
"""),
  ('after', r"""#if !FAST1_ON""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA"""),
  ('before', r"""  for (int i = tid; i < (OKC_PRMTSC ? 0 : 256); i += NT) {""", r"""#endif
"""),
  ('after', r"""    }
  }
#endif""", r"""
#if OKC_XPF2"""),
  ('before', r"""      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif""", r"""#endif
"""),
  ('after', r"""#if OKC_SPLIT || !OKC_F1EARLY
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);""", r"""
#if OKC_WS"""),
  ('before', r"""#endif
    if (ln == 0) {""", r"""#else
    uint32_t m = 0;
      const int e = __ldg(p.topk_ids + i) - p.lo;
      if (e >= 0 && e < p.nloc) m |= 1u << e;
    }
    m = __reduce_or_sync(0xffffffffu, m);
"""),
  ('before', r"""      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }""", r"""#endif
"""),
  ('after', r"""    if (ln == 0) {
      for (int k = 0; k < 12; ++k) mbar_init(S.bars + k, 1);""", r"""
#if OKC_WS"""),
  ('after', r"""#if OKC_WGS
  if (tid >= NT) {""", r"""
#if LDD19_DSMEM
    asm volatile("barrier.cluster.arrive.relaxed;\n");
#endif"""),
  ('before', r"""}
__device__ __forceinline__ float* ss_cur() {""", r"""#else
#if OKC_TINY
  if (nF > 0) __trap();
#else
  if (nF > 0) { if (BIG) fat_stream<PB, true>(p, S, xbuf, nF, ubs); else fat_stream_ni<PB>(p, S, xbuf, nF, ubs); }
#endif
#endif
"""),
  ('after', r"""  const int nS = njS * rbs, nF = njF * ubs;""", r"""
#if OKC_FATSTATIC"""),
  (r"""  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);""", r"""#if !(LDD19_UCOST)
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);
#endif"""),
  ('after', r"""  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);""", r"""
#if LDD19_DSMEM
  ldd19_dsu(S, 0);
  ldd19_dsu(S, 1);
#endif
#if OKC_DEBUG && OKC_WGSFIT
  if (p.dbg) {
    for (int pb = 0; pb < 2; ++pb) {
      unsigned long long pk = 0;
      for (int gs = S.wgr[2 * pb]; gs < S.wgr[2 * pb + 1]; ++gs) {
        const int nt = (S.jC[gs >> (9 - pb)] + 7) >> 3;
        pk += 1ull << (8 * (nt - 1));
      }
      p.dbg[blockIdx.x * 8 + 6 + pb] = pk;
    }
  }
#endif
}
__device__ __noinline__ void wgs_ranges_warp(const Params& p, Smem& S) {
  const int lane = TIDX & 31, nj = S.njF + S.njS;
  uint32_t carA = 0, carB = 0;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + lane;
    const int c = j < nj ? S.jC[j] : 0;
#if LDD19_UCOST
    const uint32_t a = j < nj ? ((uint32_t)wgs_cost(0, c) * 16u + LDD19_UA) << 5 : 0u;
    const uint32_t b = j < nj ? ((uint32_t)wgs_cost(1, c) * 16u + LDD19_UB) << 4 : 0u;
#else
    const uint32_t a = j < nj ? (uint32_t)wgs_cost(0, c) << 9 : 0u, b = j < nj ? (uint32_t)wgs_cost(1, c) << 8 : 0u;
#endif
    uint32_t ia = a, ib = b;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t ta = __shfl_up_sync(0xffffffffu, ia, d), tb = __shfl_up_sync(0xffffffffu, ib, d);
      if (lane >= d) { ia += ta; ib += tb; }
    }
    if (j < nj) { S.wcp[j] = (int)(carA + ia - a); S.wcpB[j] = (int)(carB + ib - b); }
    carA += __shfl_sync(0xffffffffu, ia, 31);
    carB += __shfl_sync(0xffffffffu, ib, 31);
  }
  if (lane == 0) {
    S.wcp[nj] = (int)carA; S.wcpB[nj] = (int)carB;
    S.wtot = carA; S.wtotB = carB;
  }
  __syncwarp();
  if (lane < 4) {
    const int pb = lane >> 1, c = (int)blockIdx.x + (lane & 1), G = gridDim.x;
    S.wgr[lane] = ((lane & 1) && c == G) ? (nj << (9 - pb))
                                         : wgs_stage_at(S, nj, pb, (pb ? S.wtotB : S.wtot) * (uint32_t)c / (uint32_t)G);
  }
  __syncwarp();
#if LDD19_DSMEM
  if (lane < 2) ldd19_dsu(S, lane);
  __syncwarp();
#endif"""),
  ('before', r"""__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {""", r"""#if LDD19_DSMEM
__device__ __forceinline__ void ldd19_dsu(Smem& S, int pb) {
  uint32_t ncr;
  asm volatile("mov.u32 %0, %%cluster_nctarank;\n" : "=r"(ncr));
  const int c = blockIdx.x, G = gridDim.x, nj = S.njF + S.njS;
  int dsu = -1;
  if (ncr == 2 && !(G & 1) && nj > 0) {
    const uint32_t tot = pb ? S.wtotB : S.wtot;
    const int c0 = c & ~1;
    const int bo = (c & 1) ? wgs_stage_at(S, nj, pb, tot * (uint32_t)c0 / (uint32_t)G)
                           : (c0 + 2 >= G ? (nj << (9 - pb)) : wgs_stage_at(S, nj, pb, tot * (uint32_t)(c0 + 2) / (uint32_t)G));
    const int b0 = (c & 1) ? bo : S.wgr[2 * pb], b1 = (c & 1) ? S.wgr[2 * pb] : S.wgr[2 * pb + 1];
    const int b2 = (c & 1) ? S.wgr[2 * pb + 1] : bo;
    if (b1 & 15) {
      const int u = b1 >> 4;
      if (b0 <= (u << 4) && b2 >= ((u + 1) << 4)) dsu = u;
    }
  }
  S.dsu[pb] = dsu;
}
#endif
"""),
  ('after', r"""  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);""", r"""
#endif"""),
  ('before', r"""  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);""", r"""#if LDD19_UCOST
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = unit_cost(pb), cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
  const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);
  return (lo << (9 - pb)) + (int)(un * 16u + st);
#else
"""),
  ('before', r"""  const int* pre = pb ? S.wcpB : S.wcp;""", r"""#if OKC_MFIN || LDD19_SC6 || LDD19_UCOST
  if (nj == 0) return 0;
#endif
"""),
  ('after', r"""    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;""", r"""
#endif"""),
  ('before', r"""    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;""", r"""#if !(LDD19_UCOST)
"""),
  ('after', r"""  wgs_flush<PB>(p, S, pd);
}""", r"""
// LDD19: fixed unit overhead from keell0renz's CEPI scheduling model (Apache-2.0).
#ifndef LDD19_UCOST
#define LDD19_UCOST 0
#endif
#ifndef LDD19_UA
#define LDD19_UA 80
#endif
#ifndef LDD19_UB
#define LDD19_UB 70
#endif
__device__ __forceinline__ uint32_t unit_cost(int pb) { return pb ? LDD19_UB : LDD19_UA; }"""),
  ('before', r"""    }
    gs += s1 - s0;""", r"""#endif
"""),
  ('before', r"""      case 5: wgs_seg<PB, 5>(p, S, ring, u, s0, s1, slot, seq, pd); break;""", r"""#endif
#if OKC_FT_T > 32
"""),
  ('before', r"""      case 2: wgs_seg<PB, 2>(p, S, ring, u, s0, s1, slot, seq, pd); break;""", r"""#if OKC_FT_T > 8
"""),
  ('after', r"""  if (!PB) pd = WgsPend{1, u, -1, 0};""", r"""
#if OKC_MFIN && (OKC_MFP & 1)
  else pd = WgsPend{4, u, (OKC_MFP & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1, 0};
#endif"""),
  (r"""  if (slot >= 0) {""", r"""#if LDD19_DSMEM
  bool dsf = false;
  if constexpr (NTT <= 4) {
    if (slot >= 0 && u == S.dsu[PB]) {
      const uint32_t rk = blockIdx.x & 1u;
      const uint32_t buf = xring_s + (uint32_t)((wg >> 1) * WGS_XS + 16384 + (wg & 1) * 8192 + wt * 16);
      const uint32_t bar = smem_u32(S.dsb + (PB ? 4 : 0) + wg);
      bool send = rk != 0u;
      if (!PB) {
        if (wt == 0) {
          uint32_t ca, old;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ca) : "r"(smem_u32(S.dsc + wg)), "r"(1u));
          asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], 1;\n" : "=r"(old) : "r"(ca) : "memory");
          S.wglast[wg] = old != 0u;
        }
        asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
        send = !S.wglast[wg];
      }
      if (send) {
        uint32_t rbuf, rbar;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rbuf) : "r"(buf), "r"(rk ^ 1u));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(rbar) : "r"(bar), "r"(rk ^ 1u));
#pragma unroll
        for (int nt = 0; nt < NTT; ++nt)
          asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
                       ::"r"(rbuf + nt * 2048), "f"(d[nt][0]), "f"(d[nt][1]), "f"(d[nt][2]), "f"(d[nt][3]), "r"(rbar)
                       : "memory");
        return;
      }
      if (wt == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(NTT * 2048) : "memory");
      mbar_wait(S.dsb + (PB ? 4 : 0) + wg, 0u);
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        float v[4];
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]) : "r"(buf + nt * 2048) : "memory");
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float a;
          asm("add.rn.ftz.f32 %0, %1, %2;\n" : "=f"(a) : "f"(0.f), "f"(d[nt][i]));
          asm("add.rn.ftz.f32 %0, %1, %2;\n" : "=f"(d[nt][i]) : "f"(a), "f"(v[i]));
        }
      }
      dsf = true;
    }
  }
  if (slot >= 0 && !dsf) {
#else
  if (slot >= 0) {
#endif"""),
  ('before', r"""        }
        wg_fence();""", r"""#endif
"""),
  ('before', r"""          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);""", r"""#endif
"""),
  ('after', r"""          dq4(q, r0, r1, r2, r3);""", r"""
#if LDD19_KPERM
          const uint32_t sgp = sgp4[js];
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);
          af[ci & 1][jj][2] = bmul2l(r2, sgp); af[ci & 1][jj][3] = bmul2h(r3, sgp);
#elif OKC_WGSSC2
#if LDD19_SC6
          const uint32_t sgp = sgp4[js];
#else"""),
  (r"""        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);""", r"""#if !LDD19_KPERM
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
#endif"""),
  (r"""      const uint2 scg = scn;""", r"""#if !LDD19_KPERM
      const uint2 scg = scn;
#endif"""),
  (r"""    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);""", r"""#if OKC_WGSDESC0
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
#else
    const unsigned char* pc = ring + ws * UNIT_BYTES + wg * WGS_PIECE;
#endif
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
#if OKC_WGSDESC0
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
#else
    const uint64_t xd = wg_desc(xring_s + xs * WGS_XS, 128, 4096);
#endif
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
#if LDD19_KPERM
    uint32_t sgp4[4];
#if LDD19_SC6
    // (bytes of 4 v, see keell0renz SC6) are built here once and stay in registers for the 4 load groups
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      sgp4[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgp4[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgp4[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgp4[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#else
    {
      const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
#pragma unroll
      for (int js = 0; js < 4; ++js) {
        const uint32_t sw = js < 2 ? scs.x : scs.y, sb = (uint32_t)(js & 1) * 2u;
        sgp4[js] = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
      }
    }
#endif
#else
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#endif"""),
  ('before', r"""  auto stage = [&](auto first) {""", r"""#endif
"""),
  ('after', r"""  int prev = -1;""", r"""
#if OKC_WGSDESC0"""),
  (r"""  const uint32_t k16c = 16u + p.zero;""", r"""  const uint32_t k16c = (LDD19_SC6 ? 4u : 16u) + p.zero;"""),
  (r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }""", r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }"""),
  ('after', r"""  if (wt == 0) *fc = 0;""", r"""
#if OKC_MFIN && (OKC_MFP & 2)
  if (PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");
  }
#endif"""),
  (r"""  if (pd.kind == 1) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");""", r"""#if OKC_MFIN && (OKC_MFP & 1)
  if (pd.kind == 4) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j]))), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
#endif
  if (pd.kind == 1) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");"""),
  ('before', r"""  }
}
template <bool PB>""", r"""#if LDD19_B4FIN
    {
      const float a0 = d4[0] * w0, b0 = d4[2] * w1, a1 = d4[1] * w0, b1 = d4[3] * w1;
      float s0 = a0 * a0 + b0 * b0, s1 = a1 * a1 + b1 * b1;
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        s0 += __shfl_xor_sync(0xffffffffu, s0, o);
        s1 += __shfl_xor_sync(0xffffffffu, s1, o);
      }
      if (g == 0) {
        float* pp = ldd19_b4_part(p) + (size_t)(jst + tok0) * 256 + r16;
        if (tok0 < cnt) st_keep_f32(pp, s0);
        if (tok0 + 1 < cnt) st_keep_f32(pp + 256, s1);
      }
    }
#endif
#endif
"""),
  ('after', r"""    }
  } else {""", r"""
#if OKC_WGSFINSS"""),
  (r"""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");""", r"""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");"""),
  (r"""    int jj = 0;""", r"""#if LDD19_GBS
    int jj = q0 < R ? gath_job(S, nj, q0) : 0;
#else
    int jj = 0;
#endif"""),
  ('after', r"""      if (d[k]) *d[k] = v[k];
  }""", r"""
#endif"""),
  (r"""          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);""", r"""#if LDD19_KPERM
          {
            const int G = tid & 31;
            const uint32_t* xw = reinterpret_cast<const uint32_t*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) +
                                 (tid >> 5) * 128 + 8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1);
            v[k] = make_uint4(__ldg(xw), __ldg(xw + 32), __ldg(xw + 64), __ldg(xw + 96));
          }
#endif"""),
  (r"""  int j = 0;""", r"""#if LDD19_GBS
  int j = q0 < R ? gath_job(S, nj, q0) : 0;
#else
  int j = 0;
#endif
#if OKC_GATHB"""),
  ('before', r"""      if (++xs == WGS_NX) { xs = 0; xph ^= 1; }""", r"""      if (LDD19_XFIRST && n == 0) *reinterpret_cast<volatile int*>(&S.preissuedB) = 1;
"""),
  (r"""        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }
      }
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }""", r"""        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }
      }
      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }"""),
  ('before', r"""    int curj = -1, npad = 8;""", r"""    if (LDD19_XFIRST && pb == 0 && g1 == g0) *reinterpret_cast<volatile int*>(&S.preissuedB) = 1;
"""),
  (r"""  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
  if ((TIDX & 31) == 0) wgs_ranges(p, S);
  __syncwarp();""", r"""#if OKC_WGSPR
  asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
#if LDD19_WRANGE
  wgs_ranges_warp(p, S);
#endif
  __syncwarp();
#endif"""),
  (r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");""", r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
#if LDD19_XFIRST
      if (pb == 0 && n == 0) { while (*reinterpret_cast<volatile int*>(&S.preissuedB) == 0) __nanosleep(20); }
#endif"""),
  ('after', r"""__device__ void wgs_ranges(const Params& p, Smem& S);""", r"""
__device__ void wgs_ranges_warp(const Params& p, Smem& S);
__device__ __forceinline__ int gath_job(const Smem& S, int nj, int q) {
  int lo = 0, hi = nj;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (S.jrow[mid] <= q) lo = mid; else hi = mid; }
  return lo;
}"""),
  ('before', r"""  }
};
__device__ __forceinline__ int wgs_split(int pb, int g0, int g1) {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('before', r"""  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)""", r"""#endif
"""),
  ('after', r"""template <> struct WGSMMA<8> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)"""),
  ('before', r"""  }
};
template <> struct WGSMMA<8> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('before', r'''  }
  static __device__ __forceinline__ void mma(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
    asm volatile("wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, 1, 1, 1, 0;\n"''', r"""#endif
"""),
  ('after', r"""template <> struct WGSMMA<7> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)"""),
  ('before', r"""  }
};
template <> struct WGSMMA<7> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]), "=f"(d[5][0]), "=f"(d[5][1]), "=f"(d[5][2]), "=f"(d[5][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));""", r"""
#endif"""),
  ('after', r"""template <> struct WGSMMA<6> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)"""),
  ('before', r"""  }
};
template <> struct WGSMMA<6> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]), "=f"(d[4][0]), "=f"(d[4][1]), "=f"(d[4][2]), "=f"(d[4][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));""", r"""
#endif"""),
  ('after', r"""template <> struct WGSMMA<5> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)"""),
  ('before', r"""  }
};
template <> struct WGSMMA<5> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]), "=f"(d[3][0]), "=f"(d[3][1]), "=f"(d[3][2]), "=f"(d[3][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));""", r"""
#endif"""),
  ('after', r"""template <> struct WGSMMA<4> {
  static __device__ __forceinline__ void mma0(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)"""),
  ('before', r"""  }
};
template <> struct WGSMMA<4> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]), "=f"(d[2][0]), "=f"(d[2][1]), "=f"(d[2][2]), "=f"(d[2][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));""", r"""
#endif"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('before', r"""  }
};
template <> struct WGSMMA<3> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]), "=f"(d[1][0]), "=f"(d[1][1]), "=f"(d[1][2]), "=f"(d[1][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));""", r"""
#endif"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 0, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('before', r"""  }
};
template <> struct WGSMMA<2> {""", r"""#endif
"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""                 : "=f"(d[0][0]), "=f"(d[0][1]), "=f"(d[0][2]), "=f"(d[0][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));""", r"""
#endif"""),
  ('before', r'''    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n"''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !(OKC_WGSX & 1)
"""),
  ('after', r"""  mbar_wait(bar, parity);""", r"""
#endif"""),
  (r"""#define OKC_WGSREORD 1
#define OKC_GATHB 8
#define OKC_WGSPR 1
#define OKC_WGSDFIN 2
#define OKC_WGSFCI 3
#define OKC_WGSGS 2
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {""", r"""#ifndef OKC_WGSPROF
#define OKC_WGSPROF 0
#endif
#ifndef OKC_WGSREORD
#define OKC_WGSREORD 1
#endif
#ifndef OKC_GATHB
#define OKC_GATHB 8
#endif
#ifndef OKC_WGSPR
#define OKC_WGSPR 1
#endif
#ifndef OKC_WGSDFIN
#define OKC_WGSDFIN 2
#endif
#ifndef OKC_WGSFCI
#define OKC_WGSFCI 3
#endif
#ifndef OKC_WGSGS
#define OKC_WGSGS 2
#endif
#ifndef OKC_WGSFIT
#define OKC_WGSFIT 0
#endif
#ifndef OKC_WGSDBG
#define OKC_WGSDBG 0
#endif
__device__ __forceinline__ void wgs_wait(uint64_t* bar, uint32_t parity, int tag) {
#if !(OKC_WGSDBG)"""),
  (r"""#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174""", r"""static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
__device__ __forceinline__ int mf_gen(const Smem& S) {
#if OKC_MFQ & 32
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
#endif
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) { return p.xready + ((OKC_MFIN && (OKC_MFQ & 2)) ? mf_gen(S) * MF_XRS : 0); }
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) { return p.doneAs + ((OKC_MFIN && (OKC_MFQ & 1)) ? mf_gen(S) * (JMAX * 16) : 0); }
#ifndef OKC_WGSA
#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207
#endif
#ifndef OKC_WGSB
#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174
#endif"""),
  ('before', r"""__device__ __forceinline__ int* wgs_cnt(const Params& p, int pb) {""", r"""#if LDD19_B4FIN
__device__ __forceinline__ float* ldd19_b4_part(const Params& p) { return p.ybuf + (size_t)(EMAX - 128) * HD; }
#endif
"""),
  (r"""constexpr int WGS_PIECE = 9216;""", r"""// LDD19 (SC6): exact 6-bit scale packing, adapted from team keell0renz (Apache-2.0).
#ifndef LDD19_SC6
#define LDD19_SC6 0
#endif
#if LDD19_SC6 && !LDD19_KPERM
#error SC6 requires the KPERM scale layout
#endif
constexpr int WGS_STG = LDD19_SC6 ? 35840 : 36864;
constexpr int WGS_PIECE = LDD19_SC6 ? 8960 : 9216;"""),
  (r"""#define OKC_WGSNX 3""", r"""#ifndef OKC_WGSNX
#define OKC_WGSNX 3
#endif"""),
  ('after', r"""      const float gsc = p.gsB[ce] * TWO62;
      const int n0 = rb * 16 + gg;""", r"""
#endif"""),
  ('after', r"""        }
      }
    } else {""", r"""
#if OKC_WSH0
      const int n0 = rb * 16 + gg;
      const float gsc = p.gsB[ce] * TWO62;
#else"""),
  ('before', r"""    const bool PB = q >= nA;""", r"""#endif
"""),
  ('after', r"""  for (int q = 0; q < Nh; ++q) {""", r"""
#if !OKC_NOLOAD"""),
  ('before', r"""  seq = seq0 + nU;
}""", r"""#endif
"""),
  ('before', r"""#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 2 * tt + h;""", r"""#endif
"""),
  ('before', r"""    const float gsc = lgs * TWO62;""", r"""#if OKC_WSBL == 2
"""),
  (r"""  }
  if (PB && warp < OKC_WSBLK && warp < nU) {""", r"""#endif
  }
#if OKC_WSB && OKC_WSBL
  if (PB && warp < OKC_WSBLK && warp < nU) {"""),
  ('after', r"""#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];""", r"""
#else
    acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3];
#endif
#if OKC_WSB"""),
  ('before', r"""#pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];""", r"""#if OKC_NACC == 4 && !OKC_XNOMMA
"""),
  (r"""        dq4(ww[st], r0, r1, r2, r3);
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);""", r"""#if !(OKC_XNODQ)
        dq4(ww[st], r0, r1, r2, r3);
#endif
#if OKC_NACC == 4
        mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
#endif"""),
  (r"""      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif""", r"""#if !(OKC_XNOLUT)
      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];
#endif
#endif"""),
  (r"""    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll""", r"""#if OKC_NW == 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
#pragma unroll"""),
  ('after', r"""    const int sl = q % OKC_NS;""", r"""
#if OKC_NOLOAD
    if (q < OKC_NS)
#endif"""),
  ('before', r"""    if (j != curj) {
      curj = j;
      if (PB) {""", r"""#endif
"""),
  ('after', r"""    const int j = njF + v / rbs;""", r"""
#if OKC_WSB && OKC_WSBL == 2"""),
  ('after', r"""  uint32_t lnw0 = 0u, lnw1 = 0u;""", r"""
#endif"""),
  ('before', r"""  float lgs = 0.f;""", r"""#if OKC_WSB && OKC_WSBL
"""),
  (r"""static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#define OKC_WSRF 0""", r"""static_assert(!OKC_WS || !OKC_WSBL || OKC_WSBLK <= OKC_WSNR, "OKC_WSBLK must not exceed OKC_WSNR");
#ifndef OKC_WSRF
#define OKC_WSRF 0
#endif"""),
  (r"""  }
}
#if OKC_WS""", r"""  }
#else
  const int n = min(OKC_NS, v1 - v0);
  for (int i = 0; i < n; ++i) ring_issue_all<false>(p, S, v0 + i, ring + i * UNIT_BYTES, S.bars + i);
  if (TIDX == 0) S.preissued = n > 0 ? n : 0;
#endif
}
#if OKC_WS"""),
  ('before', r"""  const int n = min(OKC_EARLYN, v1 - v0);""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
"""),
  (r"""  cp_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");""", r"""  cp_wait<0>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif"""),
  (r"""      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      if (nat) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif"""),
  (r"""    const int un = u + gridDim.x;""", r"""#if OKC_FATSTATIC
    const int un = u + gridDim.x;
#endif"""),
  (r"""  };
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");""", r"""  };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif"""),
  (r"""  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  auto is_nat = [&](int j) { return BIG && OKC_WGMIN <= 8 && ((S.jC[j] + 7) >> 3) >= OKC_WGMIN; };
#else
  auto is_nat = [&](int j) { (void)j; return false; };
#endif"""),
  ('after', r"""  (void)ctr;""", r"""
#else
  if (tid == 0) S.unit = atomicAdd(ctr, 1);
  CTA_SYNC();
  int u = S.unit;
  CTA_SYNC();
#endif"""),
  ('before', r"""  int u = blockIdx.x;""", r"""#if OKC_FATSTATIC
"""),
  (r"""    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];""", r"""#if OKC_NW == 8
    const uint32_t sw1 = (cc < 2 ? sc1.x : sc1.y) >> sh;
#endif
    const uint32_t sg0 = S.lut[sw0 & 0xff], sg80 = S.lut[(sw0 >> 8) & 0xff];
#if OKC_NW == 8
    const uint32_t sg1 = S.lut[sw1 & 0xff], sg81 = S.lut[(sw1 >> 8) & 0xff];
#endif"""),
  ('after', r"""#define FMG81(r) bmul2((r), sg81)""", r"""
#endif
#else
#define FMG0(r) bmul2((r), sg0)
#define FMG80(r) bmul2((r), sg80)
#define FMG1(r) bmul2((r), sg1)
#define FMG81(r) bmul2((r), sg81)
#endif"""),
  ('before', r"""      const uint32_t sg0 = __byte_perm(p0, p0, 0x1010u), sg80 = __byte_perm(p0, p0, 0x3232u);""", r"""#if !(OKC_FTSC2)
"""),
  ('before', r"""      const uint32_t K92 = 0x6D806D80u;""", r"""#if OKC_WGALU
"""),
  ('before', r"""  }
};
template <int NTT>""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<8> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<7> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<6> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<5> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<4> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<3> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""  }
};
template <> struct WG<2> {""", r"""#endif
"""),
  ('before', r'''    asm volatile(
        "{\n .reg .pred p;\n setp.ne.b32 p, 1, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "''', r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
"""),
  ('before', r"""}
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo) {""", r"""}
__device__ __forceinline__ void wg_pin(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
  asm volatile("" : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
"""),
  ('before', r"""}
__device__ __forceinline__ uint64_t wg_desc_sw128(uint32_t saddr) {""", r"""#else
  for (int i = TIDX; i < UNIT_BYTES / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(slot + i * 16)), "l"(src + i * 16));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
"""),
  ('before', r"""  if (TIDX == 0) {""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
"""),
  ('after', r"""  CTA_SYNC();
}
#endif""", r"""
#ifndef OKC_NSF
#define OKC_NSF 3
#endif"""),
  (r"""  }
  seq = seq0 + nU;""", r"""  }
#if OKC_NOLOAD
  seq = seq0 + min(nU, OKC_NS);
#else
  seq = seq0 + nU;
#endif"""),
  ('before', r"""    nwc0 = nwn0; nwc1 = nwn1;
  }""", r"""#endif
"""),
  (r"""      ++preB;
    }
    if (tid < (two ? 128 : 64)) {""", r"""      ++preB;
    }
#endif
#if OKC_EPI4
#else
    if (tid < (two ? 128 : 64)) {"""),
  (r"""    CTA_SYNC();
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);""", r"""    CTA_SYNC();
#if !OKC_NOLOAD
    if (i + OKC_NS < nU) ring_issue_all<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl);"""),
  ('after', r"""      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }""", r"""
#elif OKC_TWOACC
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
#endif"""),
  (r"""#undef MULG8
    if (!two) {
#pragma unroll""", r"""#undef MULG8
#if OKC_NACC == 4
    if (!two) {
#pragma unroll"""),
  ('before', r"""#undef MULG
#undef MULG8
#if OKC_NACC == 4""", r"""#endif
"""),
  (r"""          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);""", r"""#if OKC_NACC == 4
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#elif OKC_TWOACC
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#endif"""),
  ('after', r"""    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);
      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)""", r"""
#endif"""),
  ('before', r"""      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);
#define MULG(r) bmul2((r), sg)
#define MULG8(r) bmul2((r), sg8)
#endif""", r"""#if !(OKC_SKSC2)
"""),
  (r"""    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);""", r"""#if !OKC_NOLOAD
    mbar_wait(bars + sl, par);
#endif
    const unsigned char* slot = ring + sl * UNIT_BYTES;
#if OKC_NW == 16
    const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UNIT_CODES + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
    const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
    float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#if !OKC_NOCOMPUTE"""),
  (r"""  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  float nwc0 = 0.f, nwc1 = 0.f;
  const int egg = (tid & 31) >> 2;""", r"""  }
  uint4 x0[2 * NCH];
  int curj = -1;
  bool two = false;
  __nv_bfloat16* x1s = x1base + warp * 8 * X1P;
  float nwc0 = 0.f, nwc1 = 0.f;
#if !(OKC_EPI4)
  const int egg = (tid & 31) >> 2;
#endif"""),
  (r"""  if (nU <= 0) return;
  const int seq0 = seq;
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");""", r"""  if (nU <= 0) return;
  const int seq0 = seq;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif"""),
  ('before', r"""    nwc0 = nwn0; nwc1 = nwn1;
    ++nproc;""", r"""#endif
"""),
  (r"""    }
    if (tid < (two ? 128 : 64)) {""", r"""    }
#if OKC_EPI4
#else
    if (tid < (two ? 128 : 64)) {"""),
  ('before', r"""    CTA_SYNC();
    if (tid == 0) {
      if (i + OKC_NS < nU) { ring_issue<PB>(p, S, v + OKC_NS, ring + sl * UNIT_BYTES, bars + sl); ++snext; }""", r"""#if OKC_SKT > 8
    if (two) *reinterpret_cast<float4*>(rd + 4) = make_float4(acc2[0], acc2[1], acc2[2], acc2[3]);
#endif
"""),
  (r"""#undef MULG8
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }""", r"""#undef MULG8
#if OKC_NACC == 4
    if (!two) {
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];
    }
#elif OKC_TWOACC
    if (!two) { acc[0] += acc2[0]; acc[1] += acc2[1]; acc[2] += acc2[2]; acc[3] += acc2[3]; }
#endif"""),
  (r"""    }
#undef MULG""", r"""    }
#endif
#undef MULG"""),
  (r"""          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
        }""", r"""#if OKC_NACC == 4
          mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#elif OKC_TWOACC
          mma_bf16((st & 1) ? acc2 : acc, MULG(r0), MULG8(r1), MULG(r2), MULG8(r3), b[2 * st], b[2 * st + 1]);
#endif
        }"""),
  (r"""#define MULG8(r) bmul2((r), sg8)
#else""", r"""#define MULG8(r) bmul2((r), sg8)
#endif
#else"""),
  ('after', r"""    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + c) * 512 + lane * 16);
#if OKC_PRMTSC
      const uint32_t swd = c < 4 ? (c < 2 ? sc.x : sc.y) : (c < 6 ? sc.z : sc.w);""", r"""
#if !(OKC_SKSC2)"""),
  (r"""    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#pragma unroll""", r"""#if OKC_NACC == 4
    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};
#endif
    const __nv_bfloat16* x1r = x1s + (lane >> 2) * X1P + 16 * (lane & 3);
#if !OKC_NOCOMPUTE
#pragma unroll"""),
  (r"""    mbar_wait(bars + sl, par);
    const unsigned char* slot = ring + sl * UNIT_BYTES;""", r"""#if !OKC_NOLOAD
    mbar_wait(bars + sl, par);
#endif
    const unsigned char* slot = ring + sl * UNIT_BYTES;"""),
  (r"""  const int egg = (tid & 31) >> 2;
  if (PB && tid < 128) {""", r"""#if !(OKC_EPI4)
  const int egg = (tid & 31) >> 2;
#endif
  if (PB && tid < 128) {"""),
  (r"""  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  int preB = 0;""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (tid == 0) asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
  int preB = 0;"""),
  (r"""  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);""", r"""#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA
  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);
#else
  ring_issue<PB>(p, S, v, slot, bar);
#endif"""),
  ('before', r"""}
template <bool PB>
__device__ __forceinline__ void ring_issue_all(const Params& p, const Smem& S, int v, unsigned char* slot, uint64_t* bar) {""", r"""#else
  for (int i = TIDX; i < UNIT_BYTES / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(slot + i * 16)), "l"(src + i * 16));
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_u32(bar)) : "memory");
#endif
"""),
  ('after', r"""  const unsigned char* src = (PB ? p.wBu : p.wAu) + blk * (size_t)UNIT_BYTES;""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && OKC_USE_TMA"""),
  ('after', r"""      ::"r"(smem_u32(bar)), "r"(parity) : "memory");""", r"""
#else
  asm volatile(
      "{\n .reg .pred p;\n WAIT_%=:\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
      ::"r"(smem_u32(bar)), "r"(parity) : "memory");
#endif"""),
  ('after', r"""__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"""),
  ('after', r"""      if (tok0 + h < ntok) atomicAdd(&ss_cur()[pstart + tok0 + h], sq[h]);
  }
}""", r"""
__device__ __forceinline__ int fat_cost(int cnt) {
  int nt = (cnt + 7) >> 3;
  int c = 22 + 8 * nt;
  return c < 32 ? 32 : c;
}
struct UnitIter {
  int j, nj, njF, ubs, rbs, ul, uh;
  long long Jc, lo, hi;
  __device__ int cost(const Smem& S, int jj) const { return jj < njF ? fat_cost(S.jC[jj]) : 4; }
  __device__ int count(int jj) const { return jj < njF ? ubs : rbs; }
  __device__ bool seek(const Smem& S) {
    while (j < nj && Jc < hi) {
      int c = cost(S, j), n = count(j);
      long long a = lo - Jc, b = hi - Jc;
      int l = a <= 0 ? 0 : (int)min((long long)n, (a + c - 1) / c);
      int h = b <= 0 ? 0 : (int)min((long long)n, (b + c - 1) / c);
      if (l < h) { ul = l; uh = h; return true; }
      Jc += (long long)c * n;
      ++j;
    }
    j = nj;
    return false;
  }
  __device__ bool next(const Smem& S, int u, int& jn, int& un) {
    if (u + 1 < uh) { jn = j; un = u + 1; return true; }
    Jc += (long long)cost(S, j) * count(j);
    ++j;
    if (seek(S)) { jn = j; un = ul; return true; }
    return false;
  }
};"""),
  (r"""__device__ __forceinline__ float probe_q(float v) {
  return v;""", r"""template <bool PB>
__device__ __forceinline__ void sk_load_w(const Params& p, int e, int rb, int lane, int warp, uint4 (&w)[NCH], uint4& s) {
  const int rbs = PB ? RB_B : RB_A;
  const uint4* wb = (PB ? p.wB : p.wA) + ((size_t)(e * rbs + rb) * NW + warp) * (NCH * 32) + lane;
  const uint4* sb = (PB ? p.sB : p.sA) + ((size_t)(e * rbs + rb) * NW + warp) * 32 + lane;
#pragma unroll
  for (int c = 0; c < NCH; ++c) w[c] = ldg_stream(wb + c * 32);
  s = ldg_stream(sb);
}
template <bool PB>
__device__ __forceinline__ void sk_stage_x(const Params& p, const Smem& S, int j, __nv_bfloat16* xs, int nrows) {
  const int tid = TIDX;
  const int cnt = S.jC[j], st = S.jS[j];
  const int tot = nrows * (HD / 8);
#pragma unroll 4
  for (int i = tid; i < tot; i += NT) {
    int r = i >> 9, c = i & 511;
    bool v = r < cnt;
    const __nv_bfloat16* src;
    if (PB) src = p.inter + (size_t)(st + (v ? r : 0)) * ID + c * 8;
    else src = p.x + (size_t)(v ? (S.ent[st + r] >> 3) : 0) * HD + c * 8;
    cp_async16(xs + r * XP_SK + c * 8, src, v);
  }
  cp_commit();
}
__device__ __forceinline__ float probe_q(float v) {
#if !(OKC_PROBE)
  return v;
#endif"""),
  ('after', r"""    __syncwarp();
  }""", r"""
#endif"""),
  ('after', r"""  early_issue(p, S);""", r"""
#if OKC_WGS && LDD19_RANK
  for (int i = i0 + lane; i < i1; i += 32) {
    const int e = (int)S.eid[i];
    if (e >= 0) {
      const int q = S.wc[warp][e] + (int)S.pos[i];
      S.ent[q] = (short)i;
      S.pos[i] = (short)q;
    } else {
      S.pos[i] = -1;
    }
  }
#else"""),
  ('before', r"""    if (lane == 0) { S.njF = nfTot; S.njS = nsTot; }""", r"""#if OKC_WGS && LDD19_JROW
    {
      const int pr = ((tot + 7) >> 3) << 3;
      const int rf = nfj ? pr : 0, rs = nsj ? pr : 0;
      int incF = rf, incS = rs;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const int a = __shfl_up_sync(0xffffffffu, incF, d), b = __shfl_up_sync(0xffffffffu, incS, d);
        if (lane >= d) { incF += a; incS += b; }
      }
      const int TF = __shfl_sync(0xffffffffu, incF, 31), TS = __shfl_sync(0xffffffffu, incS, 31);
      for (int q = 0; q < nfj; ++q) S.jrow[fj + q] = incF - rf + q * FT_T;
      if (nsj) S.jrow[sj] = TF + incS - rs;
      if (lane == 0) S.jrow[nfTot + nsTot] = TF + TS;
    }
#endif
"""),
  (r"""    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
  }""", r"""#if OKC_WGS && LDD19_RANK
    const int lead = __ffs(peers) - 1;
    int old = 0;
    if (e >= 0 && lane == lead) { old = S.wc[warp][e]; S.wc[warp][e] = old + __popc(peers); }
    old = __shfl_sync(0xffffffffu, old, lead);
    if (e >= 0) S.pos[i] = (short)(old + __popc(peers & ((1u << lane) - 1u)));
#else
    if (e >= 0 && lane == __ffs(peers) - 1) S.wc[warp][e] += __popc(peers);
#endif
  }"""),
  (r"""    }
#pragma unroll
    for (int k = 0; k < LPT; ++k) {""", r"""    }
#if LDD19_GEN
    if (tid == 0) {
      #if OKC_MFIN
      const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
      const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
#endif
      S.gen = gen;
      S.ssp = (gen & 1) ? p.ss1 : p.ss;
    }
#endif
#pragma unroll
    for (int k = 0; k < LPT; ++k) {"""),
  (r"""  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);""", r"""#endif
#if OKC_DQIMAD
  r3 = ((q * 16u) & 0x81808180u) | (__funnelshift_r(q, q, 4) & 0x00400040u);
#endif
}
__device__ __forceinline__ void chunk_mma(uint4 w, uint32_t sg, uint32_t sg8, const uint32_t* b, float* acc) {
  const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int st = 0; st < 4; ++st) {
    uint32_t r0, r1, r2, r3;
    dq4(ww[st], r0, r1, r2, r3);
    mma_bf16(acc, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);
  }"""),
  ('before', r"""  r1 = (q * 16u) & M;""", r"""#if OKC_DQ7
"""),
  ('before', r"""  r0 = q & M;""", r"""#if LDD19_DQ3
  r0 = q & M;
  r1 = __funnelshift_l(q, q, 13) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = ((q & 0x001B001Bu) * 0x1084u) & M;
  return;
#endif
"""),
  ('after', r"""      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));""", r"""
}
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;"""),
  ('before', r"""__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,""", r"""__device__ __forceinline__ uint32_t dq_hi(uint32_t q) { return (q & 0x80008000u) | ((q & 0x70007000u) >> 6); }
"""),
  ('after', r"""  int wgr[4];
#endif""", r"""
#if OKC_WGS && LDD19_DSMEM
  uint64_t dsb[8];
  int dsu[2];
  int dsc[4];
#endif"""),
  ('after', r"""  int* doneAs;""", r"""
#if OKC_MFIN
  int* mfl;
#endif"""),
  ('before', r"""struct Params {""", r"""constexpr int MF_CE = 0, MF_EXIT = 64, MF_XR = 128, MF_XRS = 96, MF_DA = 384, MF_INTS = 4096;
"""),
  (r"""#define OKC_SKT 8""", r"""#ifndef OKC_SKT
#define OKC_SKT 8
#endif
#if OKC_SKT > 8 && OKC_EPI4
#error "OKC_EPI4 assumes single-tile skinny jobs"
#endif"""),
  (r"""#define TIDX ((int)((threadIdx.x + NT) % NTL))""", r"""// LDD19: OKC_WSH0 from Madhavan113/kerneloptimizer (Apache-2.0): in the WS builds the helper warp is HARDWARE warp 0
#ifndef OKC_WSH0
#define OKC_WSH0 0
#endif
#if OKC_WSH0
#define TIDX ((threadIdx.x + NT) % NTL)
#else
#define TIDX threadIdx.x
#endif"""),
  (r"""#define OKC_WSNR 4
#define OKC_NOLOAD 0""", r"""#ifndef OKC_XNOLUT
#define OKC_XNOLUT 0
#endif
#ifndef OKC_XNODQ
#define OKC_XNODQ 0
#endif
#ifndef OKC_XNOMMA
#define OKC_XNOMMA 0
#endif
#ifndef OKC_WSNR
#define OKC_WSNR 4
#endif
#ifndef OKC_NOLOAD
#define OKC_NOLOAD 0
#endif
#ifndef OKC_NOCOMPUTE
#define OKC_NOCOMPUTE 0
#endif"""),
  ('before', r"""#ifndef OKC_PRMTSC""", r"""#ifndef OKC_DQ7
#define OKC_DQ7 0
#endif
#ifndef OKC_DQIMAD
#define OKC_DQIMAD 1
#endif
"""),
  (r"""#define SILU_DIV(a, b) __fdividef((a), (b))""", r"""#ifndef LDD19_DSMEM
#define LDD19_DSMEM 0
#endif
#ifndef OKC_FDIV
#define OKC_FDIV 1
#endif
#if OKC_FDIV
#define SILU_DIV(a, b) __fdividef((a), (b))
#endif"""),
  (r"""#define OKC_WGSX 0""", r"""#ifndef OKC_LAUNCHCHK
#define OKC_LAUNCHCHK 0
#endif
#ifndef OKC_WGSX
#define OKC_WGSX 0
#endif
"""),
  ('after', r"""#define OKC_FCOSTB 40""", r"""
#endif
#ifndef OKC_FCOSTB2
#define OKC_FCOSTB2 OKC_FCOSTB"""),
  (r"""#define OKC_FATSTATIC 1
#define OKC_FCOST 40""", r"""#ifndef OKC_FATSTATIC
#define OKC_FATSTATIC 1
#endif
#ifndef OKC_FCOST
#define OKC_FCOST 40
#endif
"""),
  (r"""#define OKC_BIGM 64
#define OKC_XPF 0""", r"""#ifndef OKC_BIGM
#define OKC_BIGM 64
#endif
#ifndef OKC_XPF2
#define OKC_XPF2 1
#endif
#ifndef OKC_XPF
#define OKC_XPF 0
#endif"""),
  (r"""#define OKC_TDYN 0
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#define OKC_FT_T 64""", r"""#ifndef OKC_TDYN
#define OKC_TDYN 0
#endif
#define SKDYN (OKC_DYN && (!OKC_TINY || OKC_TDYN) && FAST1_ARCH && OKC_USE_TMA && OKC_SKT == 8 && !OKC_XPF && !OKC_NOLOAD)
#ifndef OKC_FT_T
#define OKC_FT_T 64
#endif"""),
  (r"""#define OKC_FCOSTSB OKC_FCOST
#define OKC_DYN 1""", r"""#ifndef OKC_FCOSTSB
#define OKC_FCOSTSB OKC_FCOST
#endif
#ifndef OKC_DYN
#define OKC_DYN 1
#endif"""),
  (r"""#define OKC_WGMIN 9
#define OKC_XPRE 1
#define FAST1_ARCH 1""", r"""#ifndef OKC_WGMIN
#define OKC_WGMIN 9
#endif
#ifndef OKC_WGALU
#define OKC_WGALU 1
#endif
#ifndef OKC_XPRE
#define OKC_XPRE 1
#endif
#ifndef OKC_PROBE
#define OKC_PROBE 0
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define FAST1_ARCH 1
#else
#define FAST1_ARCH 0
#endif"""),
  (r"""#define OKC_DEBUG 0
#define OKC_USE_TMA 1
#define OKC_RLOAD 1
#define OKC_F1EARLY 1
#define OKC_WGSFINSS 1""", r"""#ifndef OKC_DEBUG
#define OKC_DEBUG 0
#endif
#ifndef OKC_USE_TMA
#define OKC_USE_TMA 1
#endif
#ifndef OKC_TWOACC
#define OKC_TWOACC 1
#endif
#ifndef OKC_NACC
#define OKC_NACC 2
#endif
#ifndef OKC_FPF
#define OKC_FPF 4
#endif
#ifndef OKC_RLOAD
#define OKC_RLOAD 1
#endif
#ifndef OKC_SKSC2
#define OKC_SKSC2 0
#endif
#ifndef OKC_FTSC2
#define OKC_FTSC2 0
#endif
#ifndef OKC_F1EARLY
#define OKC_F1EARLY 1
#endif
#ifndef OKC_WGSFINSS
// LDD19 (MFIN): readiness counters adapted from keell0renz, Apache-2.0.
#ifndef OKC_MFIN
#define OKC_MFIN 0
#endif
#define OKC_MFP 15
#define OKC_MFQ 47
#define OKC_MFS 0
#if !OKC_WGS || LDD19_DSMEM || LDD19_B4FIN
#undef OKC_MFIN
#define OKC_MFIN 0
#endif
#define OKC_WGSFINSS 1
#endif
#ifndef OKC_WGSDESC0
#define OKC_WGSDESC0 1
#endif
#ifndef OKC_WGSSC2
#define OKC_WGSSC2 1
#endif
#ifndef LDD19_KPERM
#define LDD19_KPERM 0  // LDD19: kperm (KPERM3, idea alygg77): the WGS copies' K permuted per stage (set by _moe_build)
#endif
#ifndef LDD19_B4FIN
#define LDD19_B4FIN 0  // LDD19: b4fin (idea alygg77): tile ss partials + one-round-trip WGS final pass (M <= 256 only);
#endif"""),
  ('before', r"""#include <cuda_bf16.h>""", r"""#ifndef LDD19_MOEPRO
#define LDD19_MOEPRO 0
#endif
#ifndef LDD19_GEN
#define LDD19_GEN LDD19_MOEPRO
#endif
#ifndef LDD19_RANK
#define LDD19_RANK LDD19_MOEPRO
#endif
#ifndef LDD19_JROW
#define LDD19_JROW LDD19_MOEPRO
#endif
#ifndef LDD19_WRANGE
#define LDD19_WRANGE LDD19_MOEPRO
#endif
#ifndef LDD19_GBS
#define LDD19_GBS LDD19_MOEPRO
#endif
#ifndef LDD19_XFIRST
#define LDD19_XFIRST LDD19_MOEPRO
#endif
"""),
])
# LDD19 v24r: the small-M MoE source is MOE_CUDA_SRC with these exact edits (readable template).
# LDD19 v24r: the small-M MoE source is MOE_CUDA_SRC with these exact edits (readable template).
_MA_SMALL_CUDA_SRC = _ldd19_derive(MOE_CUDA_SRC, [
  (r"""#if OKC_MFIN
  p.xready = p.mfl + okc::MF_XR;
  p.doneAs = p.mfl + okc::MF_DA;
#endif
#else""", r"""#else"""),
  (r"""#if OKC_MFIN
  p.mfl = ib + 8 + 1024 + okc::EMAX;
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), "MFIN flag region");
#endif
  p.ybuf = p.ss + okc::EMAX;""", r"""  p.ybuf = p.ss + okc::EMAX;"""),
  (r"""#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
      if (tk == blockIdx.x) DBG(4);
#endif
      int qv[8];""", r"""      int qv[8];"""),
  (r"""#endif
  }
#endif""", r"""#endif
  }"""),
  (r"""#endif
#if !OKC_MFIN""", r"""#endif"""),
  (r"""  }
#endif
#if OKC_WS""", r"""  }
#if OKC_WS"""),
  (r"""#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)
  {
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32) p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS) p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
  }
#else
  {""", r"""  {"""),
  (r"""    #if OKC_MFIN
      const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else
      const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
#endif
    S.gen = gen;""", r"""    const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);
    S.gen = gen;"""),
  (r"""#endif
  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);""", r"""  int c = (int)(((xe - 1) * (unsigned long long)G) / tot);"""),
  (r"""  const int j = s >> (9 - pb);
#if !(LDD19_UCOST)""", r"""  const int j = s >> (9 - pb);"""),
  (r"""#endif
    uint32_t ia = a, ib = b;""", r"""    uint32_t ia = a, ib = b;"""),
  (r"""    const int c = j < nj ? S.jC[j] : 0;
#if LDD19_UCOST
    const uint32_t a = j < nj ? ((uint32_t)wgs_cost(0, c) * 16u + LDD19_UA) << 5 : 0u;
    const uint32_t b = j < nj ? ((uint32_t)wgs_cost(1, c) * 16u + LDD19_UB) << 4 : 0u;
#else""", r"""    const int c = j < nj ? S.jC[j] : 0;"""),
  (r"""  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);
#endif""", r"""  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);"""),
  (r"""#if LDD19_UCOST
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = unit_cost(pb), cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
  const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);
  return (lo << (9 - pb)) + (int)(un * 16u + st);
#else
  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);""", r"""  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);"""),
  (r"""#if OKC_MFIN || LDD19_SC6 || LDD19_UCOST
  if (nj == 0) return 0;
#endif
  const int* pre = pb ? S.wcpB : S.wcp;""", r"""  const int* pre = pb ? S.wcpB : S.wcp;"""),
  (r"""    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;
#endif""", r"""    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;"""),
  (r"""#if !(LDD19_UCOST)
    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;""", r"""    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;"""),
  (r"""}
// LDD19: fixed unit overhead from keell0renz's CEPI scheduling model (Apache-2.0).
#ifndef LDD19_UCOST
#define LDD19_UCOST 0
#endif
#ifndef LDD19_UA
#define LDD19_UA 80
#endif
#ifndef LDD19_UB
#define LDD19_UB 70
#endif
__device__ __forceinline__ uint32_t unit_cost(int pb) { return pb ? LDD19_UB : LDD19_UA; }""", r"""}"""),
  (r"""#if OKC_MFIN && (OKC_MFP & 1)
  else pd = WgsPend{4, u, (OKC_MFP & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1, 0};
#endif
}""", r"""}"""),
  (r"""#endif
          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);""", r"""          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);"""),
  (r"""#elif OKC_WGSSC2
#if LDD19_SC6
          const uint32_t sgp = sgp4[js];
#else""", r"""#elif OKC_WGSSC2"""),
  (r"""#endif
#else
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);""", r"""#else
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);"""),
  (r"""#if LDD19_SC6
    // (bytes of 4 v, see keell0renz SC6) are built here once and stay in registers for the 4 load groups
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      sgp4[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgp4[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgp4[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgp4[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#else
    {""", r"""    {"""),
  (r"""  const uint32_t k16c = (LDD19_SC6 ? 4u : 16u) + p.zero;""", r"""  const uint32_t k16c = 16u + p.zero;"""),
  (r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }""", r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }"""),
  (r"""#if OKC_MFIN && (OKC_MFP & 2)
  if (PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");
  }
#endif
  if (!PB) {""", r"""  if (!PB) {"""),
  (r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");""", r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");"""),
  (r"""#if OKC_MFIN && (OKC_MFP & 1)
  if (pd.kind == 4) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j]))), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
#endif
  if (pd.kind == 1) {""", r"""  if (pd.kind == 1) {"""),
  (r"""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");""", r"""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");"""),
  (r"""      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }""", r"""      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }"""),
  (r"""        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }""", r"""        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }"""),
  (r"""                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");""", r"""                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");"""),
  (r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");""", r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");"""),
  (r"""}
static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
__device__ __forceinline__ int mf_gen(const Smem& S) {
#if OKC_MFQ & 32
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
#endif
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) { return p.xready + ((OKC_MFIN && (OKC_MFQ & 2)) ? mf_gen(S) * MF_XRS : 0); }
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) { return p.doneAs + ((OKC_MFIN && (OKC_MFQ & 1)) ? mf_gen(S) * (JMAX * 16) : 0); }""", r"""}"""),
  (r"""// LDD19 (SC6): exact 6-bit scale packing, adapted from team keell0renz (Apache-2.0).
#ifndef LDD19_SC6
#define LDD19_SC6 0
#endif
#if LDD19_SC6 && !LDD19_KPERM
#error SC6 requires the KPERM scale layout
#endif
constexpr int WGS_STG = LDD19_SC6 ? 35840 : 36864;
constexpr int WGS_PIECE = LDD19_SC6 ? 8960 : 9216;""", r"""constexpr int WGS_PIECE = 9216;"""),
  (r"""#endif
      S.gen = gen;""", r"""      S.gen = gen;"""),
  (r"""    if (tid == 0) {
      #if OKC_MFIN
      const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else""", r"""    if (tid == 0) {"""),
  (r"""  int* doneAs;
#if OKC_MFIN
  int* mfl;
#endif""", r"""  int* doneAs;"""),
  (r"""constexpr int MF_CE = 0, MF_EXIT = 64, MF_XR = 128, MF_XRS = 96, MF_DA = 384, MF_INTS = 4096;
struct Params {""", r"""struct Params {"""),
  (r"""#ifndef OKC_WGSFINSS
// LDD19 (MFIN): readiness counters adapted from keell0renz, Apache-2.0.
#ifndef OKC_MFIN
#define OKC_MFIN 0
#endif
#define OKC_MFP 15
#define OKC_MFQ 47
#define OKC_MFS 0
#if !OKC_WGS || LDD19_DSMEM || LDD19_B4FIN
#undef OKC_MFIN
#define OKC_MFIN 0
#endif""", r"""#ifndef OKC_WGSFINSS"""),
])
MOE_CPP_SRC = r"""
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
"""


_MOE_TINY = None
_MOE_TINY2 = None
_MOE_MID = None
MOE_DEFINES = ("-DOKC_NW=16", "-DOKC_FAST1=0", "-DOKC_EPI4=0", "-DOKC_WGS=1")
MOE_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2", "-DOKC_FINPF=1")    # their small-M build ( () = absent )  # LDD19: + OKC_FINPF
MOE_TINY_MAX_M = 4
MOE_TINY2_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=1', '-DOKC_FINPF=1')
MOE_TINY2_MAX_M = 8
MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '-DOKC_WSH0=1')      # their mid-M build ( () = absent )  # LDD19: + OKC_WSH0 (Madhavan113)
MOE_MID_MAX_M = 16


_MA_PRMT_SCALES = True
_MA_SFX_IF_PRMT = ("-DOKC_PRMTSC=1",)


def _ma_scales_prmt_exact(*sfs):
  for sf in sfs:
    b = sf.contiguous().view(torch.uint8)
    if bool(((b < 0x08) | (b > 0x7E)).any()):
      return False
  return True


MOE_HAS_PDL = True


# LDD19: moe_large_t sib: with -DLDD19_SIB=1 the main WGS build fills the weight stages of split experts with
# EVICT_NORMAL so the sibling job's re-read hits L2 (cache-hint operand only). LDD19_SIB=0 restores the previous build.
LDD19_SIB = int(__import__("os").environ.get("LDD19_SIB", "1"))
MOE_CUDA_SRC = _ldd19_derive(MOE_CUDA_SRC, [
  (r"""#ifndef LDD19_XFIRST
""", r"""#ifndef LDD19_SIB
#define LDD19_SIB 0
#endif
#ifndef LDD19_XFIRST
"""),
  (r"""      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
""", r"""      const int j = u >> ubl, ub = u & ((1 << ubl) - 1);
#if LDD19_SIB && OKC_L2H
      // LDD19: sib: the policy is read before the slot wait, off the release -> re-issue turnaround
      const uint64_t pol = (LDD19_SIB == 2 || S.cnt[S.jE[j]] > FT_T) ? 0x1000000000000000ull : 0x12F0000000000000ull;
#endif
      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
"""),
  (r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
""", r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");
#if LDD19_SIB && OKC_L2H
      {   // LDD19: sib: split experts (cnt > FT_T -> sibling jobs re-read the expert) fill L2 with EVICT_NORMAL
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
                     ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)), "l"(pol) : "memory");
      }
#else
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\n"
                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");
#endif
"""),
])


# LDD19: moe_large_t fin1 / fin2 (main non-DSMEM WGS build, M > 32): FIN1 keeps each token's 8 ybuf rows in registers
# for the weighted sum; FIN2 finishes first the token whose 8 experts are complete. 0 restores the previous final pass.
LDD19_FIN1 = int(__import__("os").environ.get("LDD19_FIN1", "1"))
LDD19_FIN2 = int(__import__("os").environ.get("LDD19_FIN2", "1"))
MOE_CUDA_SRC = _ldd19_derive(MOE_CUDA_SRC, [
  (r"""#ifndef LDD19_XFIRST
""", r"""#ifndef LDD19_FIN1
#define LDD19_FIN1 0
#endif
#ifndef LDD19_XFIRST
"""),
  (r"""      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
""", r"""      float ps[8];
#if LDD19_FIN1
      float4 ya[8], yb[8];   // LDD19: fin1: one L2 read of each ybuf row, kept for the weighted sum
#endif
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
#if LDD19_FIN1
        ya[sl] = yb[sl] = make_float4(0.f, 0.f, 0.f, 0.f);
#endif
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
#if LDD19_FIN1
          ya[sl] = a; yb[sl] = b;
#endif
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
"""),
  (r"""          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;
          const float4 a = __ldcg(src), b = __ldcg(src + 1);
          acc[0] += r * a.x;""", r"""          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
#if LDD19_FIN1
          const float4 a = ya[sl], b = yb[sl];
#endif
          acc[0] += r * a.x;"""),
  (r"""#ifndef LDD19_XFIRST
""", r"""#ifndef LDD19_FIN2
#define LDD19_FIN2 0
#endif
#ifndef LDD19_XFIRST
"""),
  (r"""    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
""", r"""#if LDD19_FIN2
    const int ntk = p.M > (int)blockIdx.x ? (p.M - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    unsigned left = ntk >= 4 ? 15u : ((1u << ntk) - 1u);
    for (int it = 0; it < ntk; ++it) {
      int tk = blockIdx.x + it * gridDim.x;
      if (ntk <= 4) {   // LDD19: fin2: pick the first remaining token whose 8 experts are complete
        if (warp == 0) {
          const int k = lane >> 3, s = lane & 7;
          const int tq = blockIdx.x + k * gridDim.x;
          const int* cp = nullptr;
          int target = 0;
          if (k < ntk && ((left >> k) & 1u) && S.pos[tq * 8 + s] >= 0) {
            const int e = S.eid[tq * 8 + s], tot = S.cnt[e];
            target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
            cp = p.mfl + MF_CE + S.gen * 32 + e;
          }
          int pick = -1;
          for (;;) {
            const unsigned b = __ballot_sync(0xffffffffu, cp == nullptr || ld_acquire(cp) >= target);
            for (int kk = 0; kk < ntk; ++kk)
              if (((left >> kk) & 1u) && ((b >> (8 * kk)) & 0xffu) == 0xffu) { pick = kk; break; }
            if (pick >= 0) break;
            if (OKC_MFS) __nanosleep(OKC_MFS);
          }
          if (lane == 0) reinterpret_cast<volatile int*>(red)[NW * 8 + 64] = pick;
        }
        CTA_SYNC();
        const int pk = reinterpret_cast<volatile int*>(red)[NW * 8 + 64];
        left &= ~(1u << pk);
        tk = blockIdx.x + pk * gridDim.x;
      } else {
        if (tid < 8 && S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          while (ld_acquire(cp) < target) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
        CTA_SYNC();
      }
#else"""),
])


# LDD19 (MP, moe_porter_integrator run 3 it 4): ports of the sibling MoE-board run's MAIN-family items (~/eat_moe: FNF from
# architect_large it 6; bounded BHOLD from scout it 3-4; FCI 4 + MAIN BHOLD 1 from tuner run 2 it 4) into the v24r MAIN WGS
# source (T = 128/256 main build only; the DSMEM M <= 32 variant compiles the same text with every flag at 0).
#   FNF: the WGS DF finisher drops its WG-wide fence.acq_rel.gpu (thread 0's acquire poll / acq_rel atom + the WG bar.sync
#        already order the partial loads).
#   BHOLD: the weight producer holds the last LDD19_BHOLD pre-boundary phase-B stages until the 4 warpgroups have issued
#        their final phase-A releases (an issue-order hint only, bounded to LDD19_BHOLD_SPIN polls: never a hang).
#   FCI: the deferred flush of the previous segment fires behind wgmma commit group OKC_WGSFCI (was 3).
# Scheduling / fence only: outputs bit-identical. 0 for a flag restores the f12 main build exactly.
LDD19_MP_FNF = 1
LDD19_MP_BHOLD = 1
LDD19_MP_FCI = 4
LDD19_MP_SPIN = 65536
LDD19_MP_TAGMAX = 160   # LDD19 (MP/TAG): extension tags longer than this are hashed (incumbent tags are <= ~150)
LDD19_MP_PMB8 = 163   # LDD19 (MP/PM): phase-B 8-n-tile stage cost of the main build at M > 192 (174 = previous, one build)


def _ldd19_mp_src(c):
  def rep(old, new, lo=None, hi=None):
    nonlocal c
    a = 0 if lo is None else c.index(lo)
    b = len(c) if hi is None else c.index(hi, a)
    seg = c[a:b]
    assert seg.count(old) == 1, (seg.count(old), old[:80])
    c = c[:a] + seg.replace(old, new) + c[b:]
  rep("""#ifndef LDD19_FIN1
#define LDD19_FIN1 0
#endif
""", """#ifndef LDD19_FIN1
#define LDD19_FIN1 0
#endif
#ifndef LDD19_FNF
#define LDD19_FNF 0
#endif
#ifndef LDD19_BHOLD
#define LDD19_BHOLD 0
#endif
#ifndef LDD19_BHOLD_SPIN
#define LDD19_BHOLD_SPIN 65536
#endif
""")
  rep('#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, 174\n',
      '#ifndef LDD19_PMB8\n#define LDD19_PMB8 174  // LDD19 (MP/PM): -D per build (M > 192)\n#endif\n'
      '#define OKC_WGSB 100, 104, 115, 123, 132, 144, 156, LDD19_PMB8\n')
  rep('  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n',
      '#if !LDD19_FNF  // LDD19 (MP/FNF): no WG-wide fence; thread 0\'s acquire + the WG bar.sync order the partial loads\n'
      '  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n#endif\n',
      lo="__device__ __forceinline__ void wgs_flush(", hi="\n}\n")
  rep("""      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
      uint64_t* bar = S.bars + BW_F + ws;
""", """      if (n >= WGS_NS) wgs_wait(S.bars + BW_E + ws, wph ^ 1, 1000 + gs);
#if LDD19_BHOLD  // LDD19 (MP/BHOLD): the last LDD19_BHOLD B stages issued before phase A ends wait (bounded) for the 4 WGs' final A releases
      if (pb == 1 && i >= WGS_NS - LDD19_BHOLD && i < WGS_NS) {
        int bsp = 0;
        while (*reinterpret_cast<volatile int*>(&S.preissued) < 4 && bsp < LDD19_BHOLD_SPIN) { __nanosleep(20); ++bsp; }
      }
#endif
      uint64_t* bar = S.bars + BW_F + ws;
""")
  rep("""    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
    DBG(2);
""", """    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);
#if LDD19_BHOLD  // LDD19 (MP/BHOLD): this warpgroup's last phase-A release (thread 0 of the WG, in wgs_flush) has been issued
    if ((TIDX & 127) == 0) atomicAdd(&S.preissued, 1);
#endif
    DBG(2);
""")
  return c


if LDD19_MP_FNF or LDD19_MP_BHOLD or LDD19_MP_FCI != 3:
  MOE_CUDA_SRC = _ldd19_mp_src(MOE_CUDA_SRC)


# LDD19 (MP/FINJ, from run 1 moe_large_t it 6 finj_patch.py): JOINT two-token final pass at T = 256 when both of a CTA's tokens are
# ready in the same poll (one latency chain instead of two); FIN1 + FIN2 otherwise. -DLDD19_FINJ=1 on the main build only.
LDD19_MP_FINJ = 1
def _ldd19_finj_one(src, old, new, count=1):
    n = src.count(old)
    assert n == count, (old[:90], n)
    return src.replace(old, new)


def _ldd19_finj_src(src):
    src = _ldd19_finj_one(src, '#ifndef LDD19_XFIRST\n', '#ifndef LDD19_FINJ\n#define LDD19_FINJ 0\n#endif\n#ifndef LDD19_XFIRST\n')
    src = _ldd19_finj_one(src, '''            for (int kk = 0; kk < ntk; ++kk)
              if (((left >> kk) & 1u) && ((b >> (8 * kk)) & 0xffu) == 0xffu) { pick = kk; break; }
''', '''            for (int kk = 0; kk < ntk; ++kk)
              if (((left >> kk) & 1u) && ((b >> (8 * kk)) & 0xffu) == 0xffu) { pick = kk; break; }
#if LDD19_FINJ
            if (ntk == 2 && left == 3u && (b & 0xffffu) == 0xffffu) pick = 2;   // LDD19: finj: both ready -> joint pass
#endif
''')
    src = _ldd19_finj_one(src, '''        const int pk = reinterpret_cast<volatile int*>(red)[NW * 8 + 64];
        left &= ~(1u << pk);
''', '''        const int pk = reinterpret_cast<volatile int*>(red)[NW * 8 + 64];
#if LDD19_FINJ
        if (pk == 2) {   // LDD19: finj: one pass for both tokens, two L2 reads, no row registers
          float* r2 = red + 256;
          const int tq0 = blockIdx.x, tq1 = blockIdx.x + gridDim.x;
          float ps[16];
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int sl = 0; sl < 8; ++sl) {
              const int q = S.pos[(h ? tq1 : tq0) * 8 + sl];
              float v = 0.f;
              if (q >= 0) {
                const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 2;
                const float4 a = __ldcg(src), b = __ldcg(src + 1);
                v = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
              }
              ps[h * 8 + sl] = v;
            }
#pragma unroll
          for (int i = 0; i < 16; ++i) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ps[i] += __shfl_xor_sync(0xffffffffu, ps[i], o);
          }
          if (lane == 0) {
#pragma unroll
            for (int i = 0; i < 16; ++i) r2[warp * 16 + i] = ps[i];
          }
          CTA_SYNC();
          if (tid < 16) {
            float tsum = 0.f;
            for (int w = 0; w < NW; ++w) tsum += r2[w * 16 + tid];
            r2[NW * 16 + tid] = tsum;
          }
          CTA_SYNC();
#pragma unroll 1
          for (int h = 0; h < 2; ++h) {
            const int tq = h ? tq1 : tq0;
            float acc[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
            for (int sl = 0; sl < 8; ++sl) {
              const int q = S.pos[tq * 8 + sl];
              if (q >= 0) {
                const float r = p.topk_w[tq * 8 + sl] * rsqrtf(r2[NW * 16 + h * 8 + sl] * (1.f / HD) + 1e-8f);
                const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)q * HD) + tid * 2;
                const float4 a = __ldcg(src), b = __ldcg(src + 1);
                acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
                acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
              }
            }
            uint4 o;
            uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
              ow[i] = *reinterpret_cast<uint32_t*>(&v);
            }
            *reinterpret_cast<uint4*>(p.out + (size_t)tq * HD + tid * 8) = o;
          }
          CTA_SYNC();
          break;
        }
#endif
        left &= ~(1u << pk);
''')
    return src


if LDD19_MP_FINJ:
  MOE_CUDA_SRC = _ldd19_finj_src(MOE_CUDA_SRC)


_MA_SFX = ()


_MA_EXTS = {}
_MA_ALT_SRCS = {}
_MA_MAIN_NS = 3
_MA_PDL = False


LDD19_MA_LOCK = 1   # LDD19 integrator r5i2: 1 = per-name lock in _ma_ext (prepare only); 0 = previous behaviour
_MA_LOCKS = {}
_MA_LOCKS_GUARD = __import__("threading").Lock()



LDD19_RSRED = 3   # LDD19: fix-up slots reset by a red of -v instead of a store: bit 1 MAIN family, bit 2 Keell B4 (0 = stock sources)


def _ldd19_rsred_src(h):   # LDD19: RSRED: x + (-x) == +0 exactly; the slot stays all-zero between calls as the store left it
  a = """        *reinterpret_cast<float4*>(fb + (nt0 + k) * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
"""
  assert h.count(a) == 1
  return h.replace(a, """        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + (nt0 + k) * 512), "f"(-v[k].x), "f"(-v[k].y), "f"(-v[k].z), "f"(-v[k].w) : "memory");
""")


LDD19_FXS = 1   # LDD19: phase-B fix-ups one-sided (store + finisher read, no reds, no slot reset): bit 1 MAIN family, bit 2 Keell B4 (0 = stock sources)


def _ldd19_fxs_src(h):   # LDD19: FXS: the slot holds stale data between calls; only the B protocol touches the B region and the finisher waits for the partner's release before reading
  a = """    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
"""
  assert h.count(a) == 1
  c = """    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
"""
  assert h.count(c) == 1
  h = h.replace(c, """    pd.kind = 0;
    return;
  }
  if constexpr (PB) return;   // LDD19: FXS: the finisher works inside wgs_seg
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
""")
  h = h.replace(a, a + """    if constexpr (PB) {
      if (s0 != 0) {
#pragma unroll
        for (int nt = 0; nt < NTT; ++nt)
          asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]), "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
        pd = WgsPend{2, u, slot, s1 - s0};
        return;
      }
      int* fc = wgs_cnt(p, PB) + slot * 4 + wg;
      if (wt == 0) { while (ld_acquire(fc) < 16 - (s1 - s0)) __nanosleep(32); }
      asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
#if !defined(LDD19_FNF) || !LDD19_FNF
      asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
#endif
      const int jx = u >> UBL, r16x = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const float4 vx = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
        d[nt][0] += vx.x; d[nt][1] += vx.y; d[nt][2] += vx.z; d[nt][3] += vx.w;
        wgs_epi_tile<PB>(p, S, jx, r16x, nt, d[nt]);
      }
      if (wt == 0) *fc = 0;
#if OKC_MFIN && (OKC_MFP & 2)
      asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
      if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[jx]), "r"(1) : "memory");
#endif
      return;
    }
""")
  # narrow rule: every B range holds >= 15 stages when W = tot / G >= 17 stages of the heaviest job present. mbx = that job's whole B cost (256 stages, unit overhead included)
  a0 = """  uint32_t carA = 0, carB = 0;
"""
  b0 = """  uint32_t carryA = 0, carryB = 0;
"""
  assert h.count(a0) + h.count(b0) == 1
  h = h.replace(a0, a0 + """  uint32_t mbx = 0;   // LDD19: FXS
""").replace(b0, b0 + """  uint32_t mbx = 0;   // LDD19: FXS
""")
  a1 = """    const uint32_t a = j < nj ? (uint32_t)wgs_cost(0, c) << 9 : 0u, b = j < nj ? (uint32_t)wgs_cost(1, c) << 8 : 0u;
"""
  b1 = """    uint32_t sa = ia, sb = ib;
"""
  assert h.count(a1) + h.count(b1) == 1
  h = h.replace(a1, a1 + """    mbx = max(mbx, b);   // LDD19: FXS
""").replace(b1, """    mbx = max(mbx, ib);   // LDD19: FXS
""" + b1)
  a2 = """  __syncwarp();
  if (lane < 4) {
    const int pb = lane >> 1, c = (int)blockIdx.x + (lane & 1), G = gridDim.x;
"""
  b2 = """  __syncwarp();
  const int l4 = lane & 3, pbl = l4 >> 1, kk = l4 & 1;
"""
  assert h.count(a2) + h.count(b2) == 1
  red = """#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mbx = max(mbx, (uint32_t)__shfl_xor_sync(0xffffffffu, mbx, o));   // LDD19: FXS
"""
  h = h.replace(a2, """  __syncwarp();
""" + red + """  if (lane < 4) {
    const int pb = lane >> 1, c = (int)blockIdx.x + (lane & 1), G = gridDim.x;
""").replace(b2, """  __syncwarp();
""" + red + """  const int l4 = lane & 3, pbl = l4 >> 1, kk = l4 & 1;
""")
  a = """                                         : wgs_stage_at(S, nj, pb, (pb ? S.wtotB : S.wtot) * (uint32_t)c / (uint32_t)G);
"""
  b = """  if (lane < 4) S.wgr[lane] = res;
"""
  assert h.count(a) + h.count(b) == 1
  h = h.replace(a, a + """    if (pb && S.wtotB < (uint32_t)G * ((mbx >> 4) + (mbx >> 8))) S.wgr[lane] &= ~15;   // LDD19: FXS: narrow call, whole units
""")
  h = h.replace(b, """  if (pbl && carryB < (uint32_t)G * ((mbx >> 4) + (mbx >> 8))) res &= ~15;   // LDD19: FXS: narrow call, whole units
""" + b)
  return h


LDD19_FXH = 1   # LDD19: unit epilogue of the MAIN family: gs, post-norm and job fields loaded once per segment (0 = stock sources)


def _ldd19_fxh_src(h):   # LDD19: FXH
  a = """  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
"""
  assert h.count(a) == 1
  h = h.replace(a, """  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
  wgs_epi_unit<PB, NTT>(p, S, j, r16, d);
""")
  a = """#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const float4 vx = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
        d[nt][0] += vx.x; d[nt][1] += vx.y; d[nt][2] += vx.z; d[nt][3] += vx.w;
        wgs_epi_tile<PB>(p, S, jx, r16x, nt, d[nt]);
      }
"""
  assert h.count(a) == 1
  h = h.replace(a, """#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const float4 vx = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
        d[nt][0] += vx.x; d[nt][1] += vx.y; d[nt][2] += vx.z; d[nt][3] += vx.w;
      }
      wgs_epi_unit<PB, NTT>(p, S, jx, r16x, d);
""")
  a = """template <bool PB>
__device__ __forceinline__ void wgs_flush(const Params& p, Smem& S, WgsPend& pd) {
"""
  assert h.count(a) == 1
  h = h.replace(a, """template <bool PB, int NTT>
__device__ __forceinline__ void wgs_epi_unit(const Params& p, const Smem& S, int j, int r16, float (&d)[WGS_NT][4]) {
#if LDD19_B4FIN || !OKC_WGSFINSS
#pragma unroll
  for (int nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);
#else
  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], e = S.jE[j];
  if constexpr (!PB) {
    const int col = r16 * 8 + g;
    const float gsc = p.gsA[e] * TWO62;
    const uint32_t xo = (uint32_t)S.jrow[j] * HD + (col & 7) + (uint32_t)((((col >> 8) * NTT) * 32 + ((col & 255) >> 3)) * 64);
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = nt * 8 + 2 * t + h;
        if (tk < cnt) {
          const float gt = d[nt][h] * gsc, up = d[nt][2 + h] * gsc;
          const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
          st_keep_bf16(p.xB + (size_t)(xo + (uint32_t)((tk >> 3) * 32 * 64 + (tk & 7) * 8)), __float2bfloat16(probe_q(v)));
        }
      }
    }
  } else {
    const int n0 = r16 * 16 + g;
    const float gsc = p.gsB[e] * TWO62;
    const float w0 = (1.f + __bfloat162float(__ldg(p.normw + n0))) * gsc, w1 = (1.f + __bfloat162float(__ldg(p.normw + n0 + 8))) * gsc;
    const uint32_t yo = (uint32_t)S.jS[j] * HD + n0;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int tk = nt * 8 + 2 * t + h;
        if (tk < cnt) {
          st_keep_f32(p.ybuf + (size_t)(yo + (uint32_t)tk * HD), d[nt][h] * w0);
          st_keep_f32(p.ybuf + (size_t)(yo + (uint32_t)tk * HD + 8), d[nt][2 + h] * w1);
        }
      }
    }
  }
#endif
}
""" + a)
  return h

def _ma_ext(defines=None):
  defines = tuple(MOE_DEFINES if defines is None else defines)
  if LDD19_FXH and (LDD19_FXS & 1) and "-DOKC_WGS=1" in defines and "-DLDD19_FXH=1" not in defines:   # LDD19: FXH
    defines = defines + ("-DLDD19_FXH=1",)
  if (LDD19_FXS & 1) and "-DOKC_WGS=1" in defines and "-DLDD19_FXS=1" not in defines:   # LDD19: FXS
    defines = defines + ("-DLDD19_FXS=1",)
  if (LDD19_RSRED & 1) and "-DOKC_WGS=1" in defines and "-DLDD19_RSRED=1" not in defines:   # LDD19: RSRED
    defines = defines + ("-DLDD19_RSRED=1",)
  if DQ7 and "-DOKC_DQ7=1" not in defines:
    defines = defines + ("-DOKC_DQ7=1",)
  if not LDD19_MA_LOCK:
    return _ma_ext_build(defines)
  with _MA_LOCKS_GUARD:   # LDD19 integrator r5i2: one builder per name; a concurrent load_inline of the same name imports a .so that is not built yet
    lock = _MA_LOCKS.setdefault(defines, __import__("threading").Lock())
  with lock:
    return _ma_ext_build(defines)


def _ma_ext_build(defines):
  if defines not in _MA_EXTS:
    src = MOE_CUDA_SRC if "-DOKC_WGS=1" in defines else _MA_SMALL_CUDA_SRC
    if "-DLDD19_RSRED=1" in defines:   # LDD19: RSRED
      src = _ldd19_rsred_src(src)
    if "-DLDD19_FXS=1" in defines:   # LDD19: FXS
      src = _ldd19_fxs_src(src)
    if "-DLDD19_FXH=1" in defines:   # LDD19: FXH
      src = _ldd19_fxh_src(src)
    defs = [d for d in defines if not d.startswith("@")]
    if torch.cuda.get_device_capability()[0] < 9:
      defs = [d for d in defs if not d.startswith("-DOKC_NS=") and not d.startswith("-DOKC_NSF=")]
    for d in defines:
      if d.startswith("@"):
        src = _MA_ALT_SRCS[d[1:]]
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
    if any(d.startswith('@') for d in defines):
      tag = 'keell_' + __import__('hashlib').sha256((src + repr(defines)).encode()).hexdigest()[:20]
    if len(tag) > LDD19_MP_TAGMAX:   # LDD19 (MP/TAG): judge filename limit (255 B): hash long define lists
      tag = 'mp_' + __import__('hashlib').sha256((src + repr(defines)).encode()).hexdigest()[:20]
    _MA_EXTS[defines] = _ldd19_inline(
        name=f"dl_okc_moe_w4a16_v216x_{tag}_sm{arch}",
        cpp_sources=MOE_CPP_SRC,
        cuda_sources=src,
        functions=["okc_run", "okc_set_pdl"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                           *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={_MA_MAIN_NS if major >= 9 else 1}"]),
                           f"-DOKC_FNST={4 if major >= 9 else 3}",
                           *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
        verbose=False,
    )
    _MA_EXTS[defines].okc_set_pdl(1 if _MA_PDL else 0)
  return _MA_EXTS[defines]


def _moe_build(defines):
  if LDD19_KPERM and "-DOKC_WGS=1" in tuple(defines):   # LDD19: kperm: matches _ma_prepare's WGS copies
    defines = tuple(defines) + ("-DLDD19_KPERM=1",)
  if LDD19_DQ3:   # LDD19: v20 dq3: matches _ma_prepare's code-word bit layout
    defines = tuple(defines) + ("-DLDD19_DQ3=1",)
  if '-DOKC_WGS=1' in tuple(defines):
    if _MA_SC6:
      defines = tuple(defines) + ('-DLDD19_SC6=1',)
    if LDD19_GF2:   # LDD19: GF2: matches _wgs_copy's mixed code words
      defines = tuple(defines) + ('-DLDD19_GF2=1',)
    if LDD19_MFIN and not any(d.startswith(('-DLDD19_DSMEM=', '-DLDD19_B4FIN=')) for d in defines):
      defines = tuple(defines) + ('-DOKC_MFIN=1',)
  ext = _ma_ext(tuple(defines) + tuple(_MA_SFX))
  if MOE_HAS_PDL:
    ext.okc_set_pdl(0)
  return ext


# LDD19: norm1, the MAIN build with the standalone MoE prologue switches (-DLDD19_MOEPRO=1; 0 = v12's MAIN build).
LDD19_MOEPRO = int(__import__("os").environ.get("LDD19_MOEPRO", "1"))
# LDD19 (MP/BW1, moe_porter_integrator run 3 it 4): Triton _boundary1 num_warps per T above LDD19_GLUE_MAX_T (16 = previous)
LDD19_MP_BW1 = {128: 32}
# LDD19 (MP/CEL): the T = 1 gle route stores the carry s1 with eviction_policy evict_last (0 = previous store)
LDD19_MP_CEL = 1
# LDD19: moe4, the MAIN build with the DSMEM split-unit finish (-DLDD19_DSMEM=1) for M <= 32; read when _moe picks
LDD19_DSMEM = int(__import__("os").environ.get("LDD19_DSMEM", "1"))   # LDD19: v15: the 2-CTA-cluster DSMEM finish on by default (moe4)
# LDD19: b4fin (idea alygg77's B4FIN): the DSMEM build at M <= 32 with -DLDD19_B4FIN=v (one-round-trip final pass;
LDD19_B4FIN = int(__import__("os").environ.get("LDD19_B4FIN", "2"))   # LDD19: b4fin, halves won the A/B (agents/b4fin/NOTES.md)


def _moe(M=None):
  if M is not None and MOE_TINY_DEFINES and M <= MOE_TINY_MAX_M:
    return _moe_build(MOE_TINY_DEFINES)
  if M is not None and _moe_tiny2_ok() and M <= MOE_TINY2_MAX_M:
    return _moe_build(MOE_TINY2_DEFINES)
  if M is not None and MOE_MID_DEFINES and M <= MOE_MID_MAX_M:
    return _moe_build(MOE_MID_DEFINES)
  d = tuple(_main_defines(_MA_SFX))
  if LDD19_MOEPRO and '-DOKC_WGS=1' in d:
    d += ('-DLDD19_MOEPRO=1',)
    if LDD19_DSMEM and M is not None and M <= 32:
      d += ('-DLDD19_DSMEM=1',)
      if LDD19_B4FIN:d += (f'-DLDD19_B4FIN={LDD19_B4FIN}',)
  if LDD19_SIB and '-DOKC_WGS=1' in d and '-DLDD19_DSMEM=1' not in d:   # LDD19: sib, main WGS build only
    d += ('-DLDD19_SIB=1',)
  if '-DOKC_WGS=1' in d and '-DLDD19_DSMEM=1' not in d:   # LDD19: fin1 / fin2, main WGS build only
    d += (('-DLDD19_FIN1=1',) if LDD19_FIN1 else ()) + (('-DLDD19_FIN2=1',) if LDD19_FIN2 else ())
    if globals().get('LDD19_MP_FNF') or globals().get('LDD19_MP_BHOLD') or globals().get('LDD19_MP_FCI', 3) != 3:  # LDD19 (MP)
      d += ((('-DLDD19_FNF=1',) if LDD19_MP_FNF else ()) + ((f'-DLDD19_BHOLD={LDD19_MP_BHOLD}', f'-DLDD19_BHOLD_SPIN={LDD19_MP_SPIN}') if LDD19_MP_BHOLD else ())
            + ((f'-DOKC_WGSFCI={LDD19_MP_FCI}',) if LDD19_MP_FCI != 3 else ()))
    if globals().get('LDD19_MP_FINJ'):  # LDD19 (MP/FINJ)
      d += ('-DLDD19_FINJ=1',)
  if '-DOKC_WGS=1' in d and '-DLDD19_DSMEM=1' not in d and M is not None and M > 192 and globals().get('LDD19_MP_PMB8', 174) != 174:
    d += (f'-DLDD19_PMB8={LDD19_MP_PMB8}',)   # LDD19 (MP/PM): batch-size-only choice (a launch shape, nothing from the inputs)
  if LDD19_UNIT_COST and M == 32 and '-DOKC_WGS=1' in d:
    d += ('-DLDD19_UCOST=1','-DLDD19_UA=80','-DLDD19_UB=70')
  return _moe_build(d)


def _moe_mid():
  global _MOE_MID
  if MOE_MID_DEFINES and _MOE_MID is None:
    _MOE_MID = _moe_build(MOE_MID_DEFINES)
  return _MOE_MID


def _moe_tiny2_ok():
  return bool(MOE_TINY2_DEFINES) and (bool(_MA_SFX) or "-DOKC_WGS=1" not in tuple(MOE_TINY2_DEFINES))


def _moe_tiny2():
  global _MOE_TINY2
  if _moe_tiny2_ok() and _MOE_TINY2 is None:
    _MOE_TINY2 = _moe_build(MOE_TINY2_DEFINES)
  return _MOE_TINY2


def _moe_tiny():
  global _MOE_TINY
  if MOE_TINY_DEFINES and _MOE_TINY is None:
    _MOE_TINY = _moe_build(MOE_TINY_DEFINES)
  return _MOE_TINY


_REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


DQ7 = True
LDD19_DQ3 = int(__import__("os").environ.get("LDD19_DQ3", "1"))   # LDD19: v20: dq3 bit-layout in prepare() + _moe_build


def _stored_pos(dq3=False):
  pos = {}
  for R in range(4):
    for up in range(2):
      outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}
      for b, o in outb.items():
        if dq3:   # LDD19: v20 dq3: r3 multiplies low 5-bit lane fields by (1<<2) | (1<<7) | (1<<12)
          d = (0, 19, 22, (30 if b == 0 else 25 if b in (1, 2) else 20))[R]
        elif DQ7:
          d = (0, 28, 22, 18 if b == 2 else 26)[R]
        elif R == 0:
          d = 0
        elif R == 1:
          d = 6
        elif R == 2:
          d = 10
        else:
          d = 4 if b == 0 else 28
        pos[(R, up, b)] = (o + d) % 32
  assert sorted(pos.values()) == list(range(32))
  return pos


_POS = _stored_pos(False)
_POS_DQ3 = _stored_pos(True)   # LDD19: v20


def _pack_codes(codes, rows):
  c = codes.index_select(0, rows)
  R = c.shape[0]
  nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)
  nb = R // 16
  n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)
  n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)
  pos = _POS_DQ3 if LDD19_DQ3 else _POS   # LDD19: v20
  word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
  for Rg, (h, j0) in enumerate(_REG_HJ):
    for up in range(2):
      code = n[..., h, j0 + up]
      for b in range(4):
        word |= ((code >> b) & 1) << pos[(Rg, up, b)]
  word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
  return word.contiguous().view(-1)


def _pack_scales(sf, rows):
  s = sf.view(torch.uint8).index_select(0, rows)
  R = s.shape[0]
  nb = R // 16
  s = s.view(nb, 2, 8, 8, 8, 4)
  s = s.permute(0, 3, 2, 5, 4, 1)
  return s.contiguous().view(-1)


def _fat_order(codes, scales):
  nub = codes.numel() // (16 * 8 * 8 * 32 * 4)
  c = codes.view(nub, 16, 8, 2, 4, 32, 4).permute(0, 2, 3, 1, 4, 5, 6).contiguous()
  c = c.view(torch.uint8).view(nub, 16, 32768)
  s = scales.view(nub, 16, 8, 32, 2, 4, 2).permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(nub, 16, 4096)
  return torch.cat([c, s], dim=2).contiguous().view(-1)


def _unit_order(codes, scales):
  nblk = codes.numel() // (8 * 8 * 32 * 4)
  c = codes.contiguous().view(torch.uint8).view(nblk, 32768)
  s = scales.contiguous().view(nblk, 4096)
  return torch.cat([c, s], dim=1).contiguous().view(-1)


def _pack_codes_nat(codes, rows):
  c = codes.index_select(0, rows)
  R = c.shape[0]
  nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)
  nb = R // 16
  n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)
  n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)
  pos = _POS_DQ3 if LDD19_DQ3 else _POS   # LDD19: v20
  word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
  for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
    for up in range(2):
      code = n[..., h, q, up]
      for b in range(4):
        word |= ((code >> b) & 1) << pos[(Rg, up, b)]
  word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
  return word.contiguous().view(-1)


# LDD19: kperm (KPERM3, idea from team alygg77, https://github.com/alygg77/okc-muse-spark-kernels, Apache-2.0; our own
LDD19_KPERM = int(__import__("os").environ.get("LDD19_KPERM", "1"))


def _ldd19_kperm_pos(device=None):
  """LDD19: kperm: the logical column (0..255) at each physical position p of a 256-K stage."""
  p = torch.arange(256, device=device)
  s, c = p // 16, p % 16
  lg, j, t, i = s // 4, s % 4, (c // 2) % 4, c % 2 + 2 * (c // 8)
  return 16 * (4 * t + j) + 4 * lg + i


def _ldd19_kperm_codes(codes):
  """LDD19: kperm: packed fp4 codes [R, K / 2] (one byte = columns 2m, 2m + 1) with K permuted inside each stage. The
    permutation keeps column pairs together (p even -> an even column, p + 1 -> the next), so it is a byte gather."""
  k = _ldd19_kperm_pos()
  assert torch.equal(k[1::2], k[0::2] + 1) and bool((k[0::2] % 2 == 0).all()) and k.sort().values.equal(torch.arange(256))
  nst = codes.shape[1] // 128
  idx = (torch.arange(nst)[:, None] * 128 + (k[0::2] // 2)[None, :]).reshape(-1).to(codes.device)
  return codes.index_select(1, idx)


def _ldd19_kperm_rows(rowsA, inter):
  """LDD19: kperm: W13 row order for the WGS copy. Phase A's epilogue writes the row pair at inter position P
    (16-row block P >> 3, lane g = P & 7: gate row, up row) to xB position P; phase B's permuted W2 reads logical inter
    column colmap(P) = 256 (P >> 8) + pos[P & 255] there. So the pair at P must compute column colmap(P): gate row
    colmap(P) and up row inter + colmap(P) (gate and up move together: the SwiGLU pairing holds)."""
  P = torch.arange(inter, device=rowsA.device)
  cm = (P // 256) * 256 + _ldd19_kperm_pos(rowsA.device)[P % 256]
  up = rowsA >= inter
  return torch.where(up, inter + cm[(rowsA - inter).clamp(min=0)], cm[rowsA.clamp(max=inter - 1)])


def _sc6_ok(*sfs):
  for sf in sfs:
    b = sf.contiguous().view(torch.uint8)
    if bool(((b < 0x40) | (b > 0x7F)).any()):
      return False
  return True

def _sc6_wgs(s):
  lead = s.shape[:-1]
  v = s.view(*lead, 128, 8).to(torch.int32) - 0x40
  assert int(v.min()) >= 0 and int(v.max()) <= 63
  v6, v7 = v[..., 6], v[..., 7]
  piece = [(v6 >> 0) & 3, (v6 >> 2) & 3, (v6 >> 4) & 3, v7 & 3, (v7 >> 2) & 3, (v7 >> 4) & 3]
  wb = torch.stack([(v[..., n] << 2) | piece[n] for n in range(4)], -1).to(torch.uint8)
  hb = torch.stack([(v[..., 4 + n] << 2) | piece[4 + n] for n in range(2)], -1).to(torch.uint8)
  return torch.cat([wb.reshape(*lead, 512), hb.reshape(*lead, 256)], -1).contiguous()


LDD19_SC6 = True
LDD19_MFIN = True
LDD19_UNIT_COST = False
_MA_SC6 = False

def _wgs_copy(codes, sf, rows, kperm=False, sc6=False):
  """WGS copy (RS-wgmma tiny band, moe -DOKC_WGS=1): [256-row unit][stage (16)][warpgroup q (4)][9216 B]. Warpgroup q
    owns rows 64q..64q+63 of the unit; stage st holds k16 steps st*16 + (0..15): codes [grp (4)][warp (4)][lane (32)]
    [4 words] (word (grp, j) = k16 step st*16 + grp*4 + j, natural mma k order), then scales [grp][warp][g][j][h] (the kernel's OKC_WGSSC2 layout).
    LDD19: kperm=True: K permuted inside each stage (_ldd19_kperm_pos), scales [warp][g][t][j][h] (block 4 t + j)."""
  wn = _pack_codes_nat(_ldd19_kperm_codes(codes) if kperm else codes, rows)
  if LDD19_GF2:   # LDD19: GF2: the WGS consumer decodes with dq4g
    wn = _ldd19_gf2_map(wn)
  R = rows.numel()
  nb, nub = R // 16, R // 256
  w = wn.view(nb, 8, 8, 8, 4, 4)
  w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
  w = w.view(nub, 4, 4, 16, 4, 4, 32)
  w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()
  cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
  s = sf.view(torch.uint8).index_select(0, rows)
  if kperm:   # LDD19: kperm: lane (g, t) of warp w reads scale block 4 t + j of rows g / g + 8 for its step j
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)
    s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)
  else:
    s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)
    s = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 16, 4, 1024)
  if sc6:
    s = _sc6_wgs(s)
  return torch.cat([cb, s], dim=3).contiguous().view(-1)


MAIN_FALLBACK_DEFINES = ("-DOKC_CPF=0", "-DOKC_CPFF=0", "-DOKC_DYNK=7", "-DOKC_EARLYN=2", "-DOKC_EPI4=0", "-DOKC_FCOSTB=35",
                         "-DOKC_FCOSTBB=36", "-DOKC_FINCOL=0", "-DOKC_FPF=5")


def _main_defines(sfx):
  if not sfx and "-DOKC_WGS=1" in tuple(MOE_DEFINES):
    return MAIN_FALLBACK_DEFINES
  return MOE_DEFINES


def _wgs_on():
  return any("-DOKC_WGS=1" in tuple(d) for d in (MOE_DEFINES, MOE_TINY_DEFINES, MOE_TINY2_DEFINES, MOE_MID_DEFINES))



# LDD19: GF2 (port of LDD19's muse_moe_w4a16 GF(2) re-layout): the WGS builds' code words get a one-time, weight-independent
# 32x32 GF(2) bit mix T in prepare (_wgs_copy); the WGS consumer decodes them with dq4g (r0 = w & M, r1 = (w & M>>4) << 4,
# r2 / r3 = ((w ^ w >> 8) & M>>2 / M>>6) << 2 / << 6: 4 LOP3 + 1 SHF + 3 IMAD.SHL), returning exactly DQ3's dq4 of the
# unmixed word. T = Dg^-1 o D3 is derived and asserted below. Unit-order copies and the non-WGS sources are untouched.
LDD19_GF2 = int(__import__("os").environ.get("LDD19_GF2", "1")) if LDD19_DQ3 else 0


def _ldd19_gf2_cols():
    F, M = 0xFFFFFFFF, 0x81C081C0
    d3 = lambda q: (q & M, ((q << 13) | (q >> 19)) & M, ((q << 10) | (q >> 22)) & M, ((q & 0x001B001B) * 0x1084) & M)
    dg = lambda w: (w & M, (w & M >> 4) << 4, ((w ^ w >> 8) & M >> 2) << 2, ((w ^ w >> 8) & M >> 6) << 6 & F)
    vec = lambda r: sum(((x >> p) & 1) << (8 * i + j) for i, x in enumerate(r) for j, p in enumerate((6, 7, 8, 15, 22, 23, 24, 31)))
    A = [[(vec(dg(1 << c)) >> r) & 1 for c in range(32)] + [int(r == c) for c in range(32)] for r in range(32)]
    for c in range(32):   # GF(2) inverse of Dg
        p = next(r for r in range(c, 32) if A[r][c])
        A[c], A[p] = A[p], A[c]
        A = [[a ^ b for a, b in zip(A[r], A[c])] if r != c and A[r][c] else A[r] for r in range(32)]
    ap = lambda cols, x: __import__("functools").reduce(lambda a, i: a ^ cols[i] if x >> i & 1 else a, range(32), 0)
    gi = [sum(A[r][32 + c] << r for r in range(32)) for c in range(32)]
    cols = [ap(gi, vec(d3(1 << i))) for i in range(32)]
    rnd = __import__("random").Random(19)
    for q in [1 << i for i in range(32)] + [rnd.getrandbits(32) for _ in range(512)]:
        assert dg(ap(cols, q)) == d3(q)
    return cols


_LDD19_GF2_T = _ldd19_gf2_cols() if LDD19_GF2 else None


def _ldd19_gf2_map(wn):
    x, out = wn.to(torch.int64) & 0xFFFFFFFF, torch.zeros(wn.shape, dtype=torch.int64, device=wn.device)
    for i, c in enumerate(_LDD19_GF2_T):
        out ^= ((x >> i) & 1) * c
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)


if LDD19_GF2:
    MOE_CUDA_SRC = _ldd19_derive(MOE_CUDA_SRC, [
      ('before', r"""__device__ void early_issue(const Params& p, Smem& S);""", r"""#ifndef LDD19_GF2
#define LDD19_GF2 0
#endif
template <uint32_t C>
__device__ __forceinline__ uint32_t gf2_xm(uint32_t a, uint32_t b) {  // LDD19: GF2: (a ^ b) & C as one LOP3
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(d) : "r"(a), "r"(b), "n"(C));
  return d;
}
__device__ __forceinline__ void dq4g(uint32_t w, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;   // LDD19: GF2: == dq4 (DQ3) of the unmixed word
  const uint32_t t = w >> 8;
  r0 = w & M;
  r1 = (w & (M >> 4)) * 16u;
  r2 = gf2_xm<(M >> 2)>(w, t) * 4u;
  r3 = gf2_xm<(M >> 6)>(w, t) * 64u;
}
"""),
      (r"""          dq4(q, r0, r1, r2, r3);
#if LDD19_KPERM""", r"""#if LDD19_GF2
          dq4g(q, r0, r1, r2, r3);   // LDD19: GF2
#else
          dq4(q, r0, r1, r2, r3);
#endif
#if LDD19_KPERM"""),
    ])




# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open: skinny swap-AB tensor-core GEMV/GEMM with
losslessly compressed BF16 weights.

out[m, n] = x[m, k] @ W[n, k]^T.  For m <= 64 every weight byte is streamed once from HBM
straight into registers, in mma.m16n8k16 A-fragment order (prepared once), and multiplied
against x (staged in shared memory) on BF16 tensor cores with FP32 accumulation.

Lossless 12-bit weights (BF16 projections): each BF16 weight is stored as its low byte
(exponent bit 0 + 7 mantissa bits, raw) and a 4-bit code for its high byte (sign + exponent
bits 7..1, as an offset into an 8-value window chosen per matrix).  The kernel rebuilds the
exact BF16 bits in registers (a few integer ops per weight).  The rare weights whose exponent
falls outside the window (tiny magnitudes) are exact-corrected: the kernel adds
x[:, k] * (w - w_decoded) for each of them (both factors exact in FP32), so every output
equals the product with the original weights up to FP32 accumulation order.

Split-K across warps (shared-memory reduction) and across CTAs (FP32 partials + a
self-resetting semaphore; the last CTA of a row block reduces) keeps ~128 CTAs streaming on
every shape.  The router's FP32 weights run as a BF16 hi + lo pair (concatenated along K)
with FP32 accumulation, as SGLang does above 8 tokens.
"""

import torch
import torch.nn.functional as F
import sys
import os as _os
bfp__SK2_DELAY = int(_os.environ.get("SK2_DELAY_ESC", "0"))


# LDD19 v24: bf16_proj sources = Keell's copies of the same file with these exact edits.
bfp_CUDA_SRC = _ldd19_derive(kl_bfp_CUDA_SRC, [
  ('after', r"""#include <stdint.h>""", r"""
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
#if __CUDA_ARCH__ >= 800
  uint64_t pol; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(pol));
#else
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
#endif
  return r;
}"""),
])

bfp_MBAR_HELPERS = _ldd19_derive(kl_bfp_MBAR_HELPERS, [
  ('after', r"""                 " selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
  } while (!ok);
}""", r"""
#endif"""),
  ('after', r"""  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}""", r"""
#if __CUDA_ARCH__ >= 900"""),
  ('before', r"""}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {""", r"""#else
  mbar_wait(bar, parity);
#endif
"""),
  ('after', r"""__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {""", r"""
#if __CUDA_ARCH__ >= 900"""),
  ('before', r"""  return ok != 0;""", r"""#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
#endif
"""),
  ('after', r"""  uint32_t ok;""", r"""
#if __CUDA_ARCH__ >= 900"""),
])

bfp_WG_HELPERS = _ldd19_derive(kl_bfp_WG_HELPERS, [
  ('before', r"""__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t saddr) {""", r"""__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
"""),
])


def bfp_wgmma_defs(bns):
  out = []
  for bn in bns:
    nacc = bn // 2
    regs = ", ".join(f"%{i}" for i in range(nacc))
    a0 = nacc
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
    pi = a0 + 5
    out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_bn(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{pi}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, {{%{a0}, %{a0+1}, %{a0+2}, %{a0+3}}}, %{a0+4}, p, 1, 1, 0;\\n"
      "}}\\n"
      : {cons}
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(1));
}}""")
  return "\n".join(out)


bfp_TMA_SRC = _ldd19_derive(kl_bfp_TMA_SRC, [
  ('before', r"""  if constexpr (WGM) {
#pragma unroll""", r"""#endif
"""),
  ('after', r"""          }
        }
      }
    }
  }""", r"""
#if __CUDA_ARCH__ >= 900"""),
  ('before', r"""    } else {
#pragma unroll""", r"""#else
      const int g8 = lane >> 2, t8 = lane & 3;
#pragma unroll
      for (int u = 0; u < KS / 4; ++u) {
        const uint4 L0 = wr[3 * u], L1 = wr[3 * u + 1], C = wr[3 * u + 2];
        const int kl = kw0 + (st * KS + u * 4) * 16;
        const unsigned char* tile = reinterpret_cast<const unsigned char*>(xs) + (kl >> 6) * TOK * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[4];
          decode8(j < 2 ? (j == 0 ? L0.x : L0.z) : (j == 2 ? L1.x : L1.z),
                  j < 2 ? (j == 0 ? L0.y : L0.w) : (j == 2 ? L1.y : L1.w),
                  j == 0 ? C.x : j == 1 ? C.y : j == 2 ? C.z : C.w, base4, af);
#pragma unroll
          for (int jj = 0; jj < MT; ++jj) {
            const int tok = jj * 8 + g8;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t8);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t8);
            mma_bf16(wacc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
        }
      }
#endif
"""),
  ('after', r"""    } else if constexpr (WGM) {""", r"""
#if __CUDA_ARCH__ >= 900"""),
  (r"""        if constexpr (WGM) fence_async_smem();""", r"""#if __CUDA_ARCH__ >= 900
        if constexpr (WGM) fence_async_smem();
#endif"""),
  (r"""      }
    }
    return;""", r"""      }
#else
      for (int off = lane * 16; off < SLOT; off += 512)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst + off), "l"(src + off) : "memory");
      asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
    }
    return;"""),
  ('before', r"""      if (lane == 0) {""", r"""#if __CUDA_ARCH__ >= 900
"""),
  (r"""    const int st0 = NSTAGE < nst ? NSTAGE : nst;""", r"""#if __CUDA_ARCH__ >= 900
    const int st0 = NSTAGE < nst ? NSTAGE : nst;
#else
    const int st0 = 0;
#endif"""),
  ('after', r"""      bulk_g2s(ring + st * SLOT, wsrc + (size_t)st * SLOT, SLOT, full0 + 8 * st);
    }""", r"""
#endif"""),
  ('after', r"""    for (int i = 0; i < XP; ++i) mbar_init(xbar0 + 8 * i, NC);""", r"""
#if __CUDA_ARCH__ >= 900"""),
  (r"""      mbar_init(full0 + 8 * i, 1);""", r"""#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif"""),
])

bfp_MAP_SRC = _ldd19_derive(kl_bfp_MAP_SRC, [
  ('before', r"""static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {""", r"""#endif
"""),
  ('after', r"""#include <cstring>""", r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(map), "r"(bar), "r"(c0), "r"(c1) : "memory");
}"""),
])

bfp_GEMM_SRC = r"""@@WGMMA_SS_DEFS@@
template <int NWG, int BN, int STAGES, bool AS, typename OutT>
__global__ void __launch_bounds__(NWG * 128 + 128, 1)
gemm90_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
              const uint8_t* __restrict__ Wst,
              OutT* __restrict__ out, float* __restrict__ ws, int* __restrict__ counters,
              int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  constexpr int A_BYTES = BM * 128;
  constexpr int B_BYTES = BN * 128;
  constexpr int ST_BYTES = A_BYTES + B_BYTES;
  constexpr int NACC = BN / 2;
  constexpr int NT = NWG * 128;
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  const uint32_t bars = sbase + STAGES * ST_BYTES;
  const uint32_t full0 = bars, empty0 = bars + 8 * STAGES;
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int rb = blockIdx.x, sp = blockIdx.y;
  const int nkb_all = Kp >> 6;
  const int kb0 = (int)(((long long)sp * nkb_all) / splits);
  const int kb1 = (int)(((long long)(sp + 1) * nkb_all) / splits);
  const int nkb = kb1 - kb0;
  if (tid == 0) {
    for (int i = 0; i < STAGES; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NWG); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (splits > 1) asm volatile("prefetch.global.L2 [%0];" :: "l"(counters + rb));
  }
  __syncthreads();
  if (warp >= NWG * 4) {
    if (warp == NWG * 4 && lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapA) : "memory");
      asm volatile("prefetch.tensormap [%0];" :: "l"(&mapB) : "memory");
      for (int i = 0; i < nkb; ++i) {
        const int slot = i % STAGES;
        if (i >= STAGES) mbar_wait(empty0 + 8 * slot, ((i / STAGES) - 1) & 1);
        mbar_expect_tx(full0 + 8 * slot, ST_BYTES);
        const int kg = (kb0 + i) * 64;
        if constexpr (AS)
          bulk_g2s(sbase + slot * ST_BYTES, Wst + ((size_t)rb * nkb_all + kb0 + i) * A_BYTES, A_BYTES, full0 + 8 * slot);
        else
          tma_load_2d_ef(sbase + slot * ST_BYTES, &mapA, full0 + 8 * slot, kg, rb * BM);
        tma_load_2d_el(sbase + slot * ST_BYTES + A_BYTES, &mapB, full0 + 8 * slot, kg % Kx, 0);
      }
    }
    return;
  }
  const int wg = warp >> 2;
  float acc[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) acc[i] = 0.f;
  for (int i = 0; i < nkb; ++i) {
    const int slot = i % STAGES;
    mbar_wait(full0 + 8 * slot, (i / STAGES) & 1);
    const uint32_t a_st = sbase + slot * ST_BYTES + wg * (64 * 128);
    const uint32_t b_st = sbase + slot * ST_BYTES + A_BYTES;
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    wg_fence();
#pragma unroll
    for (int j = 0; j < 4; ++j)
      wgmma_ss<BN>(acc, make_desc_sw128(a_st + j * 32), make_desc_sw128(b_st + j * 32));
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
    if (i > 0 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % STAGES));
  }
  wg_wait<0>();
#pragma unroll
  for (int q = 0; q < NACC; ++q) asm volatile("" : "+f"(acc[q]) :: "memory");
  const int g = lane >> 2, t4 = lane & 3;
  const int row_w = rb * BM + (warp & 3) * 16 + wg * 64;
  if (splits == 1) {
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
      const int t0 = 8 * j + 2 * t4;
      if (t0 < M) {
        out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(acc[4 * j]);
        out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 2]);
      }
      if (t0 + 1 < M) {
        out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(acc[4 * j + 1]);
        out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(acc[4 * j + 3]);
      }
    }
    return;
  }
  {
    float4* wsp = reinterpret_cast<float4*>(ws) + (size_t)(rb * splits + sp) * (NACC / 4) * NT;
#pragma unroll
    for (int j = 0; j < NACC / 4; ++j)
      __stcg(wsp + j * NT + tid, make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]));
  }
  __threadfence();
  named_bar(1, NT);
  if (tid == 0) {
    int old = atomicAdd(counters + rb, 1);
    int target = (old / splits + 1) * splits;
    okc_spin_ge(counters + rb, target, 32);
  }
  named_bar(1, NT);
  __threadfence();
  const int j0 = (int)(((long long)sp * (NACC / 4)) / splits);
  const int j1 = (int)(((long long)(sp + 1) * (NACC / 4)) / splits);
  const float4* base = reinterpret_cast<const float4*>(ws) + (size_t)rb * splits * (NACC / 4) * NT;
  for (int j = j0; j < j1; ++j) {
    float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
    int q = 0;
    for (; q + 4 <= splits; q += 4) {
      float4 v0 = __ldcg(base + ((size_t)(q + 0) * (NACC / 4) + j) * NT + tid);
      float4 v1 = __ldcg(base + ((size_t)(q + 1) * (NACC / 4) + j) * NT + tid);
      float4 v2 = __ldcg(base + ((size_t)(q + 2) * (NACC / 4) + j) * NT + tid);
      float4 v3 = __ldcg(base + ((size_t)(q + 3) * (NACC / 4) + j) * NT + tid);
      s.x += v0.x + v1.x + v2.x + v3.x; s.y += v0.y + v1.y + v2.y + v3.y;
      s.z += v0.z + v1.z + v2.z + v3.z; s.w += v0.w + v1.w + v2.w + v3.w;
    }
    for (; q < splits; ++q) {
      float4 v = __ldcg(base + ((size_t)q * (NACC / 4) + j) * NT + tid);
      s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
    }
    const int t0 = 8 * j + 2 * t4;
    if (t0 < M) {
      out[(size_t)t0 * N + row_w + g] = cvt_out<OutT>(s.x);
      out[(size_t)t0 * N + row_w + g + 8] = cvt_out<OutT>(s.z);
    }
    if (t0 + 1 < M) {
      out[(size_t)(t0 + 1) * N + row_w + g] = cvt_out<OutT>(s.y);
      out[(size_t)(t0 + 1) * N + row_w + g + 8] = cvt_out<OutT>(s.w);
    }
  }
}
template <int NWG, int BN, int STAGES, bool AS, typename OutT>
static void launch_gemm90(const at::Tensor& W, const at::Tensor& Wst, const at::Tensor& x, at::Tensor& out, at::Tensor& ws,
                          at::Tensor& counters, int M, int N, int Kp, int Kx, int splits) {
  constexpr int BM = NWG * 64;
  size_t smem = 1024 + (size_t)STAGES * (BM * 128 + BN * 128) + 16 * STAGES + 64;
  auto kern = gemm90_kernel<NWG, BN, STAGES, AS, OutT>;
  static bool attr_done = false;
  if (!attr_done) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    attr_done = true;
  }
  CUtensorMap mA = make_map_2d(W.data_ptr(), (uint64_t)W.size(0), (uint64_t)W.size(1), BM);
  CUtensorMap mB = make_map_2d(x.data_ptr(), (uint64_t)M, (uint64_t)Kx, BN);
  dim3 grid(N / BM, splits);
  kern<<<grid, NWG * 128 + 128, smem, at::cuda::getCurrentCUDAStream()>>>(
      mA, mB, reinterpret_cast<const uint8_t*>(Wst.data_ptr()),
      reinterpret_cast<OutT*>(out.data_ptr()), ws.data_ptr<float>(), counters.data_ptr<int>(),
      M, N, Kp, Kx, splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void gemm90(at::Tensor W, at::Tensor Wst, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as) {
  bool out_f32 = out.scalar_type() == at::kFloat;
  @@GEMM_DISPATCH@@
  TORCH_CHECK(false, "gemm90: no config");
}
"""


bfp_SK2_SRC = _ldd19_derive(kl_bfp_SK2_SRC, [
  (r"""  }
}""", r"""  }
#endif
}"""),
  (r"""      if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }""", r"""    }"""),
  (r"""  if constexpr (CL == 1) {
    if (tid == 0 && !(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
  }
  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");""", r"""  if constexpr (CL > 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");"""),
  (r"""    } else {
      if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      if (!(dgv & 4))
        for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
      if (COMP) {
        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }
    }""", r"""    }"""),
  (r"""        mbar_expect_tx(pbar, pmax * 8);
        bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
      }""", r"""      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {"""),
  (r"""    if constexpr (CL == 1) {
      if (COMP) {""", r"""    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
    if (!(dgv & 4))
      for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);
    if (COMP) {
      if constexpr (CL == 1) {"""),
  (r"""  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;""", r"""  const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;"""),
  ('before', r"""  const int dgv = SK2_DIAG ? dg : 0;""", r"""#if __CUDA_ARCH__ >= 900
"""),
  (r"""#define SK2_DIAG 0""", r"""#ifndef SK2_DIAG
#define SK2_DIAG 0
#endif"""),
  (r"""#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))""", r"""#ifndef SK2_DELAY_ESC
#define SK2_DELAY_ESC 0
#endif
#if !(SK2_DELAY_ESC)
#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#endif
"""),
])


bfp_WIDE_SRC = _ldd19_derive(kl_bfp_WIDE_SRC, [
  ('before', r"""  named_bar(1, NT);
  {""", r"""#endif
"""),
  (r"""  }
  wg_wait<0>();""", r"""#else
    named_bar(1, NT);
    if (producer && i + STAGES < nst) issue(i + STAGES);
#endif
  }
#if __CUDA_ARCH__ >= 900
  wg_wait<0>();"""),
  ('before', r"""    if (i > 0) {""", r"""#if __CUDA_ARCH__ >= 900
"""),
  ('before', r"""    }
    }
#if __CUDA_ARCH__ >= 900""", r"""#else
      const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
      for (int jj = 0; jj < TOK / 8; ++jj) {
        const int tok = jj * 8 + g;
        const unsigned char* row = tile + tok * 128;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
        mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
      }
#endif
"""),
  ('after', r"""      decode8(lw0, lw1, cw, base4, af);""", r"""
#if __CUDA_ARCH__ >= 900"""),
  ('before', r"""        }""", r"""#else
          const unsigned char* tile = gbase + slot * ST_BYTES;
#pragma unroll
          for (int jj = 0; jj < TOK / 8; ++jj) {
            const int tok = jj * 8 + g;
            const unsigned char* row = tile + tok * 128;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(row + (((2 * j) ^ (tok & 7)) << 4) + 4 * t4);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(row + (((2 * j + 1) ^ (tok & 7)) << 4) + 4 * t4);
            mma_bf16(acc + 4 * jj, af[0], af[1], af[2], af[3], b0, b1);
          }
#endif
"""),
  ('after', r"""                       : "r"(wa + h * 2048 + j * 512));""", r"""
#if __CUDA_ARCH__ >= 900"""),
  ('after', r"""  __syncthreads();""", r"""
#else
  __syncthreads();
  if (producer)
    for (int i = 0; i < STAGES && i < nst; ++i) issue(i);
#endif"""),
  (r"""  };
  const bool producer = tid == 0;""", r"""#else
    for (int idx = lane; idx < TOK * 8; idx += 32) {
      const int t = idx >> 3, c = idx & 7;
      const uint32_t dst = st + t * 128 + ((c ^ (t & 7)) << 4);
      const __nv_bfloat16* src = x + (size_t)(t0z + (t < Mz ? t : 0)) * K + (u0 + i) * 64 + c * 8;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(t < Mz ? 16 : 0) : "memory");
    }
    const char* wa = reinterpret_cast<const char*>(wsrc) + (size_t)i * A_BYTES;
    for (int off = lane * 16; off < A_BYTES; off += 512)
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(st + B_BYTES + off), "l"(wa + off) : "memory");
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(full0 + 8 * slot) : "memory");
#endif
  };
#if __CUDA_ARCH__ >= 900
  const bool producer = tid == 0;
#else
  const bool producer = warp == 0;
#endif
#if __CUDA_ARCH__ >= 900"""),
  ('before', r"""    mbar_expect_tx(full0 + 8 * slot, ST_BYTES);""", r"""#if __CUDA_ARCH__ >= 900
"""),
  (r"""    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");""", r"""#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif"""),
  (r"""      mbar_init(full0 + 8 * i, 1);""", r"""#if __CUDA_ARCH__ >= 900
      mbar_init(full0 + 8 * i, 1);
#else
      mbar_init(full0 + 8 * i, 32);
#endif"""),
])


def bfp_wgmma_ss_defs(bns):
  out = []
  for bn in bns:
    nacc = bn // 2
    regs = ", ".join(f"%{i}" for i in range(nacc))
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(nacc))
    a, b, p = nacc, nacc + 1, nacc + 2
    out.append(f"""
template <int BN_> __device__ __forceinline__ typename std::enable_if<BN_ == {bn}>::type
wgmma_ss(float* d, uint64_t da, uint64_t db) {{
  asm volatile("{{\\n"
      ".reg .pred p;\\n"
      "setp.ne.b32 p, %{p}, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{bn}k16.f32.bf16.bf16 "
      "{{{regs}}}, %{a}, %{b}, p, 1, 1, 0, 0;\\n"
      "}}\\n"
      : {cons}
      : "l"(da), "l"(db), "r"(1));
}}""")
  return "\n".join(out)


bfp_CPP_SRC = r"""void skinny_tma_a(torch::Tensor Wp, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
                torch::Tensor ptab, int64_t pmax, int64_t base4,
                int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t mt, int64_t rg, int64_t wsn,
                int64_t comp, int64_t nrg, int64_t ks, int64_t dg);
void gemm90(torch::Tensor W, torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters,
            int64_t M, int64_t N, int64_t Kp, int64_t Kx, int64_t splits, int64_t nwg, int64_t bn, int64_t as);
void wide(torch::Tensor Wst, torch::Tensor x, torch::Tensor out, torch::Tensor ws, torch::Tensor counters, torch::Tensor ptab,
          int64_t pmax, int64_t base4, int64_t M, int64_t N, int64_t K, int64_t splits, int64_t nwg, int64_t tok);
void gq(torch::Tensor W, torch::Tensor x, torch::Tensor out, int64_t M, int64_t N, int64_t K, int64_t bn, int64_t nwg,
        int64_t cl, int64_t nstage);
void sk2(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, torch::Tensor dbgt);
void sk2f(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr, torch::Tensor ptab,
          int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M, int64_t tokr);
void sk2_set_pdl(int64_t n);
"""
bfp_w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
bfp_COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}

bfp__ext = None
bfp_TMA_CONFIGS = {
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
bfp_USE_WGM = True
bfp_WGM_SM80_TEST = False


def bfp__tma_nstage(nw, comp, sm90, ks, hl=False):
  chunk = (ks // 4) * 1536 if comp else ks * 512 * (2 if hl else 1)
  budget = (112 if sm90 else 40) * 1024
  return max(2, min(8, budget // (nw * chunk)))


bfp_TMA_TU = {}


bfp_TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def bfp__tma_cfg_items():
  items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(bfp_TMA_CONFIGS.items())]
  items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(bfp_TMA_ALT.items())]
  return items


def bfp__tma_srcs(sm90):
  lines = {}
  for key, (splits, rg, wsn, ks) in bfp__tma_cfg_items():
    if key[0] == "alt":
      layer, mtk = key[1], bfp__mt_key(key[2])
    else:
      layer, mtk = key
    mt = max(mtk, 1)
    f32 = layer == "router"
    ot = "float" if f32 else "__nv_bfloat16"
    comp = 0 if f32 else 1
    hl = f32
    ns = bfp__tma_nstage(rg * wsn, comp, sm90, ks, hl)
    if f32 and mt >= 4:
      ns = min(ns, 3)
    n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else bfp_w_rows[layer]
    wgm = bool((sm90 or bfp_WGM_SM80_TEST) and comp and bfp_USE_WGM and rg % 4 == 0 and mtk >= 2)
    line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
            f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
            f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
    lines.setdefault(line, []).append(key)
  keys = list(lines)
  for ln in keys:
    for cfg in lines[ln]:
      bfp_TMA_TU[cfg] = 0
  return bfp_TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


bfp_GEMM_CONFIGS = {
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


bfp_DIAG_LAYERS = ()
bfp_GEMM_DIAG = False


def bfp_swizzle_tiles(w, bm):
  n, k = w.shape
  t = w.view(n // bm, bm, k // 64, 8, 8).permute(0, 2, 1, 3, 4)
  r = torch.arange(bm, device=w.device).view(bm, 1)
  p = torch.arange(8, device=w.device).view(1, 8)
  src_c = (p ^ (r % 8)).view(1, 1, bm, 8, 1).expand(t.shape[0], t.shape[1], bm, 8, 8)
  return torch.gather(t, 3, src_c).contiguous()


def bfp__gemm_src(sm90):
  if not sm90:
    return ('void gemm90(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, '
            'int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) { TORCH_CHECK(false, "gemm90 needs sm_90"); }')
  lines = []
  for (layer, bn), (nwg, splits, stages) in sorted(bfp_GEMM_CONFIGS.items()):
    f32 = layer == "router"
    ot = "float" if f32 else "__nv_bfloat16"
    for a_s in ((0, 1) if (not f32 and bfp_GEMM_DIAG) else (0,)):
      lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                   f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
  bns = sorted({bn for (_, bn) in bfp_GEMM_CONFIGS})
  return (bfp_GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", bfp_wgmma_ss_defs(bns))
          .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))


bfp_GQ_SRC = _ldd19_derive(kl_bfp_GQ_SRC, [
  (r"""    }
  }
}""", r"""    }
  }
#endif
}"""),
  ('after', r"""          __nv_bfloat16* __restrict__ out, int M, int N, int K) {""", r"""
#if __CUDA_ARCH__ >= 900"""),
])

bfp_GQ_CONFIGS = {
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 7),
}
bfp_USE_GQ = True


def bfp__gq_src(sm90):
  if not sm90:
    return ('void gq(at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, '
            'int64_t) { TORCH_CHECK(false, "gq needs sm_90"); }')
  lines = []
  for (layer, bn), (nwg, cl, ns) in sorted(bfp_GQ_CONFIGS.items()):
    lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                 f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
  return bfp_GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__gq_bn(m):
  return 64 if m <= 64 else 128 if m <= 128 else 256


def bfp__run_gq(x, layer, m, n, k, state):
  nwg, cl, ns = bfp_GQ_CONFIGS[(layer, bfp__gq_bn(m))]
  out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
  bfp__get_ext().gq(state["raw"][layer], x, out, m, n, k, bfp__gq_bn(m), nwg, cl, ns)
  return out



bfp_WIDE_CONFIGS = {
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


def bfp__wide_key(m):
  return (64, 1) if m <= 64 else (128, 1) if m <= 128 else (128, 2)


def bfp__wide_src(sm90):
  lines = []
  for (layer, tok, nz), (nwg, splits, stages) in sorted(bfp_WIDE_CONFIGS.items()):
    hl = layer == "router"
    if not sm90:
      stages = min(stages, 3 if not hl else 2)
    ot = "float" if hl else "__nv_bfloat16"
    lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                 f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                 f"M, N, K, splits); return; }}")
  return bfp_WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def bfp__get_ext():
  global bfp__ext
  if bfp__ext is None:
    major, minor = torch.cuda.get_device_capability()
    sm90 = major >= 9
    arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
    common = bfp_CUDA_SRC.replace("@@MBAR@@", bfp_MBAR_HELPERS).replace("@@BODY@@", bfp_WG_HELPERS if sm90 else "")
    defs = bfp_wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
    cuda_src = (common + defs + bfp__tma_srcs(sm90) + bfp_MAP_SRC + bfp__gemm_src(sm90) + bfp__wide_src(sm90) + bfp__sk2_src(sm90)
                + bfp__gq_src(sm90))
    bfp__ext = _ldd19_inline(
        name=f"dl_okc_bf16proj_v86y4_{arch}" + (f"_dly{bfp__SK2_DELAY}" if bfp__SK2_DELAY else ""),
        cpp_sources="#include <torch/extension.h>\n" + bfp_CPP_SRC,
        cuda_sources=cuda_src,
        functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],
        extra_cuda_cflags=["-O3", "-split-compile=0", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={bfp__SK2_DELAY}"] if bfp__SK2_DELAY else []),   # LDD19: v20
        no_implicit_headers=True,
        verbose=False,
    )
  return bfp__ext


def bfp_pack_fragments(w: torch.Tensor) -> torch.Tensor:
  n, k = w.shape
  t = w.view(n // 16, 2, 8, k // 16, 2, 4, 2)
  t = t.permute(0, 3, 2, 5, 4, 1, 6)
  return t.contiguous().view(n // 16, k // 16, 32, 8)


def bfp_pack_compressed(w: torch.Tensor):
  n, k = w.shape
  if n % 16 or k % 64:
    return None
  bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
  hi = bits >> 8
  e71 = hi & 0x7F
  hist = torch.bincount(e71.flatten(), minlength=128)
  win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
  base = int(win.argmax())
  o = (e71 - base).clamp(0, 7)
  code = ((hi >> 7) << 3) | o
  dec_hi = ((hi >> 7) << 7) | (base + o)
  dec = ((dec_hi << 8) | (bits & 0xFF)).to(torch.int16).view(torch.bfloat16)
  esc = (e71 < base) | (e71 > base + 7)
  n_esc = int(esc.sum())
  if n_esc > n * k // 200:
    return None
  rows, cols = esc.nonzero(as_tuple=True)
  wtrue = w[rows, cols].float()
  wdec = dec[rows, cols].float()
  delta = wtrue - wdec
  if not bool(((wdec + delta) == wtrue).all()):
    return None
  lo_f = bfp_pack_fragments((bits & 0xFF).to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xFF
  cd_f = bfp_pack_fragments(code.to(torch.int16).view(torch.bfloat16)).view(torch.int16) & 0xF
  nrg, S = n // 16, k // 16
  lo_f = lo_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
  cd_f = cd_f.view(nrg, S // 4, 4, 32, 8).to(torch.uint8)
  lo_p = lo_f.view(nrg, S // 4, 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).reshape(nrg, S // 4, 2, 32, 16)
  cw = (cd_f[..., 0:4] | (cd_f[..., 4:8] << 4))
  cw = cw.permute(0, 1, 3, 2, 4).reshape(nrg, S // 4, 1, 32, 16)
  packed = torch.cat([lo_p, cw], dim=2).transpose(0, 1).contiguous()
  order = torch.argsort(rows * k + cols)
  rows, cols, delta = rows[order], cols[order], delta[order]
  rg_idx = rows // 16
  poff = torch.zeros(nrg + 1, dtype=torch.int32, device=w.device)
  poff[1:] = torch.cumsum(torch.bincount(rg_idx, minlength=nrg), 0).to(torch.int32)
  return {
      "packed": packed,
      "base4": base * 0x01010101,
      "poff": poff,
      "pk": cols.to(torch.int32).contiguous(),
      "pr": (rows % 16).to(torch.int32).contiguous(),
      "pd": delta.to(torch.float32).contiguous(),
      "n_esc": n_esc,
  }


def bfp_build_patch_table(c, n_cfg, k, splits, rg):
  dev = c["pk"].device
  rows_cta = rg * 16
  nrb = n_cfg // rows_cta
  nunits = k // 64
  poff = c["poff"].long()
  nrg_all = poff.numel() - 1
  rgi = torch.repeat_interleave(torch.arange(nrg_all, device=dev), poff[1:] - poff[:-1])
  rows = rgi * 16 + c["pr"].long()
  cols = c["pk"].long()
  keep = rows < n_cfg
  rows, cols, delta = rows[keep], cols[keep], c["pd"][keep]
  unit = cols // 64
  spl = ((unit + 1) * splits + nunits - 1) // nunits - 1
  k_start = (spl * nunits // splits) * 64
  cta = spl * nrb + rows // rows_cta
  ncta = nrb * splits
  counts = torch.bincount(cta, minlength=ncta)
  pmax = int(counts.max().item()) + 1 if rows.numel() else 1
  pmax += pmax % 2
  order = torch.argsort(cta, stable=True)
  cta, rows, cols, delta, k_start = cta[order], rows[order], cols[order], delta[order], k_start[order]
  start = torch.zeros(ncta + 1, dtype=torch.long, device=dev)
  start[1:] = torch.cumsum(counts, 0)
  slot = torch.arange(cta.numel(), device=dev) - start[cta] + 1
  tab = torch.zeros(ncta, pmax, 2, dtype=torch.int32, device=dev)
  tab[:, 0, 0] = counts.to(torch.int32)
  tab[cta, slot, 0] = (((rows % rows_cta) << 16) | (cols - k_start)).to(torch.int32)
  tab[cta, slot, 1] = delta.contiguous().view(torch.int32)
  return tab.contiguous(), pmax


def bfp_build_stream(src, comp, n_cfg, k, splits, rg, wsn, ks):
  dev = src.device
  nw = rg * wsn
  nrb = (n_cfg // 16) // rg
  ar = lambda n: torch.arange(n, device=dev)
  if wsn == 1:
    s_all = (k // 16) // ks
    rb = ar(nrb).view(-1, 1, 1, 1)
    st = ar(s_all).view(1, -1, 1, 1)
    w = ar(nw).view(1, 1, -1, 1)
    rg_idx = rb * rg + w
    if comp:
      u_idx = st * (ks // 4) + ar(ks // 4).view(1, 1, 1, -1)
      return src[rg_idx, u_idx].contiguous()
    s_idx = st * ks + ar(ks).view(1, 1, 1, -1)
    return src[rg_idx, s_idx].contiguous()
  kc = k // splits
  ns = kc // (16 * wsn)
  nst = ns // ks
  sp = ar(splits).view(-1, 1, 1, 1, 1)
  rb = ar(nrb).view(1, -1, 1, 1, 1)
  st = ar(nst).view(1, 1, -1, 1, 1)
  w = ar(nw).view(1, 1, 1, -1, 1)
  rg_idx = rb * rg + w % rg
  if comp:
    ku = ar(ks // 4).view(1, 1, 1, 1, -1)
    u_idx = sp * (kc // 64) + (w // rg) * (ns // 4) + st * (ks // 4) + ku
    out = src[rg_idx, u_idx]
  else:
    j = ar(ks).view(1, 1, 1, 1, -1)
    s_idx = sp * (kc // 16) + (w // rg) * ns + st * ks + j
    out = src[rg_idx, s_idx]
  return out.contiguous()


bfp_SK2_CONFIGS = [
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv16_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler16h_r100', 'unpooler', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
]
bfp_SK2_PICK = {
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o8h_r88',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8_r176',
    ('qkv_gate', 16): 'qkv16_r176',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16h_r100',
    ('unpooler', 32): 'unpooler32_r144',
}
bfp_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
bfp_USE_SK2 = True
bfp_WIDE_BF16 = False


def bfp__sk2_name(layer, m):
  return bfp_SK2_PICK[(layer, bfp__sk2_tokr(m))]


def bfp__sk2_tokr(m):
  return 8 if m <= 8 else 16 if m <= 16 else 32


def bfp__sk2_geom(cfg, tokr):
  name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
  hl = layer == "router"
  tu = 2048 if hl else 768
  ups = WSK * UPW
  wby = NWR * WSK * UPW * NT * tu
  xby = ups * tokr * 128
  sst = (xby + wby + 1023) // 1024 * 1024
  nst = -(-(bfp_SK2_K[layer] // (ups * 64)) // CL)
  mt = 2 if tokr == 32 else 1
  toks = 16 * mt
  cp = R + 1
  nesc = 4 if not hl else 1
  rp = R + ((40 - (R % 32)) % 32)
  extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
           + 64 * 8 + 128)
  cap = 200
  tail = name.rsplit("_r", 1)
  if len(tail) == 2 and tail[1].isdigit():
    cap = int(tail[1])
  ring = min(cap * 1024, 232448 - extra)
  nstage = max(2, min(16, nst, ring // sst))
  return hl, nstage


def bfp__sk2_src(sm90):
  if not sm90:
    return ('void sk2(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, '
            'int64_t, int64_t, int64_t, at::Tensor) { TORCH_CHECK(false, "sk2 needs sm_90"); }\n'
            'void sk2f(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, '
            'at::Tensor, at::Tensor, int64_t, int64_t) { TORCH_CHECK(false, "sk2f needs sm_90"); }\n'
            'void sk2_set_pdl(int64_t) {}')
  lines = []
  for ci, cfg in enumerate(bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    for tokr in (8, 16, 32):
      if bfp_SK2_PICK.get((layer, tokr)) != name:
        continue
      hl, nstage = bfp__sk2_geom(cfg, tokr)
      mt = 2 if tokr == 32 else 1
      ot = "float" if hl else "__nv_bfloat16"
      lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                   f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
  flines = []
  for ci, cfg in enumerate(bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    if name != "pooler":
      continue
    for tokr in (8, 16, 32):
      hl, nstage = bfp__sk2_geom(cfg, tokr)
      mt = 2 if tokr == 32 else 1
      nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
      flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                    f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                    f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                    f"rcnt.data_ptr<int>()); return; }}")
  return (bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))
          .replace("@@SK2F_DISPATCH@@", "\n  ".join(flines)))


def bfp_sk2_tileunits(w):
  n, k = w.shape
  t = w.view(n // 8, 8, k // 64, 4, 2, 4, 2)
  return t.permute(0, 2, 1, 5, 3, 4, 6).reshape(n // 8, k // 64, 32, 16)


def bfp_sk2_compress(w):
  n, k = w.shape
  tu = bfp_sk2_tileunits(w)
  bits = tu.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
  hi = bits >> 8
  low = bits & 0xFF
  e71 = hi & 0x7F
  sign = hi >> 7
  hist = torch.bincount(e71.flatten(), minlength=128)
  win = torch.stack([hist[b:b + 8].sum() for b in range(121)])
  base = int(win.argmax())
  o = (e71 - base).clamp(0, 7)
  code = (sign << 3) | o
  esc = (e71 < base) | (e71 > base + 7)
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
  return {"packed": packed, "base4": base * 0x01010101, "rows": rows, "cols": cols, "delta": delta.contiguous()}


def bfp_sk2_stream(tub, n_cfg, R, NT, NWR, WSK, UPW):
  dev = tub.device
  U = tub.shape[1]
  ups = WSK * UPW
  ar = lambda v: torch.arange(v, device=dev)
  s = ar(U // ups).view(-1, 1, 1, 1, 1)
  c = ar(n_cfg // R).view(1, -1, 1, 1, 1)
  w = ar(NWR * WSK).view(1, 1, -1, 1, 1)
  uu = ar(UPW).view(1, 1, 1, -1, 1)
  nt = ar(NT).view(1, 1, 1, 1, -1)
  tile = c * (R // 8) + (w // WSK) * NT + nt
  unit = s * ups + (w % WSK) * UPW + uu
  return tub[tile, unit].contiguous()


def bfp_sk2_table(comp, n_cfg, R):
  dev = comp["rows"].device
  rows, cols, delta = comp["rows"], comp["cols"], comp["delta"]
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
  tab[cta, slot, 1] = delta.view(torch.int32)
  return tab.contiguous(), pmax


def bfp_sk2_prepare(weights, state):
  state["sk2"] = {}
  if not (bfp_USE_SK2 and state["sm90"]):
    return
  dev = weights["o"].device
  shared = {}
  for ci, cfg in enumerate(bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    w = weights[layer]
    if layer == "router":
      hi = w.to(torch.bfloat16)
      lo = (w - hi.float()).to(torch.bfloat16)
      T, U = w.shape[0] // 8, w.shape[1] // 64
      tub = torch.cat([bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                       bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
      stream = bfp_sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
      tab, pmax, b4 = state["dummy"], 0, 0
    else:
      comp = state.setdefault("sk2_comp", {}).get(layer)
      if comp is None:
        comp = bfp_sk2_compress(w.contiguous())
        state["sk2_comp"][layer] = comp
      if comp is None:
        continue
      lkey = (layer, n_rows, R, NT, NWR, WSK, UPW)
      if lkey not in shared:
        shared[lkey] = (bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW),) + tuple(bfp_sk2_table(comp, n_rows, R))
      stream, tab, pmax = shared[lkey]
      b4 = comp["base4"]
    state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
  state.pop("sk2_comp", None)


def bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
  name = bfp__sk2_name(layer, m)
  ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
  out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
  ext = bfp__get_ext()
  ext.sk2_set_pdl(bfp_SK2_PDL)
  ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
  return out


def bfp_fused_pooler_router(x, state):
  m = x.shape[0]
  assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
  ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
  rstream = state["sk2"]["router"][1]
  if "fused_part" not in state:
    state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
    state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
  out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
  outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
  ext = bfp__get_ext()
  ext.sk2_set_pdl(bfp_SK2_PDL)
  ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, bfp__sk2_tokr(m))
  return out, outr


bfp_SK2_PDL = 0


def bfp__mt_key(m):
  return 0 if m == 1 else 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 8


def bfp_prepare(weights):
  ext = bfp__get_ext()
  dev = weights["qkv_gate"].device
  sm90_ = torch.cuda.get_device_capability()[0] >= 9
  old_paths = not (sm90_ and bfp_USE_SK2)
  state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
  state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
  ws_floats = 0
  for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
    mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
    ws_floats = max(ws_floats, splits * mmax * bfp_w_rows[lname])
  for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
    ws_floats = max(ws_floats, splits * mm * bfp_w_rows[lname])
  state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
  gemm_floats = 0
  for (lname, bn), (nwg, splits, stages) in bfp_GEMM_CONFIGS.items():
    gemm_floats = max(gemm_floats, (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
  for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
    gemm_floats = max(gemm_floats, nz * (bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
  state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
  if bfp_GEMM_DIAG:
    state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
  for name, w in weights.items():
    if name == "router":
      hi = w.to(torch.bfloat16)
      lo = (w - hi.float()).to(torch.bfloat16)
      state["router_hi_t"] = hi.t().contiguous()
      state["router_lo_t"] = lo.t().contiguous()
      state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()
      src = torch.stack([bfp_pack_fragments(hi), bfp_pack_fragments(lo)], dim=2).contiguous()
      comp, kk = False, w.shape[1]
      c = None
      nrg_r = w.shape[0] // 16
      u_all = w.shape[1] // 64
      t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()
      nwg_r = bfp_WIDE_CONFIGS[("router", 128, 1)][0]
      t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
      state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
      for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
        if lname == "router":
          state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
    else:
      w = w.contiguous()
      state["raw"][name] = w
      c = bfp_pack_compressed(w) if name in bfp_COMPRESS else None
      if c is not None:
        src, comp = c["packed"].transpose(0, 1).contiguous(), True
        state["base4"][name] = c["base4"]
        del c["packed"]
      else:
        src, comp = bfp_pack_fragments(w), False
      kk = w.shape[1]
    n = w.shape[0]
    state["comp"][name] = comp
    for (lname, mtk), (splits, rgc, wsn, ks) in bfp_TMA_CONFIGS.items():
      if lname != name or not old_paths:
        continue
      n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
      key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
      if key not in state["streams"]:
        state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
      ckey = (name, splits, rgc, wsn, ks, n_cfg)
      if ckey not in state["cnt"]:
        state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
      if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
        state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
    if comp and old_paths:
      for (lname, mm), (splits, rgc, wsn, ks) in bfp_TMA_ALT.items():
        if lname != name:
          continue
        n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
        key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
        if key not in state["streams"]:
          state["streams"][key] = bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
        ckey = (name, splits, rgc, wsn, ks, n_cfg)
        if ckey not in state["cnt"]:
          state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        if (name, splits, rgc, n_cfg) not in state["tabs"]:
          state["tabs"][(name, splits, rgc, n_cfg)] = bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
    for (lname, tok, nz), (nwg, splits, stages) in bfp_WIDE_CONFIGS.items():
      if lname != name or not comp:
        continue
      rgc = 4 * nwg
      key = (name, 0, rgc, 1, 4, n)
      if key not in state["streams"]:
        state["streams"][key] = bfp_build_stream(src, comp, n, kk, splits, rgc, 1, 4)
      if (name, splits, rgc, n) not in state["tabs"]:
        state["tabs"][(name, splits, rgc, n)] = bfp_build_patch_table(c, n, kk, splits, rgc)
      state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    del src
    state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
  state["sm90"] = sm90_
  state["old_paths"] = old_paths
  bfp_sk2_prepare(weights, state)
  if bfp_GEMM_DIAG and state["sm90"]:
    state["wtile"], state["diag_cnt"] = {}, {}
    for layer in bfp_DIAG_LAYERS:
      state["wtile"][layer] = bfp_swizzle_tiles(state["raw"][layer], 128)
      for a_s in (0, 1):
        for sp_ in sorted({bfp_GEMM_CONFIGS[(layer, 128)][1], bfp_GEMM_CONFIGS[(layer, 256)][1], 1}):
          state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(bfp_w_rows[layer] // 16, dtype=torch.int32, device=dev)
  bfp__selftest(weights, state)
  bfp__autotune(weights, state)
  torch.cuda.empty_cache()
  return state


bfp_ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
bfp_WIDE_SM80_TEST = False
bfp_ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def bfp__route(layer, m, sm90):
  for mmax, r in (bfp_ROUTE_ROUTER if layer == "router" else bfp_ROUTE_BF16):
    if m <= mmax:
      if r == "gemm" and not sm90:
        return "torch"
      if r == "wide" and not (sm90 or bfp_WIDE_SM80_TEST):
        return "torch"
      return r
  return "torch"


def bfp__route_key(layer, m, route):
  if route == "sk2":
    return (layer, route, bfp__sk2_name(layer, m), bfp__sk2_tokr(m))
  if route == "skinny":
    return (layer, route, bfp__mt_key(m))
  if route == "wide":
    return (layer, route) + bfp__wide_key(m)
  return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


bfp_SELFTEST_M = (1, 8, 16, 32, 33, 128, 200, 256)


def bfp__selftest(weights, state):
  state["disabled"] = set()
  g = torch.Generator(device=weights["o"].device)
  g.manual_seed(1234)
  for layer, w in weights.items():
    n, k = w.shape
    for m in bfp_SELFTEST_M:
      nn = 1024 if (layer == "unpooler" and m <= 8) else n
      x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
      ref = x.double() @ w[:nn].double().T
      if layer != "router":
        ref = ref.to(torch.bfloat16).double()
      tol = 2e-5 if layer == "router" else 5e-3
      for ch in bfp__candidates(layer, m, state):
        if ch[0] == "torch":
          continue
        key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
               else bfp__route_key(layer, m, ch[0]))
        try:
          out = bfp__run(ch, x, layer, m, nn, k, state).double()
          torch.cuda.synchronize()
          err = ((out - ref).norm() / ref.norm()).item()
        except RuntimeError as e:
          err = float("inf")
          if False: print(f"[okc] selftest {layer} m={m} {ch[0]}: {e}", file=sys.stderr)
        if not err < tol:
          state["disabled"].add(key)
          if False: print(f"[okc] selftest {layer} m={m} {ch[0]}: rel err {err:.3e} -> disabled", file=sys.stderr)


def bfp__run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
  splits, rg, wsn, ks = cfg
  mtk = bfp__mt_key(m)
  out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
  stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
  cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
  fn = bfp__get_ext().skinny_tma_a
  if state["comp"][layer]:
    tab, pmax = state["tabs"][(layer, splits, rg, n)]
    fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
       max(mtk, 1), rg, wsn, 1, bfp_w_rows[layer] // 16, ks, dg)
  else:
    fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
       max(mtk, 1), rg, wsn, 0, bfp_w_rows[layer] // 16, ks, dg)
  return out


def bfp__run_wide(x, layer, m, n, k, state):
  tok, nz = bfp__wide_key(m)
  nwg, splits, _ = bfp_WIDE_CONFIGS[(layer, tok, nz)]
  rgc = 4 * nwg
  if layer == "router":
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    tab, pmax, b4 = state["dummy"], 0, 0
  else:
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
  bfp__get_ext().wide(state["streams"][(layer, 0, rgc, 1, 4, n)], x, out, state["mid_ws_shared"],
                  state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
  return out


def bfp__run_gemm(x, layer, m, n, k, state):
  bn = 64 if m <= 64 else 128 if m <= 128 else 256
  nwg, splits, _ = bfp_GEMM_CONFIGS[(layer, bn)]
  out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
  w = state["router_cat"] if layer == "router" else state["raw"][layer]
  bfp__get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                    splits, nwg, bn, 0)
  return out


def bfp__run_torch(x, layer, m, n, k, state):
  if layer == "router":
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
  return F.linear(x, state["raw"][layer][:n])


def bfp__run(choice, x, layer, m, n, k, state):
  kind = choice[0]
  if kind == "skinny":
    return bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
  if kind == "wide":
    return bfp__run_wide(x, layer, m, n, k, state)
  if kind == "sk2":
    return bfp__run_sk2(x, layer, m, n, k, state)
  if kind == "gemm":
    return bfp__run_gemm(x, layer, m, n, k, state)
  if kind == "gq":
    return bfp__run_gq(x, layer, m, n, k, state)
  return bfp__run_torch(x, layer, m, n, k, state)


def bfp__static_choice(layer, m, state):
  route = bfp__route(layer, m, state["sm90"])
  if not state.get("old_paths", True):
    if route == "skinny":
      route = "sk2" if (bfp__sk2_name(layer, m) in state.get("sk2", {})
                        and bfp__route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
    elif route == "wide" and layer != "router" and not bfp_WIDE_BF16:
      route = "torch"
  if route != "torch" and bfp__route_key(layer, m, route) in state.get("disabled", ()):
    route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                       and bfp__route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
  if route == "skinny":
    mtk = bfp__mt_key(m)
    return ("skinny", bfp_TMA_CONFIGS[(layer, mtk)], (layer, mtk))
  return (route,)


def bfp__candidates(layer, m, state):
  c = [bfp__static_choice(layer, m, state)]
  if c[0][0] == "skinny" and (layer, m) in bfp_TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
    c.append(("skinny", bfp_TMA_ALT[(layer, m)], ("alt", layer, m)))
  if (c[0][0] == "wide" and layer == "router" and state["sm90"]
          and bfp__route_key(layer, m, "gemm") not in state["disabled"]):
    c.append(("gemm",))
  if (m <= 32 and c[0][0] != "sk2" and bfp__sk2_name(layer, m) in state.get("sk2", {})
          and bfp__route_key(layer, m, "sk2") not in state["disabled"]):
    c.insert(0, ("sk2",))
  if (bfp_USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, bfp__gq_bn(m)) in bfp_GQ_CONFIGS
          and bfp__route_key(layer, m, "gq") not in state["disabled"]):
    c.insert(0, ("gq",))
  if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
    c.append(("torch",))
  return c


bfp_AUTOTUNE = True
bfp_TUNE_M = (1, 16, 32, 8, 128, 256)


def bfp__autotune(weights, state):
  state["choice"] = {}
  if not bfp_AUTOTUNE:
    return
  dev = weights["o"].device
  flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)
  sink = torch.empty(1024, dtype=torch.float32, device=dev)
  g = torch.Generator(device=dev)
  g.manual_seed(4321)
  reps = 24
  for m in bfp_TUNE_M:
    for layer, w in weights.items():
      n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
      k = w.shape[1]
      cands = bfp__candidates(layer, m, state)
      if len(cands) < 2:
        continue
      x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
      graphs = []
      for ch in cands:
        bfp__run(ch, x, layer, m, n, k, state)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
          bfp__run(ch, x, layer, m, n, k, state)
        graphs.append(gr)
      evs = []
      for r in range(reps):
        order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
        for i in order:
          torch.sum(flush, dim=0, out=sink)
          e0 = torch.cuda.Event(enable_timing=True)
          e1 = torch.cuda.Event(enable_timing=True)
          e0.record()
          graphs[i].replay()
          e1.record()
          evs.append((i, e0, e1))
      torch.cuda.synchronize()
      per = [[] for _ in cands]
      for i, e0, e1 in evs:
        per[i].append(e0.elapsed_time(e1))
      med = [sorted(v)[len(v) // 2] for v in per]
      best = min(range(len(cands)), key=lambda i: med[i])
      state["choice"][(layer, m)] = cands[best]
      state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
      del graphs
  del flush
  torch.cuda.synchronize()
  for layer, m, tl in state.get("tune_log", []):
    ch = state["choice"][(layer, m)]
    desc = " ".join(f"{name}{'/' + 'x'.join(map(str, c[1])) if c[0] == 'skinny' else ''}={t}"
                    for (name, t), c in zip(tl, bfp__candidates(layer, m, state)))
    pick = ch[0] + ("/" + "x".join(map(str, ch[1])) if ch[0] == "skinny" else "")
    if False: print(f"[okc] tune {layer} m={m}: {desc} -> {pick}", file=sys.stderr)
  if False: print(f"[okc] disabled: {sorted(state['disabled'])}", file=sys.stderr)


def bfp__nearby_choice(layer, m, state):
  tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
  if not tm:
    return None
  ch = state.get("choice", {}).get((layer, tm))
  if ch is None:
    return None
  if ch[0] == "gq" and (layer, bfp__gq_bn(m)) not in bfp_GQ_CONFIGS:
    return None
  return ch


def bfp_kernel_fn(data, state):
  x = data["x"]
  cfg = data["config"]
  layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
  choice = state.get("choice", {}).get((layer, m)) or bfp__nearby_choice(layer, m, state) or bfp__static_choice(layer, m, state)
  out = bfp__run(choice, x, layer, m, n, k, state)
  if bfp_GEMM_DIAG:
    bfp__diag(x, layer, m, n, k, state, choice)
  return out


def bfp__diag(x, layer, m, n, k, state, choice):
  if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
    for dg in (1, 2, 3):
      bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)



def _bfp_prepare(weights, st):
  st["bfp"] = None
  try:
    w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    st["bfp"] = bfp_prepare(w5)
  except Exception as ex:
    print("bfp disabled:", repr(ex)[:160], flush=True)


_BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


BFP_PDL_N = 2


def _bfp_pdl(T):
  global bfp_SK2_PDL
  bfp_SK2_PDL = BFP_PDL_N if _pdl(T) else 0


def _bfp_gemm(st, name, x, m, n, k):
  _bfp_pdl(m)
  return bfp_kernel_fn({"x": x, "config": {"layer": _BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


_MA_MMAX = 384
_MA_EMAX = _MA_MMAX * 8


WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
WGS_XROWS = _MA_EMAX + 8 * (32 + _MA_EMAX // 32 + 8)


def _ma_make_scratch(dev):
  return [torch.zeros(8 + 1024 + _MA_EMAX + 4096, dtype=torch.int32, device=dev),
          torch.zeros(_MA_EMAX + _MA_EMAX * 4096 + WGS_FIXF, dtype=torch.float32, device=dev),
          torch.zeros((_MA_EMAX + 2 * WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev)]

SCRATCH_GUARD = 1 << 16
SCRATCH_SENTINEL = 0x5EADBEEF


def _ma_prepare(weights, st):
  dev = weights["w13"].device
  I = H = 4096
  rb = torch.arange(I // 8, device=dev)
  g = torch.arange(8, device=dev)
  rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
  rowsB = torch.arange(H, device=dev)
  wA, sA, wB, sB = [], [], [], []
  for e in range(32):
    wA.append(_pack_codes(weights["w13"][e].view(torch.uint8), rowsA))
    sA.append(_pack_scales(weights["w13_sf"][e], rowsA))
    wB.append(_pack_codes(weights["w2"][e].view(torch.uint8), rowsB))
    sB.append(_pack_scales(weights["w2_sf"][e], rowsB))
  lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
         * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
  lut = (lut | (lut << 16)).to(torch.int32)
  st["ma_w"] = [torch.cat(wA), torch.cat(sA), weights["w13_gs"].float().contiguous(),
                torch.cat(wB), torch.cat(sB), weights["w2_gs"].float().contiguous(),
                lut.contiguous(), weights["post_expert_norm"].contiguous(),
                torch.cat([_fat_order(x, y) for x, y in zip(wA, sA)]),
                torch.cat([_fat_order(x, y) for x, y in zip(wB, sB)]),
                torch.cat([_unit_order(x, y) for x, y in zip(wA, sA)]),
                torch.cat([_unit_order(x, y) for x, y in zip(wB, sB)])]
  if "_wgs_on" in globals() and _wgs_on():
    kp = bool(LDD19_KPERM)   # LDD19: kperm (the WGS builds compile with -DLDD19_KPERM=1 to match: _moe_build)
    rowsK = _ldd19_kperm_rows(rowsA, I) if kp else rowsA
    st["ldd19_kperm"] = kp
    st["ma_w"].append(torch.cat([_wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], rowsK, kp, _MA_SC6)
                                 for e in range(32)]))
    st["ma_w"].append(torch.cat([_wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, kp, _MA_SC6)
                                 for e in range(32)]))
  st["ldd19_sc6"] = _MA_SC6
  st["ldd19_dq3"] = bool(LDD19_DQ3)   # LDD19: v20
  st["ma_scratch"] = [_guarded(t) for t in _ma_make_scratch(dev)]






def _to_i32(x64):
  x64 = x64 & 0xFFFFFFFF
  return (x64 - (x64 >= 2**31).to(torch.int64) * 2**32).to(torch.int32)


def _repack_q(wp, interleave):
  E, R, _ = wp.shape
  if interleave:
    wp = wp.view(E, 2, R // 32, 16, 2048).transpose(1, 2).reshape(E, R, 2048)
  x = wp.view(E, R // 16, 16, 256, 8).permute(0, 1, 3, 2, 4).to(torch.int64)
  B3 = x[..., :8, :4]
  B2 = x[..., 8:, :4]
  B1 = x[..., :8, 4:]
  B0 = x[..., 8:, 4:]
  w = ((B0 & 15) | ((B1 & 15) << 4) | ((B2 & 15) << 8) | ((B3 & 15) << 12)
       | ((B0 >> 4) << 16) | ((B1 >> 4) << 20) | ((B2 >> 4) << 24) | ((B3 >> 4) << 28))
  w = _to_i32(w).reshape(E, R // 16, 64, 4, 32).permute(0, 1, 2, 4, 3).contiguous()
  return w


def _repack_s(sf, interleave):
  E, R, _ = sf.shape
  if interleave:
    sf = sf.view(E, 2, R // 32, 16, 256).transpose(1, 2).reshape(E, R, 256)
  x = sf.view(E, R // 16, 16, 64, 4).permute(0, 1, 3, 2, 4)
  return torch.cat([x[:, :, :, :8, :], x[:, :, :, 8:, :]], dim=-1).contiguous()


@triton.jit
def _gemm_splitk(X, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                 BK: tl.constexpr, USE_PDL: tl.constexpr = False):
  pn = tl.program_id(0)
  ps = tl.program_id(1)
  pt = tl.program_id(2)
  rn = pn * BN + tl.arange(0, BN)
  rt = pt * BT + tl.arange(0, BT)
  rk = tl.arange(0, BK)
  kbase = ps * KSPLIT
  w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
  x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
  tmask = rt[None, :] < T
  acc = tl.zeros([BN, BT], dtype=tl.float32)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  for _k in range(0, KSPLIT, BK):
    w = tl.load(w_ptr)
    x = tl.load(x_ptr, mask=tmask, other=0.0)
    acc = tl.dot(w, x, acc)
    w_ptr += BK
    x_ptr += BK
  tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _gemm_splitk_ng(X, G, W, P, T, N, K, KSPLIT: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
                    BK: tl.constexpr, USE_PDL: tl.constexpr = False):
  pn = tl.program_id(0)
  ps = tl.program_id(1)
  pt = tl.program_id(2)
  rn = pn * BN + tl.arange(0, BN)
  rt = pt * BT + tl.arange(0, BT)
  rk = tl.arange(0, BK)
  kbase = ps * KSPLIT
  w_ptr = W + rn[:, None].to(tl.int64) * K + (kbase + rk)[None, :]
  x_ptr = X + rt[None, :] * K + (kbase + rk)[:, None]
  g_ptr = G + rt[None, :] * K + (kbase + rk)[:, None]
  tmask = rt[None, :] < T
  acc = tl.zeros([BN, BT], dtype=tl.float32)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  for _k in range(0, KSPLIT, BK):
    w = tl.load(w_ptr)
    x = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
    gt = tl.load(g_ptr, mask=tmask, other=0.0)
    x3 = tl.reshape(x, [BK // 64, 64, BT])
    r = tl.rsqrt(tl.sum(x3 * x3, 1) / 64.0 + 1e-5)
    xn = tl.reshape(x3 * r[:, None, :], [BK, BT]) * tl.sigmoid(gt)
    acc = tl.dot(w, xn.to(tl.bfloat16), acc)
    w_ptr += BK
    x_ptr += BK
    g_ptr += BK
  tl.store(P + ps * T * N + rt[None, :] * N + rn[:, None], acc, mask=tmask)


@triton.jit
def _normgate(X, G, OUT, USE_PDL: tl.constexpr = False):
  t = tl.program_id(0)
  h = tl.arange(0, 16)[:, None]
  d = tl.arange(0, 64)[None, :]
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  x = tl.load(X + t * 1024 + h * 64 + d).to(tl.float32)
  x = x * tl.rsqrt(tl.sum(x * x, 1) / 64.0 + 1e-5)[:, None]
  x = x * tl.sigmoid(tl.load(G + t * 1024 + h * 64 + d))
  tl.store(OUT + t * 1024 + h * 64 + d, x.to(tl.bfloat16))


def dense_ng(x, gate, w, T, state=None):
  N, K = w.shape
  if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
    xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
    _normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
    return dense(xn, w, T, state, "o")
  BN, BK = 64, 128
  BT = 16 if T <= 16 else 32
  S = 1
  P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
  grid = (N // BN, S, triton.cdiv(T, BT))
  _gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
  return P, S


TUNED_T = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _bucket(T):
  for t in TUNED_T:
    if T <= t:
      return t
  return TUNED_T[-1]


BFP_MAX_T = 32


def dense(x, w, T, state=None, name=None):
  N, K = w.shape
  if T > BFP_MAX_T and state is not None and name in ("o", "un", "pr"):   # LDD19: lt128 (rq / cuBLASLt / BF16 out); big2: + pr (cuBLASLt, FP32 out)
    out = _ldd19_big(state, name, x, T, w)
    if out is not None:
      return out.view(1, T, N), 1
  pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
  if pick == "bfp" and T <= BFP_MAX_T and name in _BFP_LAYER and state.get("bfp") is not None:
    return _bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
  if pick is True or pick == "bfp":
    out = torch.mm(x, w.t(), out_dtype=torch.float32)
    return out.view(1, T, N), 1
  if T > 32:
    out = torch.mm(x, w.t(), out_dtype=torch.float32)
    return out.view(1, T, N), 1
  BN, BK = 64, 128
  BT = 16 if T <= 16 else 32
  tiles = (N // BN) * triton.cdiv(T, BT)
  S = 1
  while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
    S *= 2
  P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
  grid = (N // BN, S, triton.cdiv(T, BT))
  _gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
  return P, S


@triton.jit
def _f32_to_e4m3(x):
  x = tl.minimum(tl.maximum(x, -448.0), 448.0)
  bits = x.to(tl.int32, bitcast=True)
  sign = (bits >> 24) & 0x80
  a = tl.abs(x)
  sub = (a * 512.0 + 0.5).to(tl.int32)
  ab = a.to(tl.int32, bitcast=True)
  r = ab + 0x7FFFF + ((ab >> 20) & 1)
  r = r & 0x7FF00000
  e = (r >> 23) - 120
  m = (r >> 20) & 7
  nrm = (e << 3) | m
  nrm = tl.minimum(nrm, 0x7E)
  code = tl.where(a < 0.015625, sub, nrm)
  return code | sign


@triton.jit
def _qk_prep(P, S: tl.constexpr, T, POS, LOC, INVF, KC, VC, KS, VS, QOUT, Q8OUT, GOUT, qk_factor,
             ROPE: tl.constexpr, USE_PDL: tl.constexpr = False):
  t = tl.program_id(0)
  base = P + t * 2304
  hq = tl.arange(0, 16)[:, None]
  hk = tl.arange(0, 2)[:, None]
  i = tl.arange(0, 32)[None, :]
  d = tl.arange(0, 64)[None, :]
  qe = tl.zeros([16, 32], dtype=tl.float32)
  qo = tl.zeros([16, 32], dtype=tl.float32)
  ke = tl.zeros([2, 32], dtype=tl.float32)
  ko = tl.zeros([2, 32], dtype=tl.float32)
  v = tl.zeros([2, 64], dtype=tl.float32)
  g = tl.zeros([1024], dtype=tl.float32)
  gi = tl.arange(0, 1024)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  for s in range(S):
    b = base + s * T * 2304
    qe += tl.load(b + hq * 64 + 2 * i)
    qo += tl.load(b + hq * 64 + 2 * i + 1)
    ke += tl.load(b + 1024 + hk * 64 + 2 * i)
    ko += tl.load(b + 1024 + hk * 64 + 2 * i + 1)
    v += tl.load(b + 1152 + hk * 64 + d)
    g += tl.load(b + 1280 + gi)
  rq = tl.rsqrt(tl.sum(qe * qe + qo * qo, 1) / 64.0 + 1e-5)[:, None]
  rk = tl.rsqrt(tl.sum(ke * ke + ko * ko, 1) / 64.0 + 1e-5)[:, None]
  qe = qe * rq
  qo = qo * rq
  ke = ke * rk
  ko = ko * rk
  if ROPE:
    pos = tl.load(POS + t).to(tl.float32)
    ang = pos * tl.load(INVF + i)
    c = tl.cos(ang)
    sn = tl.sin(ang)
    qe, qo = qe * c - qo * sn, qe * sn + qo * c
    ke, ko = ke * c - ko * sn, ke * sn + ko * c
  qb = QOUT + t * 1024 + hq * 64 + 2 * i
  tl.store(qb, (qe * qk_factor).to(tl.bfloat16))
  tl.store(qb + 1, (qo * qk_factor).to(tl.bfloat16))
  q8b = Q8OUT + t * 1024 + hq * 64 + 2 * i
  tl.store(q8b, _f32_to_e4m3(qe).to(tl.uint8))
  tl.store(q8b + 1, _f32_to_e4m3(qo).to(tl.uint8))
  tl.store(GOUT + t * 1024 + gi, g)
  loc = tl.load(LOC + t).to(tl.int64)
  kinv = 1.0 / tl.load(KS)
  vinv = 1.0 / tl.load(VS)
  kb = KC + loc * 128 + hk * 64 + 2 * i
  tl.store(kb, _f32_to_e4m3(ke * kinv).to(tl.uint8))
  tl.store(kb + 1, _f32_to_e4m3(ko * kinv).to(tl.uint8))
  tl.store(VC + loc * 128 + hk * 64 + d, _f32_to_e4m3(v * vinv).to(tl.uint8))


@triton.jit
def _attn_split(Q, KC, VC, PT, SEQ, KS, ACC, ML, pt_stride, page_size, window, sm_scale,
                L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SLIDING: tl.constexpr):
  b = tl.program_id(0)
  s = tl.program_id(1)
  seq = tl.load(SEQ + b)
  m_off = tl.arange(0, BM)
  l_idx = m_off // 8
  hh = m_off % 8
  row_ok = m_off < L * 8
  qpos = seq + l_idx
  d = tl.arange(0, 64)
  qrow = (b * L + l_idx) * 16 + hh
  q0 = tl.load(Q + qrow[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
  q1 = tl.load(Q + (qrow + 8)[:, None] * 64 + d[None, :], mask=row_ok[:, None], other=0.0).to(tl.float16)
  total = seq + L
  if SLIDING:
    lo = tl.maximum(seq - window + 1, 0)
  else:
    lo = seq * 0
  n = total - lo
  per = tl.cdiv(tl.cdiv(n, NSPLIT), BN) * BN
  kst = lo + s * per
  ken = tl.minimum(total, kst + per)
  scale = sm_scale * 256.0 * tl.load(KS) * 1.4426950408889634
  m0 = tl.full([BM], float("-inf"), dtype=tl.float32)
  m1 = tl.full([BM], float("-inf"), dtype=tl.float32)
  l0 = tl.zeros([BM], dtype=tl.float32)
  l1 = tl.zeros([BM], dtype=tl.float32)
  a0 = tl.zeros([BM, 64], dtype=tl.float32)
  a1 = tl.zeros([BM, 64], dtype=tl.float32)
  for kb in range(kst, ken, BN):
    offs = kb + tl.arange(0, BN)
    kv_ok = offs < ken
    page = tl.load(PT + b * pt_stride + offs // page_size, mask=kv_ok, other=0)
    row = page.to(tl.int64) * page_size + offs % page_size
    kptr = row[:, None] * 128 + d[None, :]
    ok = kv_ok[None, :] & (offs[None, :] <= qpos[:, None])
    if SLIDING:
      ok = ok & (offs[None, :] > qpos[:, None] - window)
    k8 = tl.load(KC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
    k16 = (((k8 & 0x7F) << 7) | ((k8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
    k8b = tl.load(KC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
    k16b = (((k8b & 0x7F) << 7) | ((k8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
    v8 = tl.load(VC + kptr, mask=kv_ok[:, None], other=0).to(tl.int32)
    v16 = (((v8 & 0x7F) << 7) | ((v8 & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
    v8b = tl.load(VC + kptr + 64, mask=kv_ok[:, None], other=0).to(tl.int32)
    v16b = (((v8b & 0x7F) << 7) | ((v8b & 0x80) << 8)).to(tl.int16).to(tl.float16, bitcast=True)
    sc = tl.where(ok, tl.dot(q0, tl.trans(k16)) * scale, float("-inf"))
    mn = tl.maximum(m0, tl.max(sc, 1))
    mu = tl.where(mn == float("-inf"), 0.0, mn)
    p = tl.exp2(sc - mu[:, None])
    al = tl.exp2(m0 - mu)
    l0 = l0 * al + tl.sum(p, 1)
    a0 = a0 * al[:, None] + tl.dot(p.to(tl.float16), v16)
    m0 = mn
    sc = tl.where(ok, tl.dot(q1, tl.trans(k16b)) * scale, float("-inf"))
    mn = tl.maximum(m1, tl.max(sc, 1))
    mu = tl.where(mn == float("-inf"), 0.0, mn)
    p = tl.exp2(sc - mu[:, None])
    al = tl.exp2(m1 - mu)
    l1 = l1 * al + tl.sum(p, 1)
    a1 = a1 * al[:, None] + tl.dot(p.to(tl.float16), v16b)
    m1 = mn
  ob0 = ((b * 2 + 0) * NSPLIT + s) * BM + m_off
  ob1 = ((b * 2 + 1) * NSPLIT + s) * BM + m_off
  tl.store(ACC + ob0[:, None] * 64 + d[None, :], a0)
  tl.store(ACC + ob1[:, None] * 64 + d[None, :], a1)
  tl.store(ML + ob0 * 2, m0)
  tl.store(ML + ob0 * 2 + 1, l0)
  tl.store(ML + ob1 * 2, m1)
  tl.store(ML + ob1 * 2 + 1, l1)


@triton.jit
def _attn_combine(ACC, ML, GATE, VS, OUT, L: tl.constexpr, NSPLIT: tl.constexpr, BM: tl.constexpr,
                  SB: tl.constexpr, NORMGATE: tl.constexpr = True):
  t = tl.program_id(0)
  h = tl.program_id(1)
  b = t // L
  l = t % L
  g = h // 8
  m_row = l * 8 + h % 8
  d = tl.arange(0, 64)
  base = (b * 2 + g) * NSPLIT
  M = tl.full([SB], float("-inf"), dtype=tl.float32)
  for s0 in range(0, NSPLIT, SB):
    sidx = s0 + tl.arange(0, SB)
    sm = sidx < NSPLIT
    M = tl.maximum(M, tl.load(ML + ((base + sidx) * BM + m_row) * 2, mask=sm, other=float("-inf")))
  Mx = tl.max(M, 0)
  Mx = tl.where(Mx == float("-inf"), 0.0, Mx)
  den = tl.zeros([SB], dtype=tl.float32)
  num = tl.zeros([64], dtype=tl.float32)
  for s0 in range(0, NSPLIT, SB):
    sidx = s0 + tl.arange(0, SB)
    sm = sidx < NSPLIT
    rows = (base + sidx) * BM + m_row
    m_s = tl.load(ML + rows * 2, mask=sm, other=float("-inf"))
    l_s = tl.load(ML + rows * 2 + 1, mask=sm, other=0.0)
    w = tl.exp2(m_s - Mx)
    den += w * l_s
    a = tl.load(ACC + rows[:, None] * 64 + d[None, :], mask=sm[:, None], other=0.0)
    num += tl.sum(w[:, None] * a, 0)
  o = num / tl.sum(den, 0) * (256.0 * tl.load(VS))
  if NORMGATE:
    o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
    gt = tl.load(GATE + t * 1024 + h * 64 + d)
    o = o * tl.sigmoid(gt)
  tl.store(OUT + t * 1024 + h * 64 + d, o.to(tl.bfloat16))


@triton.jit
def _gemm_o(X, G, W, AOUT, SUMS, STREAM, A1, B1, T, N, K: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
            BK: tl.constexpr, NORMGATE: tl.constexpr):
  pn = tl.program_id(0)
  pt = tl.program_id(1)
  rn = pn * BN + tl.arange(0, BN)
  rt = pt * BT + tl.arange(0, BT)
  rk = tl.arange(0, BK)
  tmask = rt[None, :] < T
  acc = tl.zeros([BN, BT], dtype=tl.float32)
  w_ptr = W + rn[:, None].to(tl.int64) * K + rk[None, :]
  x_ptr = X + rt[None, :] * K + rk[:, None]
  for _k in range(0, K, BK):
    w = tl.load(w_ptr)
    if NORMGATE:
      xv = tl.load(x_ptr, mask=tmask, other=0.0).to(tl.float32)
      gv = tl.load(G + rt[None, :] * K + (rk[:, None] + _k))
      xr = tl.reshape(xv, [BK // 64, 64, BT])
      rs = tl.rsqrt(tl.sum(xr * xr, 1) / 64.0 + 1e-5)
      xv = tl.reshape(xr * rs[:, None, :], [BK, BT]) * tl.sigmoid(gv)
      x = xv.to(tl.bfloat16)
    else:
      x = tl.load(x_ptr, mask=tmask, other=0.0)
    acc = tl.dot(w, x, acc)
    w_ptr += BK
    x_ptr += BK
  tl.store(AOUT + rt[None, :] * N + rn[:, None], acc, mask=tmask)
  st = tl.load(STREAM + rt[None, :] * N + rn[:, None], mask=tmask, other=0.0)
  a1v, b1v = _gate_ab(A1, rn)
  as_ = a1v[:, None] * st
  ba = b1v[:, None] * acc
  ok = (rt[None, :] < T) + tl.zeros([BN, BT], tl.int1)
  q0 = tl.sum(tl.where(ok, acc * acc, 0.0), 0)
  q1 = tl.sum(tl.where(ok, as_ * as_, 0.0), 0)
  q2 = tl.sum(tl.where(ok, as_ * ba, 0.0), 0)
  q3 = tl.sum(tl.where(ok, ba * ba, 0.0), 0)
  m = rt < T
  tl.atomic_add(SUMS + rt * 4 + 0, q0, mask=m)
  tl.atomic_add(SUMS + rt * 4 + 1, q1, mask=m)
  tl.atomic_add(SUMS + rt * 4 + 2, q2, mask=m)
  tl.atomic_add(SUMS + rt * 4 + 3, q3, mask=m)


@triton.jit
def _bnd1_apply(AOUT, SUMS, STREAM, A1, B1, G, S1OUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
  t = tl.program_id(0)
  c0 = tl.program_id(1) * BC
  offs = c0 + tl.arange(0, BC)
  s0 = tl.load(SUMS + t * 4 + 0)
  s1q = tl.load(SUMS + t * 4 + 1)
  s2q = tl.load(SUMS + t * 4 + 2)
  s3q = tl.load(SUMS + t * 4 + 3)
  r = tl.rsqrt(s0 / N + 1e-8)
  r1 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
  a = tl.load(AOUT + t * N + offs)
  a1v, b1v = _gate_ab(A1, offs)
  s1 = a1v * tl.load(STREAM + t * N + offs) + b1v * a * r
  tl.store(S1OUT + t * N + offs, s1)
  tl.store(HOUT + t * N + offs, (s1 * r1 * (1.0 + tl.load(G + offs).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _route_a(P, S: tl.constexpr, T, BIAS, IDS, WTS, LS, LIDS, POOL, PSUM, BC: tl.constexpr):
  t = tl.program_id(0)
  j = tl.program_id(1)
  c = j * BC + tl.arange(0, BC)
  pooled = tl.zeros([BC], dtype=tl.float32)
  for s in range(S):
    pooled += tl.load(P + (s * T + t) * 4608 + c)
  tl.store(POOL + t * 4096 + c, pooled)
  tl.atomic_add(PSUM + t, tl.sum(pooled * pooled, 0))
  if j == 0:
    e = tl.arange(0, 256)
    logit = tl.zeros([256], dtype=tl.float32)
    for s in range(S):
      b = P + (s * T + t) * 4608
      logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + tl.load(BIAS + e)
    k8 = tl.arange(0, 8)
    ids = tl.zeros([8], dtype=tl.int32)
    wv = tl.zeros([8], dtype=tl.float32)
    for k in tl.static_range(8):
      idx = tl.argmax(sel, 0)
      val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
      ids = tl.where(k8 == k, idx, ids)
      wv = tl.where(k8 == k, val, wv)
      sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - tl.load(LS) + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)


@triton.jit
def _route_b(POOL, PSUM, GPRE, XOUT, BC: tl.constexpr, PERMUTED: tl.constexpr):
  t = tl.program_id(0)
  c = tl.program_id(1) * BC + tl.arange(0, BC)
  r = tl.rsqrt(tl.load(PSUM + t) / 4096.0 + 1e-5)
  x = tl.load(POOL + t * 4096 + c) * r * (1.0 + tl.load(GPRE + c).to(tl.float32))
  if PERMUTED:
    ch = c // 64
    kk = c % 64
    kb = kk // 16
    rr = kk % 16
    po = ((t * 64 + ch) * 4 + (rr % 8) // 2) * 16 + kb * 4 + (rr // 8) * 2 + (rr % 2)
    tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
  else:
    tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _bnd2_sums(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, FOUT, SUMS, BC: tl.constexpr):
  t = tl.program_id(0)
  c = tl.program_id(1) * BC + tl.arange(0, BC)
  m = c < NU
  f = tl.zeros([BC], dtype=tl.float32)
  for s in range(S):
    f += tl.load(P + (s * T + t) * NU + c, mask=m, other=0.0)
  f = f * (1.0 + tl.load(GPOST + c).to(tl.float32))
  tl.store(FOUT + t * 8192 + c, f)
  st = tl.load(S1 + t * 8192 + c)
  a2v, b2v = _gate_ab(A2, c)
  as_ = a2v * st
  bf = b2v * f
  tl.atomic_add(SUMS + t * 4 + 0, tl.sum(f * f, 0))
  tl.atomic_add(SUMS + t * 4 + 1, tl.sum(as_ * as_, 0))
  tl.atomic_add(SUMS + t * 4 + 2, tl.sum(as_ * bf, 0))
  tl.atomic_add(SUMS + t * 4 + 3, tl.sum(bf * bf, 0))


@triton.jit
def _bnd2_apply(FOUT, SUMS, S1, A2, B2, GNEXT, SOUT, HOUT, N: tl.constexpr, BC: tl.constexpr):
  t = tl.program_id(0)
  c = tl.program_id(1) * BC + tl.arange(0, BC)
  s0 = tl.load(SUMS + t * 4 + 0)
  s1q = tl.load(SUMS + t * 4 + 1)
  s2q = tl.load(SUMS + t * 4 + 2)
  s3q = tl.load(SUMS + t * 4 + 3)
  r = tl.rsqrt(s0 / N + 1e-8)
  r2 = tl.rsqrt((s1q + 2.0 * r * s2q + r * r * s3q) / N + 1e-5)
  a2v, b2v = _gate_ab(A2, c)
  s2 = a2v * tl.load(S1 + t * N + c) + b2v * tl.load(FOUT + t * N + c) * r
  tl.store(SOUT + t * N + c, s2)
  tl.store(HOUT + t * N + c, (s2 * r2 * (1.0 + tl.load(GNEXT + c).to(tl.float32))).to(tl.bfloat16))


@triton.jit
def _gate_ab(G, offs):
  w = tl.load(G + offs).to(tl.float32) / 0.3
  beta = tl.sigmoid(w)
  alpha = tl.sqrt(tl.maximum(tl.sigmoid(-w) * (1.0 + beta), 1e-3))
  return alpha, beta


# LDD19 (dense_pricer r6i2): LDD19_GT128: Triton boundaries at T = 128 read gate tables made in prepare, 0 = sigmoid + sqrt per call
LDD19_GT128 = 1
@triton.jit
def _boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, USE_PDL: tl.constexpr = False,
               DELAY: tl.constexpr = False, GT: tl.constexpr = False):   # LDD19: big2: + DELAY; GT128
  t = tl.program_id(0)
  offs = tl.arange(0, 8192)
  if GT:   # LDD19: GT128
    a1v = tl.load(A1 + offs)
    b1v = tl.load(B1 + offs)
  else:
    a1v, b1v = _gate_ab(A1, offs)
  stv = tl.load(STREAM + t * 8192 + offs)
  if not DELAY:
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  a = tl.zeros([8192], dtype=tl.float32)
  for s in range(S):
    a += tl.load(P + (s * T + t) * 8192 + offs)
  n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
  s1 = a1v * stv + b1v * n
  tl.store(S1OUT + t * 8192 + offs, s1)
  if DELAY:   # LDD19: big2: G loaded after both reductions (16 warps: 83 -> 64 registers, two CTAs per SM)
    r1 = tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5)
    gw = 1.0 + tl.load(G + offs).to(tl.float32)
    hh = s1 * r1 * gw
  else:
    hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
  tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


@triton.jit
def _route(P, S: tl.constexpr, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
  t = tl.program_id(0)
  e = tl.arange(0, 256)
  logit = tl.zeros([256], dtype=tl.float32)
  pooled = tl.zeros([4096], dtype=tl.float32)
  c = tl.arange(0, 4096)
  biasv = tl.load(BIAS + e)
  gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
  lsv = tl.load(LS)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  for s in range(S):
    b = P + (s * T + t) * 4608
    logit += tl.load(b + 4096 + e) + tl.load(b + 4352 + e)
    pooled += tl.load(b + c)
  sc = tl.sigmoid(logit * 0.1767766952966369)
  sel = sc + biasv
  k8 = tl.arange(0, 8)
  ids = tl.zeros([8], dtype=tl.int32)
  wv = tl.zeros([8], dtype=tl.float32)
  for k in tl.static_range(8):
    idx = tl.argmax(sel, 0)
    val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
    ids = tl.where(k8 == k, idx, ids)
    wv = tl.where(k8 == k, val, wv)
    sel = tl.where(e == idx, float("-inf"), sel)
  wv = wv / (tl.sum(wv, 0) + 1e-15)
  tl.store(IDS + t * 8 + k8, ids)
  lid = (ids - lsv + 256) % 256
  tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
  tl.store(WTS + t * 8 + k8, wv)
  x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
  ch = c // 64
  kk = c % 64
  kb = kk // 16
  r = kk % 16
  if PERMUTED:
    po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
    tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
  else:
    tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _route_split(POOLED, LOGITS, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, PERMUTED: tl.constexpr, USE_PDL: tl.constexpr = False):
  t = tl.program_id(0)
  e = tl.arange(0, 256)
  c = tl.arange(0, 4096)
  biasv = tl.load(BIAS + e)
  gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
  lsv = tl.load(LS)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  logit = tl.load(LOGITS + t * 256 + e)
  pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
  sc = tl.sigmoid(logit * 0.1767766952966369)
  sel = sc + biasv
  k8 = tl.arange(0, 8)
  ids = tl.zeros([8], dtype=tl.int32)
  wv = tl.zeros([8], dtype=tl.float32)
  for k in tl.static_range(8):
    idx = tl.argmax(sel, 0)
    val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
    ids = tl.where(k8 == k, idx, ids)
    wv = tl.where(k8 == k, val, wv)
    sel = tl.where(e == idx, float("-inf"), sel)
  wv = wv / (tl.sum(wv, 0) + 1e-15)
  tl.store(IDS + t * 8 + k8, ids)
  lid = (ids - lsv + 256) % 256
  tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
  tl.store(WTS + t * 8 + k8, wv)
  x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
  ch = c // 64
  kk = c % 64
  kb = kk // 16
  r = kk % 16
  if PERMUTED:
    po = ((t * 64 + ch) * 4 + (r % 8) // 2) * 16 + kb * 4 + (r // 8) * 2 + (r % 2)
    tl.store(XOUT + po, x.to(tl.bfloat16).to(tl.float16))
  else:
    tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))


@triton.jit
def _boundary2(P, S: tl.constexpr, T, NU: tl.constexpr, S1, A2, B2, GPOST, GNEXT, SOUT, HOUT,
               USE_PDL: tl.constexpr = False, GT: tl.constexpr = False):   # LDD19: GT128
  t = tl.program_id(0)
  offs = tl.arange(0, 8192)
  m = offs < NU
  gp = 1.0 + tl.load(GPOST + offs).to(tl.float32)
  gn = 1.0 + tl.load(GNEXT + offs).to(tl.float32)
  if GT:   # LDD19: GT128
    a2v = tl.load(A2 + offs)
    b2v = tl.load(B2 + offs)
  else:
    a2v, b2v = _gate_ab(A2, offs)
  s1v = tl.load(S1 + t * 8192 + offs)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  f = tl.zeros([8192], dtype=tl.float32)
  for s in range(S):
    f += tl.load(P + (s * T + t) * NU + offs, mask=m, other=0.0)
  f = f * gp
  n = f * tl.rsqrt(tl.sum(f * f, 0) / 8192.0 + 1e-8)
  s2 = a2v * s1v + b2v * n
  tl.store(SOUT + t * 8192 + offs, s2)
  hh = s2 * tl.rsqrt(tl.sum(s2 * s2, 0) / 8192.0 + 1e-5) * gn
  tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))


# LDD19: the three per-token glue kernels (_boundary1, _route_split, _boundary2) at T <= LDD19_GLUE_MAX_T as CUDA,
# each boundary token split over a thread-block cluster of LDD19_CL CTAs (8192 / LDD19_CL columns each, the two
import os as _ldd19_os

LDD19_CLUSTER_GLUE = _ldd19_os.environ.get("LDD19_GLUE", "1") == "1"   # LDD19: A/B switch (env var for the harness)
LDD19_GLUE_MAX_T = 32
LDD19_CL = int(_ldd19_os.environ.get("LDD19_CL", "8"))   # LDD19: CTAs per token in the boundary clusters (1/2/4/8/16)
LDD19_ROUTE = _ldd19_os.environ.get("LDD19_ROUTE", "1") == "1"   # LDD19: the CUDA routing kernel (else Triton's)
LDD19_BND_ONE = _ldd19_os.environ.get("LDD19_BND_ONE", "1") == "1"   # LDD19: both boundaries with one reduction
LDD19_BGD = int(_ldd19_os.environ.get("LDD19_BGD", "2"))   # LDD19: big2: _boundary1's delayed gain (bit 1 T <= 128, 2 T > 128)
LDD19_RT32 = int(_ldd19_os.environ.get("LDD19_RT32", "1"))   # LDD19: big2: CUDA _route above 32 tokens (0: Triton _route)

LDD19_GLUE_CUDA_SRC = r"""#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace ldd19 {
__device__ __forceinline__ float t_exp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
}
__device__ __forceinline__ float t_div(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ __forceinline__ float t_sqrt(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float t_rsqrt(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float t_sigmoid(float x) { return t_div(1.0f, 1.0f + t_exp(-x)); }
__device__ __forceinline__ void gate_ab(float g, float& al, float& be) {
  const float w = t_div(g, 0.3f);
  be = t_sigmoid(w);
  al = t_sqrt(fmaxf(t_sigmoid(-w) * (1.0f + be), 1e-3f));
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t cl_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
__device__ __forceinline__ void st_cluster(float* local, uint32_t rank, float v) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(local);
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(r), "f"(v) : "memory");
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float bf_lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ void ld_bf4(const __nv_bfloat16* p, float* v) {
  const uint2 u = __ldg(reinterpret_cast<const uint2*>(p));
  v[0] = bf_lo(u.x); v[1] = bf_hi(u.x); v[2] = bf_lo(u.y); v[3] = bf_hi(u.y);
}
__device__ __forceinline__ uint32_t pack_bf2(float a, float b) {
  const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(a));
  const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(b));
  return lo | (hi << 16);
}
__device__ __forceinline__ void ld_p4(const float* p, float* v) {
  const float4 u = __ldcg(reinterpret_cast<const float4*>(p));
  v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
}
__device__ __forceinline__ void ld_p4(const __nv_bfloat16* p, float* v) {
  const uint2 u = __ldcg(reinterpret_cast<const uint2*>(p));
  v[0] = bf_lo(u.x); v[1] = bf_hi(u.x); v[2] = bf_lo(u.y); v[3] = bf_hi(u.y);
}
template <int NT, int CL>
__device__ __forceinline__ float cta_cluster_sum(float v, float* ws, float* slots) {
  constexpr int NW = NT / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  v = warp_sum(v);
  if (lane == 0) ws[warp] = v;
  if constexpr (CL == 1) {
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) s += ws[i];
    return s;
  } else {
    __syncthreads();
    if (warp == 0) {
      float p = 0.f;
#pragma unroll
      for (int i = 0; i < NW; ++i) p += ws[i];
      if (lane < CL) st_cluster(slots + cl_rank(), lane, p);
    }
    cl_arrive();
    cl_wait();
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < CL; ++i) s += slots[i];
    return s;
  }
}
template <typename PT, bool GP, int CL, int NT>
__global__ void __launch_bounds__(NT) bnd_kernel(const PT* __restrict__ P, int S, int T, int NU,
                                                 const float* __restrict__ X, const __nv_bfloat16* __restrict__ GATE,
                                                 const __nv_bfloat16* __restrict__ GPOST,
                                                 const __nv_bfloat16* __restrict__ G, float* __restrict__ SOUT,
                                                 __nv_bfloat16* __restrict__ HOUT) {
  constexpr int COLS = 8192 / CL, V = COLS / NT, NC = V / 4;
  static_assert(V % 4 == 0 && NC >= 1, "4-column chunks");
  __shared__ float ws[2][NT / 32];
  __shared__ float slots[2][CL];
  if constexpr (CL > 1) cl_arrive_relaxed();
  pdl_trigger();
  const int t = blockIdx.x / CL;
  const int c0 = (int)(blockIdx.x % CL) * COLS + threadIdx.x * 4;
  float x[NC][4], al[NC][4], be[NC][4], gw[NC][4], gp[NC][4];
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = c0 + j * NT * 4;
    const float4 xv = __ldcg(reinterpret_cast<const float4*>(X + (size_t)t * 8192 + c));
    x[j][0] = xv.x; x[j][1] = xv.y; x[j][2] = xv.z; x[j][3] = xv.w;
    float g4[4];
    ld_bf4(GATE + c, g4);
#pragma unroll
    for (int k = 0; k < 4; ++k) gate_ab(g4[k], al[j][k], be[j][k]);
    ld_bf4(G + c, g4);
#pragma unroll
    for (int k = 0; k < 4; ++k) gw[j][k] = 1.0f + g4[k];
    if constexpr (GP) {
      ld_bf4(GPOST + c, g4);
#pragma unroll
      for (int k = 0; k < 4; ++k) gp[j][k] = 1.0f + g4[k];
    }
  }
  pdl_wait();
  float a[NC][4];
#pragma unroll
  for (int j = 0; j < NC; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) a[j][k] = 0.f;
  for (int s = 0; s < S; ++s) {
#pragma unroll
    for (int j = 0; j < NC; ++j) {
      const int c = c0 + j * NT * 4;
      if (c < NU) {
        float v[4];
        ld_p4(P + ((size_t)s * T + t) * NU + c, v);
#pragma unroll
        for (int k = 0; k < 4; ++k) a[j][k] += v[k];
      }
    }
  }
  float q = 0.f;
#pragma unroll
  for (int j = 0; j < NC; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if constexpr (GP) a[j][k] = a[j][k] * gp[j][k];
      q += a[j][k] * a[j][k];
    }
  if constexpr (CL > 1) cl_wait();
  const float r = t_rsqrt(t_div(cta_cluster_sum<NT, CL>(q, ws[0], slots[0]), 8192.0f) + 1e-8f);
  float q2 = 0.f;
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = c0 + j * NT * 4;
    float sv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float n = a[j][k] * r;
      sv[k] = al[j][k] * x[j][k] + be[j][k] * n;
      q2 += sv[k] * sv[k];
      x[j][k] = sv[k];
    }
    *reinterpret_cast<float4*>(SOUT + (size_t)t * 8192 + c) = make_float4(sv[0], sv[1], sv[2], sv[3]);
  }
  const float r2 = t_rsqrt(t_div(cta_cluster_sum<NT, CL>(q2, ws[1], slots[1]), 8192.0f) + 1e-5f);
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = c0 + j * NT * 4;
    uint2 o;
    o.x = pack_bf2(x[j][0] * r2 * gw[j][0], x[j][1] * r2 * gw[j][1]);
    o.y = pack_bf2(x[j][2] * r2 * gw[j][2], x[j][3] * r2 * gw[j][3]);
    *reinterpret_cast<uint2*>(HOUT + (size_t)t * 8192 + c) = o;
  }
}
template <int NT, int CL>
__device__ __forceinline__ void cta_cluster_sum4(const float* v, float* out, float (*ws)[NT / 32], float (*slots)[CL]) {
  constexpr int NW = NT / 32;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) w[k] = warp_sum(v[k]);
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < 4; ++k) ws[k][warp] = w[k];
  }
  __syncthreads();
  if constexpr (CL == 1) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < NW; ++i) s += ws[k][i];
      out[k] = s;
    }
  } else {
    if (warp == 0) {
      float p[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        p[k] = 0.f;
#pragma unroll
        for (int i = 0; i < NW; ++i) p[k] += ws[k][i];
      }
      if (lane < CL) {
#pragma unroll
        for (int k = 0; k < 4; ++k) st_cluster(&slots[k][0] + cl_rank(), lane, p[k]);
      }
    }
    cl_arrive();
    cl_wait();
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < CL; ++i) s += slots[k][i];
      out[k] = s;
    }
  }
}
template <typename PT, bool GP, int CL, int NT>
__global__ void __launch_bounds__(NT) bnd1r_kernel(const PT* __restrict__ P, int S, int T, int NU,
                                                   const float* __restrict__ X, const __nv_bfloat16* __restrict__ GATE,
                                                   const __nv_bfloat16* __restrict__ GPOST,
                                                   const __nv_bfloat16* __restrict__ G, float* __restrict__ SOUT,
                                                   __nv_bfloat16* __restrict__ HOUT) {
  constexpr int COLS = 8192 / CL, V = COLS / NT, NC = V / 4;
  static_assert(V % 4 == 0 && NC >= 1, "4-column chunks");
  __shared__ float ws[4][NT / 32];
  __shared__ float slots[4][CL];
  if constexpr (CL > 1) cl_arrive_relaxed();
  pdl_trigger();
  const int t = blockIdx.x / CL;
  const int c0 = (int)(blockIdx.x % CL) * COLS + threadIdx.x * 4;
  float x[NC][4], al[NC][4], be[NC][4], gw[NC][4], gp[NC][4];
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = c0 + j * NT * 4;
    const float4 xv = __ldcg(reinterpret_cast<const float4*>(X + (size_t)t * 8192 + c));
    x[j][0] = xv.x; x[j][1] = xv.y; x[j][2] = xv.z; x[j][3] = xv.w;
    float g4[4];
    ld_bf4(GATE + c, g4);
#pragma unroll
    for (int k = 0; k < 4; ++k) gate_ab(g4[k], al[j][k], be[j][k]);
    ld_bf4(G + c, g4);
#pragma unroll
    for (int k = 0; k < 4; ++k) gw[j][k] = 1.0f + g4[k];
    if constexpr (GP) {
      ld_bf4(GPOST + c, g4);
#pragma unroll
      for (int k = 0; k < 4; ++k) gp[j][k] = 1.0f + g4[k];
    }
  }
  pdl_wait();
  float a[NC][4];
#pragma unroll
  for (int j = 0; j < NC; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) a[j][k] = 0.f;
  for (int s = 0; s < S; ++s) {
#pragma unroll
    for (int j = 0; j < NC; ++j) {
      const int c = c0 + j * NT * 4;
      if (c < NU) {
        float v[4];
        ld_p4(P + ((size_t)s * T + t) * NU + c, v);
#pragma unroll
        for (int k = 0; k < 4; ++k) a[j][k] += v[k];
      }
    }
  }
  float q[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int j = 0; j < NC; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if constexpr (GP) a[j][k] = a[j][k] * gp[j][k];
      const float as_ = al[j][k] * x[j][k], bn = be[j][k] * a[j][k];
      q[0] += a[j][k] * a[j][k];
      q[1] += as_ * as_;
      q[2] += as_ * bn;
      q[3] += bn * bn;
    }
  if constexpr (CL > 1) cl_wait();
  float tot[4];
  cta_cluster_sum4<NT, CL>(q, tot, ws, slots);
  const float r = t_rsqrt(t_div(tot[0], 8192.0f) + 1e-8f);
  const float r2 = t_rsqrt(t_div(tot[1] + 2.0f * r * tot[2] + r * r * tot[3], 8192.0f) + 1e-5f);
#pragma unroll
  for (int j = 0; j < NC; ++j) {
    const int c = c0 + j * NT * 4;
    float sv[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float n = a[j][k] * r;
      sv[k] = al[j][k] * x[j][k] + be[j][k] * n;
    }
    *reinterpret_cast<float4*>(SOUT + (size_t)t * 8192 + c) = make_float4(sv[0], sv[1], sv[2], sv[3]);
    uint2 o;
    o.x = pack_bf2(sv[0] * r2 * gw[j][0], sv[1] * r2 * gw[j][1]);
    o.y = pack_bf2(sv[2] * r2 * gw[j][2], sv[3] * r2 * gw[j][3]);
    *reinterpret_cast<uint2*>(HOUT + (size_t)t * 8192 + c) = o;
  }
}
__device__ __forceinline__ uint32_t f2key(float f) {
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
constexpr int RT_NTN = 256;
__global__ void __launch_bounds__(RT_NTN + 32) route_kernel(const __nv_bfloat16* __restrict__ POOLED,
                                                            const float* __restrict__ LOGITS,
                                                            const float* __restrict__ BIAS,
                                                            const __nv_bfloat16* __restrict__ GPRE,
                                                            int* __restrict__ IDS, float* __restrict__ WTS,
                                                            __nv_bfloat16* __restrict__ XOUT,
                                                            const int* __restrict__ LS, int* __restrict__ LIDS) {
  __shared__ float ws[RT_NTN / 32];
  const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  pdl_trigger();
  if (warp == RT_NTN / 32) {
    float bias[8], lg[8], sc[8], sel[8];
    {
      const float4 b0 = __ldg(reinterpret_cast<const float4*>(BIAS + lane * 8));
      const float4 b1 = __ldg(reinterpret_cast<const float4*>(BIAS + lane * 8 + 4));
      bias[0] = b0.x; bias[1] = b0.y; bias[2] = b0.z; bias[3] = b0.w;
      bias[4] = b1.x; bias[5] = b1.y; bias[6] = b1.z; bias[7] = b1.w;
    }
    const int lsv = __ldg(LS);
    pdl_wait();
    {
      const float4 l0 = __ldcg(reinterpret_cast<const float4*>(LOGITS + (size_t)t * 256 + lane * 8));
      const float4 l1 = __ldcg(reinterpret_cast<const float4*>(LOGITS + (size_t)t * 256 + lane * 8 + 4));
      lg[0] = l0.x; lg[1] = l0.y; lg[2] = l0.z; lg[3] = l0.w;
      lg[4] = l1.x; lg[5] = l1.y; lg[6] = l1.z; lg[7] = l1.w;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sc[j] = t_sigmoid(lg[j] * 0.1767766952966369f);
      sel[j] = sc[j] + bias[j];
    }
    int my_id = 0;
    float my_w = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bv = sel[0], bs = sc[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (sel[j] > bv) bv = sel[j], bs = sc[j], bj = j;
      const uint32_t key = f2key(bv);
      const uint32_t m = __reduce_max_sync(0xffffffffu, key);
      const int wl = __ffs(__ballot_sync(0xffffffffu, key == m)) - 1;
      const int widx = __shfl_sync(0xffffffffu, lane * 8 + bj, wl);
      const float wsc = __shfl_sync(0xffffffffu, bs, wl);
      if (lane == wl) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j == bj) sel[j] = -INFINITY;
      }
      if (lane == k) my_id = widx, my_w = wsc;
    }
    float s = my_w;
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    if (lane < 8) {
      IDS[t * 8 + lane] = my_id;
      const int lid = (my_id - lsv + 256) % 256;
      LIDS[t * 8 + lane] = lid < 32 ? lid : -1;
      WTS[t * 8 + lane] = t_div(my_w, s + 1e-15f);
    }
    return;
  }
  const int c0 = threadIdx.x * 8, c1 = 2048 + threadIdx.x * 8;
  float g[16], p[16];
  {
    const uint4 u0 = __ldg(reinterpret_cast<const uint4*>(GPRE + c0));
    const uint4 u1 = __ldg(reinterpret_cast<const uint4*>(GPRE + c1));
    const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) g[2 * i] = 1.0f + bf_lo(w[i]), g[2 * i + 1] = 1.0f + bf_hi(w[i]);
  }
  pdl_wait();
  {
    const uint4 u0 = __ldcg(reinterpret_cast<const uint4*>(POOLED + (size_t)t * 4096 + c0));
    const uint4 u1 = __ldcg(reinterpret_cast<const uint4*>(POOLED + (size_t)t * 4096 + c1));
    const uint32_t w[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) p[2 * i] = bf_lo(w[i]), p[2 * i + 1] = bf_hi(w[i]);
  }
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) q += p[i] * p[i];
  q = warp_sum(q);
  if (lane == 0) ws[warp] = q;
  asm volatile("bar.sync 1, %0;" ::"n"(RT_NTN) : "memory");
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < RT_NTN / 32; ++i) tot += ws[i];
  const float r = t_rsqrt(t_div(tot, 4096.0f) + 1e-5f);
  uint4 o0, o1;
  o0.x = pack_bf2(p[0] * r * g[0], p[1] * r * g[1]);
  o0.y = pack_bf2(p[2] * r * g[2], p[3] * r * g[3]);
  o0.z = pack_bf2(p[4] * r * g[4], p[5] * r * g[5]);
  o0.w = pack_bf2(p[6] * r * g[6], p[7] * r * g[7]);
  o1.x = pack_bf2(p[8] * r * g[8], p[9] * r * g[9]);
  o1.y = pack_bf2(p[10] * r * g[10], p[11] * r * g[11]);
  o1.z = pack_bf2(p[12] * r * g[12], p[13] * r * g[13]);
  o1.w = pack_bf2(p[14] * r * g[14], p[15] * r * g[15]);
  *reinterpret_cast<uint4*>(XOUT + (size_t)t * 4096 + c0) = o0;
  *reinterpret_cast<uint4*>(XOUT + (size_t)t * 4096 + c1) = o1;
}
template <int NTN>
__global__ void __launch_bounds__(NTN + 32) route32_kernel(const float* __restrict__ P, const float* __restrict__ BIAS,
                                                           const __nv_bfloat16* __restrict__ GPRE,
                                                           int* __restrict__ IDS, float* __restrict__ WTS,
                                                           __nv_bfloat16* __restrict__ XOUT,
                                                           const int* __restrict__ LS, int* __restrict__ LIDS) {
  constexpr int NV = 4096 / (4 * NTN);
  __shared__ float ws[NTN / 32];
  const int t = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float* row = P + (size_t)t * 4608;
  pdl_trigger();
  if (warp == NTN / 32) {
    float bias[8], lg[8], sc[8], sel[8];
    {
      const float4 b0 = __ldg(reinterpret_cast<const float4*>(BIAS + lane * 8));
      const float4 b1 = __ldg(reinterpret_cast<const float4*>(BIAS + lane * 8 + 4));
      bias[0] = b0.x; bias[1] = b0.y; bias[2] = b0.z; bias[3] = b0.w;
      bias[4] = b1.x; bias[5] = b1.y; bias[6] = b1.z; bias[7] = b1.w;
    }
    const int lsv = __ldg(LS);
    pdl_wait();
    {
      const float4 h0 = __ldcg(reinterpret_cast<const float4*>(row + 4096 + lane * 8));
      const float4 h1 = __ldcg(reinterpret_cast<const float4*>(row + 4096 + lane * 8 + 4));
      const float4 l0 = __ldcg(reinterpret_cast<const float4*>(row + 4352 + lane * 8));
      const float4 l1 = __ldcg(reinterpret_cast<const float4*>(row + 4352 + lane * 8 + 4));
      lg[0] = h0.x + l0.x; lg[1] = h0.y + l0.y; lg[2] = h0.z + l0.z; lg[3] = h0.w + l0.w;
      lg[4] = h1.x + l1.x; lg[5] = h1.y + l1.y; lg[6] = h1.z + l1.z; lg[7] = h1.w + l1.w;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sc[j] = t_sigmoid(lg[j] * 0.1767766952966369f);
      sel[j] = sc[j] + bias[j];
    }
    int my_id = 0;
    float my_w = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bv = sel[0], bs = sc[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (sel[j] > bv) bv = sel[j], bs = sc[j], bj = j;
      const uint32_t key = f2key(bv);
      const uint32_t m = __reduce_max_sync(0xffffffffu, key);
      const int wl = __ffs(__ballot_sync(0xffffffffu, key == m)) - 1;
      const int widx = __shfl_sync(0xffffffffu, lane * 8 + bj, wl);
      const float wsc = __shfl_sync(0xffffffffu, bs, wl);
      if (lane == wl) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j == bj) sel[j] = -INFINITY;
      }
      if (lane == k) my_id = widx, my_w = wsc;
    }
    float s = my_w;
    s += __shfl_xor_sync(0xffffffffu, s, 4);
    s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 1);
    if (lane < 8) {
      IDS[t * 8 + lane] = my_id;
      const int lid = (my_id - lsv + 256) % 256;
      LIDS[t * 8 + lane] = lid < 32 ? lid : -1;
      WTS[t * 8 + lane] = t_div(my_w, s + 1e-15f);
    }
    return;
  }
  float g[4 * NV], p[4 * NV];
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(GPRE + (i * NTN + threadIdx.x) * 4));
    g[4 * i] = 1.0f + bf_lo(u.x); g[4 * i + 1] = 1.0f + bf_hi(u.x);
    g[4 * i + 2] = 1.0f + bf_lo(u.y); g[4 * i + 3] = 1.0f + bf_hi(u.y);
  }
  pdl_wait();
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    const float4 u = __ldcg(reinterpret_cast<const float4*>(row + (i * NTN + threadIdx.x) * 4));
    p[4 * i] = u.x; p[4 * i + 1] = u.y; p[4 * i + 2] = u.z; p[4 * i + 3] = u.w;
  }
  float q = 0.f;
#pragma unroll
  for (int i = 0; i < 4 * NV; ++i) q += p[i] * p[i];
  q = warp_sum(q);
  if (lane == 0) ws[warp] = q;
  asm volatile("bar.sync 1, %0;" ::"n"(NTN) : "memory");
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NTN / 32; ++i) tot += ws[i];
  const float r = t_rsqrt(t_div(tot, 4096.0f) + 1e-5f);
#pragma unroll
  for (int i = 0; i < NV; ++i) {
    uint2 o;
    o.x = pack_bf2(p[4 * i] * r * g[4 * i], p[4 * i + 1] * r * g[4 * i + 1]);
    o.y = pack_bf2(p[4 * i + 2] * r * g[4 * i + 2], p[4 * i + 3] * r * g[4 * i + 3]);
    *reinterpret_cast<uint2*>(XOUT + (size_t)t * 4096 + (i * NTN + threadIdx.x) * 4) = o;
  }
}
template <typename K, typename... A>
static void launch(K kern, int grid, int block, int cl, bool pdl, A... args) {
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(grid);
  lc.blockDim = dim3(block);
  lc.dynamicSmemBytes = 0;
  lc.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at_[2];
  int na = 0;
  if (cl > 1) {
    at_[na].id = cudaLaunchAttributeClusterDimension;
    at_[na].val.clusterDim.x = cl;
    at_[na].val.clusterDim.y = 1;
    at_[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    at_[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  lc.attrs = at_;
  lc.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, args...));
}
static bool g_bnd_one = false;
template <typename PT, bool GP, int CL, int NT>
static void bnd_go(const at::Tensor& P, int S, int T, int NU, const at::Tensor& X, const at::Tensor& gate,
                   const at::Tensor& gpost, const at::Tensor& g, const at::Tensor& sout, const at::Tensor& hout, bool pdl) {
  auto kern = g_bnd_one ? bnd1r_kernel<PT, GP, CL, NT> : bnd_kernel<PT, GP, CL, NT>;
  if (CL > 8) {
    static bool done = false;
    if (!done) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(bnd_kernel<PT, GP, CL, NT>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      C10_CUDA_CHECK(cudaFuncSetAttribute(bnd1r_kernel<PT, GP, CL, NT>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      done = true;
    }
  }
  launch(kern, T * CL, NT, CL, pdl, reinterpret_cast<const PT*>(P.data_ptr()), S, T, NU,
         reinterpret_cast<const float*>(X.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
         reinterpret_cast<const __nv_bfloat16*>(gpost.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(g.data_ptr()),
         reinterpret_cast<float*>(sout.data_ptr()), reinterpret_cast<__nv_bfloat16*>(hout.data_ptr()));
}
template <typename PT, bool GP>
static void bnd_cl(int cl, const at::Tensor& P, int S, int T, int NU, const at::Tensor& X, const at::Tensor& gate,
                   const at::Tensor& gpost, const at::Tensor& g, const at::Tensor& sout, const at::Tensor& hout, bool pdl) {
  switch (cl) {
    case 1: bnd_go<PT, GP, 1, 512>(P, S, T, NU, X, gate, gpost, g, sout, hout, pdl); return;
    case 2: bnd_go<PT, GP, 2, 512>(P, S, T, NU, X, gate, gpost, g, sout, hout, pdl); return;
    case 4: bnd_go<PT, GP, 4, 256>(P, S, T, NU, X, gate, gpost, g, sout, hout, pdl); return;
    case 8: bnd_go<PT, GP, 8, 256>(P, S, T, NU, X, gate, gpost, g, sout, hout, pdl); return;
    case 16: bnd_go<PT, GP, 16, 128>(P, S, T, NU, X, gate, gpost, g, sout, hout, pdl); return;
    default: TORCH_CHECK(false, "ldd19 bnd: cluster size");
  }
}
}
void ldd19_bnd(at::Tensor P, int64_t S, int64_t T, int64_t NU, at::Tensor X, at::Tensor gate, at::Tensor gpost,
               int64_t has_gp, at::Tensor g, at::Tensor sout, at::Tensor hout, int64_t cl, int64_t pdl) {
  const bool bf = P.scalar_type() == at::kBFloat16;
  TORCH_CHECK(bf || P.scalar_type() == at::kFloat, "ldd19 bnd: P dtype");
  if (bf) {
    if (has_gp) ldd19::bnd_cl<__nv_bfloat16, true>((int)cl, P, (int)S, (int)T, (int)NU, X, gate, gpost, g, sout, hout, pdl != 0);
    else ldd19::bnd_cl<__nv_bfloat16, false>((int)cl, P, (int)S, (int)T, (int)NU, X, gate, gpost, g, sout, hout, pdl != 0);
  } else {
    if (has_gp) ldd19::bnd_cl<float, true>((int)cl, P, (int)S, (int)T, (int)NU, X, gate, gpost, g, sout, hout, pdl != 0);
    else ldd19::bnd_cl<float, false>((int)cl, P, (int)S, (int)T, (int)NU, X, gate, gpost, g, sout, hout, pdl != 0);
  }
}
void ldd19_bnd_one(int64_t on) { ldd19::g_bnd_one = on != 0; }
void ldd19_route(at::Tensor pooled, at::Tensor logits, at::Tensor bias, at::Tensor gpre, at::Tensor ids, at::Tensor wts,
                 at::Tensor xm, at::Tensor ls, at::Tensor lids, int64_t T, int64_t pdl) {
  ldd19::launch(ldd19::route_kernel, (int)T, ldd19::RT_NTN + 32, 1, pdl != 0,
                reinterpret_cast<const __nv_bfloat16*>(pooled.data_ptr()), reinterpret_cast<const float*>(logits.data_ptr()),
                reinterpret_cast<const float*>(bias.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gpre.data_ptr()),
                reinterpret_cast<int*>(ids.data_ptr()), reinterpret_cast<float*>(wts.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(xm.data_ptr()), reinterpret_cast<const int*>(ls.data_ptr()),
                reinterpret_cast<int*>(lids.data_ptr()));
}
void ldd19_route32(at::Tensor P, at::Tensor bias, at::Tensor gpre, at::Tensor ids, at::Tensor wts, at::Tensor xm,
                   at::Tensor ls, at::Tensor lids, int64_t T) {
  ldd19::launch(ldd19::route32_kernel<256>, (int)T, 256 + 32, 1, false, reinterpret_cast<const float*>(P.data_ptr()),
                reinterpret_cast<const float*>(bias.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(gpre.data_ptr()),
                reinterpret_cast<int*>(ids.data_ptr()), reinterpret_cast<float*>(wts.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(xm.data_ptr()), reinterpret_cast<const int*>(ls.data_ptr()),
                reinterpret_cast<int*>(lids.data_ptr()));
}
"""

LDD19_GLUE_CPP_SRC = """#include <torch/extension.h>
void ldd19_bnd(torch::Tensor P, int64_t S, int64_t T, int64_t NU, torch::Tensor X, torch::Tensor gate, torch::Tensor gpost,
               int64_t has_gp, torch::Tensor g, torch::Tensor sout, torch::Tensor hout, int64_t cl, int64_t pdl);
void ldd19_route(torch::Tensor pooled, torch::Tensor logits, torch::Tensor bias, torch::Tensor gpre, torch::Tensor ids,
                 torch::Tensor wts, torch::Tensor xm, torch::Tensor ls, torch::Tensor lids, int64_t T, int64_t pdl);
void ldd19_bnd_one(int64_t on);   // LDD19: fold agent
void ldd19_route32(torch::Tensor P, torch::Tensor bias, torch::Tensor gpre, torch::Tensor ids, torch::Tensor wts,
                   torch::Tensor xm, torch::Tensor ls, torch::Tensor lids, int64_t T);   // LDD19: big2
"""

_LDD19_EXT = None
_LDD19_TRIED = False


def _ldd19_glue():
  """LDD19: build the glue extension (sm_90a only); any failure leaves the Triton kernels in place."""
  global _LDD19_EXT, _LDD19_TRIED
  if _LDD19_TRIED:
    return _LDD19_EXT
  _LDD19_TRIED = True
  try:
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
      return None
    from torch.utils.cpp_extension import load_inline
    arch = f"{major}{minor}a"
    _LDD19_EXT = _ldd19_inline(
        name=f"ldd19_glue_v4_{arch}", cpp_sources=LDD19_GLUE_CPP_SRC, cuda_sources=LDD19_GLUE_CUDA_SRC,   # LDD19: v4 (big2)
        functions=["ldd19_bnd", "ldd19_route", "ldd19_bnd_one", "ldd19_route32"],   # LDD19: + ldd19_bnd_one; big2: + ldd19_route32
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
        no_implicit_headers=True, verbose=False)
  except Exception as ex:
    print("ldd19 glue disabled:", repr(ex)[:300], flush=True)
    _LDD19_EXT = None
  return _LDD19_EXT


def _ldd19_on(T, *ts):
  """LDD19: the CUDA glue applies (flag, built, token count, and the dtypes / layouts it is written for)."""
  if not (LDD19_CLUSTER_GLUE and _LDD19_EXT is not None and T <= LDD19_GLUE_MAX_T):
    return False
  return all(x.is_cuda and x.is_contiguous() for x in ts)


def _ldd19_wts_ok(st):
  """LDD19: the gains / gates are BF16 vectors and the router bias FP32 (as the checkpoint gives them)."""
  bf = torch.bfloat16
  return (all(st[k].dtype == bf and st[k].is_contiguous() for k in ("gate1", "gate2", "g_preffn", "g_preexp", "g_postffn", "g_next"))
          and st["bias"].dtype == torch.float32)


# LDD19: lt128, the dense projections at T = 128 / 256 (verify 16 / 32), where v13 ran torch.mm (qkv BF16 out; o and the
#   LDD19_RQ    T = 128 only: chenwainuo's rq kernel (12-bit lossless W in wgmma A registers, x as the smem B operand,
#   LDD19_LT    T = 128 / 256: NathenFernandes' pinned cuBLASLt algorithms (W pre-transposed where the layout says so,
#   LDD19_BF16  o / unpooler: v13's torch.mm with a BF16 output (the baseline's output precision) where neither serves.
LDD19_LT = int(_ldd19_os.environ.get("LDD19_LT", "13"))   # LDD19: big2: bit 8 = the combined pooler + router (FP32 out)
LDD19_RQ = int(_ldd19_os.environ.get("LDD19_RQ", "5"))
LDD19_BF16 = int(_ldd19_os.environ.get("LDD19_BF16", "2"))
LDD19_BIG_BITS = {"qkv": 1, "o": 2, "un": 4, "pr": 8}   # LDD19: big2: + pr (cuBLASLt only; neither rq nor the BF16 fallback)
# LDD19: big3: PDL at T = 128 per edge whose two kernels are ours (each measured alone in the layer, then bundled):
LDD19_PDL_EDGES = int(_ldd19_os.environ.get("LDD19_PDL_EDGES", "63"))   # LDD19: v20: + 16 (attn->ol128) + 32 (ol128->_boundary1)


LDD19_EDGE2_256 = True   # LDD19: iter1 solo: edge 2 (route32 -> moe, both ours) also at T = 256 (paired +0.2% verify 32)


def _ldd19_edge(bit, T):
  """LDD19: big3: the PDL edge bit is on (T = 128 only; edge 2 also at T = 256 under LDD19_EDGE2_256)."""
  return bool(LDD19_PDL_EDGES & bit) and (T == 128 or (LDD19_EDGE2_256 and bit == 2 and T == 256))


def _ldd19_pr_lt(st, T):
  """LDD19: big2: the combined pooler + router runs on its pinned cuBLASLt plan at this T (else v15's choice)."""
  return T > BFP_MAX_T and ((bool(LDD19_LT & 8) and (T, "pr") in st.get("ldd19_lt", {}))
                            or (bool(LDD19_LTS & 8) and (T, "pr") in st.get("ldd19_lts", {})))   # LDD19: ltscan


def _ldd19_big(st, name, x, T, w):
  """LDD19: lt128, the matrix's large-token backend (BF16 [T, N]), or None (v13's path)."""
  bit = LDD19_BIG_BITS.get(name, 0)
  if not bit or x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
    return None
  if LDD19_OL & bit and T == 128 and name == "o":   # LDD19: v20: ol_projection + PDL edges 16 / 32
    out = _ldd19_ol_run(st, x, (1 if _ldd19_edge(16, T) else 0) | (2 if _ldd19_edge(32, T) else 0))
    if out is not None:
      return out
  if LDD19_LTS & bit and (T, name) in st.get("ldd19_lts", {}):   # LDD19: ltscan: the rescanned plan
    out = _ldd19_lt_run(st, name, x, T, "ldd19_lts")
    if out is not None:
      return out
  if LDD19_RQ & bit and T == 128 and name != "pr":   # LDD19: big2: no rq route for pr
    out = _ldd19_rq_run(st, name, x, name == "un" and _ldd19_edge(4, T))   # LDD19: big3: edge 4 (PDL behind the experts)
    if out is not None:
      return out
  if LDD19_LT & bit:
    out = _ldd19_lt_run(st, name, x, T)
    if out is not None:
      return out
  if LDD19_BF16 & bit & 6:
    return torch.mm(x, w.t())
  return None


# LDD19: the pinned cuBLASLt path, from our muse_bf16_proj sub_v22 (tt_lt_*: NathenFernandes' v18, MIT; the unpooler
# m = 128 descriptor is chenwainuo's pick, Apache-2.0); the extension pruned to what a pinned plan uses (no enumeration).
LDD19_LT_CPP = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <array>
#include <memory>
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
  torch::Tensor weight;
  int m, n, k, layout;
  cudaDataType_t out_type;
  std::vector<cublasLtMatmulAlgo_t> algorithms;
  std::vector<size_t> workspaces;
  LtPlan(std::shared_ptr<LtContext> ctx, torch::Tensor w, int rows, int cols, int inner, int variant, int out_f32,
         int nb)
      : context(ctx), weight(w), m(rows), n(cols), k(inner), layout(variant),
        out_type(out_f32 ? CUDA_R_32F : CUDA_R_16BF) {
    c10::cuda::CUDAGuard guard(w.device());
    TORCH_CHECK(nb >= 1 && cols % nb == 0, "LDD19: ltscan: batch count must divide N");
    const int nn = cols / nb;
    LT_OK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    int8_t fast = 0;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
    bool is_row = layout % 2;
    bool is_packed = layout >= 2;
    cublasOperation_t trans_a = !is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b = is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    const int64_t wstride = is_packed ? (int64_t)nn : (int64_t)nn * k;
    auto batch = [nb](cublasLtMatrixLayout_t d, int64_t stride) {
      if (nb == 1) return;
      int32_t c = nb;
      LT_OK(cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &c, sizeof(c)));
      LT_OK(cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    };
    if (is_row) {
      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, m, k, k));
      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, is_packed ? k : nn,
                                      is_packed ? nn : k, is_packed ? n : k));
      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, out_type, m, nn, n));
      cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
      for (auto d : {a_layout, b_layout, d_layout})
        LT_OK(cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
      batch(a_layout, 0); batch(b_layout, wstride); batch(d_layout, nn);
    } else {
      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, is_packed ? nn : k,
                                      is_packed ? k : nn, is_packed ? n : k));
      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, k, m, k));
      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, out_type, nn, m, n));
      batch(a_layout, wstride); batch(b_layout, 0); batch(d_layout, nn);
    }
    cublasLtEpilogue_t e = CUBLASLT_EPILOGUE_DEFAULT;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
  }
  ~LtPlan() {
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
  }
  bool add(const cublasLtMatmulAlgo_t& algo) {
    Config c = config_of(algo);
    if (c[3] != CUBLASLT_REDUCTION_SCHEME_NONE && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
      return false;
    if (c[2] > 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return false;
    uint64_t flags = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                         &flags, sizeof(flags), &written) != CUBLAS_STATUS_SUCCESS) return false;
    if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)
      return false;
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
        CUDA_R_16BF, CUDA_R_16BF, out_type, out_type, values[0], &algo) != CUBLAS_STATUS_SUCCESS) return -1;
    for (int i = 1; i < 9; ++i) {
      if (!set_config(algo, i, values[i]) && values[i] != 0) return -1;
    }
    if (!add(algo)) return -1;
    return algorithms.size() - 1;
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
  pybind11::class_<LtContext, std::shared_ptr<LtContext>>(m, "Context", pybind11::module_local())
    .def(pybind11::init<torch::Tensor>());
  pybind11::class_<LtPlan, std::shared_ptr<LtPlan>>(m, "Plan", pybind11::module_local())
    .def(pybind11::init<std::shared_ptr<LtContext>, torch::Tensor, int, int, int, int, int, int>())
    .def("metadata", &LtPlan::metadata).def("restore", &LtPlan::restore).def("run", &LtPlan::run);
}
"""
LDD19_LT_VERSION = 130101
LDD19_LT_WS = 128 << 20
LDD19_LT_FIXED = {(128, "qkv"): (0, 0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (256, "qkv"): (0, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2]),
                  (128, "un"): (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5]), (256, "un"): (3, 0, [66, 20, 1, 0, 0, 4, 35, 0, 2]),
                  # LDD19: big2: the combined pooler + router, FP32 out, natural weight layout, no split-K (nvjet
                  (128, "pr"): (0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 4]), (256, "pr"): (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 2])}
# LDD19: ltscan: plans from a full rescan (every algo x tile x stages x cluster AlgoCheck accepts, then swizzle / custom /
# They run in place of v18's backend for (T, matrix) when the matrix's LDD19_LTS bit is on (read at launch).
LDD19_LTS = int(_ldd19_os.environ.get("LDD19_LTS", "2"))
LDD19_LTS_FIXED = {(256, "o"): (1, 1, [66, 20, 1, 0, 0, 6, 35, 0, 2])}   # LDD19: ltscan: o at T = 256, row-major natural W, nvjet 128x128_64x6 coopA, bit-identical to torch.mm (isolated 9.98 / 10.19 vs 10.27 / 10.62 us); no batched layout beat batch 1 on any (T, matrix)
_LDD19_LT_EXT = None


def _ldd19_lt_build():
  """LDD19: build the cuBLASLt extension (sm_90 only, only when a flag uses it); a failure leaves torch.mm."""
  global _LDD19_LT_EXT
  if not LDD19_LT or _LDD19_LT_EXT is not None:
    return _LDD19_LT_EXT
  try:
    if torch.cuda.get_device_capability() != (9, 0):
      return None
    from torch.utils.cpp_extension import load_inline
    _LDD19_LT_EXT = _ldd19_inline(name="ldd19_cublaslt_v3", cpp_sources=LDD19_LT_CPP, with_cuda=True,   # LDD19: big2: v2; ltscan: v3
                                extra_cflags=["-O3", "-std=c++20"], extra_ldflags=["-l:libcublasLt.so.13"], verbose=False)
  except Exception as ex:
    print("ldd19 cublaslt disabled:", repr(ex)[:300], flush=True)
    _LDD19_LT_EXT = None
  return _LDD19_LT_EXT


def _ldd19_lt_prepare(st):
  """LDD19: one plan per pinned (T, matrix) the flags use: the algorithm restored from its config (and the config read
    back), one run on random activations against an FP64 reference; anything that fails keeps torch.mm."""
  st["ldd19_lt"] = {}
  ext = _LDD19_LT_EXT
  if ext is None:
    return
  try:
    if int(ext.version()) != LDD19_LT_VERSION:
      print(f"ldd19 cublaslt: version {ext.version()} != {LDD19_LT_VERSION}, torch.mm kept", flush=True)
      return
    dev = st["qkv"].device
    ws = torch.empty(LDD19_LT_WS, device=dev, dtype=torch.uint8)
    ctx = ext.Context(ws)
    mats = {"qkv": st["qkv"], "o": st["o"], "un": st["unpool"], "pr": st["pr"]}   # LDD19: big2: + pr
    packed = {}
    g = torch.Generator(device=dev).manual_seed(4242)
    st["ldd19_lts"] = {}   # LDD19: ltscan
    jobs = [((T, name), layout, 1, cfg, False) for (T, name), (layout, epi, cfg) in LDD19_LT_FIXED.items()]
    jobs += [((T, name), layout, nb, cfg, True) for (T, name), (layout, nb, cfg) in LDD19_LTS_FIXED.items()]
    for (T, name), layout, nb, cfg, lts in jobs:   # LDD19: ltscan: v18's pins, then the rescanned ones
      if not (LDD19_LTS if lts else LDD19_LT) & LDD19_BIG_BITS[name]:
        continue
      w = mats[name]
      n, k = w.shape
      if layout >= 2 and name not in packed:
        packed[name] = w.t().contiguous()
      f32 = name == "pr"   # LDD19: big2: the pooler + router keeps v15's FP32 output
      plan = ext.Plan(ctx, packed[name] if layout >= 2 else w, T, n, k, layout, int(f32), nb)   # LDD19: ltscan: + nb
      idx = plan.restore(list(cfg))
      if idx < 0 or list(plan.metadata()[idx][:9]) != list(cfg):
        print(f"ldd19 cublaslt {name} T={T}: config not restored, torch.mm kept", flush=True)
        continue
      x = (torch.randn(T, k, generator=g, device=dev) * 0.5).to(torch.bfloat16)
      out = torch.empty((T, n), device=dev, dtype=torch.float32 if f32 else torch.bfloat16)   # LDD19: big2
      rc = plan.run(x, out, idx)
      torch.cuda.synchronize()
      ref = x.double() @ w.double().t()
      err = float((out.double() - ref).norm() / ref.norm()) if rc == 0 else float("inf")
      if f32 and rc == 0:   # LDD19: big2: pooled vs FP64, and the routing logits (hi + lo) vs FP64 and torch.mm
        mm = torch.mm(x, w.t(), out_dtype=torch.float32).double()
        err = float((out[:, :4096].double() - ref[:, :4096]).norm() / ref[:, :4096].norm())
        lg = out[:, 4096:4352].double() + out[:, 4352:].double()
        lr = ref[:, 4096:4352] + ref[:, 4352:]
        lm = mm[:, 4096:4352] + mm[:, 4352:]
        el = float((lg - lr).norm() / lr.norm())
        em = float((lg - lm).abs().max() / lm.abs().max())
        if not (el < 1e-5 and em < 1e-5):
          err = float("inf")
        print(f"ldd19 cublaslt pr T={T}: pooled rel L2 {err:.2e}, logits rel L2 {el:.2e} vs FP64, max rel "
              f"{em:.1e} vs torch.mm, bit-identical to torch.mm {bool((out.double() == mm).all())}", flush=True)
      if not err < 4e-3:
        print(f"ldd19 cublaslt {name} T={T}: rc {rc} rel err {err:.2e}, torch.mm kept", flush=True)
        continue
      if lts:   # LDD19: ltscan: kept only if bit-identical to torch.mm (BF16 out: exactly what v18 runs for o)
        if not f32:
          same = bool(torch.equal(out, torch.mm(x, w.t())))
          print(f"ldd19 cublaslt ltscan {name} T={T} L{layout} b{nb} {cfg}: rel L2 {err:.2e} vs FP64, "
                f"bit-identical to torch.mm {same}{'' if same else ', not used'}", flush=True)
          if not same:
            continue
        st["ldd19_lts"][(T, name)] = (plan, idx, n)
        continue
      st["ldd19_lt"][(T, name)] = (plan, idx, n)
    st["ldd19_lt_ws"] = (ws, ctx, packed)
  except Exception as ex:
    print("ldd19 cublaslt disabled:", repr(ex)[:300], flush=True)
    st["ldd19_lt"] = {}


def _ldd19_lt_run(st, name, x, T, table="ldd19_lt"):   # LDD19: ltscan: + table
  ent = st.get(table, {}).get((T, name))
  if ent is None or x.data_ptr() % 256:
    return None
  plan, idx, n = ent
  out = torch.empty((T, n), device=x.device, dtype=torch.float32 if name == "pr" else torch.bfloat16)   # LDD19: big2
  rc = plan.run(x, out, idx)
  if rc:
    raise RuntimeError(f"cuBLASLt launch failed: status {rc}")
  return out


# LDD19: chenwainuo's rq kernel (chenwainuo/okc, Apache-2.0; see the header), its wgmma helper and 12-bit pack / escape
LDD19_RQ_CU = r"""#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>
struct alignas(64) RqTmap { unsigned long long v[16]; };
struct RqP {
  const uint8_t* W;
  const uint32_t* esc;
  int ecap;
  __nv_bfloat16* out;
  int* cnt;
  unsigned long long* prof;
  uint32_t* sink;
  int m, N, K, KC, NB, walias, ncnt;
  uint32_t Tb;
  int pdl;
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
__device__ __forceinline__ uint4 rq_ldg(const void* a) {
  uint4 v; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a)); return v;
}
__device__ __forceinline__ uint32_t rq_prmt(uint32_t a, uint32_t b, uint32_t s) { uint32_t d; asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void rq_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t Tb, uint32_t* a) {
  const uint32_t hl = (C & 0x87878787u) + Tb, hh = (__funnelshift_r(C, C, 4) & 0x87878787u) + Tb;
  a[0] = rq_prmt(L0, hl, 0x5140); a[1] = rq_prmt(L0, hl, 0x7362); a[2] = rq_prmt(L1, hh, 0x5140); a[3] = rq_prmt(L1, hh, 0x7362);
}
__device__ __forceinline__ void rq_unit(const uint4& q0, const uint4& q1, const uint4& q2, uint32_t Tb, uint32_t* A) {
  rq_dec(q0.x, q0.y, q2.x, Tb, A); rq_dec(q0.z, q0.w, q2.y, Tb, A + 4);
  rq_dec(q1.x, q1.y, q2.z, Tb, A + 8); rq_dec(q1.z, q1.w, q2.w, Tb, A + 12);
}
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
__device__ __forceinline__ uint64_t rq_desc(uint32_t saddr) {
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
__device__ __forceinline__ uint32_t rq_pk(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
template <int NT> __device__ __forceinline__ void rq_wgmma(float (&D)[NT / 2], const uint32_t* a, uint64_t desc, int scd);
template <int NT> __device__ __forceinline__ void rq_wgss(float (&D)[NT / 2], uint64_t da, uint64_t db, int scd);
@@WGMMA@@
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
struct RqC {
  static constexpr int XB = NT * 128, WB = (FL & 16) ? 6144 : 12288, ND = NT / 2;
  static constexpr int SMEM = 1024 + ST * (XB + WB) + 24 * ST + 16;
};
template <int NT, int ST, int GS, int PRB, int WM, int INS, int FL>
__global__ void __launch_bounds__(384, 1) rq(const RqP p, const __grid_constant__ RqTmap tx, const __grid_constant__ RqTmap to) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  using C = RqC<NT, ST, GS, PRB, WM, INS, FL>;
  constexpr int XB = C::XB, WB = C::WB, ND = C::ND;
  constexpr bool HINT = FL & 1, ZI = (FL & 2) && PRB == 0;
  constexpr bool RES = PRB == 8 || PRB == 10;
  constexpr bool XON = PRB != 1 && PRB != 5 && PRB != 7 && !RES;
  constexpr bool WON = PRB != 7 && !RES;
  constexpr bool SELF = PRB == 5 || PRB == 6;
  constexpr bool HALF = (FL & 16) != 0;
  static_assert(!(FL & 65536) || (NT == 128 && ST == 5 && GS == 1 && PRB == 0 && WM == 0 && INS == 0 && (FL == 65538 || FL == 65539)), "EH backport approved M128/ST5/selfzero full-output variants only");
  constexpr int NW = HALF ? 4 : 8, CW = HALF ? 64 : 128;
  constexpr bool EARLY = (FL & 32) != 0 && PRB == 0;
  constexpr bool RG = (FL & 16777216) != 0;
  static_assert(!RG || (GS == 1 && WM == 0 && PRB == 0 && INS == 0), "RG: plain variants only");
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem), sbase = (sraw + 1023u) & ~1023u;
  const uint32_t xring = sbase, wring = sbase + ST * XB, bars = wring + ST * WB;
  const uint32_t full = bars, wempty = bars + 8 * ST, xempty = bars + 16 * ST, zflag = bars + 24 * ST;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nb = HALF ? (int)(blockIdx.x >> 1) : (int)blockIdx.x, hh = HALF ? (int)(blockIdx.x & 1) : 0, ks = blockIdx.y, KS = gridDim.y;
  const int cb = nb * 128 + 64 * hh;
  const int s0 = (int)((long long)ks * p.KC / KS), KCS = (int)((long long)(ks + 1) * p.KC / KS) - s0;
  const unsigned long long gt0 = INS ? rq_gt() : 0;
  const uint32_t* el = p.esc + (size_t)((ks * p.NB + nb) * 8 + (HALF ? 4 * hh + (warp & 3) : (warp & 7))) * p.ecap;
  uint32_t ec = warp < NW ? __ldg(el + lane) : 0u;
  uint4 QR0 = make_uint4(0, 0, 0, 0), QR1 = QR0, QR2 = QR0;
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
      if (p.pdl) {
        for (int s = 0; s < ST && s < KCS; s++) {
          if (RG && s == 0) {
            rq_arrive(full);
          } else {
            rq_expect(full + 8 * s, WB);
            if (HINT) rq_bulkh(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s, pw);
            else rq_bulk(wring + s * WB, src + (size_t)s * 12288, WB, full + 8 * s);
          }
        }
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
        for (int s = 0; s < ST && s < KCS; s++) {
          rq_expect(full + 8 * s, XB);
          if (HINT) rq_tileh(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s, px);
          else rq_tile(xring + s * XB, &tx, (s0 + s) * 64, 0, full + 8 * s);
        }
      } else
      for (int s = 0; s < ST && s < KCS; s++) {
        if (RG && s == 0) rq_arrive(full);
        else rq_expect(full + 8 * s, WB);
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
  if (warp >= 8) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
    if (warp == 8 && lane == 0 && WON) {
      const int nbw = p.walias ? nb % p.walias : nb;
      const uint8_t* src = p.W + ((size_t)nbw * p.KC + s0) * 12288 + hh * 6144;
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;
      int slot = 0, r = EARLY ? 1 : 0;
      for (int s = se; s < KCS; s++) {
        if (s >= ST) {
          const unsigned long long t0 = INS ? clock64() : 0;
          rq_wait((SELF ? full : wempty) + 8 * slot, (r & 1) ^ 1);
          if (INS) tw += clock64() - t0;
        }
        if (RG && s == 0) {
          rq_arrive(full);
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
      if (p.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");
      uint64_t pol = 0;
      if (HINT) asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
      unsigned long long tw = 0;
      const int se = EARLY ? (KCS < ST ? KCS : ST) : 0;
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
    } else if (ZI && warp >= 10) {
      const int t = tid - 320, r0 = (int)((long long)ks * p.m / KS), r1 = (int)((long long)(ks + 1) * p.m / KS);
      if (p.pdl) asm volatile("griddepcontrol.wait;\n" ::: "memory");
      for (int e = t; e < (r1 - r0) * (CW / 8); e += 64) {
        __nv_bfloat16* d = p.out + (size_t)(r0 + e / (CW / 8)) * p.N + cb + (e % (CW / 8)) * 8;
        asm volatile("st.global.v4.u32 [%0], {%1, %1, %1, %1};\n" :: "l"(d), "r"(0) : "memory");
      }
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
      asm volatile("bar.sync 4, 64;\n" ::: "memory");
      if (t == 0) {
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;\n" :: "l"(p.cnt + blockIdx.x) : "memory");
        uint32_t c = 0;
        for (int it = 0; it < (1 << 22); it++) {
          asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(c) : "l"(p.cnt + blockIdx.x) : "memory");
          if ((int)c >= KS) break;
          __nanosleep(128);
        }
        if ((int)c < KS) asm volatile("trap;");
        asm volatile("fence.proxy.async.global;\n" ::: "memory");
        asm volatile("st.release.cta.shared.u32 [%0], 1;\n" :: "r"(zflag) : "memory");
        if (atomicAdd(p.cnt + gridDim.x + blockIdx.x, 1) == KS - 1) { p.cnt[blockIdx.x] = 0; p.cnt[gridDim.x + blockIdx.x] = 0; }
      }
    }
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");
  if (SELF || (HALF && warp >= 4)) return;
  const int g = warp >> 2, w = warp & 3;
  int ei = 0;
  uint32_t nx = __shfl_sync(0xffffffffu, ec, 0), nxs = nx >> 17;
  uint32_t Tb = p.Tb;
  asm volatile("" : "+r"(Tb));
  const uint32_t wl = wring + warp * 1536 + lane * 16;
  const uint64_t xd0 = rq_desc(xring);
  float D[ND];
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
  const int NGF = KCS / GS, NGT = (KCS + GS - 1) / GS;
  auto ldr = [&](int G) {
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
    if (INS) {
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
  auto mma = [&](int G, auto NSC, uint32_t (&A)[GS][16]) {
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
    if constexpr (RG) {
      rq_wait(full, 0);
      Q[0][0] = QR0; Q[0][1] = QR1; Q[0][2] = QR2;
      __syncwarp();
      if (lane == 0) rq_arrive(wempty);
    } else ldr(0);
    if (WM != 0) dec(0, GS, A0);
    int G = 0;
    if (WM == 2) {
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
      if (GS > 1 && NGT > NGF) {
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
  if (PRB == 0 || PRB == 7) {
    if (PRB == 7) {
#pragma unroll
      for (int i = 0; i < ND; i++) D[i] = (float)((i + tid) & 7) * 0.125f;
    }
    if constexpr (FL & 65536) {
    const unsigned long long ce0 = INS ? clock64() : 0;
    unsigned long long ce[5] = {0, 0, 0, 0, 0};
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");
    if (INS) ce[0] = clock64() - ce0;
    const uint32_t slab = xring + g * XB;
    const int j = lane >> 3, rr = lane & 7;
    constexpr bool EH = (FL & 65536) != 0, EP = false;
    const bool issuer = (w == 0 && lane == 0);
    auto zpoll = [&]() {
      if (ZI) {
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
    asm volatile("bar.sync 1, %0;\n" :: "r"(NW * 32) : "memory");
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
      if (ZI) {
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
      asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
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
  if (p.pdl) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = dim3(p.NB * ((FL & 16) ? 2 : 1), KS);
    lc.blockDim = dim3(384);
    lc.dynamicSmemBytes = C::SMEM;
    lc.stream = st;
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    cudaLaunchKernelEx(&lc, k, p, tx, to);
  } else
  k<<<dim3(p.NB * ((FL & 16) ? 2 : 1), KS), dim3(384), C::SMEM, st>>>(p, tx, to);
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : (int)e;
}
int64_t v19_rqrun(int64_t v, int64_t W, int64_t NB, int64_t KC, int64_t x, int64_t m, int64_t K, int64_t out, int64_t N,
                  int64_t esc, int64_t ecap, int64_t cnt, int64_t ncnt, int64_t KS, int64_t Tb, int64_t sink, int64_t stream,
                  int64_t info, int64_t pdl) {
  RqP p = {};
  if (!info) {
    if (K != KC * 64 || N != NB * 128 || m < 1 || ecap < 32 || ecap % 32) return -11;
    p.W = (const uint8_t*)W; p.esc = (const uint32_t*)esc; p.ecap = (int)ecap;
    p.out = (__nv_bfloat16*)out; p.cnt = (int*)cnt; p.ncnt = (int)ncnt; p.prof = nullptr; p.sink = (uint32_t*)sink;
    p.m = (int)m; p.K = (int)K; p.N = (int)N; p.KC = (int)KC; p.NB = (int)NB; p.walias = 0; p.Tb = (uint32_t)(Tb & 0xffffffff);
    p.pdl = (int)pdl;
  }
  switch (v) {
@@CASES@@
  }
  return -2;
}
"""


# ---- chenwainuo/okc (Apache-2.0): wgmma helper and the 12-bit pack / escape-list helpers, unchanged
def _rqwg(n):
  nd = n // 2
  d = ", ".join("%%%d" % i for i in range(nd))
  a = ", ".join("%%%d" % (nd + i) for i in range(4))
  outs = ", ".join('"+f"(D[%d])' % i for i in range(nd))
  return (("template <> __device__ __forceinline__ void rq_wgmma<%d>(float (&D)[%d], const uint32_t* a, uint64_t desc, int scd) {\n"
           "  asm volatile(\"{\\n.reg .pred p;\\nsetp.ne.b32 p, %%%d, 0;\\n"
           "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, p, 1, 1, 0;\\n}\\n\"\n"
           "    : %s\n    : \"r\"(a[0]), \"r\"(a[1]), \"r\"(a[2]), \"r\"(a[3]), \"l\"(desc), \"r\"(scd));\n}\n")
          % (n, nd, nd + 5, n, d, a, nd + 4, outs))


_OS = ((0, 4, 8, 12, 16, 20, 24, 28), (0, 8, 16, 24, 4, 12, 20, 28))
_SS = ((3, 11, 19, 27, 7, 15, 23, 31), (7, 15, 23, 31, 11, 19, 27, 3))


def _c12(fr, dv=1):
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
  N, K = w.shape
  W = w.view(N // 16, 2, 8, K // 64, 4, 2, 4, 2)
  return W.permute(0, 3, 4, 2, 6, 5, 1, 7).reshape(N // 16, K // 64, 4, 32, 8).contiguous()


def _rqpack(w):
  N, K = w.shape
  c = _c12(_pack_nat(w), 1)
  if c is None or N % 128:
    return None
  data, tab, esc = c
  T, KC = data.shape[:2]
  lay = data.reshape(N // 128, 8, KC, 1536).permute(0, 2, 1, 3).contiguous()
  return lay, tab, esc


def _rqunpack(lay, tab, esc, N, K):
  NB, KC = lay.shape[:2]
  data = lay.permute(0, 2, 1, 3).reshape(NB * 8, KC, 3, 32, 16)
  v = _unc12(data, tab, esc, 1)
  W = v.view(N // 16, K // 64, 4, 8, 4, 2, 2, 2)
  return W.permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(N, K)


def _rqesc(esc6, NB, KC, KS, dev):
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


LDD19_RQ_VARIANTS = [(128, 5, 1, 0, 0, 0, 16777251), (128, 5, 1, 0, 0, 0, 3)]
LDD19_RQ_ROUTES = {"qkv": [(0, 7), (1, 7)], "un": [(0, 2), (1, 2)]}
_LDD19_RQ = {}


def _ldd19_rq_build():
  """LDD19: build the two rq instances (sm_90a only, only when a flag uses them); a failure leaves the other paths."""
  if not LDD19_RQ or _LDD19_RQ:
    return
  try:
    if torch.cuda.get_device_capability() != (9, 0):
      return
    import hashlib
    from torch.utils.cpp_extension import load_inline
    cases = "\n".join("    case %d: return rq_go<%s>(p, (const void*)x, (void*)out, (int)KS, (int)info, (cudaStream_t)stream);"
                      % (i, ", ".join(map(str, v))) for i, v in enumerate(LDD19_RQ_VARIANTS))
    src = LDD19_RQ_CU.replace("@@WGMMA@@", "".join(_rqwg(n) for n in sorted({v[0] for v in LDD19_RQ_VARIANTS})))
    src = src.replace("@@CASES@@", cases)
    cpp = "#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t v19_rqrun(" + ", ".join(["int64_t"] * 19) + ");\n"   # LDD19: big3: + pdl
    ext = _ldd19_inline("ldd19_rq_" + hashlib.sha256(src.encode()).hexdigest()[:12], cpp, cuda_sources=src,
                      functions=["v19_rqrun"],
                      extra_cuda_cflags=["-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a"],
                      no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
    _LDD19_RQ["fn"] = ext.v19_rqrun
  except Exception as ex:
    print("ldd19 rq disabled:", repr(ex)[:300], flush=True)


def _ldd19_rqcall(st, p, x, out, pdl=False):   # LDD19: big3: + pdl (programmatic dependent launch)
  return st["fn"](p["vi"], p["lay"].data_ptr(), p["NB"], p["KC"], x.data_ptr(), x.shape[0], p["K"], out.data_ptr(), p["n"],
                  p["esc"].data_ptr(), p["ecap"], p["cnt"].data_ptr(), p["cnt"].numel(), p["KS"], p["tab"],
                  st["sink"].data_ptr(), torch.cuda.current_stream(x.device).cuda_stream, 0, int(pdl))


def _ldd19_rq_verify(st, p, w, m):
  """sub_v22's v19_rq_verify at m exactly: a zero input over a NaN-filled output (the in-kernel zeroing), every K column
    as a one-hot row (bit-exact against W, counters back to 0), random rows (normal, then 1% outlier columns x30) with a
    16.0 spike row: overall rel L2 <= chenwainuo's per-KS bound (<= 8e-3, judge 1e-2), every row <= 1e-2, spike <= 2.5e-4."""
  n, K = w.shape
  dev = w.device
  out = torch.full((m, n), float('nan'), dtype=torch.bfloat16, device=dev)
  rc = _ldd19_rqcall(st, p, torch.zeros((m, K), dtype=torch.bfloat16, device=dev), out)
  if rc:
    return False, f'launch rc {rc}'
  if not bool((out == 0).all()):
    return False, 'zero input: output not zeroed'
  eye = torch.eye(K, dtype=torch.bfloat16, device=dev)
  got = torch.empty((K // m, m, n), dtype=torch.bfloat16, device=dev)
  for i in range(K // m):
    if _ldd19_rqcall(st, p, eye[i * m:(i + 1) * m], got[i]) != 0:
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
    if _ldd19_rqcall(st, p, x, o) != 0:
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


def _ldd19_rq_prepare(st):
  """LDD19: sub_v22's v19_prepare_rq_layer per matrix the flags use: the 12-bit pack and its independent unpack must
    reproduce W exactly; a candidate whose one-wave grid exceeds this GPU's SM count is skipped (the kernel needs every
    CTA resident); the first candidate passing _ldd19_rq_verify serves T = 128, else the other backends."""
  st["ldd19_rq"] = {}
  if "fn" not in _LDD19_RQ:
    return
  rq = {"fn": _LDD19_RQ["fn"], "sink": torch.zeros(4, dtype=torch.int32, device=st["qkv"].device), "plans": {}}
  nsm = torch.cuda.get_device_properties(st["qkv"].device).multi_processor_count
  m = 128
  for name, w in (("qkv", st["qkv"]), ("un", st["unpool"])):
    if not LDD19_RQ & LDD19_BIG_BITS[name]:
      continue
    try:
      w = w.contiguous()
      n, K = w.shape
      dev = w.device
      pk = _rqpack(w)
      if pk is None or not torch.equal(_rqunpack(*pk, n, K), w.view(torch.int16).int() & 0xffff):
        print(f"ldd19 rq {name}: 12-bit pack not exact", flush=True)
        continue
      lay, tab, esc6 = pk
      NB, KC = lay.shape[:2]
      for vi, KS in LDD19_RQ_ROUTES[name]:
        gx = NB * (2 if LDD19_RQ_VARIANTS[vi][6] & 16 else 1)
        if gx * KS > nsm:
          print(f"ldd19 rq {name}: grid {gx}x{KS} not one wave on {nsm} SMs", flush=True)
          continue
        esc, _ = _rqesc(esc6, NB, KC, KS, dev)
        nl = KS * NB * 8
        if (esc.numel() - 32) % nl:
          continue
        p = {'vi': vi, 'lay': lay, 'NB': NB, 'KC': KC, 'K': K, 'n': n, 'esc': esc, 'ecap': (esc.numel() - 32) // nl,
             'cnt': torch.zeros(2 * gx, dtype=torch.int32, device=dev), 'KS': KS, 'tab': int(tab), 'm': m}
        ok, msg = _ldd19_rq_verify(rq, p, w, m)
        if ok:
          rq["plans"][name] = p
          break
        print(f"ldd19 rq {name} rq{LDD19_RQ_VARIANTS[vi]} KS={KS}: {msg}", flush=True)
    except Exception as ex:
      print(f"ldd19 rq {name} disabled:", repr(ex)[:300], flush=True)
  torch.cuda.synchronize()
  st["ldd19_rq"] = rq


def _ldd19_rq_plan(st, name, x):
  """LDD19: big3: the rq plan serving this matrix and input, or None (shared by the launch and the PDL edge tests)."""
  rq = st.get("ldd19_rq")
  p = rq["plans"].get(name) if rq else None
  if p is None or x.shape[0] != p["m"] or x.shape[1] != p["K"] or x.data_ptr() % 16 or x.device != p["lay"].device:
    return None
  return p


def _ldd19_rq_run(st, name, x, pdl=False):   # LDD19: big3: + pdl
  rq = st.get("ldd19_rq")
  p = _ldd19_rq_plan(st, name, x)
  if p is None:
    return None
  out = torch.empty((p["m"], p["n"]), dtype=torch.bfloat16, device=x.device)
  rc = _ldd19_rqcall(rq, p, x, out, pdl)
  if rc:
    raise RuntimeError(f"rq launch failed: {rc}")
  return out



# LDD19: v20: o projection at T = 128 via ol_projection<128,128,4,3,1,1,1> + pack_olayer, adapted from
# NathenFernandes/muse-bf16-proj v18 (MIT License, see file header item (c)), extended by LDD19 with Hopper PDL edges
LDD19_OL = int(_ldd19_os.environ.get("LDD19_OL", "2"))
LDD19_OL_CU = r"""#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace {
#define OD __device__ __forceinline__
struct alignas(64) OlTmap { uint64_t w[16]; };
static_assert(sizeof(CUtensorMap) == sizeof(OlTmap), "CUtensorMap size");
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b, int ph) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b), "r"(ph) : "memory");
}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a & 0x3ffff) >> 4) | (uint64_t(64) << 32) | (uint64_t(1) << 62); }
OD void ol_tma(uint32_t d, const void* map, int row, int kb, uint32_t b, uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;"
               :: "r"(d), "l"(map), "r"(row), "r"(kb), "r"(b), "l"(p) : "memory");
}
OD void ol_mma128(float* a, uint64_t w, uint64_t x) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 0;"
               : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]),"+f"(a[16]),"+f"(a[17]),"+f"(a[18]),"+f"(a[19]),"+f"(a[20]),"+f"(a[21]),"+f"(a[22]),"+f"(a[23]),"+f"(a[24]),"+f"(a[25]),"+f"(a[26]),"+f"(a[27]),"+f"(a[28]),"+f"(a[29]),"+f"(a[30]),"+f"(a[31]),"+f"(a[32]),"+f"(a[33]),"+f"(a[34]),"+f"(a[35]),"+f"(a[36]),"+f"(a[37]),"+f"(a[38]),"+f"(a[39]),"+f"(a[40]),"+f"(a[41]),"+f"(a[42]),"+f"(a[43]),"+f"(a[44]),"+f"(a[45]),"+f"(a[46]),"+f"(a[47]),"+f"(a[48]),"+f"(a[49]),"+f"(a[50]),"+f"(a[51]),"+f"(a[52]),"+f"(a[53]),"+f"(a[54]),"+f"(a[55]),"+f"(a[56]),"+f"(a[57]),"+f"(a[58]),"+f"(a[59]),"+f"(a[60]),"+f"(a[61]),"+f"(a[62]),"+f"(a[63]) : "l"(w), "l"(x));
}
template<int BN, int BK, int ST, int PIPE, int PACK, int HINT, int TW>
__global__ __launch_bounds__(128 * TW + ((PIPE & 2) ? 128 : 0)) void ol_projection(
    const __grid_constant__ OlTmap tx, const uint8_t* w, __nv_bfloat16* out, int m, int pdl) {
  constexpr int T = BN * TW, R = 64, CT = 128 * TW;
  constexpr int AB = 64 * BK * 2, SB = (64 + T) * BK * 2, IT = 1024 / BK, TH = CT + ((PIPE & 2) ? 128 : 0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base = ol_addr(sm), full = base + ST * SB, empty = full + ST * 8;
  int tid = threadIdx.x, nt = blockIdx.x, mt = blockIdx.y;
  uint64_t pw, px;
  if constexpr (HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));
    px = pw;
  }
  auto load = [&](int i) {
    int s = i % ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full + s * 8), "r"(SB) : "memory");
    const uint8_t* src = w + ((long)nt * IT + i) * AB;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;"
                 :: "r"(base + s * SB), "l"(src), "r"(AB), "r"(full + s * 8), "l"(pw) : "memory");
    ol_tma(base + s * SB + AB, &tx, mt * T, i * (BK / 64), full + s * 8, px);
  };
  if (tid == ((PIPE & 2) ? CT : 0)) {
    #pragma unroll
    for (int s = 0; s < ST; s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full + s * 8) : "memory");
      if constexpr (PIPE & 2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty + s * 8), "n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    if (pdl & 1) {
      #pragma unroll
      for (int i = 0; i < ST && i < IT; i++) {
        const uint8_t* src = w + ((long)nt * IT + i) * AB;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full + i * 8), "r"(SB) : "memory");
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;"
                     :: "r"(base + i * SB), "l"(src), "r"(AB), "r"(full + i * 8), "l"(pw) : "memory");
      }
    } else {
      #pragma unroll
      for (int i = 0; i < ST && i < IT; i++) load(i);
    }
  }
  if (pdl & 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (tid == ((PIPE & 2) ? CT : 0)) {
      #pragma unroll
      for (int i = 0; i < ST && i < IT; i++) {
        ol_tma(base + i * SB + AB, &tx, mt * T, i * (BK / 64), full + i * 8, px);
      }
    }
  }
  __syncthreads();
  if ((PIPE & 2) && tid >= CT) {
    if (tid == CT) {
      #pragma unroll 1
      for (int i = ST; i < IT; i++) { ol_wait(empty + (i % ST) * 8, ((i / ST) - 1) & 1); load(i); }
    }
  } else {
    float a[BN / 2];
    #pragma unroll
    for (int j = 0; j < BN / 2; j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for (int i = 0; i < IT; i++) {
      int s = i % ST;
      ol_wait(full + s * 8, (i / ST) & 1);
      #pragma unroll
      for (int j = 0; j < BN / 2; j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for (int kk = 0; kk < BK / 16; kk++) {
        uint64_t aw = ol_desc(base + s * SB + (kk / 4) * 8192 + (kk % 4) * 32);
        uint64_t ax = ol_desc(base + s * SB + AB + (kk / 4) * T * 128 + (tid / 128) * BN * 128 + (kk % 4) * 32);
        ol_mma128(a, aw, ax);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr (PIPE & 1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for (int j = 0; j < BN / 2; j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done = i - ((PIPE & 1) ? 1 : 0);
      if (done >= 0 && done + ST < IT && tid % 128 == 0) {
        if constexpr (PIPE & 2) ol_arrive(empty + (done % ST) * 8);
        else load(done + ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for (int j = 0; j < BN / 2; j++) asm volatile("" : "+f"(a[j]) :: "memory");
    asm volatile("bar.sync 1,%0;" :: "n"(CT) : "memory");
    #pragma unroll
    for (int j = 0; j < BN / 8; j++) {
      __nv_bfloat162 lo = __floats2bfloat162_rn(a[4 * j], a[4 * j + 1]);
      __nv_bfloat162 hi = __floats2bfloat162_rn(a[4 * j + 2], a[4 * j + 3]);
      int rb = ((tid % 128) / 32) * 16 + ((tid % 16) / 8) * 8, cb = (tid / 128) * BN + 8 * j + (tid % 8);
      uint32_t d = base + 2 * (cb * 72 + rb);
      asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0],{%1,%2};"
                   :: "r"(d), "r"(*(uint32_t*)&lo), "r"(*(uint32_t*)&hi) : "memory");
    }
  }
  __syncthreads();
  if ((pdl & 2) && tid == 0) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  for (int q = tid; q < T * (R / 8); q += TH) {
    int token = q / (R / 8), row = (q % (R / 8)) * 8;
    if (mt * T + token < m) *(uint4*)(out + (long)(mt * T + token) * 8192 + nt * R + row) = *(uint4*)((__nv_bfloat16*)sm + token * 72 + row);
  }
}
typedef CUresult (*OlEnc)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                          const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static int ol_map_x(OlTmap* tm, const void* ptr, int rows, int box, int bk) {
  static OlEnc encode = nullptr;
  if (!encode) {
    void* fn = nullptr;
#if CUDART_VERSION >= 12050
    cudaDriverEntryPointQueryResult q;
    auto rc = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q);
#else
    cudaDriverEntryPointQueryResult q;
    auto rc = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q);
#endif
    if (rc != cudaSuccess || !fn) { cudaGetLastError(); return -100; }
    encode = (OlEnc)fn;
  }
  if (((uintptr_t)ptr) % 16) return -101;
  cuuint64_t dims[3] = {64, (cuuint64_t)rows, 16};
  cuuint64_t stride[2] = {2048, 128};
  cuuint32_t b[3] = {64, (cuuint32_t)box, (cuuint32_t)(bk / 64)}, es[3] = {1, 1, 1};
  return encode((CUtensorMap*)tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)ptr, dims, stride, b, es,
                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS ? 0 : -102;
}
}
int64_t ldd19_ol_run(int64_t x, int64_t w, int64_t out, int64_t m, int64_t stream, int64_t query, int64_t pdl) {
  constexpr int BN = 128, BK = 128, ST = 4, PIPE = 3, PACK = 1, HINT = 1, TW = 1;
  constexpr int T = BN * TW, R = 64;
  constexpr int MAIN = ST * ((64 + T) * BK * 2 + 16), EPI = T * 72 * 2, SMEM = MAIN > EPI ? MAIN : EPI;
  auto fn = ol_projection<BN, BK, ST, PIPE, PACK, HINT, TW>;
  static bool configured = false;
  if (!configured) {
    int rc = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    if (rc) return rc;
    configured = true;
  }
  if (query) {
    cudaFuncAttributes a;
    int rc = (int)cudaFuncGetAttributes(&a, fn);
    return rc ? -rc : a.numRegs * 10000 + (int)a.localSizeBytes;
  }
  OlTmap tx;
  int rc = ol_map_x(&tx, (const void*)x, (int)m, T, BK);
  if (rc) return rc;
  dim3 grid(8192 / R, ((int)m + T - 1) / T);
  dim3 block(128 * TW + ((PIPE & 2) ? 128 : 0));
  cudaStream_t st = (cudaStream_t)stream;
  if (pdl) {
    cudaLaunchConfig_t lc = {};
    lc.gridDim = grid;
    lc.blockDim = block;
    lc.dynamicSmemBytes = SMEM;
    lc.stream = st;
    cudaLaunchAttribute at_[1];
    at_[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at_[0].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at_;
    lc.numAttrs = 1;
    cudaLaunchKernelEx(&lc, fn, tx, (const uint8_t*)w, (__nv_bfloat16*)out, (int)m, (int)pdl);
  } else {
    fn<<<grid, block, SMEM, st>>>(tx, (const uint8_t*)w, (__nv_bfloat16*)out, (int)m, 0);
  }
  return (int64_t)cudaGetLastError();
}
"""
_LDD19_OL = {}


def _ldd19_ol_pack(weight):
  cols = torch.arange(64, device=weight.device)[None, :] ^ ((torch.arange(64, device=weight.device)[:, None] & 7) * 8)
  tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
  return tiled.gather(-1, cols.expand(128, 16, 64, 64)).contiguous()


def _ldd19_ol_build():
  """LDD19: v20: compile ol_projection<128,128,4,3,1,1,1> (sm_90a only)."""
  if not LDD19_OL or "fn" in _LDD19_OL:
    return
  try:
    if torch.cuda.get_device_capability() != (9, 0):
      return
    import hashlib
    from torch.utils.cpp_extension import load_inline
    cpp = "#include <pybind11/pybind11.h>\n#include <stdint.h>\nint64_t ldd19_ol_run(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n"
    name = "ldd19_ol_" + hashlib.sha256(LDD19_OL_CU.encode()).hexdigest()[:12]
    ext = _ldd19_inline(name, cpp, cuda_sources=LDD19_OL_CU, functions=["ldd19_ol_run"],
                      extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"],
                      no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
    _LDD19_OL["fn"] = ext.ldd19_ol_run
  except Exception as ex:
    print("ldd19 ol disabled:", repr(ex)[:300], flush=True)


def _ldd19_ol_prepare(st):
  """LDD19: v20: pack st['o'] and verify m = 128 against FP64 reference (plain and PDL paths)."""
  st["ldd19_ol"] = {}
  fn = _LDD19_OL.get("fn")
  w = st.get("o")
  if fn is None or not (LDD19_OL & 2) or w is None or w.shape != (8192, 1024) or w.dtype != torch.bfloat16:
    return
  try:
    wp = _ldd19_ol_pack(w.contiguous())
    dev = w.device
    gen = torch.Generator(device=dev).manual_seed(2026)
    x = torch.randn((128, 1024), generator=gen, device=dev, dtype=torch.bfloat16)
    ref = (x.double() @ w.double().t()).to(torch.bfloat16)
    stream = torch.cuda.current_stream(dev).cuda_stream
    for pdl in (0, 1, 2, 3):
      out = torch.full((128, 8192), float("nan"), device=dev, dtype=torch.bfloat16)
      rc = fn(x.data_ptr(), wp.data_ptr(), out.data_ptr(), 128, stream, 0, pdl)
      torch.cuda.synchronize()
      if rc or not bool(torch.isfinite(out).all()):
        print(f"ldd19 ol verify failed (pdl={pdl}, rc={rc})", flush=True)
        return
      err = float((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))
      if err > 2e-3:
        print(f"ldd19 ol verify rel L2 {err:.3e} > 2e-3", flush=True)
        return
    st["ldd19_ol"] = {"fn": fn, "w": wp}
  except Exception as ex:
    print("ldd19 ol prepare disabled:", repr(ex)[:300], flush=True)


def _ldd19_ol_ok(st, x):
  ol = st.get("ldd19_ol")
  return (bool(ol) and x.shape == (128, 1024) and x.dtype == torch.bfloat16
          and x.is_contiguous() and x.data_ptr() % 16 == 0 and x.device == ol["w"].device)


def _ldd19_ol_run(st, x, pdl=0):
  if not _ldd19_ol_ok(st, x):
    return None
  ol = st["ldd19_ol"]
  out = torch.empty((128, 8192), dtype=torch.bfloat16, device=x.device)
  rc = ol["fn"](x.data_ptr(), ol["w"].data_ptr(), out.data_ptr(), 128,
                torch.cuda.current_stream(x.device).cuda_stream, 0, int(pdl))
  if rc:
    raise RuntimeError(f"ldd19 ol launch failed: {rc}")
  return out


def _graph_time(fn, flush, reps=9):
  fn()
  torch.cuda.synchronize()
  g = torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
    fn()
  ts = []
  for _ in range(reps):
    _FLUSH_SINK.copy_(flush.sum())
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    ts.append(a.elapsed_time(b) * 1000.0)
  ts.sort()
  return ts[len(ts) // 2]


_FLUSH_SINK = None


def _calib_bias(st, dev, g, log=False):
  router = st["pr"][D_EXPERT:].float()
  router = router[:N_EXPERTS] + router[N_EXPERTS:]
  gc = torch.Generator(device=dev).manual_seed(987654321)
  h = torch.randn(512, HIDDEN, generator=gc, device=dev)
  h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
  sc = torch.sigmoid(h.double() @ router.double().T * 0.1767766952966369).float()
  idx = torch.arange(LOCAL_EXPERTS, device=dev).float()

  def stats(shape, delta):
    sel = sc + st["bias"]
    sel[:, :LOCAL_EXPERTS] += delta * shape
    ids = sel.topk(TOP_K, dim=-1).indices
    loc = ids < LOCAL_EXPERTS
    rpt = float(loc.float().sum(-1).mean())
    counts = [torch.unique(ids[i:i + 32][loc[i:i + 32]]).numel() for i in range(0, 512, 32)]
    d32 = sum(counts) / len(counts)
    return rpt, d32

  best = None
  for width in (2.0, 3.0, 4.5, 6.0, 9.0):
    shape = torch.exp(-idx / width)
    lo, hi = 0.0, 3.0
    for _ in range(14):
      mid = 0.5 * (lo + hi)
      if stats(shape, mid)[0] < 1.6:
        lo = mid
      else:
        hi = mid
    rpt, d32 = stats(shape, hi)
    if best is None or abs(d32 - 15.7) < abs(best[3] - 15.7):
      best = (shape, hi, rpt, d32)
  st["bias"][:LOCAL_EXPERTS] += best[1] * best[0]
  if log:
    print(f"calib delta {best[1]:.3f} routes/token {best[2]:.2f} experts@32 {best[3]:.1f}", flush=True)


def _tune(st, log=False):
  dev = st["qkv"].device
  global _FLUSH_SINK
  flush = torch.zeros(int(60e6), dtype=torch.float32, device=dev)
  _FLUSH_SINK = torch.zeros((), dtype=torch.float32, device=dev)
  g = torch.Generator(device=dev).manual_seed(12345)
  bias0 = st["bias"].clone()
  slim = LDD19_SLIM_TUNE and LDD19_LOCK and st.get("ma_ok") and st.get("bfp") is not None   # LDD19: iter2 slim
  if not slim:
    _calib_bias(st, dev, g, log)
  st["moe_pick"] = {}
  st["gluev2"] = {}
  st["marlin_cfg"] = {}
  st["mm_pick"] = {}
  pb_ok = {}
  for T in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    if slim and T <= 32:   # LDD19: iter2 slim: every pick at T <= 32 is set by _ldd19_lock
      continue
    data = {"stream": torch.randn(T, HIDDEN, generator=g, device=dev),
            "local_start": torch.zeros(1, dtype=torch.int32, device=dev)}
    o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    if T <= 4:
      want = max(2, int(round(2.57 * T / 1.0)) if T == 1 else 2 * T)
      for _ in range(40):
        _o, _s, _ids = _after_attention(data, st, o_in, None, T)
        if int(((_ids >= 0) & (_ids < LOCAL_EXPERTS)).sum()) >= min(want, 8 * T):
          break
        data["stream"] = torch.randn(T, HIDDEN, generator=g, device=dev)
        o_in = (torch.randn(T, 1024, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    best, bestt, bestv2 = None, 1e30, True
    times, dfltv2 = {}, False
    msg = []
    nloc = ndis = -1
    for name, w, K in (("qkv", st["qkv"], HIDDEN), ("o", st["o"], 1024), ("pr", st["pr"], HIDDEN),
                       ("un", st["unpool_slice"] if T <= 8 else st["unpool"], D_EXPERT)):
      if slim and (T, name) in PIN_PICK:   # LDD19: iter2 slim: pinned after the tuner
        continue
      xd = (torch.randn(T, K, generator=g, device=dev) * 0.1).to(torch.bfloat16)
      if st.get("bfp") is not None:
        st["mm_pick"][(T, name)] = "bfp"
        try:
          if name == "pr":
            t_bfp = _graph_time(lambda: (_bfp_gemm(st, "pool", xd, T, D_EXPERT, HIDDEN),
                                         _bfp_gemm(st, "router", xd, T, N_EXPERTS, HIDDEN)),
                                flush, reps=5)
          else:
            nn = w.shape[0]
            t_bfp = _graph_time(lambda: _bfp_gemm(st, name, xd, T, nn, K), flush, reps=5)
        except Exception:
          t_bfp = 1e30
      else:
        t_bfp = 1e30
      st["mm_pick"][(T, name)] = False
      try:
        t_tri = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
      except Exception:
        t_tri = 1e30
      st["mm_pick"][(T, name)] = True
      try:
        t_mm = _graph_time(lambda: dense(xd, w, T, st, name), flush, reps=5)
      except Exception:
        t_mm = 1e30
      t_bfpf = 1e30
      if name == "pr" and T <= 32 and st.get("bfp") is not None and "bfp_fused_pooler_router" in globals():
        try:
          t_bfpf = _graph_time(lambda: bfp_fused_pooler_router(xd, st["bfp"]), flush, reps=5)
        except Exception:
          t_bfpf = 1e30
      opts = {False: t_tri, True: t_mm, "bfp": t_bfp, "bfpf": t_bfpf}
      bestp = min(opts, key=lambda p: opts[p])
      st["mm_pick"][(T, name)] = bestp
      if log:
        msg.append(f"{name}{ {False: 'T', True: 'M', 'bfp': 'B', 'bfpf': 'F'}[bestp] }:{opts[bestp]:.0f}")
    cands = [c for c in ("marlin", "agent") if (c != "marlin" or st["bank"] is not None)
             and (c != "agent" or st["ma_ok"]) and not slim]   # LDD19: iter2 slim: MoE picks locked
    for moe in cands:
      if moe == "marlin" and st["bank"] is None:
        continue
      if moe == "agent" and not st["ma_ok"]:
        continue
      st["moe_pick"][T] = moe
      for v2 in (True, False):
        st["gluev2"][T] = v2
        try:
          t = _graph_time(lambda: _after_attention(data, st, o_in, None, T), flush)
        except Exception as ex:
          if log:
            msg.append(f"{moe[:3]}{2 if v2 else 1}:X{repr(ex)[:40]}")
          continue
        if log:
          msg.append(f"{moe[:3]}{2 if v2 else 1}:{t:.0f}")
          if nloc < 0:
            _o, _s, _ids = _after_attention(data, st, o_in, None, T)
            loc = _ids[(_ids >= 0) & (_ids < LOCAL_EXPERTS)]
            nloc, ndis = int(loc.numel()), int(torch.unique(loc).numel())
        if t < times.get(moe, 1e30):
          times[moe] = t
          if moe == DEFAULT_MOE:
            dfltv2 = v2
        if t < bestt:
          best, bestt, bestv2 = moe, t, v2
    if (DEFAULT_MOE in times and best is not None and best != DEFAULT_MOE
            and times[best] > TUNE_MARGIN * times[DEFAULT_MOE]):
      best, bestv2 = DEFAULT_MOE, dfltv2
    st["moe_pick"][T] = best if best is not None else DEFAULT_MOE
    st["gluev2"][T] = bestv2
    tb = 1e30
    if log:
      print(f"T{T}/{nloc}r{ndis}e " + " ".join(msg) + (f" pb:{tb:.0f}" if tb < 1e9 else ""), flush=True)
    pb_ok[T] = tb < bestt
  pbmax = 0
  for T in (1, 2, 4, 8, 16, 32):
    if pb_ok.get(T):
      pbmax = T
    else:
      break
  st["bias"].copy_(bias0)
  st["pbmax"] = pbmax if FORCE_PBMAX is None else FORCE_PBMAX
  if log:
    print("tune: pbmax", pbmax, "moe", st["moe_pick"], "gluev2", st["gluev2"], flush=True)
  del flush


def prepare(weights):
  dev = weights["qkv_gate"].device
  router = weights["router"].float()
  hi = router.to(torch.bfloat16)
  lo = (router - hi.float()).to(torch.bfloat16)
  st = {}
  st["qkv"] = weights["qkv_gate"].contiguous()
  st["o"] = weights["o"].contiguous()
  st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
  st["unpool"] = weights["unpooler"].contiguous()
  st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
  st["gate1"] = weights["attn_residual_gate"].contiguous()
  st["gate2"] = weights["ffn_residual_gate"].contiguous()
  for _gk in ("gate1", "gate2"):   # LDD19: GT128: _gate_ab's math, once
    _gw = st[_gk].float() / 0.3
    _gb = torch.sigmoid(_gw)
    st[_gk + "b"] = _gb.contiguous()
    st[_gk + "a"] = torch.sqrt(torch.clamp(torch.sigmoid(-_gw) * (1.0 + _gb), min=1e-3)).contiguous()
  st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
  st["g_preexp"] = weights["pre_expert_norm"].contiguous()
  st["g_postexp"] = weights["post_expert_norm"].contiguous()
  st["g_postffn"] = weights["post_ffn_norm"].contiguous()
  st["g_next"] = weights["next_pre_attn_norm"].contiguous()
  st["bias"] = weights["router_bias"].float().contiguous()
  st["w13q"] = _repack_q(weights["w13"].view(torch.uint8), True)
  st["w13s"] = _repack_s(weights["w13_sf"].view(torch.uint8), True)
  st["gs13"] = weights["w13_gs"].float().contiguous()
  st["w2q"] = _repack_q(weights["w2"].view(torch.uint8), False)
  st["w2s"] = _repack_s(weights["w2_sf"].view(torch.uint8), False)
  st["gs2"] = weights["w2_gs"].float().contiguous()
  st["invf0"] = torch.zeros(32, device=dev)
  st["invf"] = (1.0 / (500000.0 ** (torch.arange(0, 64, 2, device=dev).float() / 64))).contiguous()
  import threading
  errs = []

  def _build(fn):
    try:
      fn()
    except Exception as ex:
      errs.append(ex)

  global _MA_SFX, _MA_SC6
  _MA_SFX = _MA_SFX_IF_PRMT if (_MA_PRMT_SCALES and _ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
  _MA_SC6 = bool(LDD19_SC6 and LDD19_KPERM and _MA_SFX and _sc6_ok(weights["w13_sf"], weights["w2_sf"]))
  ths = [threading.Thread(target=_build, args=(f,)) for f in (_att7, _moe, lambda: _moe(32), lambda: _moe(128), lambda: _moe(256), _moe_tiny, _moe_tiny2, _moe_mid,
                                                              bfp__get_ext, _ldd19_glue, _ldd19_lt_build,
                                                              _ldd19_rq_build, _ldd19_ol_build)]   # LDD19: v20: parallel bfp + dsm_b4 + ol  # ~85 s of nvcc, in parallel (LDD19: + the glue, + cgemv, + lt128's cuBLASLt / rq)
  for th in ths:
    th.start()
  for th in ths:
    th.join()
  if errs:
    raise errs[0]
  st["ldd19_ok"] = _LDD19_EXT is not None and _ldd19_wts_ok(st)   # LDD19: the CUDA glue kernels may run
  _ma_prepare(weights, st)
  _bfp_prepare(weights, st)
  _ldd19_lt_prepare(st)   # LDD19: lt128 pinned cuBLASLt plans (version-gated, self-checked)
  _ldd19_rq_prepare(st)   # LDD19: lt128 rq 12-bit packs (exactness-checked, verified at m = 128)
  _ldd19_ol_prepare(st)   # LDD19: v20 ol_projection pack + m = 128 verification
  st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
  st["fa3"] = None
  st["bank"] = None
  st["att_ok"] = True
  st["ma_ok"] = True
  st["pbmax"] = 2
  st["moe_pick"] = {}
  try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    st["fa3"] = flash_attn_with_kvcache
    st["bank"] = None   # LDD19: no SGLang Marlin bank (every T runs the agent MoE)
  except Exception as ex:
    print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
  try:
    _tune(st, log=_TUNE_LOG)
  except Exception as ex:
    print("tune failed:", repr(ex)[:200], flush=True)
    st["pbmax"] = 2
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
  for k, v in PIN_PICK.items():
    st.setdefault("mm_pick", {})[k] = v
  st["ldd19_tuned"] = {k: dict(st.get(k, {})) for k in ("mm_pick", "gluev2", "moe_pick")}   # LDD19: the tuner's own picks
  if LDD19_LOCK:
    _ldd19_lock(st)
  torch.cuda.synchronize()
  if st.get("att_ok"):
    _attn_selftest(st)
  _check_guards()
  return st


def _attn_cfg(B, L, sliding):
  BM = 16 if L == 1 else 64
  if sliding:
    nsplit = max(1, min(32, 264 // B))
  else:
    nsplit = max(1, min(132, 528 // B))
  return BM, nsplit


_INV_FREQ7 = {}


def _inv_freq7(theta, device):
  key = (float(theta), str(device))
  t = _INV_FREQ7.get(key)
  if t is None:
    if theta:
      t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
    else:
      t = torch.zeros(32, device=device)
    _INV_FREQ7[key] = t.float().contiguous()
  return t


def _launch7(data, dbg=None, _mod=None, pdl=None, **override):
  cfg = data["config"]
  B, L = cfg["batch"], cfg["q_len"]
  hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
  qkv = data["qkv"]
  device = qkv.device
  pairs = hkv // 2
  window = cfg["window"]
  page_table = data["page_table"]
  out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
  plan = _plan7(B, pairs, L, window, page_table.shape[1], P, device)
  tuned = _TUNED7.get((B, L, bool(window), pairs))
  if tuned is None and (LDD19_ATT & 4) and (B, L, bool(window), pairs) == (16, 1, True, 1):   # LDD19: v20: d16s 8 splits
    tuned = dict(n_splits=8, nc=1)
  if tuned and not override:
    override = tuned
  if override:
    plan = dict(plan, **override)
    if "cs" in override and override["cs"] > 1:
      plan["flat"] = 0
      if "ncl" not in override:
        plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
      plan["n_splits"] = plan["cs"] * plan["ncl"]
      if "nc" not in override:
        plan["nc"] = max(1, min(plan["ncl"], 4))
    elif "n_splits" in override and "nc" not in override:
      plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
  n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
  if flat:
    coop = int(plan.get("coop", int(OKC_FLATCOOP and L == 8 and
                                    n_splits * pairs <= 2 * _num_sms7(device))))
  else:
    coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                        int(n_splits * B * pairs <= 2 * _num_sms7(device))))
  w0 = int(plan.get("w0", (OKC_W0_L1 if L == 1 else OKC_W0_L8) if (flat and (coop or L == 1)) else
                    (OKC_W0_PR if (coop and not flat and B * pairs == 1) else 1024)))
  use_pdl = OKC_PDL if pdl is None else bool(pdl)
  hs_on = bool(LDD19_ATT & 2) and L == 8 and bool(window) and not flat and cs <= 1 and B == 1   # LDD19: v20: b1split
  if hs_on and (LDD19_ATT & 8):
    use_pdl = False
  etx_on = (L == 8 and bool(window) and not flat and cs <= 1
            and ((bool(LDD19_ATT & 1) and 16 <= B * pairs < 32)
                 or (bool(LDD19_ATT & 16) and B * pairs >= 32)))   # LDD19: v20: etx2 (1=b16+PDL, 16=b32+)
  tma = (int(plan.get("tma", 0)) | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (int(use_pdl) << 16)
         | (int(etx_on) << 17) | (int(hs_on) << 18))
  if flat or cs <= 1:
    cs, ncl = 0, 1
  ws_acc, ws_l, ws_cnt = _workspace7(B * pairs * nc * L * 16, device)
  n_ctas = n_splits * (1 if flat else B) * pairs
  scratch, bar = _cluster_bufs7(n_ctas if cs > 1 else 1, L, device)
  scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
  (_mod or _att7()).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                data["k_scale"], data["v_scale"], _inv_freq7(cfg["rope_theta"], device), out, ws_acc, ws_l,
                ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                dbg, cs, ncl, scratch, bar, tma)
  return out


def _supported7(cfg):
  P = cfg["page_size"]
  return (cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
          and cfg["q_len"] in (1, 8) and cfg["batch"] <= 512 and P > 0 and (P & (P - 1)) == 0
          and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)

ATT_HAS_PDL = True


def _decode_attention7(data, state, qkv, ldd19_pdl=False):   # LDD19: big3: + ldd19_pdl (edge 1 above 32 tokens)
  cfg = data["config"]
  pdl = {"pdl": _pdl(cfg["batch"] * cfg["q_len"]) or ldd19_pdl} if ATT_HAS_PDL else {}   # their launcher's PDL switch (LDD19: big3: + edge 1)
  out = _launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                  "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                  "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
  return out.view(cfg["batch"] * cfg["q_len"], 1024)


_GUARDS = []
_GUARDED_IDS = set()


def _guarded(t):
  nbytes = t.numel() * t.element_size()
  words = (nbytes + 3) // 4
  full = torch.full((words + SCRATCH_GUARD,), SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
  full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
  v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
  _GUARDS.append((full, words))
  _GUARDED_IDS.add(id(v))
  return v


def _check_guards():
  torch.cuda.synchronize()
  for full, words in _GUARDS:
    bad = int((full[words:] != SCRATCH_SENTINEL).sum())
    if bad:
      raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


_workspace7_raw, _cluster_bufs7_raw = _workspace7, _cluster_bufs7


def _workspace7(n_rows, device):
  ws = _workspace7_raw(n_rows, device)
  if id(ws[0]) not in _GUARDED_IDS:
    ws = tuple(_guarded(t) for t in ws)
    _WS7[str(device)] = ws
  return ws


def _cluster_bufs7(n_ctas, L, device):
  cb = _cluster_bufs7_raw(n_ctas, L, device)
  if id(cb[0]) not in _GUARDED_IDS:
    cb = tuple(_guarded(t) for t in cb)
    _CL7[str(device)] = cb
  return cb


def _attn_selftest(st):
  dev = torch.device("cuda")
  g = torch.Generator(device=dev).manual_seed(4242)
  nq = st["qkv"].shape[0]
  for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                               (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                               (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                               (2048, 8, 1, 3000, 1), (2048, 8, 16, 2500, 1), (2048, 8, 32, 2500, 1),
                               (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):   # LDD19: v20: + sliding q8 (hs/etx)
    cfg = dict(batch=B, q_len=L, num_q_heads=16, num_kv_heads=2, head_dim=64, page_size=P, window=window,
               rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
    if not _supported7(cfg):
      continue
    seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    per = (ctx + L + P - 1) // P
    pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
    pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
    loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
    kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
    vc = torch.zeros_like(kc)
    qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    one = torch.ones((), dtype=torch.float32, device=dev)
    _decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                        "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def _att_fits7(data, cfg):
  kc, vc = data["k_cache"], data["v_cache"]
  rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
  return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
          and data["page_table"].numel() < (1 << 31)
          and cfg["batch"] * cfg["q_len"] * 2304 < (1 << 31)
          and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def _kernel_fn_fa3(data, state):
  st2 = dict(state)
  st2["att_ok"] = False
  return kernel_fn(data, st2)


def kernel_fn(data, state):
  cfg = data["config"]
  B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
  agent_attn = (state["att_ok"] and _supported7(cfg) and _att_fits7(data, cfg)
               )
  if agent_attn:
    qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
    qkv = _ldd19_big(state, "qkv", data["attn_input"], T, state["qkv"]) if T > BFP_MAX_T else None   # LDD19: lt128
    if qkv is not None:
      pass
    elif qpick == "bfp" and T <= BFP_MAX_T and state.get("bfp") is not None:
      qkv = _bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], HIDDEN)
    elif qpick is True or qpick == "bfp":
      qkv = torch.mm(data["attn_input"], state["qkv"].t())
    else:
      Pq, Sq = dense(data["attn_input"], state["qkv"], T)
      qkv = Pq[0] if Sq == 1 else Pq.sum(0)
      qkv = qkv.to(torch.bfloat16)
    try:
      o_in = _decode_attention7(data, state, qkv, _ldd19_edge(1, T) and bool(LDD19_RQ & 1)
                                and _ldd19_rq_plan(state, "qkv", data["attn_input"]) is not None)   # LDD19: big3
    except RuntimeError:
      return _kernel_fn_fa3(data, state)
    return _after_attention(data, state, o_in, None, T)
  dev = data["attn_input"].device
  x = data["attn_input"]
  sliding = cfg["window"] > 0
  use_fa3 = state["fa3"] is not None
  P, S = dense(x, state["qkv"], T)
  q = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
  q8 = torch.empty((T, 1024), device=dev, dtype=torch.uint8)
  gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
  kc = data["k_cache"].view(torch.uint8)
  vc = data["v_cache"].view(torch.uint8)
  _qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                 data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                 ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
  if use_fa3:
    descale = {}
    if cfg["kv_scale"] != 1.0:
      descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                     v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
    window = cfg["window"]
    out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                       v_cache=data["v_cache"], page_table=data["page_table"],
                       cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                       max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                       window_size=(window - 1, 0) if window else (-1, -1),
                       scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
      out = out[0]
    o_in = out.reshape(T, 1024)
  else:
    BM, NS = _attn_cfg(B, L, sliding)
    acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
    ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
    pt = data["page_table"]
    _attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                         pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                         L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
    o_in = torch.empty((T, 1024), device=dev, dtype=torch.bfloat16)
    _attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                           SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
  return _after_attention(data, state, o_in, gate, T)


def _after_attention(data, state, o_in, gate, T):
  dev = o_in.device
  f32 = dict(device=dev, dtype=torch.float32)
  gv = state.get("gluev2", {})
  v2 = gv.get(T, gv.get(_bucket(T), False)) if FORCE_GLUEV2 is None else FORCE_GLUEV2
  s1 = torch.empty((T, HIDDEN), **f32)
  h = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
  if v2:
    a_out = torch.empty((T, HIDDEN), **f32)
    sums = torch.zeros((T, 4), **f32)
    BN, BT, BK = 64, (16 if T <= 16 else 32), 128
    _gemm_o[(HIDDEN // BN, triton.cdiv(T, BT))](
        o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, HIDDEN,
        K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
    BC1 = 2048
    _bnd1_apply[(T, HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                    s1, h, N=HIDDEN, BC=BC1, num_warps=8)
  else:
    if gate is None:
      P, S = dense(o_in, state["o"], T, state, "o")
    else:
      P, S = dense_ng(o_in, gate, state["o"], T, state)
    if (state.get("ldd19_ok") and _ldd19_on(T, P, data["stream"]) and P.dtype in (torch.float32, torch.bfloat16)
            and data["stream"].dtype == torch.float32):
        # LDD19: the same boundary as _boundary1, one cluster of LDD19_CL CTAs per token
      _LDD19_EXT.ldd19_bnd_one(int(LDD19_BND_ONE))   # LDD19: host flag read at launch (graph-safe)
      _LDD19_EXT.ldd19_bnd(P, S, T, HIDDEN, data["stream"], state["gate1"], state["gate1"], 0, state["g_preffn"],
                           s1, h, LDD19_CL, int(_pdl(T)))
    else:
      b1p = _pdl(T) or (_ldd19_edge(32, T) and gate is None and bool(LDD19_OL & 2)
                        and _ldd19_ol_ok(state, o_in))   # LDD19: v20: edge 32 (o ran on ol_projection)
      _gt = bool(LDD19_GT128) and T == 128   # LDD19: GT128
      _boundary1[(T,)](P, S, T, data["stream"], state["gate1a"] if _gt else state["gate1"], state["gate1b"] if _gt else state["gate1"], state["g_preffn"], s1, h,
                       num_warps=LDD19_MP_BW1.get(T, 16), USE_PDL=b1p, launch_pdl=b1p,   # LDD19 (MP/BW1)
                       DELAY=T > LDD19_GLUE_MAX_T and bool(LDD19_BGD & (1 if T <= 128 else 2)), GT=_gt)   # LDD19: big2, GT128
  mmp = state.get("mm_pick", {})
  ppick = mmp.get((T, "pr"), mmp.get((_bucket(T), "pr"), False))
  if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
    _bfp_pdl(T)
    P, S = bfp_fused_pooler_router(h, state["bfp"]), -1
  elif ppick == "bfp" and state.get("bfp") is not None and not _ldd19_pr_lt(state, T):   # LDD19: big2: Lt pr first
    pooled = _bfp_gemm(state, "pool", h, T, D_EXPERT, HIDDEN)
    logits = _bfp_gemm(state, "router", h, T, N_EXPERTS, HIDDEN)
    P, S = (pooled, logits), -1
  else:
    P, S = dense(h, state["pr"], T, state, "pr")
  ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  wts = torch.empty((T, 8), **f32)
  moe = state["moe_pick"].get(T, state["moe_pick"].get(_bucket(T), "agent"))
  if moe == "marlin" and state["bank"] is None:
    moe = "agent"
  if moe != "marlin" and not state["ma_ok"]:
    moe = "marlin"
  if moe == "marlin" and state["bank"] is None:
    moe = "agent"
  use_marlin = moe == "marlin"
  use_ma = moe == "agent"
  if not (use_marlin or use_ma):
    raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
  xm = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
  rt32 = False   # LDD19: big3
  if S == -1:
    pooled, logits = P
    if (LDD19_ROUTE and state.get("ldd19_ok") and (use_marlin or use_ma) and _ldd19_on(T, pooled, logits)
            and pooled.dtype == torch.bfloat16 and logits.dtype == torch.float32
            and data["local_start"].dtype == torch.int32):
        # LDD19: _route_split's top-8 and pre-expert norm, the top-8 in its own warp
      _LDD19_EXT.ldd19_route(pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"],
                             lids, T, int(_pdl(T)))
    else:
      _route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
  elif v2:
    pool = torch.empty((T, D_EXPERT), **f32)
    psum = torch.zeros((T,), **f32)
    BCR = 1024
    _route_a[(T, D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                   BC=BCR, num_warps=8)
    _route_b[(T, D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                   PERMUTED=not (use_marlin or use_ma), num_warps=8)
  elif (LDD19_RT32 and T > LDD19_GLUE_MAX_T and S == 1 and state.get("ldd19_ok") and (use_marlin or use_ma)
        and P.dtype == torch.float32 and P.is_contiguous() and P.shape[-1] == 4608 and data["local_start"].dtype == torch.int32):
      # LDD19: big2: _route on the combined FP32 output as CUDA (route_kernel's top-8 warp, 256 norm threads)
    _LDD19_EXT.ldd19_route32(P, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids, T)
    rt32 = True   # LDD19: big3: route32 triggers its dependents at entry
  else:
    _route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                 PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=_pdl(T), launch_pdl=_pdl(T))
  if use_ma:
    m = torch.empty((T, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext = _moe(T)
    if MOE_HAS_PDL:
      ext.okc_set_pdl(1 if _pdl(T) or (rt32 and _ldd19_edge(2, T)) else 0)   # LDD19: big3: + edge 2
    for m0 in range(0, T, _MA_MMAX):
      m1 = min(T, m0 + _MA_MMAX)
      ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                  state["nsm"])
  wu = state["unpool_slice"] if T <= 8 else state["unpool"]
  P, S = dense(m, wu, T, state, "un")
  nu = wu.shape[0]
  s2 = torch.empty((T, HIDDEN), **f32)
  out = torch.empty((T, HIDDEN), device=dev, dtype=torch.bfloat16)
  if v2:
    fbuf = torch.zeros((T, HIDDEN), **f32)
    sums2 = torch.zeros((T, 4), **f32)
    BC2 = 1024
    _bnd2_sums[(T, HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                   BC=BC2, num_warps=8)
    _bnd2_apply[(T, HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                    N=HIDDEN, BC=BC2, num_warps=8)
  else:
    if state.get("ldd19_ok") and _ldd19_on(T, P, s1) and P.dtype in (torch.float32, torch.bfloat16) and nu % 4 == 0:
        # LDD19: the same boundary as _boundary2, one cluster of LDD19_CL CTAs per token
      _LDD19_EXT.ldd19_bnd_one(int(LDD19_BND_ONE))   # LDD19: fold agent
      _LDD19_EXT.ldd19_bnd(P, S, T, nu, s1, state["gate2"], state["g_postffn"], 1, state["g_next"], s2, out,
                           LDD19_CL, int(_pdl(T)))
    else:
      b2p = _pdl(T) or (_ldd19_edge(8, T) and bool(LDD19_RQ & 4) and _ldd19_rq_plan(state, "un", m) is not None
                        and use_ma)   # LDD19: big3: edge 8 (the unpooler ran on rq)
      _gt = bool(LDD19_GT128) and T == 128   # LDD19: GT128
      _boundary2[(T,)](P, S, T, nu, s1, state["gate2a"] if _gt else state["gate2"], state["gate2b"] if _gt else state["gate2"], state["g_postffn"], state["g_next"],
                       s2, out, num_warps=16, USE_PDL=b2p, launch_pdl=b2p, GT=_gt)
  return out, s2, ids

# LDD19: keell0renz's per-thread primary CUDA-context bind, already validated in BF16 v25.
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





_ldd19_pre_ctx_kernel = kernel_fn
def kernel_fn(data, state):
  _ctx_bind(data['attn_input'])
  return _ldd19_pre_ctx_kernel(data, state)

# LDD19: Keell small-token issuer/data-ready handoff ports, Apache-2.0.
# Adapted from keell0renz/muse-open-kernel-challenge, commit
# 8e30e1be994db4310794b8d04e90d18d895deb84. LICENSE/NOTICE retained.

_MA_ALT_SRCS['keell_small'] = _ldd19_derive(_MA_SMALL_CUDA_SRC, [
  ('after', r"""  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();""", r"""
#if OKC_DF
  TORCH_CHECK(scratch[2].numel() >= LDD19_DF_OFFSET + 128 * 4096, "DF private bank");
  p.inter += (size_t)LDD19_DF_OFFSET;
#endif"""),
  ('before', r"""  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]""", r"""#if OKC_DF
  const int df_rows = min(128, p.M * 8);
  uint4* df_inter = reinterpret_cast<uint4*>(p.inter);
  for (int i = blockIdx.x * NT + tid; i < df_rows * (ID / 8); i += gridDim.x * NT)
    df_inter[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
#endif
"""),
  ('after', r"""  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if OKC_WS""", r"""
#if OKC_WSI
  if (tid >= NT + 32) { ws_issuer(p, S, reinterpret_cast<unsigned char*>(xbuf)); return; }
#endif"""),
  ('before', r"""  WS_JOIN();
}""", r"""  WS_JOIN();
}
#endif
#if OKC_WS && OKC_WSI
  const int lane = TIDX & 31;
  WS_JOIN();
  const int a0 = S.wsr[0], a1 = S.wsr[1], b0 = S.wsr[2], b1 = S.wsr[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);
      mbar_wait(S.wse + q % OKC_NS, (uint32_t)((q / OKC_NS) & 1));
    }
  }
  __syncwarp();
"""),
  (r"""      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters""", r"""      if (!OKC_DF && q == nA - 1) {"""),
  (r"""  const int njF = S.njF;
  if (lane == 0)""", r"""  const int njF = S.njF;
  if (!OKC_WSI && lane == 0)"""),
  (r"""      if (q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);""", r"""      if (!OKC_WSI && q + OKC_NS < N) ws_issue(p, S, q + OKC_NS, nA, S.wsr[0], S.wsr[2], ring);"""),
  ('before', r"""#if OKC_NACC == 4 && !OKC_XNOMMA""", r"""#if OKC_WSI
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.wse + sl);
#endif
"""),
  ('before', r"""    }
    const int q = seq0 + i;""", r"""#endif
"""),
  (r"""      curj = j;
      if (PB) {""", r"""      curj = j;
#if OKC_DF
      if (PB) load_x_inter_df(p, S, j, lane, warp, x0);
      else load_x_regs<PB>(p, S, j, lane, warp, x0);
#else"""),
  ('before', r"""constexpr int X1P = 512 + 8;""", r"""#if OKC_DF
__device__ __forceinline__ uint32_t df_bad(const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) b |= (uint32_t)(((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu));
  return b;
}
__device__ __forceinline__ void load_x_inter_df(const Params& p, const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  const bool act = g < cnt;
  const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + (act ? g : 0)) * ID + warp * (64 * NCH) + 16 * t);
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      x[2 * c] = act ? __ldcg(src + c * 8) : make_uint4(0, 0, 0, 0);
      x[2 * c + 1] = act ? __ldcg(src + c * 8 + 1) : make_uint4(0, 0, 0, 0);
      bad |= df_bad(x[2 * c]) | df_bad(x[2 * c + 1]);
    }
    if (!__any_sync(0xffffffffu, bad != 0)) break;
    __nanosleep(32);
  }
}
#endif
"""),
  (r"""constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);""", r"""constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? (OKC_WSI ? 64 : 32) : 0);"""),
  ('after', r"""#define OKC_NOLOAD 0""", r"""
#ifndef OKC_WSI
#define OKC_WSI 0
#endif"""),
])
KEELL_WSI = True
KEELL_DF = True
_KEELL_TINY2_BASE = tuple(MOE_TINY2_DEFINES)
_KEELL_MID_BASE = tuple(MOE_MID_DEFINES)
_KEELL_DF_OFFSET = (_MA_EMAX + 2 * WGS_XROWS) * 4096

def _keell_set_small(wsi=True, df=True):
  global KEELL_WSI, KEELL_DF, MOE_TINY2_DEFINES, MOE_MID_DEFINES
  KEELL_WSI, KEELL_DF = bool(wsi), bool(df)
  MOE_TINY2_DEFINES = _KEELL_TINY2_BASE + (('@keell_small', '-DOKC_WSI=1') if wsi else ())
  MOE_MID_DEFINES = _KEELL_MID_BASE + (('@keell_small', '-DOKC_DF=1', '-DLDD19_DF_OFFSET=' + str(_KEELL_DF_OFFSET)) if df else ())

_keell_original_scratch = _ma_make_scratch
def _ma_make_scratch(dev):
  scratch = _keell_original_scratch(dev)
  if KEELL_DF:
    assert scratch[2].numel() == _KEELL_DF_OFFSET
    bank = torch.full((128 * 4096,), -1, device=dev, dtype=torch.int16).view(torch.bfloat16)
    scratch[2] = torch.cat((scratch[2], bank))
  return scratch

_keell_set_small(wsi=False, df=True)

# LDD19: T=1 deferred boundary port from keell0renz, Apache-2.0.
# Source revision 8e30e1be994db4310794b8d04e90d18d895deb84; LICENSE/NOTICE retained.
import hashlib as _hashlib
from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch

# LDD19 hybrid: v24's T = 1 layer (its port of Keell's gle glue, KEELL_GLE) is left out: every T = 1 call is served


# LDD19 HYBRID: team keell0renz's public muse_decode_layer pipeline, embedded below for selected (q_len, T, layer)
# routes. Source: https://github.com/keell0renz/muse-open-kernel-challenge kernels/muse_decode_layer.py at revision
# 8e30e1be994db4310794b8d04e90d18d895deb84, Copyright 2026 Bohdan Agarkov (team keell0renz), Apache License 2.0 (see
# LICENSE/NOTICE.keell0renz.txt); it carries the attributions in its own header comments (alygg77, vjeux, chenwainuo,
# Matsvei), kept verbatim. LDD19 modifications: every module-level name of that file is prefixed `kl_` so the two
# A/B diagnostics (MFAB) and its rs128 (chenwainuo) T = 128/256 bake are not run, as no routed shape uses them.
# okc: problem=muse_decode_layer
# pcx_h1: pre-wait control of the sk2 kernels: hoist (harness/make_pcx.py; team keell0renz; base stack_all1_snap1)
# + alygg77/okc-muse-spark-kernels kernels/muse_decode_layer.py (Apache-2.0, commit 6cae49a); header below is theirs. Patched by team keell0renz: trig, rq, atrig0, rtrig8, pf4.
# + chenwainuo/okc muse_bf16_proj rs128 GEMVs (Apache-2.0) at T = 128 / 256 for qkv and T = 128 for the unpooler (team keell0renz).
# Portions (the per-thread CUDA-context bind helper; cuBLASLt plan pins) follow code posted by Matsvei (kernarc user
# 'matvey') on agents.kernarc.com; OKC submissions are licensed under the Apache License 2.0 (llms.txt), and that
# Portions Copyright vjeux, from https://github.com/vjeux/okc-2026 (commit f35a41e), licensed under the Apache License 2.0.
# Submitted by alygg77 under the Apache License 2.0 with this notice retained.
"""Muse Spark decode layer: one CUDA-graph-capturable kernel_fn.

Pipeline (one decode step, T = batch * q_len tokens)
-----------------------------------------------------
1.  qkv projection (cuBLAS, the bf16_proj kernel, or a Triton split-K GEMM --
    prepare() times the three and pins the winner per (T, matrix)).
2.  Attention: the muse_attention_decode entry's fused kernel (QK-norm,
    RoPE, FP8 cache write, split-KV FP8 attention, merge, per-head
    RMSNorm * sigmoid(gate)) for the shapes it supports; otherwise the glue's
    own QK-prep Triton kernel plus SGLang's FA3 (or a Triton split-KV
    attention fallback when SGLang is unavailable).
3.  o projection, then the first residual boundary (residual add, residual
    gate, pre-FFN norm) -- one-program-per-token kernels or parallel
    atomic-sum kernels, per T, timed in prepare().
4.  Pooler + router (bf16_proj's fused launch, its two separate launches, or
    the Triton split-K GEMM), routing (top-8 of biased sigmoid scores), and
    the pre-expert norm.
5.  Experts: the muse_moe_w4a16 entry's NVFP4 kernel (three compile-time
    builds by token count), or SGLang's Marlin NVFP4 path, whichever
    prepare() timed faster per T.
6.  Unpooler, then the second residual boundary (post-FFN norm, next layer's
    pre-attention norm).

Provenance: the attention, MoE and dense kernels are our own team's entries
on this challenge's component boards, copied (not imported). The CUDA sources
(ATT7_CUDA_SRC, MOE_CUDA_SRC, the bfp_* template strings) and the bfp_*
generators are kept verbatim -- they are the tuned, proven kernels. The glue
(boundaries, routing, tuning, launchers) is this file's own code.

Precision: BF16 weights stay BF16 (or bf16_proj's lossless 12-bit encoding
with exact escape correction); NVFP4 experts stay NVFP4; norms apply
(1 + w) formed in FP32 at use; FP8 queries / FP8 KV cache as SGLang;
softmax P in f16, or e4m3 where SGLang's FA3 rounds it to e4m3 too;
FP32 accumulation throughout; activations BF16 wherever SGLang keeps BF16.

Selection: prepare() times each backend per (token count, matrix) and the
expert kernels per token count, on the judge's own GPU with synthetic
activations (it reads only weights, never data), then fixes the choices;
untuned token counts take the next tuned bucket's.

OKC rules: one Python file; native CUDA only via
torch.utils.cpp_extension.load_inline; CUDA-graph capturable (no host syncs
in timed paths -- prepare() and the tuners sync, kernel_fn does not); fresh
inputs each call, no caching across calls (module-level caches hold only
compiled extensions and per-device scratch, never activations).
Safety: every scratch/workspace buffer is followed by sentinel words;
prepare() checks them after the tuner and the attention self-test.
"""

import os as kl_os
import threading as kl_threading

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.cpp_extension import load_inline as kl_load_inline

# LDD19 hybrid: every `kl_X = X` alias below is identical to v24's definition (AST-checked); shared.
kl_HIDDEN = HIDDEN
kl_D_EXPERT = D_EXPERT
kl_LOCAL_EXPERTS = LOCAL_EXPERTS
kl_N_EXPERTS = N_EXPERTS
kl_TOP_K = TOP_K
kl_QKV_ROWS = 2304
kl_NUM_Q_HEADS = 16
kl_NUM_KV_HEADS = 2
kl_HEAD_DIM = 64
kl_Q_DIM = kl_NUM_Q_HEADS * kl_HEAD_DIM
kl_ROPE_THETA = 500000.0
kl_ROUTER_MULT = 0.1767766952966369
kl_LOG2E = 1.4426950408889634

kl__TUNE_LOG = _TUNE_LOG
kl_FORCE_PBMAX = FORCE_PBMAX
kl_PDL = PDL
kl_PDL_MAX_T = 128


def kl__pdl(T):
  return kl_PDL and T <= kl_PDL_MAX_T


kl_DEFAULT_MOE = DEFAULT_MOE
kl_TUNE_MARGIN = TUNE_MARGIN
kl_PIN_PICK = {(1, "un"): "bfp", (8, "un"): "bfp", (8, "o"): "bfp", (32, "o"): "bfp", (32, "pr"): "bfp",
            (128, "qkv"): True, (256, "qkv"): True}
kl_FORCE_GLUEV2 = FORCE_GLUEV2
kl__ATT7 = _ATT7
# LDD19 small_attention_builder (run 3 it5): kl_ATT7 with OKC_QDIRECT=1 (the q pass writes the e4m3 A tile directly, as the
# derived ATT7 and eat_attn do). Only the L == 8 per-request kernels change (v1 s / v1 f). 0 restores Keell's -DOKC_QDIRECT=0.
LDD19_KL_QDIRECT = 1


# LDD19 (W0, attention_porter run 3 it 4): kl_ATT7 built with OKC_W0ENABLE=0, which re-enables FPART + PTPF in the q1 flat
# kernel (d16 f / d32 f); w0 itself is never registered for kl-routed shapes. False restores -DOKC_W0ENABLE=1.
LDD19_KL_W0OFF = True
# LDD19 solo_optimizer (r5 it1): eat_attn's iter-5/6 items in the layer's two attention builds (make_attn.py); 0 = previous behaviour.
LDD19_KL_PSORT = 7   # LDD_PSORT in kl_ATT7 (q1 flat: d16 f, d32 f)
LDD19_KL_RBFQ = 3    # LDD_RBFQ in kl_ATT7: bit 0 <1,0,1> (d16 s, d32 s), bit 1 <1,1,0> (d16 f, d32 f)
LDD19_KL_RBF = 0     # LDD_RBF in kl_ATT7: 4 = <1,0,11> (d1 f) (2 / 3 add <8,0,11>, v1 f, 8 B stack)
LDD19_D_RBF = 1      # LDD_RBF in the derived ATT7: flat q8 (v16 f, v32 f)
LDD19_D_RBFS = 1     # LDD_RBFS in the derived ATT7: <8,0,19>, <8,0,3> (v16 s, v32 s)
LDD19_KL_ATTQ = 1    # 1 = the flagged kl_ATT7 is its own extension, used only at B > 1; B = 1 (d1, v1) keeps the flag-0 build, SASS-identical to the previous file. 0 = one build
LDD19_KL_FINWARM = 1  # LDD19: integrator_scout it3: 1 = OKC_FINWARM fin_nc1 warm-up in kl_ATT7 (T <= 32 Keell path); 0 = previous behaviour

def kl__att7_flags(q):
  on = q or not LDD19_KL_ATTQ
  return [f"-DLDD_PSORT={LDD19_KL_PSORT if on else 0}", f"-DLDD_RBFQ={LDD19_KL_RBFQ if on else 0}", f"-DLDD_RBF={LDD19_KL_RBF if on else 0}"]


def kl__att7q():   # LDD19 solo_optimizer (r5 it1): the flagged build for B > 1 (LDD19_KL_ATTQ)
  return kl__att7(True) if LDD19_KL_ATTQ else kl__att7()


kl__ATT7Q = None


def kl__att7(q=False):
  global kl__ATT7, kl__ATT7Q
  if (kl__ATT7Q if q else kl__ATT7) is None:
    import os
    from torch.utils.cpp_extension import load_inline
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}" + ("a" if major == 9 else "")
    mod = load_inline(name=f"muse_att_v298w1p{'q' if q else 'f'}_{arch}", cpp_sources=kl_ATT7_CPP_SRC, cuda_sources=kl_ATT7_CUDA_SRC,
                        functions=["run"], extra_cuda_cflags=[f"-DOKC_QDIRECT={LDD19_KL_QDIRECT}", "-DOKC_PK2=0", f"-DOKC_W0ENABLE={0 if LDD19_KL_W0OFF else 1}", f"-DLDD19_FINWARM={LDD19_KL_FINWARM}", *kl__att7_flags(q), "-O3", f"-DOKC_SM90={1 if major >= 9 else 0}", "--expt-relaxed-constexpr"],
                        verbose=False)
    if q:
      kl__ATT7Q = mod
    else:
      kl__ATT7 = mod
  return kl__ATT7Q if q else kl__ATT7


kl__WS7 = {}
kl__PF7 = None
kl__NUM_SMS7 = {}


def kl__workspace7(n_rows, device):
  key = str(device)
  ws = kl__WS7.get(key)
  if ws is None or ws[0].numel() < n_rows * 64:
    n = max(n_rows, 1 << 15)
    ws = tuple(kl__guarded(t) for t in (
        torch.zeros(n * 64, dtype=torch.float32, device=device),
        torch.zeros(n, dtype=torch.float32, device=device),
        torch.zeros(n, dtype=torch.int32, device=device)))
    kl__WS7[key] = ws
  return ws


def kl__num_sms7(device):
  key = str(device)
  if key not in kl__NUM_SMS7:
    kl__NUM_SMS7[key] = torch.cuda.get_device_properties(device).multi_processor_count
  return kl__NUM_SMS7[key]


kl__TUNED7 = {
    (32, 8, True, 1): dict(n_splits=4, nc=1, coop=1),
    (1, 1, False, 1): dict(nc=8, coop=1, n_splits=252),
}


kl_OKC_FLATCOOP = OKC_FLATCOOP
kl_OKC_PDL = OKC_PDL
kl_OKC_FLAT1B = OKC_FLAT1B
kl_OKC_PLAIN8 = OKC_PLAIN8
kl_OKC_F2 = OKC_F2
kl_OKC_S2 = OKC_S2
kl_OKC_W0_L1 = OKC_W0_L1
kl_OKC_W0_PR = OKC_W0_PR
kl_OKC_W0_L8 = OKC_W0_L8


LDD19_KL_SPL3 = True   # LDD19 (SPL3, stack_integrator run 3 it6): kl__plan7 split count at L == 1 sliding, 16 <= B*pairs <= 32 (probe_plan.py: d16 s ~-0.35 us, d32 s ~-0.5 us); False = previous behaviour


def kl__plan7(B, pairs, L, window, max_pages, P, device):
  sms = kl__num_sms7(device)
  target = 2 * sms
  if not window and 1 < B <= 512:
    ft = sms if (L == 8 and kl_OKC_F2 and torch.cuda.get_device_capability(device)[0] >= 9) else target
    return dict(n_splits=max(1, ft // pairs), flat=1, cs=0, ncl=1, nc=1)
  keys_max = min(window + L - 1, max_pages * P) if window else max_pages * P
  tiles = max(1, (keys_max + 63) // 64)
  reqs = B * pairs
  if L == 8 and kl_OKC_S2 and torch.cuda.get_device_capability(device)[0] >= 9:
    target = sms
  if not window:
    if L == 1 and B == 1 and kl_OKC_FLAT1B:
      return dict(n_splits=max(1, ((target * 21) // 22) // pairs), flat=1, cs=0, ncl=1, nc=4)
    if L == 8:
      n = max(1, min(tiles, ((target * 21) // 22) // reqs))
      return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 8)))
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)), coop=0)
  if LDD19_KL_SPL3 and L == 1 and 16 <= reqs <= 32:   # LDD19 (SPL3, stack_integrator run 3 it6): 3 CTAs/SM of split-KV work at
    n = max(1, min(tiles, (3 * sms) // reqs))           # decode 16/32 sliding (d16 s 24, d32 s 12 splits on H200); False = previous plan
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
  if L == 1 and reqs < 24:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
  if L == 8 and reqs < 16 and kl_OKC_PLAIN8:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
  if L == 8 and reqs >= 16 and kl_OKC_PLAIN8:
    n = max(1, min(tiles, (target if reqs >= 24 else sms) // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1, coop=1)
  if reqs >= 24:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=1)
  cs = 16
  limit = 2 * sms if L == 1 else sms
  if L == 8 and target == sms:
    limit = (sms * 15) // 16
  while cs > 4 and reqs * cs > limit:
    cs //= 2
  ncl = 1
  while cs * ncl > tiles and ncl > 1:
    ncl -= 1
  while cs * ncl > tiles and cs > 4:
    cs //= 2
  if cs * ncl > tiles or cs < 4:
    n = max(1, min(tiles, target // reqs))
    return dict(n_splits=n, flat=0, cs=0, ncl=1, nc=max(1, min(n, 4)))
  return dict(n_splits=cs * ncl, flat=0, cs=cs, ncl=ncl, nc=max(1, min(ncl, 4)))


kl__CL7 = {}


def kl__cluster_bufs7(n_ctas, L, device):
  key = str(device)
  need = n_ctas * (L * 16 * 64 + 1024)
  cur = kl__CL7.get(key)
  if cur is None or cur[0].numel() < need or cur[1].numel() < n_ctas:
    cur = tuple(kl__guarded(t) for t in (
        torch.zeros(max(need, 1 << 20), dtype=torch.float32, device=device),
        torch.zeros(max(n_ctas, 1 << 12), dtype=torch.int32, device=device)))
    kl__CL7[key] = cur
  return cur


# LDD19 hybrid: Keell's main MoE source = its moe_helper0 base with these exact edits (readable template).
kl_MOE_CUDA_SRC = _ldd19_derive(kl_MOE_HELPER0_CUDA_SRC, [
  ('before', r"""#else
  p.xA = p.xB = nullptr;""", r"""#if OKC_MFIN
  p.doneAs = p.mfl + okc::MF_DA;
#endif
  TORCH_CHECK((size_t)scratch[0].numel() >= (size_t)(8 + 1024 + okc::EMAX + okc::MF_INTS), "MFIN needs the flag region in scratch[0]");
"""),
  ('after', r"""  p.xready = ib + 8 + 512;""", r"""
#if OKC_MFIN
  p.xready = p.mfl + okc::MF_XR;
#endif"""),
  ('after', r"""  p.ybuf = p.ss + okc::EMAX;""", r"""
#if OKC_WGS && OKC_B4FIN
  TORCH_CHECK(x.size(0) <= 32 && x.size(0) * 8 <= okc::EMAX - 64, "B4FIN build is for M <= 32");
  p.sspart = p.ybuf + (size_t)(okc::EMAX - 64) * okc::HD;
#else
  p.sspart = nullptr;
#endif"""),
  ('before', r"""  p.ybuf = p.ss + okc::EMAX;""", r"""#if OKC_WGS
  p.mfl = ib + 8 + 1024 + okc::EMAX;
#else
  p.mfl = nullptr;
#endif
"""),
  ('after', r"""void okc_set_pdl(int64_t on) { g_okc_pdl = on ? 1 : 0; }""", r"""
std::vector<int64_t> okc_attrs() {
  cudaFuncAttributes a0, a1;
  cudaFuncGetAttributes(&a0, okc::moe_fused<false>);
  cudaFuncGetAttributes(&a1, okc::moe_fused<true>);
  return {a0.numRegs, (int64_t)a0.localSizeBytes, (int64_t)a0.sharedSizeBytes, a1.numRegs, (int64_t)a1.localSizeBytes, (int64_t)a1.sharedSizeBytes};
}"""),
  ('before', r"""          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;""", r"""#endif
"""),
  ('after', r"""          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);""", r"""
#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else"""),
  ('before', r"""          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;""", r"""#endif
"""),
  ('after', r"""        ps[sl] = 0.f;
        if (qv[sl] >= 0) {""", r"""
#if OKC_YBF
          const uint4 yr = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD) + tid);
          const float2 y0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.x)), y1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.y));
          const float2 y2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.z)), y3 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yr.w));
          const float4 a = make_float4(y0.x, y0.y, y1.x, y1.y), b = make_float4(y2.x, y2.y, y3.x, y3.y);
#else"""),
  (r"""#if OKC_WGS && OKC_WGSFINSS""", r"""#if OKC_WGS && OKC_B4FIN
  {
    static_assert(NT == 512 && NW == 16, "B4FIN: 16 warps = 8 routes x 2 halves, 512 threads x 2 columns");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    constexpr int SEG = 1024, NSEG = HD / SEG;
    for (int it = blockIdx.x; it < p.M * NSEG; it += gridDim.x) {
      const int tk = it / NSEG, col = (it % NSEG) * SEG + tid * 2;
#if OKC_MFIN && (OKC_MFP & 4)
      if (tid < 8) {
        if (S.pos[tk * 8 + tid] >= 0) {
          const int e = S.eid[tk * 8 + tid], tot = S.cnt[e];
          const int target = 64 * (tot > SK_T ? (tot + FT_T - 1) / FT_T : 1);
          const int* cp = p.mfl + MF_CE + S.gen * 32 + e;
          for (int n = 0; ld_acquire(cp) < target && n < 4000000; ++n) { if (OKC_MFS) __nanosleep(OKC_MFS); }
        }
      }
      CTA_SYNC();
#endif
#if B4_EARLY_Y
      float2 yv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        yv[sl] = q >= 0 ? __ldcg(reinterpret_cast<const float2*>(p.ybuf + (size_t)q * HD + col)) : make_float2(0.f, 0.f);
      }
#endif
      {
        const int sl = warp >> 1, hh = warp & 1;
        const int q = S.pos[tk * 8 + sl];
        float v = 0.f;
        if (q >= 0) {
          const float4 a = __ldcg(reinterpret_cast<const float4*>(p.sspart + (size_t)q * 256 + hh * 128 + lane * 4));
          v = (a.x + a.y) + (a.z + a.w);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) red[warp] = v;
      }
      CTA_SYNC();
      float2 acc = make_float2(0.f, 0.f);
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const int q = S.pos[tk * 8 + sl];
        if (q >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf((red[2 * sl] + red[2 * sl + 1]) * (1.f / HD) + 1e-8f);
#if B4_EARLY_Y
          const float2 v = yv[sl];
#endif
          acc.x += r * v.x; acc.y += r * v.y;
        }
      }
      *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)tk * HD + col) = __floats2bfloat162_rn(acc.x, acc.y);
      CTA_SYNC();
    }
  }
#elif OKC_WGS && OKC_WGSFINSS"""),
  ('before', r"""  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]""", r"""#endif
"""),
  (r"""#endif
  CTA_SYNC();
  if (tid == 0) {""", r"""#endif
#if !(OKC_MFIN && OKC_WGS && (OKC_MFQ & 4))
  CTA_SYNC();
  if (tid == 0) {"""),
  ('before', r"""#if OKC_WS
  if (tid == 0) {""", r"""#endif
"""),
  ('after', r"""  DBG(1);""", r"""
#if OKC_MFIN && OKC_WGS && (OKC_MFQ & 8)
  {
    const int o = 1 - S.gen;
    for (int i = blockIdx.x * NT + tid; i < 32 + MF_XRS + JMAX * 16; i += gridDim.x * NT) {
      if (i < 32) p.mfl[MF_CE + o * 32 + i] = 0;
      else if (i < 32 + MF_XRS) p.mfl[MF_XR + o * MF_XRS + (i - 32)] = 0;
      else p.mfl[MF_DA + o * (JMAX * 16) + (i - 32 - MF_XRS)] = 0;
    }
  }
#else"""),
  ('before', r"""    S.gen = gen;""", r"""#endif
"""),
  ('after', r"""    S.xbuf_ptr = xbuf;""", r"""
#if OKC_MFIN && (OKC_MFQ & 8)
    const int gen = atomicInc(reinterpret_cast<unsigned*>(p.mfl + MF_EXIT), 2u * gridDim.x - 1u) >= gridDim.x ? 1 : 0;
#else"""),
  (r"""  const int tid = TIDX;""", r"""  const int tid = threadIdx.x;"""),
  (r"""  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (uint32_t)((s & ((1 << (9 - pb)) - 1)) + 1) * (uint32_t)wgs_cost(pb, S.jC[j]);""", r"""  const uint32_t sj = (uint32_t)(s & ((1 << (9 - pb)) - 1)), cs = (uint32_t)wgs_cost(pb, S.jC[j]), U = (uint32_t)wgs_unitc(pb);
  const uint32_t xe = (uint32_t)(pb ? S.wcpB : S.wcp)[j] + (sj >> 4) * (16u * cs + U) + U + ((sj & 15u) + 1u) * cs;"""),
  (r"""  wgs_prefix(S, nj);
  wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);
  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);""", r"""  const int G = gridDim.x, c = blockIdx.x;
  uint32_t carryA = 0, carryB = 0;
  for (int j0 = 0; j0 < nj; j0 += 32) {
    const int j = j0 + lane;
    uint32_t ia = 0, ib = 0;
    if (j < nj) {
      ia = ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;
      ib = ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;
    }
    uint32_t sa = ia, sb = ib;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const uint32_t va = __shfl_up_sync(0xffffffffu, sa, d), vb = __shfl_up_sync(0xffffffffu, sb, d);
      if (lane >= d) { sa += va; sb += vb; }
    }
    if (j < nj) { S.wcp[j] = (int)(carryA + sa - ia); S.wcpB[j] = (int)(carryB + sb - ib); }
    carryA += __shfl_sync(0xffffffffu, sa, 31);
    carryB += __shfl_sync(0xffffffffu, sb, 31);
  }
  if (lane == 0) { S.wcp[nj] = (int)carryA; S.wcpB[nj] = (int)carryB; S.wtot = carryA; S.wtotB = carryB; }
  __syncwarp();
  const int l4 = lane & 3, pbl = l4 >> 1, kk = l4 & 1;
  const uint32_t tot = pbl ? carryB : carryA;
  int res;
  if (kk == 1 && c == G - 1) res = nj << (9 - pbl);
  else res = wgs_stage_at(S, nj, pbl, tot * (uint32_t)(c + kk) / (uint32_t)G);
  if (lane < 4) S.wgr[lane] = res;
  __syncwarp();"""),
  ('after', r"""__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {""", r"""
  const int lane = threadIdx.x & 31;"""),
  (r"""  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);
  return (lo << (9 - pb)) + (int)((x - (uint32_t)pre[lo]) / cs);""", r"""  const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]), U = (uint32_t)wgs_unitc(pb), cu = 16u * cs + U;
  const uint32_t r = x - (uint32_t)pre[lo], un = r / cu, rem = r - un * cu;
  const uint32_t st = rem < U ? 0u : min(15u, (rem - U) / cs);
  return (lo << (9 - pb)) + (int)(un * 16u + st);"""),
  (r"""    acc += (uint32_t)wgs_cost(0, S.jC[j]) << 9;
    accB += (uint32_t)wgs_cost(1, S.jC[j]) << 8;""", r"""    acc += ((uint32_t)wgs_cost(0, S.jC[j]) * 16u + OKC_WGSU) << 5;
    accB += ((uint32_t)wgs_cost(1, S.jC[j]) * 16u + OKC_WGSUB) << 4;"""),
  ('after', r"""  wgs_flush<PB>(p, S, pd);
}""", r"""
#define OKC_WGSU 160
#define OKC_WGSUB 140
__device__ __forceinline__ int wgs_unitc(int pb) { return pb ? OKC_WGSUB : OKC_WGSU; }"""),
  ('after', r"""  if (!PB) pd = WgsPend{1, u, -1, 0};""", r"""
#if OKC_MFIN && (OKC_MFP & 1)
  else pd = WgsPend{4, u, (OKC_MFP & 8) ? (int)(mf_gen(S) * 32 + S.jE[j]) : -1, 0};
#endif"""),
  ('before', r"""          af[ci & 1][jj][0] = bmul2l(r0, sgp); af[ci & 1][jj][1] = bmul2h(r1, sgp);""", r"""#endif
"""),
  ('after', r"""          dq4(q, r0, r1, r2, r3);""", r"""
#if OKC_SC6W
          const uint32_t sgp = sgs[js];
#else"""),
  (r"""        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else if (lane == 0) {""", r"""      } else if (lane == 0) {"""),
  (r"""      const uint2 scg = scn;""", r"""#if !OKC_SC6W
      const uint2 scg = scs;
#endif"""),
  (r"""    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);""", r"""#if OKC_SC6W
    uint32_t sgs[4];
    {
      const int tt = wp * 32 + lane;
      const uint32_t W = *reinterpret_cast<const uint32_t*>(pc + 8192 + tt * 4);
      const uint32_t H = static_cast<uint32_t>(*reinterpret_cast<const unsigned short*>(pc + 8192 + 512 + tt * 2));
      const uint32_t X = W & 0xFCFCFCFCu, Hm = H & 0xFCFCu;
      const uint32_t p6 = (W & 0x00030303u) * 0x04104000u;
      const uint32_t p7 = ((H & 0x0303u) | (W & 0x03000000u)) * 0x10400004u;
      sgs[0] = __byte_perm(X, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[1] = __byte_perm(X, 0u, 0x4342u) * k16c + 0x60006000u;
      sgs[2] = __byte_perm(Hm, 0u, 0x4140u) * k16c + 0x60006000u;
      sgs[3] = __byte_perm(p6, p7, 0x0703u) * k16c + 0x60006000u;
    }
#else
    const uint2 scs = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
#endif"""),
  (r"""  const uint32_t k16c = 16u + p.zero;""", r"""  const uint32_t k16c = (OKC_SC6W ? 4u : 16u) + p.zero;"""),
  (r"""  const int tid = TIDX, lane = tid & 31;""", r"""  const int tid = threadIdx.x, lane = tid & 31;"""),
  (r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }""", r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");
  }"""),
  ('after', r"""  if (wt == 0) *fc = 0;""", r"""
#if OKC_MFIN && (OKC_MFP & 2)
  if (PB) {
    asm volatile("bar.sync %0, 128;\n" ::"r"(3 + wg) : "memory");
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + mf_gen(S) * 32 + S.jE[j]), "r"(1) : "memory");
  }
#endif"""),
  (r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.doneAs + j * 16 + (ub >> 1)), "r"(1) : "memory");""", r"""    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_da(p, S) + j * 16 + (ub >> 1)), "r"(1) : "memory");"""),
  ('before', r"""  if (pd.kind == 1) {""", r"""#if OKC_MFIN && (OKC_MFP & 1)
  if (pd.kind == 4) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.mfl + MF_CE + ((OKC_MFP & 8) ? pd.slot : (int)(mf_gen(S) * 32 + S.jE[j]))), "r"(1) : "memory");
    pd.kind = 0;
    return;
  }
#endif
"""),
  (r"""  const int tid = TIDX, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;""", r"""  const int tid = threadIdx.x, warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3, wt = tid & 127;"""),
  ('before', r"""  }
}
template <bool PB>""", r"""#if OKC_B4FIN
    {
      float q0 = 0.f, q1 = 0.f;
      if (tok0 < cnt) { const float a = d4[0] * w0, b = d4[2] * w1; q0 = a * a + b * b; }
      if (tok0 + 1 < cnt) { const float a = d4[1] * w0, b = d4[3] * w1; q1 = a * a + b * b; }
#pragma unroll
      for (int o = 4; o < 32; o <<= 1) {
        q0 += __shfl_xor_sync(0xffffffffu, q0, o);
        q1 += __shfl_xor_sync(0xffffffffu, q1, o);
      }
      if (g == 0) {
        if (tok0 < cnt) p.sspart[(size_t)(jst + tok0) * 256 + r16] = q0;
        if (tok0 + 1 < cnt) p.sspart[(size_t)(jst + tok0 + 1) * 256 + r16] = q1;
      }
    }
#endif
"""),
  ('before', r"""      }
    }
#if OKC_B4FIN""", r"""#endif
"""),
  ('after', r"""      const int tk = tok0 + h;
      if (tk < cnt) {""", r"""
#if !(OKC_YBF)"""),
  (r"""  const int lane = TIDX & 31, g = lane >> 2, t = lane & 3;""", r"""  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;"""),
  ('after', r"""struct WgsPend { int kind, u, slot, nst; };
template <bool PB>""", r"""
#if !OKC_B4FIN
#define OKC_YBF 1
#else
#define OKC_YBF 0
#endif"""),
  (r"""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(p.xready + jj), "r"(e - q) : "memory");""", r"""      asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(mf_xr(p, S) + jj), "r"(e - q) : "memory");"""),
  (r"""          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);""", r"""          {
            const int G = tid & 31;
            const unsigned* xs32 = reinterpret_cast<const unsigned*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid >> 5) * 256)
                                   + (8 * ((G >> 1) & 3) + 2 * (G >> 3) + (G & 1));
            v[k] = make_uint4(__ldg(xs32), __ldg(xs32 + 32), __ldg(xs32 + 64), __ldg(xs32 + 96));
          }"""),
  (r"""  const int tid = TIDX, nj = S.njF + S.njS;""", r"""  const int tid = threadIdx.x, nj = S.njF + S.njS;"""),
  (r"""      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.doneAs + j * 16 + st) < 8) __nanosleep(32); }""", r"""      if (pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_da(p, S) + j * 16 + st) < 8) __nanosleep(32); }"""),
  (r"""        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(p.xready + j) < npad) __nanosleep(32); }""", r"""        if (!pb && !(OKC_WGSX & 8)) { while (ld_acquire(mf_xr(p, S) + j) < npad) __nanosleep(32); }"""),
  (r"""  __syncwarp();
  WS_JOIN();
  if ((TIDX & 31) != 0) return;""", r"""  __syncwarp();
  WS_JOIN();
  if ((threadIdx.x & 31) != 0) return;"""),
  (r"""  if ((TIDX & 31) == 0) wgs_ranges(p, S);""", r"""  wgs_ranges(p, S);"""),
  (r"""                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");""", r"""                   ::"r"(smem_u32(ring + ws * UNIT_BYTES)), "l"(src), "r"(WGS_STG), "r"(smem_u32(bar)) L2H_ARG : "memory");"""),
  (r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");""", r"""      const unsigned char* src = base + ((size_t)(((S.jE[j] << ubl) + ub) * 16 + st)) * (size_t)WGS_STG;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(WGS_STG) : "memory");"""),
  ('before', r"""      uint64_t* bar = S.bars + BW_F + ws;""", r"""      if (n >= 2 && p.M <= 64) { const int pn = n - 2; wgs_wait(S.bars + BW_F + pn % WGS_NS, (uint32_t)(pn / WGS_NS) & 1u, 5000 + gs); }
"""),
  (r"""  if ((TIDX & 31) != 0) return;""", r"""  if ((threadIdx.x & 31) != 0) return;"""),
  (r"""#define OKC_WGSFCI 3""", r"""#define OKC_WGSFCI 4"""),
  ('before', r"""#define OKC_WGSA 112, 135, 140, 158, 173, 185, 200, 207""", r"""constexpr int MF_CE = 0, MF_EXIT = 64, MF_XR = 128, MF_XRS = 96, MF_DA = 384, MF_INTS = 4096;
static_assert(JMAX <= MF_XRS && MF_DA + 2 * JMAX * 16 <= MF_INTS, "MFIN flag region");
__device__ __forceinline__ int mf_gen(const Smem& S) {
#if OKC_MFQ & 32
  int v;
  asm volatile("ld.volatile.shared.s32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(&S.gen)));
  return v;
#endif
}
__device__ __forceinline__ int* mf_xr(const Params& p, const Smem& S) { return p.xready + ((OKC_MFIN && (OKC_MFQ & 2)) ? mf_gen(S) * MF_XRS : 0); }
__device__ __forceinline__ int* mf_da(const Params& p, const Smem& S) { return p.doneAs + ((OKC_MFIN && (OKC_MFQ & 1)) ? mf_gen(S) * (JMAX * 16) : 0); }
"""),
  (r"""constexpr int WGS_PIECE = 9216;""", r"""constexpr int WGS_PIECE = OKC_SC6W ? 8960 : 9216;
constexpr int WGS_STG = OKC_SC6W ? 35840 : 36864;"""),
  (r"""  const int lane = TIDX & 31;""", r"""  const int lane = threadIdx.x & 31;"""),
  (r"""    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);""", r"""    mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));
    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = sc_load(slot, warp, lane);"""),
  (r"""__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;""", r"""__device__ void skinny_ring_ws(const Params& p, Smem& S, int v0, int v1, unsigned char* ring, uint64_t* bars, int& seq) {
  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;"""),
  (r"""  if (TIDX == NT - 32) {""", r"""  if (threadIdx.x == NT - 32) {"""),
  (r"""    if (TIDX == 0) S.preissued = 0;""", r"""    if (threadIdx.x == 0) S.preissued = 0;"""),
  (r"""  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;""", r"""  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const int g = lane >> 2, t = lane & 3;"""),
  (r"""  if (TIDX == 0) {""", r"""  if (threadIdx.x == 0) {"""),
  (r"""    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);""", r"""    const unsigned char* slot = ring + sl * UNIT_BYTES;
    const uint4 sc = sc_load(slot, warp, lane);"""),
  (r"""  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;
  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;""", r"""  (void)k16;
  const int rbs = PB ? RB_B : RB_A;
  const int njF = S.njF;
  const int nU = v1 - v0;
  if (nU <= 0) return;"""),
  (r"""    const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);""", r"""    const uint4 sc = sc_load(slot, warp, lane);"""),
  (r"""  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = 16u + p.zero;""", r"""  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);
  const uint32_t k16 = SCK + p.zero;"""),
  (r"""  if (TIDX == 0) ring_issue<PB>(p, S, v, slot, bar);""", r"""  if (threadIdx.x == 0) ring_issue<PB>(p, S, v, slot, bar);"""),
  ('before', r"""__device__ __forceinline__ void mbar_init(uint64_t* bar, int count) {""", r"""#endif
#define SCK ((OKC_SC6S ? 4u : 16u))
__device__ __forceinline__ uint4 sc_load(const unsigned char* slot, int warp, int lane) {
#if !(OKC_SC6S)
  return *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
}
"""),
  ('before', r"""#endif
#endif""", r"""#if OKC_SC6S
#undef SCDK
#define SCDK(word, b, k4) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k4) + 0x60006000u)
"""),
  (r"""constexpr int UNIT_CODES = 32768, UNIT_SC = 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;""", r"""#ifndef OKC_SC6S
#define OKC_SC6S 0
#endif
#ifndef OKC_SC6W
#define OKC_SC6W 0
#endif
constexpr int UNIT_CODES = 32768, UNIT_SC = OKC_SC6S ? 3072 : 4096, UNIT_BYTES = UNIT_CODES + UNIT_SC;"""),
  (r"""  const int tid = TIDX, lane = tid & 31, warp = WARP_IDX(tid);""", r"""  const int tid = threadIdx.x, lane = tid & 31, warp = WARP_IDX(tid);"""),
  (r"""#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)""", r"""#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)"""),
  ('before', r"""};
struct Smem {""", r"""  float* sspart;
"""),
  ('after', r"""  int* doneAs;""", r"""
  int* mfl;"""),
  (r"""#define TIDX ((int)((threadIdx.x + NT) % NTL))
constexpr int NRB = OKC_WS ? OKC_WSNR : 2;""", r"""constexpr int NRB = OKC_WS ? OKC_WSNR : 2;"""),
  ('before', r"""#ifndef OKC_FINCOL""", r"""#ifndef OKC_MFIN
#define OKC_MFIN 1
#endif
#ifndef OKC_B4L
#define OKC_B4L 0
#endif
#ifndef OKC_MFP
#define OKC_MFP 15
#endif
#ifndef OKC_MFS
#define OKC_MFS 0
#endif
#ifndef OKC_MFQ
#define OKC_MFQ 47
#endif
#if !OKC_WGS
#undef OKC_MFIN
#define OKC_MFIN 0
#undef OKC_B4L
#define OKC_B4L 0
#endif
#ifndef OKC_B4FIN
#define OKC_B4FIN 0
#endif
#ifndef B4_EARLY_Y
#define B4_EARLY_Y 1
#endif
"""),
])
kl_MOE_CPP_SRC = r"""
#include <torch/extension.h>
void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,
             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid);
void okc_set_pdl(int64_t on);
std::vector<int64_t> okc_attrs();
"""
kl__MOE = None
kl__MOE_B4 = None


kl__MOE_TINY = _MOE_TINY
kl__MOE_TINY2 = _MOE_TINY2
kl__MOE_MID = _MOE_MID
kl_MOE_DEFINES = MOE_DEFINES
kl_MOE_TINY_DEFINES = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4", "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2")
kl_MOE_TINY_MAX_M = MOE_TINY_MAX_M
kl_MOE_TINY2_DEFINES = MOE_TINY2_DEFINES
kl_MOE_TINY2_MAX_M = MOE_TINY2_MAX_M
LDD19_KL_MID8 = True   # LDD19: iter1 solo: Keell path runs M 5..8 on its MID build, not TINY2 (paired +1.0% verify 1)
if LDD19_KL_MID8:
  kl_MOE_TINY2_MAX_M = 0
kl_MOE_MID_DEFINES = ('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1')
kl_MOE_MID_MAX_M = MOE_MID_MAX_M


kl__MA_PRMT_SCALES = _MA_PRMT_SCALES
kl__MA_SFX_IF_PRMT = _MA_SFX_IF_PRMT


kl__ma_scales_prmt_exact = _ma_scales_prmt_exact


kl_MOE_HAS_PDL = MOE_HAS_PDL


kl_SC6_WGS = True
kl__MA_SC6W = False
kl__MA_SFX = _MA_SFX


kl__MA_EXTS = {}
kl__MA_ALT_SRCS = {}
kl__MA_MAIN_NS = _MA_MAIN_NS
kl__MA_PDL = _MA_PDL


LDD19_B4_FNF = 1   # LDD19: FNF on B4: no WG-wide fence in the Stream-K finisher, as MAIN (thread 0's acquire + the WG bar.sync order the partial loads; 0 = stock)


LDD19_B4_BHOLD = 2   # LDD19: B4: bounded hold of the last n pre-boundary B stages (0 = stock)


def _ldd19_b4h_src(c, n):
  for a, b in (("      uint64_t* bar = S.bars + BW_F + ws;\n", "      if (pb == 1 && i >= WGS_NS - %d && i < WGS_NS) {\n        int bsp = 0;\n        while (*reinterpret_cast<volatile int*>(&S.preissued) < 4 && bsp < 65536) { __nanosleep(20); ++bsp; }\n      }\n      uint64_t* bar = S.bars + BW_F + ws;\n" % n),
               ("    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n", "    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    if ((threadIdx.x & 127) == 0) atomicAdd(&S.preissued, 1);\n")):
    assert c.count(a) == 1, a[:50]
    c = c.replace(a, b)
  return c


LDD19_B4_TAB = 1   # LDD19: W32T + A3 on B4: phase-A stage costs 104, 135, 164, ... (changes the stage costs, so MoE bits differ within the census bound; 0 = stock)


def kl__ma_ext(defines=None):
  if defines is not None and LDD19_B4_TAB and "-DOKC_B4FIN=1" in tuple(defines) and "-DLDD19_B4T=1" not in tuple(defines):   # LDD19: W32T on B4
    defines = tuple(defines) + ("-DLDD19_B4T=1",)
  if defines is not None and LDD19_B4_BHOLD and "-DOKC_B4FIN=1" in tuple(defines) and "-DLDD19_B4H=1" not in tuple(defines):   # LDD19: B4 hold
    defines = tuple(defines) + ("-DLDD19_B4H=1",)
  if defines is not None and LDD19_B4_GF2 and "-DOKC_B4FIN=1" in tuple(defines) and "-DLDD19_B4X=1" not in tuple(defines):   # LDD19: GF2 on B4
    defines = tuple(defines) + ("-DLDD19_B4X=1",)
  if defines is not None and (LDD19_FXS & 2) and "-DOKC_WGS=1" in tuple(defines) and "-DOKC_B4FIN=1" in tuple(defines) and "-DLDD19_FXS=1" not in tuple(defines):   # LDD19: FXS
    defines = tuple(defines) + ("-DLDD19_FXS=1",)
  if defines is not None and LDD19_B4_FNF and "-DOKC_B4FIN=1" in tuple(defines) and "-DLDD19_B4F=1" not in tuple(defines):   # LDD19: FNF on B4
    defines = tuple(defines) + ("-DLDD19_B4F=1",)
  if defines is not None and (LDD19_RSRED & 2) and "-DOKC_WGS=1" in tuple(defines) and "-DOKC_B4FIN=1" in tuple(defines) and "-DLDD19_RSRED=1" not in tuple(defines):   # LDD19: RSRED
    defines = tuple(defines) + ("-DLDD19_RSRED=1",)
  if defines is not None and kl__MA_SC6M and "@moe_helper0df" in tuple(defines):   # LDD19: SC6M
    defines = tuple(defines) + ("-DLDD19_SC6M=1",)
  defines = tuple(kl_MOE_DEFINES if defines is None else defines)
  if kl__MA_SC6W and "-DOKC_WGS=1" in defines and "-DOKC_SC6W=1" not in defines and "-DOKC_SC6OFF=1" not in defines:
    defines = defines + ("-DOKC_SC6W=1",)
  if kl_DQ7 and "-DOKC_DQ7=1" not in defines:
    defines = defines + ("-DOKC_DQ7=1",)
  if defines not in kl__MA_EXTS:
    src = kl_MOE_CUDA_SRC
    if "-DLDD19_B4T=1" in defines:   # LDD19: W32T on B4
      assert src.count("#define OKC_WGSA 112, 135, 140, ") == 1
      src = src.replace("#define OKC_WGSA 112, 135, 140, ", "#define OKC_WGSA 104, 135, 164, ")
    if "-DLDD19_B4X=1" in defines:   # LDD19: GF2 on B4
      a = "          dq4(q, r0, r1, r2, r3);\n#if OKC_SC6W\n"
      assert src.count(a) == 1
      src = src.replace(a, "          r0 = q & 0x81C081C0u; r1 = (q & 0x081C081Cu) * 16u; { const uint32_t t = q >> 8; asm(\"lop3.b32 %0, %1, %2, %3, 0x28;\" : \"=r\"(r2) : \"r\"(q), \"r\"(t), \"n\"(0x81C081C0u >> 2)); r2 *= 4u; asm(\"lop3.b32 %0, %1, %2, %3, 0x28;\" : \"=r\"(r3) : \"r\"(q), \"r\"(t), \"n\"(0x81C081C0u >> 6)); r3 *= 64u; }\n#if OKC_SC6W\n")
    if "-DLDD19_B4H=1" in defines:   # LDD19: B4 hold
      src = _ldd19_b4h_src(src, LDD19_B4_BHOLD)
    if "-DLDD19_B4F=1" in defines:   # LDD19: FNF on B4
      a, b = src.index("__device__ __forceinline__ void wgs_flush("), '  asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");\n'
      i = src.index(b, a)
      assert src.index("\n}\n", a) > i
      src = src[:i] + src[i + len(b):]
    if "-DLDD19_RSRED=1" in defines:   # LDD19: RSRED
      src = _ldd19_rsred_src(src)
    if "-DLDD19_FXS=1" in defines:   # LDD19: FXS
      src = _ldd19_fxs_src(src)
    defs = [d for d in defines if not d.startswith("@")]
    for d in defines:
      if d.startswith("@"):
        src = kl__MA_ALT_SRCS[d[1:]]
    if "-DLDD19_SC6M=1" in defs:   # LDD19: SC6M
      src = kl__sc6m_src(src)
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major == 9 else "")
    tag = "_".join(d.replace("=", "").replace("-D", "").replace("@", "src").lower() for d in defines) or "main"
    import hashlib as _native_build_hash
    source_tag = _native_build_hash.md5((kl_MOE_CPP_SRC + src + repr((defines, kl__MA_MAIN_NS))).encode()).hexdigest()[:16]
    kl__MA_EXTS[defines] = kl_load_inline(
        name=f"glue_moe_{source_tag}_sm{arch}",
        cpp_sources=kl_MOE_CPP_SRC,
        cuda_sources=src,
        functions=["okc_run", "okc_set_pdl"] + (["okc_attrs"] if "okc_attrs" in src else []),
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                           *([] if any(d.startswith("-DOKC_NS=") for d in defs) else [f"-DOKC_NS={kl__MA_MAIN_NS if major >= 9 else 1}"]),
                           f"-DOKC_FNST={4 if major >= 9 else 3}",
                           *([] if any(d.startswith("-DOKC_NSF=") for d in defs) else [f"-DOKC_NSF={3 if major >= 9 else 1}"]), *defs],
        verbose=False,
    )
    kl__MA_EXTS[defines].okc_set_pdl(1 if kl__MA_PDL else 0)
  return kl__MA_EXTS[defines]


def kl__moe_build(defines):
  ext = kl__ma_ext(tuple(defines) + tuple(kl__MA_SFX))
  if kl_MOE_HAS_PDL:
    ext.okc_set_pdl(0)
  return ext


def kl__moe(M=None):
  global kl__MOE, kl__MOE_TINY, kl__MOE_TINY2, kl__MOE_MID, kl__MOE_B4
  if M is not None and kl_MOE_TINY_DEFINES and M <= kl_MOE_TINY_MAX_M:
    if kl__MOE_TINY is None:
      kl__MOE_TINY = kl__moe_build(kl_MOE_TINY_DEFINES)
    return kl__MOE_TINY
  if M is not None and kl__moe_tiny2_ok() and M <= kl_MOE_TINY2_MAX_M:
    if kl__MOE_TINY2 is None:
      kl__MOE_TINY2 = kl__moe_build(kl_MOE_TINY2_DEFINES)
    return kl__MOE_TINY2
  if M is not None and kl_MOE_MID_DEFINES and M <= kl_MOE_MID_MAX_M:
    if kl__MOE_MID is None:
      kl__MOE_MID = kl__moe_build(kl_MOE_MID_DEFINES)
    return kl__MOE_MID
  if (M is not None and 16 < M <= 32 and "kl__main_defines" in globals()
          and tuple(kl__main_defines(kl__MA_SFX)) == tuple(kl_MOE_DEFINES)):
    if kl__MOE_B4 is None:
      kl__MOE_B4 = kl__moe_build(tuple(kl_MOE_DEFINES) + ("-DOKC_B4FIN=1",))
    return kl__MOE_B4
  if kl__MOE is None:
    kl__MOE = kl__moe_build(kl__main_defines(kl__MA_SFX) if "kl__main_defines" in globals() else kl_MOE_DEFINES)
  return kl__MOE


def kl__moe_mid():
  global kl__MOE_MID
  if kl_MOE_MID_DEFINES and kl__MOE_MID is None:
    kl__MOE_MID = kl__moe_build(kl_MOE_MID_DEFINES)
  return kl__MOE_MID


kl__moe_tiny2_ok = _moe_tiny2_ok


def kl__moe_tiny2():
  global kl__MOE_TINY2
  if kl__moe_tiny2_ok() and kl__MOE_TINY2 is None:
    kl__MOE_TINY2 = kl__moe_build(kl_MOE_TINY2_DEFINES)
  return kl__MOE_TINY2


if LDD19_KL_MID8:   # LDD19: no routed T uses the Keell TINY2 build any more: skip its compile in prepare
  def kl__moe_tiny2():
    return None


def kl__moe_tiny():
  global kl__MOE_TINY
  if kl_MOE_TINY_DEFINES and kl__MOE_TINY is None:
    kl__MOE_TINY = kl__moe_build(kl_MOE_TINY_DEFINES)
  return kl__MOE_TINY


kl__REG_HJ = [(0, 0), (1, 0), (0, 2), (1, 2)]


kl_DQ7 = DQ7


def kl__stored_pos():
  pos = {}
  for R in range(4):
    for up in range(2):
      outb = {0: 6 + 16 * up, 1: 7 + 16 * up, 2: 8 + 16 * up, 3: 15 + 16 * up}
      for b, o in outb.items():
        if kl_DQ7:
          d = (0, 28, 22, 18 if b == 2 else 26)[R]
        elif R == 0:
          d = 0
        elif R == 1:
          d = 6
        elif R == 2:
          d = 10
        else:
          d = 4 if b == 0 else 28
        pos[(R, up, b)] = (o + d) % 32
  assert sorted(pos.values()) == list(range(32))
  return pos


kl__POS = kl__stored_pos()


def kl__pack_codes(codes, rows):
  c = codes.index_select(0, rows)
  R = c.shape[0]
  nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)
  nb = R // 16
  n = nib.view(nb, 2, 8, 8, 8, 4, 4, 4)
  n = n.permute(0, 3, 4, 2, 5, 6, 1, 7).to(torch.int64)
  word = torch.zeros(n.shape[:-2], dtype=torch.int64, device=codes.device)
  for Rg, (h, j0) in enumerate(kl__REG_HJ):
    for up in range(2):
      code = n[..., h, j0 + up]
      for b in range(4):
        word |= ((code >> b) & 1) << kl__POS[(Rg, up, b)]
  word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
  return word.contiguous().view(-1)


kl__pack_scales = _pack_scales


kl__sc6_ok = _sc6_ok


kl__sc6_wgs = _sc6_wgs



kl__fat_order = _fat_order


kl__unit_order = _unit_order


def kl__pack_codes_nat(codes, rows):
  c = codes.index_select(0, rows)
  R = c.shape[0]
  nib = torch.stack((c & 0xF, c >> 4), -1).reshape(R, -1)
  nb = R // 16
  n = nib.view(nb, 2, 8, 8, 8, 4, 2, 4, 2)
  n = n.permute(0, 3, 4, 2, 7, 5, 1, 6, 8).to(torch.int64)
  word = torch.zeros(n.shape[:-3], dtype=torch.int64, device=codes.device)
  for Rg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
    for up in range(2):
      code = n[..., h, q, up]
      for b in range(4):
        word |= ((code >> b) & 1) << kl__POS[(Rg, up, b)]
  word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
  return word.contiguous().view(-1)


def kl__kperm_src():
  p = torch.arange(256)
  s, c = p >> 4, p & 15
  lg, j, t, i = s >> 2, s & 3, (c >> 1) & 3, (c & 1) | ((c >> 3) << 1)
  k = 16 * (4 * t + j) + 4 * lg + i
  assert sorted(k.tolist()) == list(range(256))
  return k


def kl__kperm_codes(codes):
  R, K = codes.shape[0], codes.shape[1] * 2
  nib = torch.stack((codes & 0xF, codes >> 4), -1).reshape(R, K // 256, 256)
  nib = nib.index_select(2, kl__kperm_src().to(codes.device)).reshape(R, K // 2, 2)
  return (nib[..., 0] | (nib[..., 1] << 4)).contiguous()


def kl__kperm_rows(rowsA):
  r = rowsA.view(-1, 2, 8)
  gate, up = r[:, 0, :].reshape(-1), r[:, 1, :].reshape(-1)
  P = torch.arange(gate.numel(), device=rowsA.device)
  cm = (P >> 8) * 256 + kl__kperm_src().to(rowsA.device)[P & 255]
  return torch.stack((gate.index_select(0, cm).view(-1, 8), up.index_select(0, cm).view(-1, 8)), 1).reshape(-1)


LDD19_B4_GF2 = LDD19_GF2   # LDD19: GF2 on B4: GF(2)-mixed WGS code words, decoded with dq4g (bit-identical to DQ7; 0 = stock)


def kl__wgs_copy(codes, sf, rows, sc6=False):
  wn = kl__pack_codes_nat(kl__kperm_codes(codes), rows)
  if LDD19_B4_GF2:   # LDD19: GF2 on B4
    x, o = wn.to(torch.int64) & 0xFFFFFFFF, torch.zeros(wn.shape, dtype=torch.int64, device=wn.device)
    for i, c in enumerate(kl__gf2s_cols()):
      o ^= ((x >> i) & 1) * c
    wn = torch.where(o >= 2**31, o - 2**32, o).to(torch.int32)
  R = rows.numel()
  nb, nub = R // 16, R // 256
  w = wn.view(nb, 8, 8, 8, 4, 4)
  w = w.permute(0, 1, 2, 5, 3, 4).contiguous().view(nb, 256, 32)
  w = w.view(nub, 4, 4, 16, 4, 4, 32)
  w = w.permute(0, 3, 1, 4, 2, 6, 5).contiguous()
  cb = w.view(torch.uint8).view(nub, 16, 4, 8192)
  s = sf.view(torch.uint8).index_select(0, rows)
  s = s.view(nub, 4, 4, 2, 8, 16, 4, 4)
  s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)
  if sc6:
    s = kl__sc6_wgs(s)
  return torch.cat([cb, s], dim=3).contiguous().view(-1)


kl_MAIN_FALLBACK_DEFINES = MAIN_FALLBACK_DEFINES


kl__main_defines = _main_defines


def kl__wgs_on():
  return any("-DOKC_WGS=1" in tuple(d) for d in (kl_MOE_DEFINES, kl_MOE_TINY_DEFINES, kl_MOE_TINY2_DEFINES, kl_MOE_MID_DEFINES))


kl_bfp__SK2_DELAY = int(kl_os.environ.get("SK2_DELAY_ESC", "0"))


kl_bfp_wgmma_defs = bfp_wgmma_defs

kl_bfp_GEMM_SRC = bfp_GEMM_SRC


kl_bfp_wgmma_ss_defs = bfp_wgmma_ss_defs


kl_bfp_CPP_SRC = bfp_CPP_SRC

kl_bfp_w_rows = {"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192, "router": 256}
kl_bfp_COMPRESS = {"qkv_gate", "o", "pooler", "unpooler"}

kl_bfp__ext = bfp__ext


kl_bfp_TMA_CONFIGS = {
    ("pooler", 0): (1, 2, 8, 4), ("pooler", 1): (2, 4, 4, 4), ("pooler", 2): (4, 8, 2, 4), ("pooler", 4): (8, 16, 1, 4),
    ("unpooler", 0): (2, 1, 8, 4), ("unpooler", 1): (2, 1, 8, 4), ("unpooler", 2): (2, 8, 2, 4), ("unpooler", 4): (4, 16, 1, 4),
    ("qkv_gate", 0): (8, 9, 2, 4), ("qkv_gate", 1): (8, 9, 2, 4), ("qkv_gate", 2): (8, 9, 2, 4), ("qkv_gate", 4): (8, 9, 1, 8),
    ("o", 0): (1, 4, 4, 4), ("o", 1): (1, 4, 4, 4), ("o", 2): (1, 4, 4, 4), ("o", 4): (1, 4, 4, 4),
    ("router", 0): (8, 1, 8, 4), ("router", 1): (8, 1, 8, 4), ("router", 2): (8, 1, 8, 4), ("router", 4): (8, 1, 8, 4),
}
kl_bfp_USE_WGM = bfp_USE_WGM
kl_bfp_WGM_SM80_TEST = bfp_WGM_SM80_TEST


kl_bfp__tma_nstage = bfp__tma_nstage


kl_bfp_TMA_TU = {}


kl_bfp_TMA_ALT = {("pooler", 8): (8, 16, 1, 4), ("pooler", 16): (8, 16, 1, 4), ("unpooler", 1): (8, 4, 1, 4),
           ("unpooler", 8): (8, 4, 1, 4), ("unpooler", 16): (4, 16, 1, 4), ("o", 16): (4, 16, 1, 4),
           ("o", 32): (4, 16, 1, 4)}


def kl_bfp__tma_cfg_items():
  items = [((layer, mtk), cfg) for (layer, mtk), cfg in sorted(kl_bfp_TMA_CONFIGS.items())]
  items += [(("alt", layer, m), cfg) for (layer, m), cfg in sorted(kl_bfp_TMA_ALT.items())]
  return items


def kl_bfp__tma_srcs(sm90):
  lines = {}
  for key, (splits, rg, wsn, ks) in kl_bfp__tma_cfg_items():
    if key[0] == "alt":
      layer, mtk = key[1], kl_bfp__mt_key(key[2])
    else:
      layer, mtk = key
    mt = max(mtk, 1)
    f32 = layer == "router"
    ot = "float" if f32 else "__nv_bfloat16"
    comp = 0 if f32 else 1
    hl = f32
    ns = kl_bfp__tma_nstage(rg * wsn, comp, sm90, ks, hl)
    if f32 and mt >= 4:
      ns = min(ns, 3)
    n_cfg = 1024 if (layer == "unpooler" and mtk <= 1) else kl_bfp_w_rows[layer]
    wgm = bool((sm90 or kl_bfp_WGM_SM80_TEST) and comp and kl_bfp_USE_WGM and rg % 4 == 0 and mtk >= 2)
    line = (f"if (mt == {mt} && rg == {rg} && wsn == {wsn} && ks == {ks} && comp == {comp} && out_f32 == {str(f32).lower()}) "
            f"{{ launch_tma<{mt}, {rg}, {wsn}, {ks}, {ns}, {str(bool(comp)).lower()}, {str(hl).lower()}, {str(wgm).lower()}, {ot}>(Wp, x, out, ws, "
            f"counters, ptab, pmax, base4, M, N, Kp, Kx, splits, nrg, (int)dg); return; }}")
    lines.setdefault(line, []).append(key)
  keys = list(lines)
  for ln in keys:
    for cfg in lines[ln]:
      kl_bfp_TMA_TU[cfg] = 0
  return kl_bfp_TMA_SRC.replace("@@TMA_DISPATCH@@", "\n  ".join(keys)).replace("@@TMA_NAME@@", "skinny_tma_a")


kl_bfp_GEMM_CONFIGS = {
    ("pooler", 64): (2, 4, 8), ("unpooler", 64): (2, 2, 8), ("qkv_gate", 64): (2, 7, 8), ("o", 64): (2, 2, 8),
    ("router", 64): (2, 32, 8),
    ("pooler", 128): (2, 4, 6), ("unpooler", 128): (2, 2, 6), ("qkv_gate", 128): (2, 7, 6), ("o", 128): (2, 2, 6),
    ("router", 128): (2, 32, 6),
    ("pooler", 256): (2, 4, 4), ("unpooler", 256): (2, 2, 4), ("qkv_gate", 256): (2, 7, 4), ("o", 256): (2, 2, 4),
    ("router", 256): (2, 32, 4),
}


kl_bfp_DIAG_LAYERS = bfp_DIAG_LAYERS
kl_bfp_GEMM_DIAG = bfp_GEMM_DIAG


kl_bfp_swizzle_tiles = bfp_swizzle_tiles


def kl_bfp__gemm_src(sm90):
  lines = []
  for (layer, bn), (nwg, splits, stages) in sorted(kl_bfp_GEMM_CONFIGS.items()):
    f32 = layer == "router"
    ot = "float" if f32 else "__nv_bfloat16"
    for a_s in ((0, 1) if (not f32 and kl_bfp_GEMM_DIAG) else (0,)):
      lines.append(f"if (nwg == {nwg} && bn == {bn} && out_f32 == {str(f32).lower()} && as == {a_s}) "
                   f"{{ launch_gemm90<{nwg}, {bn}, {stages}, {str(bool(a_s)).lower()}, {ot}>(W, Wst, x, out, ws, counters, M, N, Kp, Kx, splits); return; }}")
  bns = sorted({bn for (_, bn) in kl_bfp_GEMM_CONFIGS})
  return (kl_bfp_GEMM_SRC.replace("@@WGMMA_SS_DEFS@@", kl_bfp_wgmma_ss_defs(bns))
          .replace("@@GEMM_DISPATCH@@", "\n  ".join(dict.fromkeys(lines))))

kl_bfp_GQ_CONFIGS = {
    ("pooler", 64): (1, 2, 10), ("pooler", 128): (1, 2, 7),
}
kl_bfp_USE_GQ = bfp_USE_GQ


def kl_bfp__gq_src(sm90):
  lines = []
  for (layer, bn), (nwg, cl, ns) in sorted(kl_bfp_GQ_CONFIGS.items()):
    lines.append(f"if (bn == {bn} && nwg == {nwg} && cl == {cl} && nstage == {ns}) "
                 f"{{ launch_gq<{bn}, {nwg}, {ns}, {cl}>(W, x, out, M, N, K); return; }}")
  return kl_bfp_GQ_SRC.replace("@@GQ_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


kl_bfp__gq_bn = bfp__gq_bn


def kl_bfp__run_gq(x, layer, m, n, k, state):
  nwg, cl, ns = kl_bfp_GQ_CONFIGS[(layer, kl_bfp__gq_bn(m))]
  out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
  kl_bfp__get_ext().gq(state["raw"][layer], x, out, m, n, k, kl_bfp__gq_bn(m), nwg, cl, ns)
  return out


kl_bfp_WIDE_CONFIGS = {
    ("pooler", 64, 1): (1, 2, 8), ("unpooler", 64, 1): (1, 1, 8), ("qkv_gate", 64, 1): (1, 3, 8), ("o", 64, 1): (1, 1, 8),
    ("pooler", 128, 1): (1, 2, 7), ("unpooler", 128, 1): (1, 1, 7), ("qkv_gate", 128, 1): (1, 3, 7), ("o", 128, 1): (1, 1, 7),
    ("pooler", 128, 2): (2, 1, 5), ("unpooler", 128, 2): (4, 1, 4), ("qkv_gate", 128, 2): (1, 1, 7), ("o", 128, 2): (4, 1, 4),
    ("router", 64, 1): (1, 32, 4), ("router", 128, 1): (1, 32, 4), ("router", 128, 2): (1, 16, 4),
}


kl_bfp__wide_key = bfp__wide_key


def kl_bfp__wide_src(sm90):
  lines = []
  for (layer, tok, nz), (nwg, splits, stages) in sorted(kl_bfp_WIDE_CONFIGS.items()):
    hl = layer == "router"
    ot = "float" if hl else "__nv_bfloat16"
    lines.append(f"if (nwg == {nwg} && tok == {tok} && hl == {int(hl)}) "
                 f"{{ launch_wide<{nwg}, {tok}, {stages}, {str(hl).lower()}, {ot}>(Wst, x, out, ws, counters, ptab, pmax, base4, "
                 f"M, N, K, splits); return; }}")
  return kl_bfp_WIDE_SRC.replace("@@WIDE_DISPATCH@@", "\n  ".join(dict.fromkeys(lines)))


def kl_bfp__get_ext():
  global kl_bfp__ext
  if kl_bfp__ext is None:
    major, minor = torch.cuda.get_device_capability()
    sm90 = major >= 9
    arch = f"{major}{minor}a" if sm90 else f"{major}{minor}"
    common = kl_bfp_CUDA_SRC.replace("@@MBAR@@", kl_bfp_MBAR_HELPERS).replace("@@BODY@@", kl_bfp_WG_HELPERS if sm90 else "")
    defs = kl_bfp_wgmma_defs([8, 16, 32, 64, 128]) if sm90 else ""
    cuda_src = (common + defs + kl_bfp__tma_srcs(sm90) + kl_bfp_MAP_SRC + kl_bfp__gemm_src(sm90) + kl_bfp__wide_src(sm90) + kl_bfp__sk2_src(sm90)
                + kl_bfp__gq_src(sm90))
    kl_bfp__ext = kl_load_inline(
        name=f"dl_okc_bf16proj_v86y4_{arch}" + (f"_dly{kl_bfp__SK2_DELAY}" if kl_bfp__SK2_DELAY else ""),
        cpp_sources="#include <torch/extension.h>\n" + kl_bfp_CPP_SRC,
        cuda_sources=cuda_src,
        functions=["skinny_tma_a", "gemm90", "wide", "sk2", "gq", "sk2f", "sk2_set_pdl"],
        extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"] + ([f"-DSK2_DELAY_ESC={kl_bfp__SK2_DELAY}"] if kl_bfp__SK2_DELAY else []),
        no_implicit_headers=True,
        verbose=False,
    )
  return kl_bfp__ext


kl_bfp_pack_fragments = bfp_pack_fragments


kl_bfp_pack_compressed = bfp_pack_compressed


kl_bfp_build_patch_table = bfp_build_patch_table


kl_bfp_build_stream = bfp_build_stream


kl_bfp_SK2_CONFIGS = [
    ('o32h_r100', 'o', 8192, 32, 2, 2, 2, 2, 1, 1),
    ('o8h_r88', 'o', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('pooler', 'pooler', 4096, 32, 4, 1, 8, 1, 2, 1),
    ('pooler32_r144', 'pooler', 4096, 32, 2, 2, 2, 2, 1, 1),
    ('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1),
    ('qkv16_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv32_r208', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1),
    ('router', 'router', 256, 8, 1, 1, 8, 1, 4, 3),
    ('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1),
    ('unpooler16h_r100', 'unpooler', 8192, 32, 4, 1, 4, 1, 2, 1),
    ('unpooler32_r144', 'unpooler', 8192, 64, 4, 2, 2, 2, 1, 1),
]
kl_bfp_SK2_PICK = {
    ('o', 8): 'o8h_r88',
    ('o', 16): 'o8h_r88',
    ('o', 32): 'o32h_r100',
    ('pooler', 8): 'pooler8h_r72',
    ('pooler', 16): 'pooler',
    ('pooler', 32): 'pooler32_r144',
    ('qkv_gate', 8): 'qkv8_r176',
    ('qkv_gate', 16): 'qkv16_r176',
    ('qkv_gate', 32): 'qkv32_r208',
    ('router', 8): 'router',
    ('router', 16): 'router',
    ('router', 32): 'router',
    ('unpooler', 8): 'slice8_r192',
    ('unpooler', 16): 'unpooler16h_r100',
    ('unpooler', 32): 'unpooler32_r144',
}
kl_bfp_SK2_K = {"pooler": 8192, "unpooler": 4096, "o": 1024, "qkv_gate": 8192, "router": 8192}
kl_bfp_USE_SK2 = bfp_USE_SK2
kl_bfp_WIDE_BF16 = bfp_WIDE_BF16


def kl_bfp__sk2_name(layer, m):
  return kl_bfp_SK2_PICK[(layer, kl_bfp__sk2_tokr(m))]


kl_bfp__sk2_tokr = bfp__sk2_tokr


def kl_bfp__sk2_geom(cfg, tokr):
  name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
  hl = layer == "router"
  tu = 2048 if hl else 768
  ups = WSK * UPW
  wby = NWR * WSK * UPW * NT * tu
  xby = ups * tokr * 128
  sst = (xby + wby + 1023) // 1024 * 1024
  nst = -(-(kl_bfp_SK2_K[layer] // (ups * 64)) // CL)
  mt = 2 if tokr == 32 else 1
  toks = 16 * mt
  cp = R + 1
  nesc = 4 if not hl else 1
  rp = R + ((40 - (R % 32)) % 32)
  extra = (1024 + 16 * 16 + 32 + nesc * ((toks * cp + 3) // 4 * 4) * 4 + ((CL - 1) * WSK * toks * rp * 4 if CL > 1 else 0)
           + 64 * 8 + 128)
  cap = 200
  tail = name.rsplit("_r", 1)
  if len(tail) == 2 and tail[1].isdigit():
    cap = int(tail[1])
  ring = min(cap * 1024, 232448 - extra)
  nstage = max(2, min(16, nst, ring // sst))
  return hl, nstage


def kl_bfp__sk2_src(sm90):
  lines = []
  for ci, cfg in enumerate(kl_bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    for tokr in (8, 16, 32):
      if kl_bfp_SK2_PICK.get((layer, tokr)) != name:
        continue
      hl, nstage = kl_bfp__sk2_geom(cfg, tokr)
      mt = 2 if tokr == 32 else 1
      ot = "float" if hl else "__nv_bfloat16"
      lines.append(f"if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, "
                   f"{str(not hl).lower()}, {str(hl).lower()}, {1 if (mt == 2 and not hl and nstage >= 4 and NT == 4) else NACC}, {CL}, {ot}>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}")
  flines = []
  for ci, cfg in enumerate(kl_bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    if name != "pooler":
      continue
    for tokr in (8, 16, 32):
      hl, nstage = kl_bfp__sk2_geom(cfg, tokr)
      mt = 2 if tokr == 32 else 1
      nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
      flines.append(f"if (tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
                    f"{nacc}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                    f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                    f"rcnt.data_ptr<int>()); return; }}")
  _s = (kl_bfp_SK2_SRC.replace("@@SK2_DISPATCH@@", "\n  ".join(lines))
          .replace("@@SK2F_DISPATCH@@", "\n  ".join(flines)))
  return kl__norc16_patch(_s) if LDD19_NORC16 else _s   # LDD19: NORC16


kl_bfp_sk2_tileunits = bfp_sk2_tileunits


kl_bfp_sk2_compress = bfp_sk2_compress


kl_bfp_sk2_stream = bfp_sk2_stream


kl_bfp_sk2_table = bfp_sk2_table



def kl__norc16_patch(s):   # LDD19: NORC16 (t8_converter, run 2 it6): stock bfp sk2f pooler skips its router counter on request
  def one(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)
  s = "#ifndef LDD19_NORC16\n#define LDD19_NORC16 1\n#endif\n" + s
  s = one(s, "           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {\n",
          "           float* __restrict__ part, int* __restrict__ rcnt, int pdl_in) {\n"
          "  const int norc_ = LDD19_NORC16 ? ((pdl_in >> 28) & 1) : 0;   // LDD19: NORC16 (bit 28 of the pdl word)\n"
          "  const int pdl = pdl_in & ~(1 << 28);\n")
  s = one(s, "    if (tid == 0) {\n      int old;\n"
             "      asm volatile(\"atom.add.release.gpu.global.s32 %0, [%1], 1;\" : \"=r\"(old) : \"l\"(rcnt + rt) : \"memory\");\n",
          "    if (tid == 0) {\n      if (norc_) { s_rlast = 0; } else {   // LDD19: NORC16 (route sums the 4 router partials)\n      int old;\n"
          "      asm volatile(\"atom.add.release.gpu.global.s32 %0, [%1], 1;\" : \"=r\"(old) : \"l\"(rcnt + rt) : \"memory\");\n")
  s = one(s, "      if (s_rlast) asm volatile(\"fence.acq_rel.gpu;\" ::: \"memory\");\n    }\n",
          "      if (s_rlast) asm volatile(\"fence.acq_rel.gpu;\" ::: \"memory\");\n      }\n    }\n")
  s = one(s, "float* part = nullptr, int* rcnt = nullptr) {\n", "float* part = nullptr, int* rcnt = nullptr, int norc = 0) {   // LDD19: NORC16\n")
  s = one(s, "wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl | (norc << 28)));   // LDD19: NORC16")
  s = one(s, "wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, norc << 28);   // LDD19: NORC16")
  s = one(s, "int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {\n",
          "int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, int64_t tokr) {\n"
          "  const int norc = (int)(tokr >> 8); tokr &= 255;   // LDD19: NORC16 (bit 8 of tokr)\n")
  s = one(s, "rcnt.data_ptr<int>()); return; }", "rcnt.data_ptr<int>(), norc); return; }", 3)
  return s

def kl_bfp_sk2_prepare(weights, state):
  state["sk2"] = {}
  if not (kl_bfp_USE_SK2 and state["sm90"]):
    return
  dev = weights["o"].device
  shared = {}
  for ci, cfg in enumerate(kl_bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    w = weights[layer]
    if layer == "router":
      hi = w.to(torch.bfloat16)
      lo = (w - hi.float()).to(torch.bfloat16)
      T, U = w.shape[0] // 8, w.shape[1] // 64
      tub = torch.cat([kl_bfp_sk2_tileunits(hi).contiguous().view(torch.uint8).reshape(T, U, 1024),
                       kl_bfp_sk2_tileunits(lo).contiguous().view(torch.uint8).reshape(T, U, 1024)], dim=-1)
      stream = kl_bfp_sk2_stream(tub, n_rows, R, NT, NWR, WSK, UPW)
      tab, pmax, b4 = state["dummy"], 0, 0
    else:
      comp = state.setdefault("sk2_comp", {}).get(layer)
      if comp is None:
        comp = kl_bfp_sk2_compress(w.contiguous())
        state["sk2_comp"][layer] = comp
      if comp is None:
        continue
      lkey = (layer, n_rows, R, NT, NWR, WSK, UPW)
      if lkey not in shared:
        shared[lkey] = (kl_bfp_sk2_stream(comp["packed"], n_rows, R, NT, NWR, WSK, UPW),) + tuple(kl_bfp_sk2_table(comp, n_rows, R))
      stream, tab, pmax = shared[lkey]
      b4 = comp["base4"]
    state["sk2"][name] = (ci, stream, tab, pmax, b4, n_rows)
  state.pop("sk2_comp", None)


def kl_bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):
  name = kl_bfp__sk2_name(layer, m)
  ci, stream, tab, pmax, b4, n_rows = state["sk2"][name]
  out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
  ext = kl_bfp__get_ext()
  ext.sk2_set_pdl(kl_bfp_SK2_PDL)
  ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, kl_bfp__sk2_tokr(m), dg, state["dummy"] if dbg is None else dbg)
  return out


def kl_bfp_fused_pooler_router(x, state, norc=False):   # LDD19: NORC16 kwarg
  m = x.shape[0]
  assert m <= 32 and state.get("sm90") and "pooler" in state["sk2"] and "router" in state["sk2"]
  ci, stream, tab, pmax, b4, n_rows = state["sk2"]["pooler"]
  rstream = state["sk2"]["router"][1]
  if "fused_part" not in state:
    state["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=x.device)
    state["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=x.device)
  out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
  outr = torch.empty((m, 256), dtype=torch.float32, device=x.device)
  ext = kl__er_ext() if (LDD19_ER32 and (kl_bfp__sk2_tokr(m) == 32 or (LDD19_ER_TN and LDD19_ER_TN16 and kl_bfp__sk2_tokr(m) == 16))) else kl_bfp__get_ext()   # LDD19: ER32, TN2
  ext.sk2_set_pdl(kl_bfp_SK2_PDL)
  ext.sk2f(stream, rstream, x, out, outr, tab, pmax, b4, state["fused_part"], state["fused_cnt"], m, kl_bfp__sk2_tokr(m) | (256 if norc else 0))   # LDD19: NORC16
  return out, (state["fused_part"] if norc else outr)   # LDD19: NORC16 (1-D partials)


kl_bfp_SK2_PDL = bfp_SK2_PDL


kl_bfp__mt_key = bfp__mt_key


def kl_bfp_prepare(weights):
  ext = kl_bfp__get_ext()
  dev = weights["qkv_gate"].device
  sm90_ = torch.cuda.get_device_capability()[0] >= 9
  old_paths = not (sm90_ and kl_bfp_USE_SK2)
  state = {"raw": {}, "cnt": {}, "comp": {}, "mid_cnt": {}, "streams": {}, "tabs": {}, "base4": {}}
  state["dummy"] = torch.zeros(2, dtype=torch.int32, device=dev)
  ws_floats = 0
  for (lname, mtk), (splits, rgc, wsn, ks) in kl_bfp_TMA_CONFIGS.items():
    mmax = {0: 1, 1: 8, 2: 16, 4: 32, 8: 64}[mtk]
    ws_floats = max(ws_floats, splits * mmax * kl_bfp_w_rows[lname])
  for (lname, mm), (splits, rgc, wsn, ks) in kl_bfp_TMA_ALT.items():
    ws_floats = max(ws_floats, splits * mm * kl_bfp_w_rows[lname])
  state["ws_shared"] = torch.empty(ws_floats, dtype=torch.float32, device=dev)
  gemm_floats = 0
  for (lname, bn), (nwg, splits, stages) in kl_bfp_GEMM_CONFIGS.items():
    gemm_floats = max(gemm_floats, (kl_bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 128) * (bn // 2))
  for (lname, tok, nz), (nwg, splits, stages) in kl_bfp_WIDE_CONFIGS.items():
    gemm_floats = max(gemm_floats, nz * (kl_bfp_w_rows[lname] // (64 * nwg)) * splits * (nwg * 64) * tok)
  state["mid_ws_shared"] = torch.empty(gemm_floats, dtype=torch.float32, device=dev)
  if kl_bfp_GEMM_DIAG:
    state["ws_shared_diag"] = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=dev)
  for name, w in weights.items():
    if name == "router":
      hi = w.to(torch.bfloat16)
      lo = (w - hi.float()).to(torch.bfloat16)
      state["router_hi_t"] = hi.t().contiguous()
      state["router_lo_t"] = lo.t().contiguous()
      state["router_cat"] = torch.cat([hi, lo], dim=1).contiguous()
      src = torch.stack([kl_bfp_pack_fragments(hi), kl_bfp_pack_fragments(lo)], dim=2).contiguous()
      comp, kk = False, w.shape[1]
      c = None
      nrg_r = w.shape[0] // 16
      u_all = w.shape[1] // 64
      t = src.view(nrg_r, u_all, 4, 2, 32, 8).permute(0, 1, 3, 2, 4, 5).contiguous()
      nwg_r = kl_bfp_WIDE_CONFIGS[("router", 128, 1)][0]
      t = t.view(nrg_r // (4 * nwg_r), 4 * nwg_r, u_all, 2 * 4 * 32 * 8).permute(0, 2, 1, 3).contiguous()
      state["streams"][("router", 0, 4 * nwg_r, 1, 4, w.shape[0])] = t
      for (lname, tok, nz), (nwg, splits, stages) in kl_bfp_WIDE_CONFIGS.items():
        if lname == "router":
          state["cnt"][("router", "wide", tok, nz)] = torch.zeros(w.shape[0] // 16, dtype=torch.int32, device=dev)
    else:
      w = w.contiguous()
      state["raw"][name] = w
      c = kl_bfp_pack_compressed(w) if name in kl_bfp_COMPRESS else None
      if c is not None:
        src, comp = c["packed"].transpose(0, 1).contiguous(), True
        state["base4"][name] = c["base4"]
        del c["packed"]
      else:
        src, comp = kl_bfp_pack_fragments(w), False
      kk = w.shape[1]
    n = w.shape[0]
    state["comp"][name] = comp
    for (lname, mtk), (splits, rgc, wsn, ks) in kl_bfp_TMA_CONFIGS.items():
      if lname != name or not old_paths:
        continue
      n_cfg = 1024 if (name == "unpooler" and mtk <= 1) else n
      key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
      if key not in state["streams"]:
        state["streams"][key] = kl_bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
      ckey = (name, splits, rgc, wsn, ks, n_cfg)
      if ckey not in state["cnt"]:
        state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
      if comp and (name, splits, rgc, n_cfg) not in state["tabs"]:
        state["tabs"][(name, splits, rgc, n_cfg)] = kl_bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
    if comp and old_paths:
      for (lname, mm), (splits, rgc, wsn, ks) in kl_bfp_TMA_ALT.items():
        if lname != name:
          continue
        n_cfg = 1024 if (name == "unpooler" and mm <= 8) else n
        key = (name, 0 if wsn == 1 else splits, rgc, wsn, ks, n_cfg)
        if key not in state["streams"]:
          state["streams"][key] = kl_bfp_build_stream(src, comp, n_cfg, kk, splits, rgc, wsn, ks)
        ckey = (name, splits, rgc, wsn, ks, n_cfg)
        if ckey not in state["cnt"]:
          state["cnt"][ckey] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
        if (name, splits, rgc, n_cfg) not in state["tabs"]:
          state["tabs"][(name, splits, rgc, n_cfg)] = kl_bfp_build_patch_table(c, n_cfg, kk, splits, rgc)
    for (lname, tok, nz), (nwg, splits, stages) in kl_bfp_WIDE_CONFIGS.items():
      if lname != name or not comp:
        continue
      rgc = 4 * nwg
      key = (name, 0, rgc, 1, 4, n)
      if key not in state["streams"]:
        state["streams"][key] = kl_bfp_build_stream(src, comp, n, kk, splits, rgc, 1, 4)
      if (name, splits, rgc, n) not in state["tabs"]:
        state["tabs"][(name, splits, rgc, n)] = kl_bfp_build_patch_table(c, n, kk, splits, rgc)
      state["cnt"][(name, "wide", tok, nz)] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
    del src
    state["mid_cnt"][name] = torch.zeros(n // 16, dtype=torch.int32, device=dev)
  state["sm90"] = sm90_
  state["old_paths"] = old_paths
  kl_bfp_sk2_prepare(weights, state)
  if kl_bfp_GEMM_DIAG and state["sm90"]:
    state["wtile"], state["diag_cnt"] = {}, {}
    for layer in kl_bfp_DIAG_LAYERS:
      state["wtile"][layer] = kl_bfp_swizzle_tiles(state["raw"][layer], 128)
      for a_s in (0, 1):
        for sp_ in sorted({kl_bfp_GEMM_CONFIGS[(layer, 128)][1], kl_bfp_GEMM_CONFIGS[(layer, 256)][1], 1}):
          state["diag_cnt"][(layer, a_s, sp_)] = torch.zeros(kl_bfp_w_rows[layer] // 16, dtype=torch.int32, device=dev)
  kl_bfp__selftest(weights, state)
  kl_bfp__autotune(weights, state)
  torch.cuda.empty_cache()
  return state


kl_bfp_ROUTE_BF16 = [(32, "skinny"), (256, "wide")]
kl_bfp_WIDE_SM80_TEST = bfp_WIDE_SM80_TEST
kl_bfp_ROUTE_ROUTER = [(32, "skinny"), (256, "wide")]


def kl_bfp__route(layer, m, sm90):
  for mmax, r in (kl_bfp_ROUTE_ROUTER if layer == "router" else kl_bfp_ROUTE_BF16):
    if m <= mmax:
      if r == "gemm" and not sm90:
        return "torch"
      if r == "wide" and not (sm90 or kl_bfp_WIDE_SM80_TEST):
        return "torch"
      return r
  return "torch"


def kl_bfp__route_key(layer, m, route):
  if route == "sk2":
    return (layer, route, kl_bfp__sk2_name(layer, m), kl_bfp__sk2_tokr(m))
  if route == "skinny":
    return (layer, route, kl_bfp__mt_key(m))
  if route == "wide":
    return (layer, route) + kl_bfp__wide_key(m)
  return (layer, route, 64 if m <= 64 else 128 if m <= 128 else 256)


kl_bfp_SELFTEST_M = bfp_SELFTEST_M


def kl_bfp__selftest(weights, state):
  state["disabled"] = set()
  g = torch.Generator(device=weights["o"].device)
  g.manual_seed(1234)
  for layer, w in weights.items():
    n, k = w.shape
    for m in kl_bfp_SELFTEST_M:
      nn = 1024 if (layer == "unpooler" and m <= 8) else n
      x = torch.randn(m, k, device=w.device, generator=g).to(torch.bfloat16)
      ref = x.double() @ w[:nn].double().T
      if layer != "router":
        ref = ref.to(torch.bfloat16).double()
      tol = 2e-5 if layer == "router" else 5e-3
      for ch in kl_bfp__candidates(layer, m, state):
        if ch[0] == "torch":
          continue
        key = (("skinny_alt", layer, m) if (ch[0] == "skinny" and ch[2][0] == "alt")
               else kl_bfp__route_key(layer, m, ch[0]))
        try:
          out = kl_bfp__run(ch, x, layer, m, nn, k, state).double()
          torch.cuda.synchronize()
          err = ((out - ref).norm() / ref.norm()).item()
        except RuntimeError:
          err = float("inf")
        if not err < tol:
          state["disabled"].add(key)


def kl_bfp__run_skinny(x, layer, m, n, k, state, cfg, tu_key, dg=0):
  splits, rg, wsn, ks = cfg
  mtk = kl_bfp__mt_key(m)
  out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
  stream = state["streams"][(layer, 0 if wsn == 1 else splits, rg, wsn, ks, n)]
  cnt = state["cnt"][(layer, splits, rg, wsn, ks, n)]
  fn = kl_bfp__get_ext().skinny_tma_a
  if state["comp"][layer]:
    tab, pmax = state["tabs"][(layer, splits, rg, n)]
    fn(stream, x, out, state["ws_shared"], cnt, tab, pmax, state["base4"][layer], m, n, k, k, splits,
       max(mtk, 1), rg, wsn, 1, kl_bfp_w_rows[layer] // 16, ks, dg)
  else:
    fn(stream, x, out, state["ws_shared"], cnt, state["dummy"], 2, 0, m, n, k, k, splits,
       max(mtk, 1), rg, wsn, 0, kl_bfp_w_rows[layer] // 16, ks, dg)
  return out


def kl_bfp__run_wide(x, layer, m, n, k, state):
  tok, nz = kl_bfp__wide_key(m)
  nwg, splits, _ = kl_bfp_WIDE_CONFIGS[(layer, tok, nz)]
  rgc = 4 * nwg
  if layer == "router":
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    tab, pmax, b4 = state["dummy"], 0, 0
  else:
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    (tab, pmax), b4 = state["tabs"][(layer, splits, rgc, n)], state["base4"][layer]
  kl_bfp__get_ext().wide(state["streams"][(layer, 0, rgc, 1, 4, n)], x, out, state["mid_ws_shared"],
                  state["cnt"][(layer, "wide", tok, nz)], tab, pmax, b4, m, n, k, splits, nwg, tok)
  return out


def kl_bfp__run_gemm(x, layer, m, n, k, state):
  bn = 64 if m <= 64 else 128 if m <= 128 else 256
  nwg, splits, _ = kl_bfp_GEMM_CONFIGS[(layer, bn)]
  out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
  w = state["router_cat"] if layer == "router" else state["raw"][layer]
  kl_bfp__get_ext().gemm90(w, state["dummy"], x, out, state["mid_ws_shared"], state["mid_cnt"][layer], m, n, w.shape[1], k,
                    splits, nwg, bn, 0)
  return out


kl_bfp__run_torch = bfp__run_torch


def kl_bfp__run(choice, x, layer, m, n, k, state):
  kind = choice[0]
  if kind == "skinny":
    return kl_bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2])
  if kind == "wide":
    return kl_bfp__run_wide(x, layer, m, n, k, state)
  if kind == "sk2":
    return kl_bfp__run_sk2(x, layer, m, n, k, state)
  if kind == "gemm":
    return kl_bfp__run_gemm(x, layer, m, n, k, state)
  if kind == "gq":
    return kl_bfp__run_gq(x, layer, m, n, k, state)
  return kl_bfp__run_torch(x, layer, m, n, k, state)


def kl_bfp__static_choice(layer, m, state):
  route = kl_bfp__route(layer, m, state["sm90"])
  if not state.get("old_paths", True):
    if route == "skinny":
      route = "sk2" if (kl_bfp__sk2_name(layer, m) in state.get("sk2", {})
                        and kl_bfp__route_key(layer, m, "sk2") not in state.get("disabled", ())) else "torch"
    elif route == "wide" and layer != "router" and not kl_bfp_WIDE_BF16:
      route = "torch"
  if route != "torch" and kl_bfp__route_key(layer, m, route) in state.get("disabled", ()):
    route = "gemm" if (layer == "router" and route == "wide" and state["sm90"]
                       and kl_bfp__route_key(layer, m, "gemm") not in state["disabled"]) else "torch"
  if route == "skinny":
    mtk = kl_bfp__mt_key(m)
    return ("skinny", kl_bfp_TMA_CONFIGS[(layer, mtk)], (layer, mtk))
  return (route,)


def kl_bfp__candidates(layer, m, state):
  c = [kl_bfp__static_choice(layer, m, state)]
  if c[0][0] == "skinny" and (layer, m) in kl_bfp_TMA_ALT and ("skinny_alt", layer, m) not in state["disabled"]:
    c.append(("skinny", kl_bfp_TMA_ALT[(layer, m)], ("alt", layer, m)))
  if (c[0][0] == "wide" and layer == "router" and state["sm90"]
          and kl_bfp__route_key(layer, m, "gemm") not in state["disabled"]):
    c.append(("gemm",))
  if (m <= 32 and c[0][0] != "sk2" and kl_bfp__sk2_name(layer, m) in state.get("sk2", {})
          and kl_bfp__route_key(layer, m, "sk2") not in state["disabled"]):
    c.insert(0, ("sk2",))
  if (kl_bfp_USE_GQ and state["sm90"] and 32 < m <= 256 and layer != "router" and (layer, kl_bfp__gq_bn(m)) in kl_bfp_GQ_CONFIGS
          and kl_bfp__route_key(layer, m, "gq") not in state["disabled"]):
    c.insert(0, ("gq",))
  if c[-1][0] != "torch" and all(ch[0] != "torch" for ch in c):
    c.append(("torch",))
  return c


kl_bfp_AUTOTUNE = bfp_AUTOTUNE
kl_bfp_TUNE_M = bfp_TUNE_M


def kl_bfp__autotune(weights, state):
  state["choice"] = {}
  if not kl_bfp_AUTOTUNE:
    return
  dev = weights["o"].device
  flush = torch.ones(64 * 1024 * 1024 // 1024, 1024, dtype=torch.float32, device=dev)
  sink = torch.empty(1024, dtype=torch.float32, device=dev)
  g = torch.Generator(device=dev)
  g.manual_seed(4321)
  reps = 24
  for m in kl_bfp_TUNE_M:
    for layer, w in weights.items():
      n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
      k = w.shape[1]
      cands = kl_bfp__candidates(layer, m, state)
      if len(cands) < 2:
        continue
      x = torch.randn(m, k, device=dev, generator=g).to(torch.bfloat16)
      graphs = []
      for ch in cands:
        kl_bfp__run(ch, x, layer, m, n, k, state)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
          kl_bfp__run(ch, x, layer, m, n, k, state)
        graphs.append(gr)
      evs = []
      for r in range(reps):
        order = range(len(cands)) if r % 2 == 0 else reversed(range(len(cands)))
        for i in order:
          torch.sum(flush, dim=0, out=sink)
          e0 = torch.cuda.Event(enable_timing=True)
          e1 = torch.cuda.Event(enable_timing=True)
          e0.record()
          graphs[i].replay()
          e1.record()
          evs.append((i, e0, e1))
      torch.cuda.synchronize()
      per = [[] for _ in cands]
      for i, e0, e1 in evs:
        per[i].append(e0.elapsed_time(e1))
      med = [sorted(v)[len(v) // 2] for v in per]
      best = min(range(len(cands)), key=lambda i: med[i])
      state["choice"][(layer, m)] = cands[best]
      state.setdefault("tune_log", []).append((layer, m, [(c[0], round(1000 * v, 2)) for c, v in zip(cands, med)]))
      del graphs
  del flush
  torch.cuda.synchronize()


def kl_bfp__nearby_choice(layer, m, state):
  tm = 1 if m <= 1 else 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 128 if m <= 128 else 256 if m <= 256 else 0
  if not tm:
    return None
  ch = state.get("choice", {}).get((layer, tm))
  if ch is None:
    return None
  if ch[0] == "gq" and (layer, kl_bfp__gq_bn(m)) not in kl_bfp_GQ_CONFIGS:
    return None
  return ch


def kl_bfp_kernel_fn(data, state):
  x = data["x"]
  cfg = data["config"]
  layer, m, n, k = cfg["layer"], cfg["m"], cfg["n"], cfg["k"]
  choice = state.get("choice", {}).get((layer, m)) or kl_bfp__nearby_choice(layer, m, state) or kl_bfp__static_choice(layer, m, state)
  out = kl_bfp__run(choice, x, layer, m, n, k, state)
  if kl_bfp_GEMM_DIAG:
    kl_bfp__diag(x, layer, m, n, k, state, choice)
  return out


def kl_bfp__diag(x, layer, m, n, k, state, choice):
  if choice[0] == "skinny" and m == 1 and layer in ("o", "unpooler") and state["comp"][layer]:
    for dg in (1, 2, 3):
      kl_bfp__run_skinny(x, layer, m, n, k, state, choice[1], choice[2], dg)


def kl__bfp_prepare(weights, st):
  st["bfp"] = None
  try:
    w5 = {k: weights[k].contiguous() for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    st["bfp"] = kl_bfp_prepare(w5)
  except Exception as ex:
    print("bfp disabled:", repr(ex)[:160], flush=True)


kl__BFP_LAYER = {"qkv": "qkv_gate", "o": "o", "un": "unpooler", "pool": "pooler", "router": "router"}


kl_BFP_PDL_N = BFP_PDL_N


kl_PCX_CFG = {}


def kl__pcx_val(role, T):
  P, d0, g, e = (tuple(kl_PCX_CFG.get((role, T), kl_PCX_CFG.get((role, 0), (kl_BFP_PDL_N, 0, 0, 0)))) + (0,))[:4]
  return (1 << 30) | P | (d0 << 4) | (g << 9) | (e << 14)


def kl__bfp_pdl(T, role=None):
  global kl_bfp_SK2_PDL
  kl_bfp_SK2_PDL = kl__pcx_val(role, T) if kl__pdl(T) else 0


def kl__bfp_gemm(st, name, x, m, n, k):
  kl__bfp_pdl(m, name)
  return kl_bfp_kernel_fn({"x": x, "config": {"layer": kl__BFP_LAYER[name], "m": m, "n": n, "k": k}}, st["bfp"])


kl__MA_MMAX = _MA_MMAX
kl__MA_EMAX = kl__MA_MMAX * 8


kl_WGS_FIXF = 2 * 160 * 4 * 4096 + 2 * 160 * 4 + 136 * 16
kl_WGS_XROWS = kl__MA_EMAX + 8 * (32 + kl__MA_EMAX // 32 + 8)


def kl__df_inter(inter):
  inter[128 * 4096:256 * 4096].view(torch.int16).fill_(-1)
  return inter


def kl__ma_make_scratch(dev):
  return [torch.zeros(8 + 1024 + kl__MA_EMAX + 4096, dtype=torch.int32, device=dev),
          torch.zeros(kl__MA_EMAX + kl__MA_EMAX * 4096 + kl_WGS_FIXF, dtype=torch.float32, device=dev),
          kl__df_inter(torch.zeros((kl__MA_EMAX + 2 * kl_WGS_XROWS) * 4096, dtype=torch.bfloat16, device=dev))]

kl_SCRATCH_GUARD = 1 << 16
kl_SCRATCH_SENTINEL = SCRATCH_SENTINEL


def kl__ma_prepare(weights, st):
  dev = weights["w13"].device
  I = H = 4096
  rb = torch.arange(I // 8, device=dev)
  g = torch.arange(8, device=dev)
  rowsA = torch.stack((8 * rb[:, None] + g[None, :], I + 8 * rb[:, None] + g[None, :]), 1).reshape(-1)
  rowsB = torch.arange(H, device=dev)
  wA, sA, wB, sB = [], [], [], []
  for e in range(32):
    wA.append(kl__pack_codes(weights["w13"][e].view(torch.uint8), rowsA))
    sA.append(kl__pack_scales(weights["w13_sf"][e], rowsA))
    wB.append(kl__pack_codes(weights["w2"][e].view(torch.uint8), rowsB))
    sB.append(kl__pack_scales(weights["w2_sf"][e], rowsB))
  lut = (torch.arange(256, dtype=torch.int32, device=dev).to(torch.uint8).view(torch.float8_e4m3fn).float()
         * 2.0**64).to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
  lut = (lut | (lut << 16)).to(torch.int32)
  st["ma_w"] = [torch.cat(wA), torch.cat(sA), weights["w13_gs"].float().contiguous(),
                torch.cat(wB), torch.cat(sB), weights["w2_gs"].float().contiguous(),
                lut.contiguous(), weights["post_expert_norm"].contiguous(),
                torch.cat([kl__fat_order(x, y) for x, y in zip(wA, sA)]),
                torch.cat([kl__fat_order(x, y) for x, y in zip(wB, sB)]),
                torch.cat([kl__unit_order(x, y) for x, y in zip(wA, sA)]),
                torch.cat([kl__unit_order(x, y) for x, y in zip(wB, sB)])]
  if "kl__wgs_on" in globals() and kl__wgs_on():
    st["ma_w"].append(torch.cat([kl__wgs_copy(weights["w13"][e].view(torch.uint8), weights["w13_sf"][e], kl__kperm_rows(rowsA), kl__MA_SC6W)
                                 for e in range(32)]))
    st["ma_w"].append(torch.cat([kl__wgs_copy(weights["w2"][e].view(torch.uint8), weights["w2_sf"][e], rowsB, kl__MA_SC6W)
                                 for e in range(32)]))
  if LDD19_GF2S:   # LDD19: GF2S: the mixed unit copies take the slots of the raw packed codes ma_w[0] / [3]; those and the raw
    # scales ma_w[1] / [4] are never read by any Keell kernel (p.wA / p.sA / p.wB / p.sB are assigned, not used), so the
    # resident memory stays ~equal (+0.9 GB measured -0.15% six-load in work_moe_engineer/ab_6.json).
    st["ma_w"][0] = kl__gf2s_units(st["ma_w"][10])
    st["ma_w"][3] = kl__gf2s_units(st["ma_w"][11])
    if kl__MA_SC6M:   # LDD19: SC6M: only the MID build reads w[0] / w[3]
      st["ma_w"][0] = kl__sc6m_units(st["ma_w"][0])
      st["ma_w"][3] = kl__sc6m_units(st["ma_w"][3])
    st["ma_w"][1] = st["ma_w"][1][:16].clone()
    st["ma_w"][4] = st["ma_w"][4][:16].clone()
  st["ma_scratch"] = [kl__guarded(t) for t in kl__ma_make_scratch(dev)]


kl__to_i32 = _to_i32


kl__repack_q = _repack_q


kl__repack_s = _repack_s


kl__gemm_splitk = _gemm_splitk


kl__gemm_splitk_ng = _gemm_splitk_ng


kl__normgate = _normgate


def kl_dense_ng(x, gate, w, T, state=None):
  N, K = w.shape
  if T > 32 or (state is not None and state.get("mm_pick", {}).get((T, "o"), False)):
    xn = torch.empty((T, K), device=x.device, dtype=torch.bfloat16)
    kl__normgate[(T,)](x, gate, xn, num_warps=4, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
    return kl_dense(xn, w, T, state, "o")
  BN, BK = 64, 128
  BT = 16 if T <= 16 else 32
  S = 1
  P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
  grid = (N // BN, S, triton.cdiv(T, BT))
  kl__gemm_splitk_ng[grid](x, gate, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  return P, S


kl_TUNED_T = TUNED_T


kl__bucket = _bucket


kl_BFP_MAX_T = BFP_MAX_T


def kl_dense(x, w, T, state=None, name=None):
  N, K = w.shape
  pick = state.get("mm_pick", {}).get((T, name), False) if state is not None else False
  if pick == "bfp" and T <= kl_BFP_MAX_T and name in kl__BFP_LAYER and state.get("bfp") is not None:
    return kl__bfp_gemm(state, name, x, T, N, K).view(1, T, N), 1
  if pick is True or pick == "bfp":
    out = torch.mm(x, w.t(), out_dtype=torch.float32)
    return out.view(1, T, N), 1
  if T > 32:
    out = torch.mm(x, w.t(), out_dtype=torch.float32)
    return out.view(1, T, N), 1
  BN, BK = 64, 128
  BT = 16 if T <= 16 else 32
  tiles = (N // BN) * triton.cdiv(T, BT)
  S = 1
  while tiles * S * 2 <= 600 and K // (S * 2) >= 2 * BK:
    S *= 2
  P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
  grid = (N // BN, S, triton.cdiv(T, BT))
  kl__gemm_splitk[grid](x, w, P, T, N, K, KSPLIT=K // S, BN=BN, BT=BT, BK=BK, num_warps=4, num_stages=4, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  return P, S


kl__f32_to_e4m3 = _f32_to_e4m3


kl__qk_prep = _qk_prep


kl__attn_split = _attn_split


kl__attn_combine = _attn_combine


kl__gemm_o = _gemm_o


kl__bnd1_apply = _bnd1_apply


kl__route_a = _route_a


kl__route_b = _route_b


kl__bnd2_sums = _bnd2_sums


kl__bnd2_apply = _bnd2_apply


kl__gate_ab = _gate_ab


@triton.jit
def kl__boundary1(P, S: tl.constexpr, T, STREAM, A1, B1, G, S1OUT, HOUT, ROUTER_OUT, WRITE_ROUTER: tl.constexpr = False, USE_PDL: tl.constexpr = False):
  t = tl.program_id(0)
  offs = tl.arange(0, 8192)
  a1v, b1v = kl__gate_ab(A1, offs)
  stv = tl.load(STREAM + t * 8192 + offs)
  gw = 1.0 + tl.load(G + offs).to(tl.float32)
  if USE_PDL:
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
  a = tl.zeros([8192], dtype=tl.float32)
  for s in range(S):
    a += tl.load(P + (s * T + t) * 8192 + offs)
  n = a * tl.rsqrt(tl.sum(a * a, 0) / 8192.0 + 1e-8)
  s1 = a1v * stv + b1v * n
  tl.store(S1OUT + t * 8192 + offs, s1)
  hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
  tl.store(HOUT + t * 8192 + offs, hh.to(tl.bfloat16))
  if WRITE_ROUTER:
    tl.store(ROUTER_OUT + t * 8192 + offs, hh)


kl__route = _route


kl__route_split = _route_split




kl__boundary2 = _boundary2




kl__graph_time = _graph_time


kl__FLUSH_SINK = _FLUSH_SINK


def kl__tune(st, log=False):
  """LDD19: iter2 integrator: Keell's prepare-time tuner REMOVED (6.5 KB). Its MoE pick could only be "agent"
    (no Marlin bank in this pipeline), gluev2 only False (= the runtime default), and every mm pick it timed at the
    routed T (1, 8, 16, 32) is overwritten later in kl_prepare (static bfp picks); T = 2/4/64/128/256 never reach
    the Keell pipeline (HY_ROUTES). The original is kl__tune in external/keell0renz (and git history)."""
  st["moe_pick"], st["gluev2"], st["marlin_cfg"], st["mm_pick"] = {}, {}, {}, {}
  st["pbmax"] = 0 if kl_FORCE_PBMAX is None else kl_FORCE_PBMAX
  if log:
    print("tune: removed (LDD19 iter2 slim)", flush=True)


def kl_prepare(weights):
  dev = weights["qkv_gate"].device
  router = weights["router"].float()
  hi = router.to(torch.bfloat16)
  lo = (router - hi.float()).to(torch.bfloat16)
  st = {}
  st["router_fp32"] = router.contiguous()
  st["qkv"] = weights["qkv_gate"].contiguous()
  st["o"] = weights["o"].contiguous()
  st["pr"] = torch.cat([weights["pooler"], hi, lo], 0).contiguous()
  st["unpool"] = weights["unpooler"].contiguous()
  st["unpool_slice"] = weights["unpooler"][:1024].contiguous()
  st["gate1"] = weights["attn_residual_gate"].contiguous()
  st["gate2"] = weights["ffn_residual_gate"].contiguous()
  st["g_preffn"] = weights["pre_ffn_norm"].contiguous()
  st["g_preexp"] = weights["pre_expert_norm"].contiguous()
  st["g_postexp"] = weights["post_expert_norm"].contiguous()
  st["g_postffn"] = weights["post_ffn_norm"].contiguous()
  st["g_next"] = weights["next_pre_attn_norm"].contiguous()
  st["bias"] = weights["router_bias"].float().contiguous()
  st["w13q"] = kl__repack_q(weights["w13"].view(torch.uint8), True)
  st["w13s"] = kl__repack_s(weights["w13_sf"].view(torch.uint8), True)
  st["gs13"] = weights["w13_gs"].float().contiguous()
  st["w2q"] = kl__repack_q(weights["w2"].view(torch.uint8), False)
  st["w2s"] = kl__repack_s(weights["w2_sf"].view(torch.uint8), False)
  st["gs2"] = weights["w2_gs"].float().contiguous()
  st["invf0"] = torch.zeros(32, device=dev)
  st["invf"] = (1.0 / (kl_ROPE_THETA ** (torch.arange(0, kl_HEAD_DIM, 2, device=dev).float() / kl_HEAD_DIM))).contiguous()
  errs = []

  def _build(fn):
    try:
      fn()
    except Exception as ex:
      errs.append(ex)

  global kl__MA_SFX, kl__MA_SC6W, kl__MA_SC6M
  kl__MA_SFX = kl__MA_SFX_IF_PRMT if (kl__MA_PRMT_SCALES and kl__ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
  kl__MA_SC6W = bool(kl_SC6_WGS and kl__MA_SFX and kl__wgs_on() and kl__sc6_ok(weights["w13_sf"], weights["w2_sf"]))
  kl__MA_SC6M = bool(LDD19_KL_SC6M and kl__MA_SFX and kl__sc6_ok(weights["w13_sf"], weights["w2_sf"]))   # LDD19: SC6M
  print(f"[sc6] scales ok -> wgs sc6 {kl__MA_SC6W}", flush=True)
  st["ma_sfx"] = tuple(kl__MA_SFX)
  st["ma_ext"] = {}
  ths = [kl_threading.Thread(target=_build, args=(f,)) for f in (kl__att7, kl__att7q, kl__moe, kl__moe_tiny, kl__moe_tiny2, kl__moe_mid, kl_bfp__get_ext, kl__graft_glue_module, kl__graft_b2_module, kl__sl_cluster_module, lambda: kl__moe(32))
         if not (LDD19_SLIM_KL and f is kl__moe)]   # LDD19: iter2 slim: no main Keell MoE build (T > 32 never routed here)
  for th in ths:
    th.start()
  for th in ths:
    th.join()
  if errs:
    raise errs[0]
  kl__ma_prepare(weights, st)
  if kl__MA_SC6W and not LDD19_SLIM_KL:   # LDD19: iter2 slim: diagnostic print only (it loads the main build)
    try:
      for d in (tuple(kl_MOE_DEFINES), tuple(kl_MOE_DEFINES) + ("-DOKC_B4FIN=1",)):
        print("[sc6] regs/local/smem (<false>, <true>)", d[-1], kl__ma_ext(d + tuple(kl__MA_SFX)).okc_attrs(), flush=True)
    except Exception as ex:
      print(f"[sc6] attrs failed: {ex!r}"[:200], flush=True)
  kl__bfp_prepare(weights, st)
  st["nsm"] = torch.cuda.get_device_properties(dev).multi_processor_count
  st["fa3"] = None
  st["bank"] = None
  st["att_ok"] = True
  st["ma_ok"] = True
  st["pbmax"] = 2
  st["moe_pick"] = {}
  try:
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    st["fa3"] = flash_attn_with_kvcache
    st["bank"] = None
  except Exception as ex:
    print("fallback (no sgl_kernel/sglang):", repr(ex)[:200])
  try:
    kl__tune(st, log=kl__TUNE_LOG)
  except Exception as ex:
    print("tune failed:", repr(ex)[:200], flush=True)
    st["pbmax"] = 2
    st["moe_pick"] = {}
    st["gluev2"] = {}
    st["marlin_cfg"] = {}
    st["mm_pick"] = {}
  for k, v in kl_PIN_PICK.items():
    st.setdefault("mm_pick", {})[k] = v
  torch.cuda.synchronize()
  if st.get("att_ok"):
    kl__attn_selftest(st)
  kl__check_guards()
  return st


kl__attn_cfg = _attn_cfg


kl__INV_FREQ_CACHE = {}


def kl__inv_freq(theta, device):
  key = (float(theta), str(device))
  t = kl__INV_FREQ_CACHE.get(key)
  if t is None:
    if theta:
      t = 1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))
    else:
      t = torch.zeros(32, device=device)
    kl__INV_FREQ_CACHE[key] = t.float().contiguous()
  return t


def kl__launch7(data, dbg=None, _mod=None, pdl=None, **override):
  cfg = data["config"]
  B, L = cfg["batch"], cfg["q_len"]
  hq, hkv, d, P = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
  qkv = data["qkv"]
  device = qkv.device
  pairs = hkv // 2
  window = cfg["window"]
  page_table = data["page_table"]
  out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device=device)
  plan = kl__plan7(B, pairs, L, window, page_table.shape[1], P, device)
  tuned = kl__TUNED7.get((B, L, bool(window), pairs))
  if tuned and not override:
    override = tuned
  if override:
    plan = dict(plan, **override)
    if "cs" in override and override["cs"] > 1:
      plan["flat"] = 0
      if "ncl" not in override:
        plan["ncl"] = max(1, plan["n_splits"] // override["cs"])
      plan["n_splits"] = plan["cs"] * plan["ncl"]
      if "nc" not in override:
        plan["nc"] = max(1, min(plan["ncl"], 4))
    elif "n_splits" in override and "nc" not in override:
      plan["nc"] = 1 if plan["flat"] else max(1, min(plan["n_splits"], 4))
  n_splits, flat, cs, ncl, nc = plan["n_splits"], plan["flat"], plan["cs"], plan["ncl"], plan["nc"]
  if flat:
    coop = int(plan.get("coop", int(kl_OKC_FLATCOOP and L == 8 and
                                    n_splits * pairs <= 2 * kl__num_sms7(device))))
  else:
    coop = int(plan.get("coop", 0 if (cs > 1 or n_splits < 64) else
                        int(n_splits * B * pairs <= 2 * kl__num_sms7(device))))
  w0 = int(plan.get("w0", (kl_OKC_W0_L1 if L == 1 else kl_OKC_W0_L8) if (flat and (coop or L == 1)) else
                    (kl_OKC_W0_PR if (coop and not flat and B * pairs == 1) else 1024)))
  use_pdl = kl_OKC_PDL if pdl is None else bool(pdl)
  tma = int(plan.get("tma", 0)) | (coop << 1) | (min(max(w0, 0), 4095) << 2) | (int(use_pdl) << 16)
  if flat or cs <= 1:
    cs, ncl = 0, 1
  ws_acc, ws_l, ws_cnt = kl__workspace7(B * pairs * nc * L * 16, device)
  n_ctas = n_splits * (1 if flat else B) * pairs
  scratch, bar = kl__cluster_bufs7(n_ctas if cs > 1 else 1, L, device)
  scale_log2 = cfg["sm_scale"] * cfg["qk_scale_factor"] * kl_LOG2E
  (_mod or (kl__att7q() if B > 1 else kl__att7())).run(qkv, data["seq_lens"], data["cache_loc"], data["k_cache"], data["v_cache"], page_table,
                data["k_scale"], data["v_scale"], kl__inv_freq(cfg["rope_theta"], device), out, ws_acc, ws_l,
                ws_cnt, B, L, hq, hkv, P, window, 1 if cfg["rope_theta"] else 0, n_splits, flat, nc, scale_log2,
                dbg, cs, ncl, scratch, bar, tma, *(kl__PF7 or (0, 0)))
  return out


kl__supported7 = _supported7

kl_ATT_HAS_PDL = ATT_HAS_PDL


def kl__decode_attention7(data, state, qkv):
  cfg = data["config"]
  pdl = {"pdl": kl__pdl(cfg["batch"] * cfg["q_len"])} if kl_ATT_HAS_PDL else {}
  global kl__PF7
  bst = state.get("bfp")
  kl__PF7 = None
  if bst is not None and "pooler" in bst.get("sk2", {}) and cfg["batch"] * cfg["q_len"] in (1, 8):
    _ps = bst["sk2"]["pooler"][1]
    kl__PF7 = (_ps.data_ptr(), min(4 << 20, _ps.numel() * _ps.element_size()))
  out = kl__launch7({"config": cfg, "qkv": qkv, "seq_lens": data["seq_lens"], "cache_loc": data["cache_loc"],
                  "k_cache": data["k_cache"], "v_cache": data["v_cache"], "page_table": data["page_table"],
                  "k_scale": data["k_scale"], "v_scale": data["v_scale"]}, **pdl)
  kl__PF7 = None
  return out.view(cfg["batch"] * cfg["q_len"], kl_Q_DIM)


kl__GUARDS = []
kl__GUARDED_IDS = set()


def kl__guarded(t):
  nbytes = t.numel() * t.element_size()
  words = (nbytes + 3) // 4
  full = torch.full((words + kl_SCRATCH_GUARD,), kl_SCRATCH_SENTINEL, dtype=torch.int32, device=t.device)
  full.view(torch.uint8)[:nbytes].copy_(t.reshape(-1).view(torch.uint8))
  v = full.view(torch.uint8)[:nbytes].view(t.dtype).view(t.shape)
  kl__GUARDS.append((full, words))
  kl__GUARDED_IDS.add(id(v))
  return v


def kl__check_guards():
  torch.cuda.synchronize()
  for full, words in kl__GUARDS:
    bad = int((full[words:] != kl_SCRATCH_SENTINEL).sum())
    if bad:
      raise RuntimeError(f"a component kernel wrote {bad} words past a {words * 4}-byte buffer")


def kl__attn_selftest(st):
  dev = torch.device("cuda")
  g = torch.Generator(device=dev).manual_seed(4242)
  nq = st["qkv"].shape[0]
  for window, L, B, ctx, P in ((2048, 1, 1, 3000, 1), (2048, 1, 16, 2500, 1), (2048, 1, 32, 2500, 1),
                               (0, 1, 1, 3000, 1), (0, 1, 16, 1500, 1), (0, 1, 33, 1000, 1),
                               (0, 8, 1, 3000, 1), (0, 8, 16, 1500, 1), (0, 8, 32, 800, 1),
                               (2048, 1, 4, 3000, 64), (0, 8, 4, 3000, 64)):
    cfg = dict(batch=B, q_len=L, num_q_heads=kl_NUM_Q_HEADS, num_kv_heads=kl_NUM_KV_HEADS, head_dim=kl_HEAD_DIM, page_size=P, window=window,
               rope_theta=kl_ROPE_THETA if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125)
    if not kl__supported7(cfg):
      continue
    seq = torch.full((B,), ctx, dtype=torch.int32, device=dev)
    per = (ctx + L + P - 1) // P
    pt = torch.arange(B * per, dtype=torch.int32, device=dev).view(B, per)
    pos = (seq[:, None].long() + torch.arange(L, device=dev)).view(B, L)
    loc = (torch.gather(pt.long(), 1, pos // P) * P + pos % P).reshape(-1)
    kc = torch.zeros((B * per, P, 2, 64), dtype=torch.float8_e4m3fn, device=dev)
    vc = torch.zeros_like(kc)
    qkv = (torch.randn(B * L, nq, generator=g, device=dev) * 0.1).to(torch.bfloat16)
    one = torch.ones((), dtype=torch.float32, device=dev)
    kl__decode_attention7({"config": cfg, "seq_lens": seq, "cache_loc": loc, "k_cache": kc, "v_cache": vc,
                        "page_table": pt, "k_scale": one, "v_scale": one}, st, qkv)

def kl__att_fits7(data, cfg):
  kc, vc = data["k_cache"], data["v_cache"]
  rows = kc.numel() // max(1, cfg["num_kv_heads"] * cfg["head_dim"])
  return (kc.numel() < (1 << 32) and vc.numel() < (1 << 32) and rows < (1 << 31)
          and data["page_table"].numel() < (1 << 31)
          and cfg["batch"] * cfg["q_len"] * kl_QKV_ROWS < (1 << 31)
          and cfg["batch"] * (cfg["num_kv_heads"] // 2) <= 4096)


def kl__kernel_fn_fa3(data, state):
  st2 = dict(state)
  st2["att_ok"] = False
  return kl_kernel_fn(data, st2)


def kl_kernel_fn(data, state):
  cfg = data["config"]
  B, L, T = cfg["batch"], cfg["q_len"], cfg["tokens"]
  agent_attn = state["att_ok"] and kl__supported7(cfg) and kl__att_fits7(data, cfg)
  if agent_attn:
    qpick = state.get("mm_pick", {}).get((T, "qkv"), True)
    if qpick == "bfp" and T <= kl_BFP_MAX_T and state.get("bfp") is not None:
      qkv = kl__bfp_gemm(state, "qkv", data["attn_input"], T, state["qkv"].shape[0], kl_HIDDEN)
    elif qpick is True or qpick == "bfp":
      qkv = torch.mm(data["attn_input"], state["qkv"].t())
    else:
      Pq, Sq = kl_dense(data["attn_input"], state["qkv"], T)
      qkv = Pq[0] if Sq == 1 else Pq.sum(0)
      qkv = qkv.to(torch.bfloat16)
    try:
      o_in = kl__decode_attention7(data, state, qkv)
    except RuntimeError:
      return kl__kernel_fn_fa3(data, state)
    return kl__after_attention(data, state, o_in, None, T)
  dev = data["attn_input"].device
  x = data["attn_input"]
  sliding = cfg["window"] > 0
  use_fa3 = state["fa3"] is not None
  P, S = kl_dense(x, state["qkv"], T)
  q = torch.empty((T, kl_Q_DIM), device=dev, dtype=torch.bfloat16)
  q8 = torch.empty((T, kl_Q_DIM), device=dev, dtype=torch.uint8)
  gate = torch.empty((T, 1024), device=dev, dtype=torch.float32)
  kc = data["k_cache"].view(torch.uint8)
  vc = data["v_cache"].view(torch.uint8)
  kl__qk_prep[(T,)](P, S, T, data["positions"], data["cache_loc"], state["invf"], kc, vc,
                 data["k_scale"], data["v_scale"], q, q8, gate, cfg["qk_scale_factor"],
                 ROPE=bool(cfg["rope_theta"]), num_warps=4, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  if use_fa3:
    descale = {}
    if cfg["kv_scale"] != 1.0:
      descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(B, 2),
                     v_descale=data["v_scale"].reshape(1, 1).expand(B, 2))
    window = cfg["window"]
    out = state["fa3"](q=q8.view(torch.float8_e4m3fn).view(T, 16, 64), k_cache=data["k_cache"],
                       v_cache=data["v_cache"], page_table=data["page_table"],
                       cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"],
                       max_seqlen_q=L, softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                       window_size=(window - 1, 0) if window else (-1, -1),
                       scheduler_metadata=data["scheduler_metadata"], **descale)
    if isinstance(out, tuple):
      out = out[0]
    o_in = out.reshape(T, 1024)
  else:
    BM, NS = kl__attn_cfg(B, L, sliding)
    acc = torch.empty((B * 2 * NS * BM, 64), device=dev, dtype=torch.float32)
    ml = torch.empty((B * 2 * NS * BM, 2), device=dev, dtype=torch.float32)
    pt = data["page_table"]
    kl__attn_split[(B, NS)](q, kc, vc, pt, data["seq_lens"], data["k_scale"], acc, ml,
                         pt.stride(0), cfg["page_size"], cfg["window"], cfg["sm_scale"],
                         L=L, NSPLIT=NS, BM=BM, BN=64, SLIDING=sliding, num_warps=4, num_stages=3)
    o_in = torch.empty((T, kl_Q_DIM), device=dev, dtype=torch.bfloat16)
    kl__attn_combine[(T, 16)](acc, ml, gate, data["v_scale"], o_in, L=L, NSPLIT=NS, BM=BM,
                           SB=min(64, triton.next_power_of_2(NS)), NORMGATE=False, num_warps=4)
  return kl__after_attention(data, state, o_in, gate, T)


def kl__pick_expert_backend(pick, marlin_available, agent_available):
  if pick == "marlin" and not marlin_available:
    pick = "agent"
  if pick != "marlin" and not agent_available:
    pick = "marlin"
  if pick == "marlin" and not marlin_available:
    pick = "agent"
  return pick


def kl__after_attention(data, state, o_in, gate, T):
  dev = o_in.device
  f32 = dict(device=dev, dtype=torch.float32)
  gv = state.get("gluev2", {})
  v2 = gv.get(T, gv.get(kl__bucket(T), False)) if kl_FORCE_GLUEV2 is None else kl_FORCE_GLUEV2
  s1 = torch.empty((T, kl_HIDDEN), **f32)
  h = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  hr = torch.empty((T, kl_HIDDEN), **f32) if T <= 8 else None
  if v2:
    a_out = torch.empty((T, kl_HIDDEN), **f32)
    sums = torch.zeros((T, 4), **f32)
    BN, BT, BK = 64, (16 if T <= 16 else 32), 128
    kl__gemm_o[(kl_HIDDEN // BN, triton.cdiv(T, BT))](
        o_in, gate, state["o"], a_out, sums, data["stream"], state["gate1"], state["gate1"], T, kl_HIDDEN,
        K=1024, BN=BN, BT=BT, BK=BK, NORMGATE=gate is not None, num_warps=4, num_stages=3)
    BC1 = 2048
    kl__bnd1_apply[(T, kl_HIDDEN // BC1)](a_out, sums, data["stream"], state["gate1"], state["gate1"], state["g_preffn"],
                                    s1, h, N=kl_HIDDEN, BC=BC1, num_warps=8)
  else:
    if gate is None:
      P, S = kl_dense(o_in, state["o"], T, state, "o")
    else:
      P, S = kl_dense_ng(o_in, gate, state["o"], T, state)
    kl__boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                     WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  if T <= 8 and v2:
    kl__router_input_norm[(T,)](s1, state["g_preffn"], hr, num_warps=8)
  mmp = state.get("mm_pick", {})
  ppick = mmp.get((T, "pr"), mmp.get((kl__bucket(T), "pr"), False))
  P, S = kl__pool_router_fp32(h, hr, T, state, ppick)
  ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  wts = torch.empty((T, 8), **f32)
  moe = kl__pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(kl__bucket(T), "agent")),
                             marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
  use_marlin = moe == "marlin"
  use_ma = moe == "agent"
  if not (use_marlin or use_ma):
    raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
  xm = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
  if S == -1:
    pooled, logits = P
    kl__route_split[(T,)](pooled, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                       data["local_start"], lids, PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  elif v2:
    pool = torch.empty((T, kl_D_EXPERT), **f32)
    psum = torch.zeros((T,), **f32)
    BCR = 1024
    kl__route_a[(T, kl_D_EXPERT // BCR)](P, S, T, state["bias"], ids, wts, data["local_start"], lids, pool, psum,
                                   BC=BCR, num_warps=8)
    kl__route_b[(T, kl_D_EXPERT // BCR)](pool, psum, state["g_preexp"], xm, BC=BCR,
                                   PERMUTED=not (use_marlin or use_ma), num_warps=8)
  else:
    kl__route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                 PERMUTED=not (use_marlin or use_ma), num_warps=8, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  if use_ma:
    m = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext = kl__moe(T, state)
    if kl_MOE_HAS_PDL:
      ext.okc_set_pdl(1 if kl__pdl(T) else 0)
    for m0 in range(0, T, kl__MA_MMAX):
      m1 = min(T, m0 + kl__MA_MMAX)
      ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                  state["nsm"])
  wu = state["unpool_slice"] if T <= 8 else state["unpool"]
  P, S = kl_dense(m, wu, T, state, "un")
  nu = wu.shape[0]
  s2 = torch.empty((T, kl_HIDDEN), **f32)
  out = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  if v2:
    fbuf = torch.zeros((T, kl_HIDDEN), **f32)
    sums2 = torch.zeros((T, 4), **f32)
    BC2 = 1024
    kl__bnd2_sums[(T, kl_HIDDEN // BC2)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], fbuf, sums2,
                                   BC=BC2, num_warps=8)
    kl__bnd2_apply[(T, kl_HIDDEN // BC2)](fbuf, sums2, s1, state["gate2"], state["gate2"], state["g_next"], s2, out,
                                    N=kl_HIDDEN, BC=BC2, num_warps=8)
  else:
    kl__boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                     s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  return out, s2, ids


# A_graft_glue_d1_v1_v32 graft block (appended to the vjeux base, everything above is unchanged).
# Portions from vjeux/okc-2026 (Apache-2.0): the base above, and _graft_after_attention below, which is
# Grafted parts are our team's own code (Apache-2.0 submission): glue_cluster_boundary (8-CTA cluster
import ctypes as kl__C
import hashlib as _hashlib
from functools import lru_cache as kl__lru_cache
from triton.language.extra.cuda import gdc_wait as _gdc_wait, gdc_launch_dependents as _gdc_launch
from triton.experimental import gluon as kl__glx
from triton.experimental.gluon import language as kl__gl

kl__GRAFT_GLUE_T = (1, 8, 16, 32, 128, 256)
kl__GRAFT_ROUTE_EARLY = False

# LDD19 hybrid: the graft glue = the SL glue (hoisted above) with these exact edits (readable template).
kl__GRAFT_GLUE_CUDA = _ldd19_derive(kl__SL_GLUE_CUDA, [
  ('after', r"""    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}""", r"""
void b2c(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)next.data_ptr(),(int)stream.size(0)};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b2c: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,true,false,false,true>,s);
  else glue_launch(a,glue_cluster_boundary<4,32,true,false,false,false>,s);
}"""),
  (r"""             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0),(const float*)alpha2.data_ptr(),(float*)tail.data_ptr()};""", r"""             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0)};"""),
  (r"""void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {""", r"""void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV router,bool write_router) {"""),
  (r"""      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);""", r"""      gsb<V>(a.out+row*8192+col,h);"""),
  (r"""    }
  }
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    #pragma unroll
    for(int j=0;j<V;j++) {
      int k=r*V+j;
      carry[k]=gm(a2[k],ns[k]);
      masked[k]=(col+j>=1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);
  if(rank==0 && tid==0) a.tail[row]=tail;""", r"""      if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
    }
  }
  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);"""),
  ('before', r"""      }
    }""", r"""        if(active) gsf<V>(a.next+row*8192+col,ns+r*V);
"""),
  (r"""    glf<V>(a.alpha2+col,a2+r*V);
    glb<V>(a.pre+col,pre+r*V);""", r"""    glb<V>(a.pre+col,pre+r*V);"""),
  (r"""  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];
  __shared__ float partials[3][NW];""", r"""  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N];
  __shared__ float partials[2][NW];"""),
  (r"""}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* px=cluster.map_shared_rank(sm,dst);
      volatile float* py=cluster.map_shared_rank(smy,dst);
      px[rank*warps_per_cta+warp]=sx;
      py[rank*warps_per_cta+warp]=sy;
    }
  }
  cluster.sync();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
}""", r"""}"""),
  (r"""  const float* alpha2;
  float* tail;
};""", r"""};"""),
])
kl__GRAFT_B2_CUDA = r"""#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;
__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
__device__ __forceinline__ void lf2(const float* p, float* x) {
  float2 v = *reinterpret_cast<const float2*>(p); x[0]=v.x; x[1]=v.y;
}
__device__ __forceinline__ void lb2(const BF* p, float* x) {
  unsigned u = *reinterpret_cast<const unsigned*>(p);
  x[0]=__uint_as_float(u<<16); x[1]=__uint_as_float(u&0xffff0000u);
}
__device__ __forceinline__ void glf2(const float* p, float* x) {
  asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
}
__device__ __forceinline__ void gsf2(float* p, const float* x) {
  asm volatile("st.global.v2.f32 [%0],{%1,%2};"::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
}
__device__ __forceinline__ void gsb2(BF* p, const float* x) {
  unsigned u;
  asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u):"f"(x[1]),"f"(x[0]));
  asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u):"memory");
}
__device__ __forceinline__ float pair_sq(float x0, float x1) { return gf(x0,x0,gm(x1,x1)); }
__device__ __forceinline__ float butterfly(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int P,bool SBF>
__global__ __launch_bounds__(P) void b2_single(const float* part, const float* stream, const BF* post,
    const BF* pre, const float* alpha, const float* beta, BF* out, float* next, int rows) {
  constexpr int J=1024/P, JL=512/P, WP=P/32;
  const int t=threadIdx.x, lane=t&31, w=t>>5, row=blockIdx.x;
  __shared__ float sm1[32], sm2[32];
  const float* srow=stream+(size_t)row*8192;
  float* nrow=next+(size_t)row*8192;
  float ns[J][8], pr[J][8];
  float s0[JL][2], al[JL][2], be[JL][2], po[JL][2];
  float q1[JL], q23[JL];
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      const int col=2*L+2048*r;
      lb2(pre+col,pr[j]+2*r);
      if(j<JL && r==0) {
        lf2(srow+col,s0[j]); lf2(alpha+col,al[j]); lf2(beta+col,be[j]); lb2(post+col,po[j]);
      } else {
        float s[2],a[2],b[2],p[2];
        lf2(srow+col,s); lf2(alpha+col,a); lf2(beta+col,b); lb2(post+col,p);
        #pragma unroll
        for(int k=0;k<2;k++) ns[j][2*r+k]=gf(a[k],s[k],gm(b[k],gm(0.f,p[k])));
      }
    }
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) if(!(j<JL && r==0)) gsf2(nrow+2*L+2048*r,ns[j]+2*r);
  }
  #pragma unroll
  for(int j=0;j<J;j++) {
    if(j<JL) {
      q1[j]=pair_sq(ns[j][2],ns[j][3]);
      q23[j]=ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7]));
    } else {
      float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),pair_sq(ns[j][2],ns[j][3])),
                 ga(pair_sq(ns[j][4],ns[j][5]),pair_sq(ns[j][6],ns[j][7])));
      v=butterfly(v);
      if(lane==0) { sm2[w+WP*j]=v; sm1[w+WP*j]=0.f; }
    }
  }
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  float b[JL][2];
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    if constexpr(SBF) {
      unsigned u;
      asm volatile("ld.global.b32 %0,[%1];":"=r"(u):"l"(reinterpret_cast<const BF*>(part)+(size_t)row*1024+col):"memory");
      b[j][0]=gb(gm(__uint_as_float(u<<16),po[j][0]));
      b[j][1]=gb(gm(__uint_as_float(u&0xffff0000u),po[j][1]));
    } else {
    float pp[8][2];
    #pragma unroll
    for(int sk=0;sk<8;sk++) glf2(part+((size_t)(sk*rows+row))*1024+col,pp[sk]);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float x=0.f;
      #pragma unroll
      for(int sk=0;sk<8;sk++) x=ga(x,pp[sk][k]);
      x=gb(x);
      b[j][k]=gb(gm(x,po[j][k]));
    }
    }
    const float z=pair_sq(0.f,0.f);
    float v=ga(ga(pair_sq(b[j][0],b[j][1]),z),ga(z,z));
    v=butterfly(v);
    if(lane==0) sm1[w+WP*j]=v;
  }
  __syncthreads();
  const float r1=gr(butterfly(sm1[lane]),1.e-8f);
  #pragma unroll
  for(int j=0;j<JL;j++) {
    const int col=2*(t+P*j);
    #pragma unroll
    for(int k=0;k<2;k++) {
      float bn=gb(gm(b[j][k],r1));
      ns[j][k]=gf(al[j][k],s0[j][k],gm(be[j][k],bn));
    }
    gsf2(nrow+col,ns[j]);
    float v=ga(ga(pair_sq(ns[j][0],ns[j][1]),q1[j]),q23[j]);
    v=butterfly(v);
    if(lane==0) sm2[w+WP*j]=v;
  }
  __syncthreads();
  const float r2=gr(butterfly(sm2[lane]),1.e-5f);
  BF* orow=out+(size_t)row*8192;
  #pragma unroll
  for(int j=0;j<J;j++) {
    const int L=t+P*j;
    #pragma unroll
    for(int r=0;r<4;r++) {
      float h[2];
      #pragma unroll
      for(int k=0;k<2;k++) h[k]=gm(gm(ns[j][2*r+k],r2),pr[j][2*r+k]);
      gsb2(orow+2*L+2048*r,h);
    }
  }
}
void b2s(TV branch,TV stream,TV post,TV pre,TV alpha,TV beta,TV out,TV next) {
  int rows=(int)stream.size(0);
  if(stream.size(1)!=8192 || branch.size(0)!=rows || branch.size(1)!=1024) throw std::runtime_error("graft b2s: shape");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(rows);cfg.blockDim=dim3(512);cfg.stream=s;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=1;
  auto e=cudaLaunchKernelEx(&cfg,b2_single<512,true>,(const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
      (const BF*)post.data_ptr(),(const BF*)pre.data_ptr(),(const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
      (BF*)out.data_ptr(),(float*)next.data_ptr(),rows);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
"""


# LDD19: B1M (solo_optimizer r4i1): one 4-moment cluster exchange for both norms of the graft b1 / b2c (see make_b1m.py)
LDD19_KL_B1M = 1


def kl__b1m_patch(s):
  edits = (
    ("template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n__global__",
     "template<int NW>\n"
     "__device__ __forceinline__ void ldd19_cluster_sum4(float (&v)[4], float* sm, cg::cluster_group cluster, int rank) {\n"
     "  constexpr int warps_per_cta=NW/8;\n"
     "  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;\n"
     "  #pragma unroll\n"
     "  for(int i=0;i<4;i++) v[i]=butterfly(v[i],16);\n"
     "  if(lane==0 && warp<warps_per_cta) {\n"
     "    #pragma unroll\n"
     "    for(int dst=0;dst<8;dst++) {\n"
     "      volatile float* p=cluster.map_shared_rank(sm,dst);\n"
     "      #pragma unroll\n"
     "      for(int i=0;i<4;i++) p[i*NW+rank*warps_per_cta+warp]=v[i];\n"
     "    }\n"
     "  }\n"
     "  cluster.sync();\n"
     "  #pragma unroll\n"
     "  for(int i=0;i<4;i++) v[i]=butterfly(sm[i*NW+(lane&(NW-1))],NW/2);\n"
     "}\n"
     "template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n__global__"),
    ("  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);\n",
     "  float r1, r2m=0.f;   // LDD19: B1M\n"
     "  __shared__ float ldd19_pm[4*NW];\n"
     "  if constexpr(!SLICED) {\n"
     "    float m4[4]={0.f,0.f,0.f,0.f};\n"
     "    #pragma unroll\n"
     "    for(int k=0;k<N;k++) {\n"
     "      float as_=gm(alpha[k],s[k]), bb_=gm(beta[k],b[k]);\n"
     "      m4[0]=gf(b[k],b[k],m4[0]); m4[1]=gf(as_,as_,m4[1]); m4[2]=gf(as_,bb_,m4[2]); m4[3]=gf(bb_,bb_,m4[3]);\n"
     "    }\n"
     "    ldd19_cluster_sum4<NW>(m4,ldd19_pm,cluster,rank);\n"
     "    r1=gr(m4[0],1.e-8f);\n"
     "    r2m=gr(gf(gm(r1,r1),m4[3],gf(gm(2.f,r1),m4[2],m4[1])),1.e-5f);\n"
     "  } else {\n"
     "    r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);\n"
     "  }\n"),
    ("  float r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);\n",
     "  float r2;   // LDD19: B1M\n"
     "  if constexpr(!SLICED) r2=r2m; else r2=gr(cluster_squares<NW>(ns,partials[1],cluster,rank),1.e-5f);\n"),
  )
  for old, new in edits:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new, 1)
  return s


if LDD19_KL_B1M:
  kl__GRAFT_GLUE_CUDA = kl__b1m_patch(kl__GRAFT_GLUE_CUDA)


@kl__lru_cache(None)
def kl__graft_glue_module():
  from tvm_ffi.cpp import load_inline as _tvm_load_inline
  return _tvm_load_inline(name='graft_glue_' + _hashlib.md5(kl__GRAFT_GLUE_CUDA.encode()).hexdigest()[:12],
                          cuda_sources=kl__GRAFT_GLUE_CUDA, functions=['b1', 'b2c'],
                          extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                             '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@kl__lru_cache(None)
def kl__graft_b2_module():
  from tvm_ffi.cpp import load_inline as _tvm_load_inline
  return _tvm_load_inline(name='graft_b2_' + _hashlib.md5(kl__GRAFT_B2_CUDA.encode()).hexdigest()[:12],
                          cuda_sources=kl__GRAFT_B2_CUDA, functions=['b2s'],
                          extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr', '--fmad=false',
                                             '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'])


@kl__glx.jit
def kl__graft_row_boundary_pre(Branch, Stream, PostW, PreW, Alpha, Beta, Out, NewStream,
                            H: kl__gl.constexpr, F: kl__gl.constexpr, POST: kl__gl.constexpr, V: kl__gl.constexpr, NW: kl__gl.constexpr):
  row = kl__gl.program_id(0)
  c = kl__gl.arange(0, H, layout=kl__gl.BlockedLayout([V], [32], [NW], [0]))
  pw = kl__gl.load(PreW + c).to(kl__gl.float32)
  a = kl__gl.load(Alpha + c)
  beta = kl__gl.load(Beta + c)
  if POST:
    post_w = kl__gl.load(PostW + c).to(kl__gl.float32)
  _gdc_wait()
  _gdc_launch()
  b = kl__gl.load(Branch + row * F + c, c < F, 0).to(kl__gl.float32)
  if POST:
    b = (b * post_w).to(kl__gl.bfloat16).to(kl__gl.float32)
  r = kl__gl.rsqrt(kl__gl.sum(b * b, 0) / H + 1.e-8)
  b = (b * r).to(kl__gl.bfloat16).to(kl__gl.float32)
  s = kl__gl.load(Stream + row * H + c)
  ns = a * s + beta * b
  kl__gl.store(NewStream + row * H + c, ns)
  r = kl__gl.rsqrt(kl__gl.sum(ns * ns, 0) / H + 1.e-5)
  h = ns * r * pw
  kl__gl.store(Out + row * H + c, h)


@triton.jit
def kl__graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, PACKED: tl.constexpr):
  t = tl.program_id(0)
  kind = tl.program_id(1)
  if EARLY:
    _gdc_launch()
  _gdc_wait()
  if not EARLY:
    _gdc_launch()
  if kind == 0:
    c = tl.arange(0, 4096)
    if SPLIT:
      pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    else:
      pooled = tl.load(ALL + t * 4608 + c)
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
  else:
    e = tl.arange(0, 256)
    if SPLIT:
      logit = tl.load(LOGITS + t * 256 + e)
    else:
      logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + tl.load(BIAS + e)
    k8 = tl.arange(0, 8)
    if PACKED != 0:
      bits = sel.to(tl.uint32, bitcast=True)
      ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
      packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
      if PACKED == 1:
        picked = tl.topk(packed, 8)
      else:
        sorted_values = tl.sort(packed, descending=True)
        picked = tl.gather(sorted_values, k8, 0)
      ids = (255 - (picked & 255)).to(tl.int32)
      wv = tl.gather(sc, ids, 0)
    else:
      ids = tl.zeros([8], dtype=tl.int32)
      wv = tl.zeros([8], dtype=tl.float32)
      for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - tl.load(LS) + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)


def kl__graft_al(*ts):
  for t in ts:
    assert t.is_contiguous() and t.data_ptr() % 16 == 0, (t.shape, t.data_ptr() % 16)


def kl__graft_boundary(P, S, T, stream, st, post, s_out, h_out, nu=kl_HIDDEN, router_out=None):
  g = st["graft"]
  if S != 1:
    return False
  br = P.view(T, -1)
  width = br.shape[1]
  pfx = "2" if post else "1"
  if T <= 16 and br.dtype == torch.bfloat16:
    kl__graft_al(br, stream, s_out, h_out)
    if not post and width == kl_HIDDEN:
      kl__graft_glue_module().b1(br, stream, g["pre1"], g["alpha1"], g["beta1"], h_out, s_out,
                              router_out if router_out is not None else s_out, router_out is not None)
      return True
    if post and width == 1024 and nu == 1024:
      kl__graft_b2_module().b2s(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
      return True
    if post and width == kl_HIDDEN and nu == kl_HIDDEN:
      kl__graft_glue_module().b2c(br, stream, g["post2"], g["pre2"], g["alpha2"], g["beta2"], h_out, s_out)
      return True
    return False
  if 16 < T <= 32 and width == kl_HIDDEN:
    kl__graft_al(br, stream, s_out, h_out)
    kl__graft_row_boundary_pre[(T,)](br, stream, g["post2"], g["pre2" if post else "pre1"], g["alpha" + pfx],
                                  g["beta" + pfx], h_out, s_out, kl_HIDDEN, width, post, 4, 32,
                                  num_warps=32, launch_pdl=True)
    return True
  return False


def kl__graft_after_attention(data, state, o_in, gate, T):
  dev = o_in.device
  f32 = dict(device=dev, dtype=torch.float32)
  s1 = torch.empty((T, kl_HIDDEN), **f32)
  h = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  hr = torch.empty((T, kl_HIDDEN), **f32) if T <= 8 else None
  if gate is None:
    P, S = kl_dense(o_in, state["o"], T, state, "o")
  else:
    P, S = kl_dense_ng(o_in, gate, state["o"], T, state)
  if not kl__graft_boundary(P, S, T, data["stream"], state, False, s1, h, router_out=hr):
    kl__boundary1[(T,)](P, S, T, data["stream"], state["gate1"], state["gate1"], state["g_preffn"], s1, h, hr,
                     WRITE_ROUTER=T <= 8, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  mmp = state.get("mm_pick", {})
  ppick = mmp.get((T, "pr"), mmp.get((kl__bucket(T), "pr"), False))
  P, S = kl__pool_router_fp32(h, hr, T, state, ppick, norc=bool(LDD19_NORC16 and T in (16, 32)))   # LDD19: NORC16
  ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  wts = torch.empty((T, 8), **f32)
  moe = kl__pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(kl__bucket(T), "agent")),
                             marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
  use_marlin = moe == "marlin"
  use_ma = moe == "agent"
  if not (use_marlin or use_ma):
    raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
  xm = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
  if S == -1:
    pooled, logits = P
    kl__graft_al(pooled, logits, xm, ids, wts, lids)
    if T in (16, 32):
      nw, parts = {16: (8, 1), 32: (8, 1)}[T]
      kl__sl_tile_route[(T, parts + 1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
          data["local_start"], lids, SPLIT=True, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True, NORC=int(logits.dim() == 1),   # LDD19: NORC16
          num_warps=nw, launch_pdl=True)
    else:
      kl__graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                       data["local_start"], lids, SPLIT=True, EARLY=kl__GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=True)
  elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
    allp = P.view(T, 4608)
    kl__graft_al(allp, xm, ids, wts, lids)
    if T in (16, 32):
      nw, parts = {16: (8, 1), 32: (8, 1)}[T]
      kl__sl_tile_route[(T, parts + 1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
          data["local_start"], lids, SPLIT=False, EARLY=False, MODE=0, PARTS=parts, PRELOAD=True, NORC=0,
          num_warps=nw, launch_pdl=True)
    else:
      kl__graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                       data["local_start"], lids, SPLIT=False, EARLY=kl__GRAFT_ROUTE_EARLY, PACKED=1 if T > 32 else 0, num_warps=4 if T > 32 else 1, launch_pdl=True)
  else:
    kl__route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                 PERMUTED=False, num_warps=8, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  if use_ma:
    m = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext = kl__moe(T, state)
    if kl_MOE_HAS_PDL:
      ext.okc_set_pdl(1 if kl__pdl(T) else 0)
    for m0 in range(0, T, kl__MA_MMAX):
      m1 = min(T, m0 + kl__MA_MMAX)
      ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                  state["nsm"])
  wu = state["unpool_slice"] if T <= 8 else state["unpool"]
  P, S = kl_dense(m, wu, T, state, "un")
  nu = wu.shape[0]
  s2 = torch.empty((T, kl_HIDDEN), **f32)
  out = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  if not kl__graft_boundary(P, S, T, s1, state, True, s2, out, nu=nu):
    kl__boundary2[(T,)](P, S, T, nu, s1, state["gate2"], state["gate2"], state["g_postffn"], state["g_next"],
                     s2, out, num_warps=state.get('compose_boundary_warps', {}).get(T, 16), USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  return out, s2, ids


kl__vj_after_attention = kl__after_attention


def kl__after_attention(data, state, o_in, gate, T):
  if "graft" in state and T in kl__GRAFT_GLUE_T and state.get("compose_graft_enabled", {}).get(T, T in (1, 8, 128, 256)):
    return kl__graft_after_attention(data, state, o_in, gate, T)
  return kl__vj_after_attention(data, state, o_in, gate, T)


kl__vj_prepare = kl_prepare


def kl_prepare(weights):
  st = kl__vj_prepare(weights)
  if kl__GRAFT_GLUE_T:
    if st.get("bfp") is not None:
      for T in kl__GRAFT_GLUE_T:
        if T <= 16:
          st["mm_pick"][(T, "o")] = "bfp"
    g = {}
    for i, name in ((1, "attn_residual_gate"), (2, "ffn_residual_gate")):
      gw = weights[name].float() / 0.3
      beta = torch.sigmoid(gw)
      g["alpha%d" % i] = torch.sqrt(torch.clamp(torch.sigmoid(-gw) * (1. + beta), min=1.e-3)).contiguous()
      g["beta%d" % i] = beta.contiguous()
    g["pre1"] = (weights["pre_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
    g["post2"] = (weights["post_ffn_norm"].float() + 1).to(torch.bfloat16).contiguous()
    g["pre2"] = (weights["next_pre_attn_norm"].float() + 1).to(torch.bfloat16).contiguous()
    for v in g.values():
      assert v.data_ptr() % 128 == 0, v.data_ptr() % 128
    st["graft"] = g
    kl__graft_glue_module()
    kl__graft_b2_module()
  st["graft_lt"] = {}   # LDD19: no cuBLASLt plans (Keell never runs T > 32)
  torch.cuda.synchronize()
  return st


# LDD19: GF2S (moe_engineer it4): GF(2) re-layout of the Keell skinny unit copies for the TINY / MID builds (see the
# CUDA comment at dq4g). 0 = off (previous behaviour, previous build hashes). The map is derived and asserted here.
LDD19_GF2S = int(__import__("os").environ.get("LDD19_GF2S", "1"))
LDD19_GF2S_TINY = int(__import__("os").environ.get("LDD19_GF2S_TINY", "0"))


def kl__gf2s_cols():
  import functools, random
  F, M = 0xFFFFFFFF, 0x81C081C0
  rot = lambda q, s: ((q << s) | (q >> (32 - s))) & F
  d7 = lambda q: (q & M, (q * 16) & M & F, rot(q, 10) & M, (((q * 64) & 0x80C080C0) | (rot(q, 14) & 0x01000100)) & F)
  dg = lambda w: (w & M, ((w & (M >> 4)) << 4) & F, (((w ^ (w >> 8)) & (M >> 2)) << 2) & F, (((w ^ (w >> 8)) & (M >> 6)) << 6) & F)
  vec = lambda r: sum(((x >> p) & 1) << (8 * i + j) for i, x in enumerate(r) for j, p in enumerate((6, 7, 8, 15, 22, 23, 24, 31)))
  A = [[(vec(dg(1 << c)) >> r) & 1 for c in range(32)] + [int(r == c) for c in range(32)] for r in range(32)]
  for c in range(32):   # GF(2) inverse of Dg
    p = next(r for r in range(c, 32) if A[r][c])
    A[c], A[p] = A[p], A[c]
    A = [[a ^ b for a, b in zip(A[r], A[c])] if r != c and A[r][c] else A[r] for r in range(32)]
  ap = lambda cs, x: functools.reduce(lambda a, i: a ^ cs[i] if x >> i & 1 else a, range(32), 0)
  gi = [sum(A[r][32 + c] << r for r in range(32)) for c in range(32)]
  cs = [ap(gi, vec(d7(1 << i))) for i in range(32)]
  rnd = random.Random(19)
  for q in [1 << i for i in range(32)] + [rnd.getrandbits(32) for _ in range(512)]:
    assert dg(ap(cs, q)) == d7(q)
  return cs


def kl__gf2s_units(u):
  """unit-order tensor (36,864 B units: 32,768 B of int32 code words, then 4,096 B of scales) -> copy with T(code)."""
  cs = kl__gf2s_cols()
  lut = []
  for b in range(4):
    v = torch.arange(256, dtype=torch.int64)
    o = torch.zeros(256, dtype=torch.int64)
    for i in range(8):
      o ^= ((v >> i) & 1) * cs[8 * b + i]
    lut.append(torch.where(o >= 2**31, o - 2**32, o).to(torch.int32).to(u.device))
  out = u.clone()
  ub = out.view(-1, 36864)
  for i0 in range(0, ub.shape[0], 2048):
    c = ub[i0:i0 + 2048, :32768].view(torch.int32)
    q = c.clone()
    c.copy_(lut[0][(q & 255).long()] ^ lut[1][((q >> 8) & 255).long()] ^ lut[2][((q >> 16) & 255).long()]
            ^ lut[3][((q >> 24) & 255).long()])
  return out


def kl__gf2s_src(h):   # LDD19: GF2S source edit (DQ4U at the 5 skinny-ring sites + okc_run redirect)
  assert h.count("dq4(ww[st], r0, r1, r2, r3);") == 5
  h = h.replace("dq4(ww[st], r0, r1, r2, r3);", "DQ4U(ww[st], r0, r1, r2, r3);")
  return _ldd19_derive(h, [
    ('before', r"""__device__ void early_issue(const Params& p, Smem& S);""", r"""// LDD19: GF2S (moe_engineer it4): the skinny unit copies hold T(q) (a weight-independent GF(2) mix of the DQ7 word,
// applied in prepare); dq4g returns exactly dq4(q) (DQ7) with 4 LOP3 + 1 shift + 3 shifts-by-multiply.
// LDD19_GF2S 1: constant multiplies; 2: multiplies by runtime 2^k (+ p.zero) so they stay IMAD (FMA pipe);
// 3: 2 + t = w >> 8 as IMAD.HI (ALU pipe: LOP3 only).
template <uint32_t C>
__device__ __forceinline__ uint32_t gf2s_xm(uint32_t a, uint32_t b) {   // (a ^ b) & C as one LOP3
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0x28;" : "=r"(d) : "r"(a), "r"(b), "n"(C));
  return d;
}
__device__ __forceinline__ void dq4g(uint32_t w, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t z) {
  constexpr uint32_t M = 0x81C081C0u;
#if !(LDD19_GF2S >= 3)
  const uint32_t t = w >> 8;
#endif
#if !(LDD19_GF2S >= 2)
  constexpr uint32_t k4 = 4u, k16 = 16u, k64 = 64u;
  (void)z;
#endif
  r0 = w & M;
#if !(LDD19_GF2S == 4)
  r1 = (w & (M >> 4)) * k16;
  r2 = gf2s_xm<(M >> 2)>(w, t) * k4;
  r3 = gf2s_xm<(M >> 6)>(w, t) * k64;
#endif
}
// LDD19: DQ7ALU (GF2S off): dq4 (DQ7) of the unmixed word with its two IMAD.SHL as rotates (SHF): (q * 16) & M ==
// rotl(q, 4) & M and (q * 64) & 0x80C080C0 == rotl(q, 6) & 0x80C080C0 because both masks are clear in bits 0..5.
__device__ __forceinline__ void dq4a(uint32_t q, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_l(q, q, 4) & M;
  r2 = __funnelshift_l(q, q, 10) & M;
  r3 = (__funnelshift_l(q, q, 6) & 0x80C080C0u) | (__funnelshift_l(q, q, 14) & 0x01000100u);
}
#if LDD19_GF2S
#define DQ4U(q, a, b, c, d) dq4g((q), a, b, c, d, (uint32_t)p.zero)
#endif
"""),
    ('after', "  else { p.wAn = p.wAf; p.wBn = p.wBf; }\n", """#if LDD19_GF2S   // LDD19: GF2S: the skinny ring streams the mixed copy (kl__ma_prepare puts it in w[0] / w[3])
  TORCH_CHECK(w[0].numel() == w[10].numel() && w[3].numel() == w[11].numel(), "LDD19 GF2S: mixed unit copies missing");
  p.wAu = (const unsigned char*)w[0].data_ptr(); p.wBu = (const unsigned char*)w[3].data_ptr();
  p.wA = nullptr; p.sA = nullptr; p.wB = nullptr; p.sB = nullptr;
#endif
"""),
  ])


if LDD19_GF2S and LDD19_GF2S_TINY:   # TINY + MID (shared base source)
  kl_MOE_HELPER0_CUDA_SRC = kl__gf2s_src(kl_MOE_HELPER0_CUDA_SRC)

kl__MA_ALT_SRCS["moe_helper0"] = kl_MOE_HELPER0_CUDA_SRC   # LDD19 hybrid: hoisted above (shared base)
kl_MOE_MID_DEFINES += ("@moe_helper0",)

def kl__df_patch(c):
  def s1(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)
  c = s1(c, "#define OKC_NOLOAD 0\n", """#define OKC_NOLOAD 0
#ifndef OKC_PRMTSC
#define OKC_PRMTSC 0
#endif
#define OKC_DF OKC_PRMTSC   // data-flag A->B hand-off (needs the PRMT scale decode builds)
#define INTER_SENT_ROWS 128
""")
  c = s1(c, "constexpr int X1P = 512 + 8;", """// DF: phase-B x rows (the job's inter) straight from L2 until every value is written (sentinel 0xFFFF = bf16 NaN)
__device__ __forceinline__ uint32_t df_bad(const uint4& v) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) b |= (uint32_t)(((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu));
  return b;
}
__device__ __forceinline__ void load_x_inter_df(const Params& p, const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH]) {
  const int g = lane >> 2, t = lane & 3;
  const int cnt = S.jC[j], st = S.jS[j];
  const bool act = g < cnt;
  const uint4* src = reinterpret_cast<const uint4*>(p.inter + (size_t)(st + (act ? g : 0)) * ID + warp * (64 * NCH) + 16 * t);
  for (int spin = 0;; ++spin) {
    uint32_t bad = 0;
#pragma unroll
    for (int c = 0; c < NCH; ++c) {
      x[2 * c] = act ? __ldcg(src + c * 8) : make_uint4(0, 0, 0, 0);
      x[2 * c + 1] = act ? __ldcg(src + c * 8 + 1) : make_uint4(0, 0, 0, 0);
      bad |= df_bad(x[2 * c]) | df_bad(x[2 * c + 1]);
    }
    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;
    __nanosleep(32);
  }
}

constexpr int X1P = 512 + 8;""")
  a = c.index("__device__ void skinny_ring_ws(const Params& p")
  b = c.index("__device__ void ws_helper(")
  seg = c[a:b]
  seg = s1(seg, """      if (PB) {
        if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
        __syncwarp();
      }
      load_x_regs<PB>(p, S, j, lane, warp, x0);""", """      if (PB && OKC_DF) {
        load_x_inter_df(p, S, j, lane, warp, x0);
      } else {
        if (PB) {
          if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
          __syncwarp();
        }
        load_x_regs<PB>(p, S, j, lane, warp, x0);
      }""")
  c = c[:a] + seg + c[b:]
  c = s1(c, "if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters",
         "if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters")
  c = s1(c, "  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]\n", """#if !OKC_WGS
  if (OKC_DF) {  // every CTA is past phase B: inter is dead until the next launch; give it back its "not yet written" sentinel
    const int rows = min(INTER_SENT_ROWS, p.M * 8);
    uint4* ip = reinterpret_cast<uint4*>(p.inter);
    for (int i = blockIdx.x * NT + tid; i < rows * (ID / 8); i += gridDim.x * NT) ip[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
  }
#endif
  // final: out[t] = sum_s w * rsqrt(ss/HD + eps) * ybuf[pos]
""")
  c = s1(c, "  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();\n", """  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
#if OKC_DF
  p.inter += (size_t)128 * 4096;   // DF sentinel rows [128, 256) of the scratch: only DF launches (MID) use them
#endif
""")
  return c


kl__MA_ALT_SRCS["moe_helper0df"] = kl__df_patch(kl__MA_ALT_SRCS["moe_helper0"])
if LDD19_GF2S and not LDD19_GF2S_TINY:   # LDD19: GF2S on the MID source only (TINY source / build hash untouched)
  kl__MA_ALT_SRCS["moe_helper0df"] = kl__gf2s_src(kl__MA_ALT_SRCS["moe_helper0df"])
kl_MOE_MID_DEFINES = tuple("@moe_helper0df" if d == "@moe_helper0" else d for d in kl_MOE_MID_DEFINES)
LDD19_KL_SC6M = 1   # LDD19: 6-bit block scales in the MID skinny units (35,840 B units, lossless; falls back when a scale byte leaves 0x40..0x7F)
kl__MA_SC6M = False


def kl__sc6m_src(h):   # LDD19: SC6M: unit size, 3-plane scale load (donor sc_load), scale decode of the 3 active skinny bodies
  def r(h, a, b, n):
    assert h.count(a) == n, a[:60]
    return h.replace(a, b)
  h = r(h, "UNIT_SC = 4096,", "UNIT_SC = LDD19_SC6M ? 3072 : 4096,", 1)
  h = r(h, "#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + 0x5C005C00u)",
        """#define SCDK(word, b, k16) (__byte_perm((word), 0u, 0x4040u | ((b) * 0x0101u)) * (k16) + SC6M_ADD)
#ifndef LDD19_SC6M
#define LDD19_SC6M 0
#endif
#if LDD19_SC6M   // LDD19: scale bytes arrive as 4 v (v = byte - 0x40): * 4 + 0x6000 per half == (0x40 + v) * 16 + 0x5C00
#define SC6M_ADD 0x60006000u
#define SC6M_K 4u
#else
#define SC6M_ADD 0x5C005C00u
#define SC6M_K 16u
#endif
__device__ __forceinline__ uint4 sc6m_load(const unsigned char* slot, int warp, int lane) {
#if LDD19_SC6M
  const unsigned char* b = slot + UNIT_CODES + warp * 128 + lane * 4;
  const uint32_t w0 = *reinterpret_cast<const uint32_t*>(b), w1 = *reinterpret_cast<const uint32_t*>(b + 1024),
                 w2 = *reinterpret_cast<const uint32_t*>(b + 2048);
  uint4 r;
  r.x = w0 & 0xFCFCFCFCu; r.y = w1 & 0xFCFCFCFCu; r.z = w2 & 0xFCFCFCFCu;
  r.w = ((w0 << 2) & 0x0C0C0C0Cu) | ((w1 << 4) & 0x30303030u) | ((w2 << 6) & 0xC0C0C0C0u);
  return r;
#else
  return *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);
#endif
}""", 1)
  h = r(h, "const uint4 sc = *reinterpret_cast<const uint4*>(slot + UNIT_CODES + warp * 512 + lane * 16);", "const uint4 sc = sc6m_load(slot, warp, lane);", 3)
  h = r(h, "* k16 + 0x5C005C00u;  // LDD19 (SCP)", "* k16 + SC6M_ADD;  // LDD19 (SCP)", 3)
  h = r(h, "const uint32_t k16 = 16u + p.zero;", "const uint32_t k16 = SC6M_K + p.zero;", 3)
  h = r(h, 'TORCH_CHECK(w[0].numel() == w[10].numel() && w[3].numel() == w[11].numel(), "LDD19 GF2S: mixed unit copies missing");',
        'TORCH_CHECK(w[0].numel() == w[10].numel() / 36864 * okc::UNIT_BYTES && w[3].numel() == w[11].numel() / 36864 * okc::UNIT_BYTES, "LDD19 GF2S: mixed unit copies missing");', 1)
  return h


def kl__sc6m_units(u):   # LDD19: 36,864 B units -> 35,840 B units; planes as the donor's _sc6_unit
  ub = u.view(-1, 36864)
  out = torch.empty((ub.shape[0], 35840), dtype=torch.uint8, device=u.device)
  out[:, :32768] = ub[:, :32768]
  for i0 in range(0, ub.shape[0], 2048):
    v = ub[i0:i0 + 2048, 32768:].view(-1, 8, 32, 16).to(torch.int32) - 0x40
    p = [torch.stack([(v[..., 4 * w + n] << 2) | ((v[..., 12 + n] >> (2 * w)) & 3) for n in range(4)], -1).to(torch.uint8) for w in range(3)]
    out[i0:i0 + 2048, 32768:] = torch.stack(p, 1).reshape(-1, 3072)
  return out.view(-1)




def kl__compose_times(fns, reps=16):
  graphs, outputs = [], []
  for fn in fns:
    fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
      out=fn()
    graphs.append(g);outputs.append(out)
  flush=torch.ones((65536,1024),device='cuda',dtype=torch.float32)
  sink=torch.empty(1024,device='cuda',dtype=torch.float32)
  events=[]
  for r in range(reps):
    order=range(len(graphs)) if r%2==0 else reversed(range(len(graphs)))
    for i in order:
      torch.sum(flush,dim=0,out=sink)
      a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
      a.record();graphs[i].replay();b.record()
      events.append((i,a,b))
  torch.cuda.synchronize()
  samples=[[] for _ in graphs]
  for i,a,b in events:samples[i].append(a.elapsed_time(b)*1000)
  return [sorted(s)[len(s)//2] for s in samples]



kl__stack_unpinned_prepare = kl_prepare

def kl_prepare(weights):
  st = kl__stack_unpinned_prepare(weights)
  bfp = st.get("bfp")
  if bfp is not None:
    for T in (1, 8, 16, 32):
      for name, layer in kl__BFP_LAYER.items():
        bfp["choice"][(layer, T)] = kl_bfp__static_choice(layer, T, bfp)
      for name in ("qkv", "o", "un"):
        st["mm_pick"][(T, name)] = "bfp"
      can_fuse = (bfp["choice"][("pooler", T)][0] == "sk2"
                  and bfp["choice"][("router", T)][0] == "sk2")
      st["mm_pick"][(T, "pr")] = "bfpf" if T <= 32 and can_fuse else "bfp"
  for T in (128, 256):
    for name in ("qkv", "o", "pr", "un"):
      st["mm_pick"][(T, name)] = True
  return st

kl__policy_parent_prepare=kl_prepare


def kl__policy_variant(st,T,mode):
  ss=dict(st)
  ss['mm_pick']=dict(st['mm_pick'])
  ss['graft_lt']=dict(st.get('graft_lt',{}))
  ss['compose_graft_enabled']=dict(st.get('compose_graft_enabled',{}))
  ss['compose_boundary_warps']=dict(st.get('compose_boundary_warps',{}))
  graft,bw,drops=mode
  ss['compose_graft_enabled'][T]=graft
  ss['compose_boundary_warps'][T]=bw
  for name in drops:ss['graft_lt'].pop((name,T),None)
  return ss


def kl_prepare(weights):
  st=kl__policy_parent_prepare(weights)
  st['compose_base_mm']=dict(st['mm_pick'])
  st['compose_base_graft_lt']=dict(st['graft_lt'])
  st['compose_active']=set()
  st['compose_selection']=[]
  st['compose_graft_enabled']={T:T in (1,8,128,256) for T in (1,8,16,32,128,256)}
  st['compose_boundary_warps']={}
  gen=torch.Generator(device=st['qkv'].device).manual_seed(1618033)
  for T in ((1,8,16,32) if LDD19_SLIM_KL else (1,8,16,32,128,256)):   # LDD19: iter2 slim
    original=st['compose_graft_enabled'][T]
    modes=[(original,16,()),(True,16,()),(False,16,()),(False,8,()),(False,4,())]
    modes=list(dict.fromkeys(modes))
    data={'stream':torch.randn(T,kl_HIDDEN,device=st['qkv'].device,generator=gen),
          'local_start':torch.zeros(1,device=st['qkv'].device,dtype=torch.int32)}
    o_in=(.1*torch.randn(T,1024,device=st['qkv'].device,generator=gen)).to(torch.bfloat16)
    ids=kl__after_attention(data,st,o_in,None,T)[2]
    counts=torch.bincount(ids.flatten().long()//32,minlength=8)
    data['local_start'].fill_(32*int(counts.argmax()))
    variants=[kl__policy_variant(st,T,mode) for mode in modes]
    ts=kl__compose_times([lambda ss=ss:kl__after_attention(data,ss,o_in,None,T) for ss in variants],reps=32)
    best=min(range(len(ts)),key=ts.__getitem__)
    if ts[best]>=ts[0]-.4:best=0
    picked=variants[best]
    st['compose_graft_enabled'][T]=picked['compose_graft_enabled'][T]
    st['compose_boundary_warps'][T]=picked['compose_boundary_warps'][T]
    st['graft_lt']=picked['graft_lt']
    st['compose_selection'].append({'T':T,'region':'chain_policy','us':ts,'modes':modes,'choice':modes[best]})
  return st

@kl__lru_cache(None)
def kl__sl_cluster_module():
  from tvm_ffi.cpp import load_inline as _tvm_load_inline
  return _tvm_load_inline(name='sl_early_tail_' + _hashlib.md5(kl__SL_GLUE_CUDA.encode()).hexdigest()[:12],
                          cuda_sources=kl__SL_GLUE_CUDA, functions=['b1'],
                          extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])

@triton.jit
def kl__sl_cluster_finish(P, T, CARRY, TAIL, BETA, POST, NEXT, SOUT, HOUT,
                       BC: tl.constexpr, USE_PDL: tl.constexpr = True):
  t = tl.program_id(0)
  tile = tl.program_id(1)
  c = tl.arange(0,1024)
  beta = tl.load(BETA+c)
  gp = tl.load(POST+c).to(tl.float32)
  carry = tl.load(CARRY+t*8192+c)
  tail = tl.load(TAIL+t)
  o = tile*BC+tl.arange(0,BC)
  outcarry = tl.load(CARRY+t*8192+o)
  gn = tl.load(NEXT+o).to(tl.float32)
  if USE_PDL:
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
  f = (tl.load(P+t*1024+c).to(tl.float32)*gp).to(tl.bfloat16).to(tl.float32)
  inv = tl.rsqrt(tl.sum(f*f,0)/8192.0+1e-8)
  n = (f*inv).to(tl.bfloat16).to(tl.float32)
  head = carry+beta*n
  invnext = tl.rsqrt((tail+tl.sum(head*head,0))/8192.0+1e-5)
  result = tl.where(o<1024,tl.gather(head,o%1024,0),outcarry)
  tl.store(SOUT+t*8192+o,result)
  tl.store(HOUT+t*8192+o,(result*invnext*gn).to(tl.bfloat16))

kl_SL_CLUSTER_CONFIG={1: (8, 256), 8: (16, 1024)}
def kl__sl_cluster_after_attention(data, state, o_in, gate, T):
  dev = o_in.device
  f32 = dict(device=dev, dtype=torch.float32)
  tail = torch.empty((T,), **f32)
  s1 = torch.empty((T, kl_HIDDEN), **f32)
  h = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  hr = torch.empty((T, kl_HIDDEN), **f32) if T <= 8 else None
  if gate is None:
    P, S = kl_dense(o_in, state["o"], T, state, "o")
  else:
    P, S = kl_dense_ng(o_in, gate, state["o"], T, state)
  if S != 1 or P.dtype != torch.bfloat16:
    return kl__graft_after_attention(data,state,o_in,gate,T)
  g=state['graft']
  kl__sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
  mmp = state.get("mm_pick", {})
  ppick = mmp.get((T, "pr"), mmp.get((kl__bucket(T), "pr"), False))
  P, S = kl__pool_router_fp32(h, hr, T, state, ppick)
  ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  wts = torch.empty((T, 8), **f32)
  moe = kl__pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(kl__bucket(T), "agent")),
                             marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
  use_marlin = moe == "marlin"
  use_ma = moe == "agent"
  if not (use_marlin or use_ma):
    raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
  xm = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
  if S == -1:
    pooled, logits = P
    kl__graft_al(pooled, logits, xm, ids, wts, lids)
    kl__sl_graft_route[(T, 2)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, SPLIT=True, EARLY=kl_SL_ROUTE_CONFIG[T][2], MODE=kl_SL_ROUTE_CONFIG[T][1], num_warps=kl_SL_ROUTE_CONFIG[T][0], launch_pdl=True)
  elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
    allp = P.view(T, 4608)
    kl__graft_al(allp, xm, ids, wts, lids)
    kl__sl_graft_route[(T, 2)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, SPLIT=False, EARLY=kl_SL_ROUTE_CONFIG[T][2], MODE=kl_SL_ROUTE_CONFIG[T][1], num_warps=kl_SL_ROUTE_CONFIG[T][0], launch_pdl=True)
  else:
    kl__route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                 PERMUTED=False, num_warps=8, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  if use_ma:
    m = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext = kl__moe(T, state)
    if kl_MOE_HAS_PDL:
      ext.okc_set_pdl(1 if kl__pdl(T) else 0)
    for m0 in range(0, T, kl__MA_MMAX):
      m1 = min(T, m0 + kl__MA_MMAX)
      ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                  state["nsm"])
  wu = state["unpool_slice"] if T <= 8 else state["unpool"]
  P, S = kl_dense(m, wu, T, state, "un")
  nu = wu.shape[0]
  s2 = torch.empty((T, kl_HIDDEN), **f32)
  out = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  if S != 1 or P.dtype != torch.bfloat16:
    raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
  nw,bc=kl_SL_CLUSTER_CONFIG[T]
  kl__sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
  return out, s2, ids


kl__sl_previous_after_attention=kl__after_attention

def kl__after_attention(data,state,o_in,gate,T):
  if T in kl_SL_CLUSTER_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
    return kl__sl_cluster_after_attention(data,state,o_in,gate,T)
  return kl__sl_previous_after_attention(data,state,o_in,gate,T)

kl__sl_previous_prepare=kl_prepare

def kl_prepare(weights):
  st=kl__sl_previous_prepare(weights)
  kl__sl_cluster_module()
  return st
@triton.jit
def kl__sl_graft_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr):
  t = tl.program_id(0)
  kind = tl.program_id(1)
  if EARLY:
    _gdc_launch()
  _gdc_wait()
  if not EARLY:
    _gdc_launch()
  if kind == 0:
    c = tl.arange(0, 4096)
    if SPLIT:
      pooled = tl.load(POOLED + t * 4096 + c).to(tl.float32)
    else:
      pooled = tl.load(ALL + t * 4608 + c)
    gpre = 1.0 + tl.load(GPRE + c).to(tl.float32)
    x = pooled * tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5) * gpre
    tl.store(XOUT + t * 4096 + c, x.to(tl.bfloat16))
  else:
    e = tl.arange(0, 256)
    if SPLIT:
      logit = tl.load(LOGITS + t * 256 + e)
    else:
      logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + tl.load(BIAS + e)
    k8 = tl.arange(0, 8)
    if MODE != 2:
      bits = sel.to(tl.uint32, bitcast=True)
      ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
      packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
      if MODE == 0:
        picked = tl.topk(packed, 8)
      else:
        sorted_values = tl.sort(packed, descending=True)
        picked = tl.gather(sorted_values, k8, 0)
      ids = (255 - (picked & 255)).to(tl.int32)
      wv = tl.gather(sc, ids, 0)
    else:
      ids = tl.zeros([8], dtype=tl.int32)
      wv = tl.zeros([8], dtype=tl.float32)
      for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - tl.load(LS) + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)

kl_SL_ROUTE_CONFIG={1: (8, 0, 0), 8: (4, 0, 0)}
@triton.jit
def kl__sl_tile_route(POOLED, LOGITS, ALL, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, SPLIT: tl.constexpr, EARLY: tl.constexpr, MODE: tl.constexpr, PARTS: tl.constexpr, PRELOAD: tl.constexpr, NORC: tl.constexpr):
  t = tl.program_id(0)
  kind = tl.program_id(1)
  if kind<PARTS:
    c=tl.arange(0,4096)
    oc=kind*(4096//PARTS)+tl.arange(0,4096//PARTS)
    if PRELOAD:
      gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
    _gdc_launch()
    _gdc_wait()
    if SPLIT:
      pooled=tl.load(POOLED+t*4096+c).to(tl.float32)
    else:
      pooled=tl.load(ALL+t*4608+c)
    inv=tl.rsqrt(tl.sum(pooled*pooled,0)/4096.0+1e-5)
    if not PRELOAD:
      gpre=1.0+tl.load(GPRE+oc).to(tl.float32)
    if PARTS==1:
      vals=pooled
    elif SPLIT:
      vals=tl.load(POOLED+t*4096+oc).to(tl.float32)
    else:
      vals=tl.load(ALL+t*4608+oc)
    tl.store(XOUT+t*4096+oc,(vals*inv*gpre).to(tl.bfloat16))
  else:
    e = tl.arange(0, 256)
    if PRELOAD:
      bias=tl.load(BIAS+e)
      ls=tl.load(LS)
    _gdc_launch()
    _gdc_wait()
    if not PRELOAD:
      bias=tl.load(BIAS+e)
      ls=tl.load(LS)
    if SPLIT:
      if NORC:   # LDD19: NORC8: LOGITS is the gle pooler's fused_part (rows = tokens); same order as its last-CTA block
        eb = (e // 8) * 1024 + (e % 8)
        logit = tl.load(LOGITS + eb + t * 8) + tl.load(LOGITS + eb + 256 + t * 8) + tl.load(LOGITS + eb + 512 + t * 8) + tl.load(LOGITS + eb + 768 + t * 8)
      else:
        logit = tl.load(LOGITS + t * 256 + e)
    else:
      logit = tl.load(ALL + t * 4608 + 4096 + e) + tl.load(ALL + t * 4608 + 4352 + e)
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + bias
    k8 = tl.arange(0, 8)
    if MODE != 2:
      bits = sel.to(tl.uint32, bitcast=True)
      ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
      packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
      if MODE == 0:
        picked = tl.topk(packed, 8)
      else:
        sorted_values = tl.sort(packed, descending=True)
        picked = tl.gather(sorted_values, k8, 0)
      ids = (255 - (picked & 255)).to(tl.int32)
      wv = tl.gather(sc, ids, 0)
    else:
      ids = tl.zeros([8], dtype=tl.int32)
      wv = tl.zeros([8], dtype=tl.float32)
      for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - ls + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)

kl_SL_TILE_CONFIG={1:(4,1,True),8:(4,4,True)}
def kl__sl_tile_after_attention(data, state, o_in, gate, T):
  dev = o_in.device
  f32 = dict(device=dev, dtype=torch.float32)
  tail = torch.empty((T,), **f32)
  s1 = torch.empty((T, kl_HIDDEN), **f32)
  h = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  hr = torch.empty((T, kl_HIDDEN), **f32) if T <= 8 else None
  if gate is None:
    P, S = kl_dense(o_in, state["o"], T, state, "o")
  else:
    P, S = kl_dense_ng(o_in, gate, state["o"], T, state)
  if S != 1 or P.dtype != torch.bfloat16:
    return kl__graft_after_attention(data,state,o_in,gate,T)
  g=state['graft']
  kl__sl_cluster_module().b1(P.view(T,8192),data['stream'],g['pre1'],g['alpha1'],g['beta1'],h,s1,g['alpha2'],tail,hr if T <= 8 else s1,T <= 8)
  mmp = state.get("mm_pick", {})
  ppick = mmp.get((T, "pr"), mmp.get((kl__bucket(T), "pr"), False))
  P, S = kl__pool_router_fp32(h, hr, T, state, ppick, norc=bool(LDD19_NORC8))   # LDD19: NORC8
  ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  wts = torch.empty((T, 8), **f32)
  moe = kl__pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(kl__bucket(T), "agent")),
                             marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
  use_marlin = moe == "marlin"
  use_ma = moe == "agent"
  if not (use_marlin or use_ma):
    raise RuntimeError("no expert backend: neither the moe agent's kernel nor SGLang's Marlin is available")
  xm = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
  if S == -1:
    pooled, logits = P
    kl__graft_al(pooled, logits, xm, ids, wts, lids)
    kl__sl_tile_route[(T,kl_SL_TILE_CONFIG[T][1]+1)](pooled, logits, logits, state["bias"], state["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, SPLIT=True, EARLY=kl_SL_ROUTE_CONFIG[T][2], MODE=kl_SL_ROUTE_CONFIG[T][1], PARTS=kl_SL_TILE_CONFIG[T][1], PRELOAD=kl_SL_TILE_CONFIG[T][2], NORC=int(logits.dim() == 1), num_warps=kl_SL_TILE_CONFIG[T][0], launch_pdl=True)
  elif S == 1 and P.shape[-1] == 4608 and P.dtype == torch.float32:
    allp = P.view(T, 4608)
    kl__graft_al(allp, xm, ids, wts, lids)
    kl__sl_tile_route[(T,kl_SL_TILE_CONFIG[T][1]+1)](allp, allp, allp, state["bias"], state["g_preexp"], ids, wts, xm,
                         data["local_start"], lids, SPLIT=False, EARLY=kl_SL_ROUTE_CONFIG[T][2], MODE=kl_SL_ROUTE_CONFIG[T][1], PARTS=kl_SL_TILE_CONFIG[T][1], PRELOAD=kl_SL_TILE_CONFIG[T][2], NORC=0, num_warps=kl_SL_TILE_CONFIG[T][0], launch_pdl=True)
  else:
    kl__route[(T,)](P, S, T, state["bias"], state["g_preexp"], ids, wts, xm, data["local_start"], lids,
                 PERMUTED=False, num_warps=8, USE_PDL=kl__pdl(T), launch_pdl=kl__pdl(T))
  if use_ma:
    m = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext = kl__moe(T, state)
    if kl_MOE_HAS_PDL:
      ext.okc_set_pdl(1 if kl__pdl(T) else 0)
    for m0 in range(0, T, kl__MA_MMAX):
      m1 = min(T, m0 + kl__MA_MMAX)
      ext.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32,
                  state["nsm"])
  wu = state["unpool_slice"] if T <= 8 else state["unpool"]
  P, S = kl_dense(m, wu, T, state, "un")
  nu = wu.shape[0]
  s2 = torch.empty((T, kl_HIDDEN), **f32)
  out = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  if S != 1 or P.dtype != torch.bfloat16:
    raise RuntimeError('small-load cluster requires BF16 sliced unpooler')
  nw,bc=kl_SL_CLUSTER_CONFIG[T]
  kl__sl_cluster_finish[(T,8192//bc)](P,T,s1,tail,g['beta2'],g['post2'],g['pre2'],s2,out,BC=bc,num_warps=nw,launch_pdl=True)
  return out, s2, ids


kl__sl_tile_prior_after=kl__after_attention
def kl__after_attention(data,state,o_in,gate,T):
  if T in kl_SL_TILE_CONFIG and 'graft' in state and state.get('compose_graft_enabled', {}).get(T, True):
    return kl__sl_tile_after_attention(data,state,o_in,gate,T)
  return kl__sl_tile_prior_after(data,state,o_in,gate,T)

kl_MOE_TINY2_DEFINES += ("@moe_helper0",)
kl_MOE_TINY_DEFINES += ("@moe_helper0", "-DOKC_FINPF=1")
# LDD19 (t8_converter, run 2 it6): TDF (integrator it4, make_tdf.py) + DFC, the TINY DF source derived when the TINY build is compiled
# (hook on kl__ma_ext: the live kl__moe calls it directly), so later edits of moe_helper0 (e.g. H2) reach it. False = f12 TINY.
LDD19_KL_TDF = True
LDD19_KL_CRED = True   # LDD19 (CRED, stack_integrator run 3 it 5): TINY DF helpers skip the dead H2 A-end credit; False = previous
LDD19_DFC = True   # LDD19: DFC (compact DF sentinel test); False = make_tdf.py's DF source
LDD19_DFC_MID = True   # LDD19: DFC also on MID's DF source (T 5..16: verify 1, decode 16), patched on its final text at build time


def kl__dfc_patch(c):   # LDD19: DFC: 2 ops per word, exact (work_t8_converter/make_tdfc.py)
  old = """  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint32_t b = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) b |= (uint32_t)(((w[i] & 0xFFFFu) == 0xFFFFu) | ((w[i] >> 16) == 0xFFFFu));
  return b;
"""
  new = """  // LDD19: DFC: bit 15 / 31 set iff a bf16 lane is 0xFFFF (SWAR zero-lane test on ~w); the caller masks with 0x80008000
  return ((0xFFFEFFFEu - v.x) & v.x) | ((0xFFFEFFFEu - v.y) & v.y) | ((0xFFFEFFFEu - v.z) & v.z) | ((0xFFFEFFFEu - v.w) & v.w);
"""
  assert c.count(old) == 1, c.count(old)
  c = c.replace(old, new)
  old = "    if (!__any_sync(0xffffffffu, bad != 0) || spin > 200000) break;\n"
  assert c.count(old) == 1, c.count(old)
  return c.replace(old, "    if (!__any_sync(0xffffffffu, (bad & 0x80008000u) != 0) || spin > 200000) break;   // LDD19: DFC mask\n")


if LDD19_KL_TDF:
  _ldd19_tdf_prev_ma_ext = kl__ma_ext

  def kl__ma_ext(defines=None):   # LDD19 (TDF, lazy): derive the TINY DF source from the final moe_helper0 text at build time
    if defines is not None and "@moe_helper0tdf" in tuple(defines):
      _s = kl__df_patch(kl__MA_ALT_SRCS["moe_helper0"])
      kl__MA_ALT_SRCS["moe_helper0tdf"] = kl__dfc_patch(_s) if LDD19_DFC else _s
      if LDD19_KL_CRED:   # LDD19 (CRED, stack_integrator run 3 it 5): the H2 A-end credit is dead under OKC_DF (TINY DF only)
        _c = kl__MA_ALT_SRCS["moe_helper0tdf"]
        _o = "      if (q + 2 >= nA) {  // LDD19 (H2)"
        assert _c.count(_o) == 1, _c.count(_o)
        kl__MA_ALT_SRCS["moe_helper0tdf"] = _c.replace(_o, "      if (!OKC_DF && q + 2 >= nA) {  // LDD19 (CRED) (H2)")
    if LDD19_DFC and LDD19_DFC_MID and defines is not None and "@moe_helper0df" in tuple(defines):   # LDD19: DFC (MID)
      _m = kl__MA_ALT_SRCS["moe_helper0df"]
      if "LDD19: DFC" not in _m:
        kl__MA_ALT_SRCS["moe_helper0df"] = kl__dfc_patch(_m)
    return _ldd19_tdf_prev_ma_ext(defines)

  kl_MOE_TINY_DEFINES = tuple("@moe_helper0tdf" if d == "@moe_helper0" else d for d in kl_MOE_TINY_DEFINES)
if LDD19_GF2S:   # LDD19: GF2S (LDD19_GF2S_TINY: also the T <= 4 TINY build; it3/it4 probes: TINY loses with every mode)
  if LDD19_GF2S_TINY:
    kl_MOE_TINY_DEFINES += ("-DLDD19_GF2S=%d" % LDD19_GF2S,)
  kl_MOE_MID_DEFINES += ("-DLDD19_GF2S=%d" % LDD19_GF2S,)

# LDD19 (run 2, solo_optimizer): two ports from the sibling MoE run (same Keell skinny-ring source family), applied to the
# derived TINY / MID source strings only (no other build's source or hash changes). False restores the f12 builds exactly.
# SCP: one bf16x2 scale pair (s_g, s_g8) per 64-K chunk (1 PRMT + 1 IMAD instead of 2 + 2); the HMUL2s broadcast its halves
#   (bmul2l / bmul2h, MAIN's form). Bit-identical: each byte * k16 < 2^16, so the halves never carry into each other.
# MIDNH: in the MID build, the weight units of a SPLIT expert's jobs (> SK_T tokens -> several jobs that each stream the
#   expert) use L2 EVICT_NORMAL instead of EVICT_FIRST so the sibling job's read hits L2; one policy word per job written at
#   routing, used as the TMA cache-hint operand (no branch on the refill thread). A cache hint only: values unchanged.
LDD19_KL_SCP = True
LDD19_KL_MIDNH = True
_LDD19_SCP_OLD = "      const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);\n"
_LDD19_SCP_NEW = ("      const uint32_t sgp = __byte_perm(swd, 0u, 0x4040u | ((uint32_t)(c & 1) * 2u) | (((uint32_t)(c & 1) * 2u + 1u) << 8)) * k16 + 0x5C005C00u;  // LDD19 (SCP)\n"
                  "#define SGL(r) bmul2l((r), sgp)\n#define SGH(r) bmul2h((r), sgp)\n")
_LDD19_SCP_LUT = "      const uint32_t sg = S.lut[sw & 0xff], sg8 = S.lut[(sw >> 8) & 0xff];\n"


def _ldd19_scp_src(s):
  n = s.count(_LDD19_SCP_OLD)
  assert n >= 1 and s.count(_LDD19_SCP_LUT) == n, (n, s.count(_LDD19_SCP_LUT))
  s = s.replace(_LDD19_SCP_OLD, _LDD19_SCP_NEW).replace(_LDD19_SCP_LUT, _LDD19_SCP_LUT + "#define SGL(r) bmul2((r), sg)\n#define SGH(r) bmul2((r), sg8)\n")
  s = s.replace("#define MULG(r) bmul2((r), sg)\n", "#define MULG(r) SGL(r)\n").replace("#define MULG8(r) bmul2((r), sg8)\n", "#define MULG8(r) SGH(r)\n")
  s = s.replace("bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8)", "SGL(r0), SGH(r1), SGL(r2), SGH(r3)")
  s = s.replace("#undef MULG8\n", "#undef MULG8\n#undef SGL\n#undef SGH\n")
  m = "    for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n"
  s = s.replace("\n" + m, "\n" + m + "#ifdef SGL\n#undef SGL\n#undef SGH\n#endif\n")
  assert "MULG(r) bmul2((r), sg)\n" not in s and "bmul2(r0, sg)," not in s
  return s


_LDD19_MIDNH_REPS = [
    ("  int njF, njS, unit;\n", "  int njF, njS, unit;\n  uint32_t jpol[JMAX];  // LDD19 (MIDNH): high word of each split job's weight L2 policy\n"),
    ("    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }\n",
     "    for (int q = 0; q < nsj; ++q) { S.jE[sj + q] = lane; S.jS[sj + q] = off + q * SK_T; S.jC[sj + q] = min(SK_T, tot - q * SK_T); }\n"
     "    for (int q = 0; q < nsj; ++q) S.jpol[sj + q] = nsj > 1 ? 0x10000000u : 0x12F00000u;  // LDD19 (MIDNH): EVICT_NORMAL / EVICT_FIRST\n"),
]
_LDD19_MIDNH_RI_OLD = """  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
}
"""
_LDD19_MIDNH_RI_NEW = """#if OKC_L2H
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)), "l"((unsigned long long)S.jpol[j] << 32) : "memory");  // LDD19 (MIDNH)
#else
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
               ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(bar)) L2H_ARG : "memory");
#endif
}
"""


def _ldd19_midnh_src(s):
  for old, new in _LDD19_MIDNH_REPS:
    assert s.count(old) == 1, (old[:60], s.count(old))
    s = s.replace(old, new)
  a = s.index("void ring_issue(")
  b = s.index(_LDD19_MIDNH_RI_OLD, a)
  assert b - a < 1200, "MIDNH: ring_issue anchor"
  return s[:b] + _LDD19_MIDNH_RI_NEW + s[b + len(_LDD19_MIDNH_RI_OLD):]


# LDD19: SCP is applied to MID only: in TINY it lost decode 1 -0.79% (run 2 it1 ab_1; 148 -> 150 regs).
if LDD19_KL_SCP:
  kl__MA_ALT_SRCS["moe_helper0df"] = _ldd19_scp_src(kl__MA_ALT_SRCS["moe_helper0df"])
if LDD19_KL_MIDNH:
  kl__MA_ALT_SRCS["moe_helper0df"] = _ldd19_midnh_src(kl__MA_ALT_SRCS["moe_helper0df"])
LDD19_KL_MIDNH1 = True   # LDD19 (MIDNH1): the FAST1 pre-issued phase-A units of split experts (S.fcnt[e] > SK_T) also NORMAL
_LDD19_MIDNH1_OLD = """        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
"""
_LDD19_MIDNH1_NEW = """#if OKC_L2H && OKC_SPLIT
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)),
                       "l"((unsigned long long)(S.fcnt[e] > SK_T ? 0x10000000u : 0x12F00000u) << 32) : "memory");  // LDD19 (MIDNH1)
#else
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes" L2H_SUFFIX " [%0], [%1], %2, [%3]" L2H_OPERAND ";\\n"
                     ::"r"(smem_u32(slot)), "l"(src), "r"(UNIT_BYTES), "r"(smem_u32(S.bars + i)) L2H_ARG : "memory");
#endif
"""
if LDD19_KL_MIDNH and LDD19_KL_MIDNH1:
  assert kl__MA_ALT_SRCS["moe_helper0df"].count(_LDD19_MIDNH1_OLD) == 1
  kl__MA_ALT_SRCS["moe_helper0df"] = kl__MA_ALT_SRCS["moe_helper0df"].replace(_LDD19_MIDNH1_OLD, _LDD19_MIDNH1_NEW)


# LDD19 (skinny_engineer, run 2 it4): H2 = two helper warps in the WS skinny builds (TINY / MID): 10 warps, helpers on
# SMSP0 and SMSP1 take alternate units, the TMA-issuing consumer warp moves to SMSP2. LDD19_KL_H2 = False restores the old sources.
LDD19_KL_H2 = True
LDD19_KL_H2_TINY = True
LDD19_KL_H2_MID = True


def _ldd19_h2_src(s, df):
  def one(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old))
    return s.replace(old, new)
  s = "#ifndef LDD19_H2\n#define LDD19_H2 1\n#endif\n" + s
  s = one(s, "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);\n",
          "constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 * (1 + LDD19_H2) : 0);  // LDD19 (H2): 2 helper warps\n")
  a = s.index("__device__ void ws_helper(")
  b = s.index("template <bool BIG, bool PB>\n__device__ __forceinline__ void skinny_split(", a)
  h = s[a:b]
  h = one(h, "  const int lane = TIDX & 31;\n",
          "  const int lane = TIDX & 31;\n  const int hw_ = LDD19_H2 ? ((TIDX - NT) >> 5) : 0;  // LDD19 (H2): helper index\n")
  h = one(h, "  if (lane == 0)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n",
          "  if (lane == 0 && hw_ == 0)  // LDD19 (H2)\n    for (int q = S.preissued; q < OKC_NS && q < N; ++q) ws_issue(p, S, q, nA, a0, b0, ring);\n")
  h = one(h, "  for (int q = 0; q < Nh; ++q) {\n",
          "#if LDD19_H2\n#if !OKC_DF\n  if (nA > 0 && hw_ >= nA) { __threadfence(); asm volatile(\"bar.sync 3, 64;\\n\" ::: \"memory\"); }  // LDD19 (H2): no A unit here\n#endif\n"
          "  for (int q = hw_; q < Nh; q += 2) {  // LDD19 (H2): alternate units\n#else\n  for (int q = 0; q < Nh; ++q) {\n#endif\n")
  if df:
    h = one(h, "      if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters\n",
            "#if LDD19_H2 && !OKC_DF\n#error \"LDD19_H2 needs the DF hand-off on this build\"\n#endif\n"
            "      if (!OKC_DF && q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters\n")
  else:
    old = ("      if (q == nA - 1) {  // this CTA's phase A is done: credit its units to the jobs' counters\n"
           "        __threadfence();\n        __syncwarp();\n        if (lane == 0) {\n")
    h = one(h, old,
            "#if LDD19_H2\n      if (q + 2 >= nA) {  // LDD19 (H2): this helper's last A unit; both helpers fence, meet, the owner of nA-1 credits\n"
            "        __threadfence();\n        asm volatile(\"bar.sync 3, 64;\\n\" ::: \"memory\");\n        if (q == nA - 1 && lane == 0) {\n"
            "#else\n" + old + "#endif\n")
  return s[:a] + h + s[b:]


if LDD19_KL_H2:
  if LDD19_KL_H2_TINY:
    kl__MA_ALT_SRCS["moe_helper0"] = _ldd19_h2_src(kl__MA_ALT_SRCS["moe_helper0"], False)
  if LDD19_KL_H2_MID:
    kl__MA_ALT_SRCS["moe_helper0df"] = _ldd19_h2_src(kl__MA_ALT_SRCS["moe_helper0df"], True)

# LDD19 (moe_ablator): NA2: two HMMA accumulator chains in the WS skinny consumer (8 FADD per unit less). Bits: 1 MID, 2 TINY. 0 = previous sources.
LDD19_KL_NA2 = 1


def _ldd19_na2_src(s):   # LDD19: NA1
  a = s.index("__device__ void skinny_ring_ws(")
  b = s.index("__device__ __forceinline__ void ws_issue(", a)
  h = s[a:b]
  o = "    float acc3[4] = {0.f, 0.f, 0.f, 0.f}, acc4[4] = {0.f, 0.f, 0.f, 0.f};\n"
  assert h.count(o) == 1
  h = h.replace(o, "")
  o = "st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4"
  assert h.count(o) == 1
  h = h.replace(o, "(st & 1) ? acc2 : acc")
  o = "for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n"
  assert h.count(o) == 1
  h = h.replace(o, "for (int k = 0; k < 4; ++k) acc[k] += acc2[k];\n")
  return s[:a] + h + s[b:]


if LDD19_KL_NA2 & 1:   # LDD19: NA1
  kl__MA_ALT_SRCS["moe_helper0df"] = _ldd19_na2_src(kl__MA_ALT_SRCS["moe_helper0df"])

kl__DE_EXT=None
kl__DE_RUN=kl_bfp__run_sk2
kl__DE_PREPARE=kl_prepare

def kl__de_source():
 cfg=('qkv8n_r144','qkv_gate',2304,24,3,1,4,2,2,1)
 _,nstage=kl_bfp__sk2_geom(cfg,8)
 dispatch=f'launch_sk2<1, 8, 3, 1, 4, 2, {nstage}, true, false, 2, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return;'
 source=kl_bfp_SK2_SRC.replace('@@SK2_DISPATCH@@',dispatch).replace('@@SK2F_DISPATCH@@','')
 source=source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)','4').replace('NW + (COMP ? NESC : 0)','NW')
 source=source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");','')
 a=source.index('      int pe = 1;');b=source.index('\n    }\n    mbar_arrive(cbar);',a)
 source=source[:a]+r"""
      for(int pe=1+ew;pe<=cnt;pe+=NESC) {
        const int2 e=ptab_s[pe];const int kk=e.x&65535;
        if(lane<M && kk>=gs0*KST && kk<(gs0+nst)*KST)
          cr[lane*CP+(e.x>>16)] += __int_as_float(e.y)*__bfloat162float(x[(size_t)lane*K+kk]);
      }
"""+source[b:]
 shared=kl_bfp_CUDA_SRC.replace('@@MBAR@@',kl_bfp_MBAR_HELPERS).replace('@@BODY@@','')
 return shared+kl_bfp_MAP_SRC+source

def kl__de_ext():
 global kl__DE_EXT
 if kl__DE_EXT is None:
  import hashlib
  source=kl__de_source();tag=hashlib.md5(source.encode()).hexdigest()[:12];source=source.replace('sk2_kernel','sk2_kernel_narrow_'+tag).replace('launch_sk2','launch_sk2_narrow_'+tag);cpp=kl_bfp_CPP_SRC[kl_bfp_CPP_SRC.index('void sk2('):kl_bfp_CPP_SRC.index('void sk2f(')]+'void sk2_set_pdl(int64_t n);'
  kl__DE_EXT=kl_load_inline('proj_dl_escape_'+hashlib.md5(source.encode()).hexdigest()[:12],'#include <torch/extension.h>\n'+cpp,
    cuda_sources=source,functions=['sk2','sk2_set_pdl'],no_implicit_headers=True,extra_cuda_cflags=['-O3','-gencode=arch=compute_90,code=sm_90'],verbose=False)
 return kl__DE_EXT

def kl_prepare(weights):
 kl__de_ext()
 return kl__DE_PREPARE(weights)

LDD19_KL_NAR1 = 1   # LDD19: NAR1 (kernel_diff_auditor r4i2): the narrow qkv sk2 also at m = 1 (0: upstream's 2 <= m <= 8)
def kl_bfp__run_sk2(x,layer,m,n,k,state,dg=0,dbg=None):
 if layer!='qkv_gate' or not (1 if LDD19_KL_NAR1 else 2)<=m<=8:return kl__DE_RUN(x,layer,m,n,k,state,dg,dbg)
 name=kl_bfp__sk2_name(layer,m);ci,stream,tab,pmax,b4,n_rows=state['sk2'][name]
 out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
 ext=kl__de_ext();ext.sk2_set_pdl(kl_bfp_SK2_PDL)
 ext.sk2(stream,x,out,tab,pmax,b4,m,n,k,ci,kl_bfp__sk2_tokr(m),dg,state['dummy'] if dbg is None else dbg)
 return out


# LDD19: NX (kernel_diff_auditor r4i2): the narrow escape transform of kl__de_source on the stock configs in LDD19_KL_NX_SET.
LDD19_KL_NX_SET = {('unpooler', 8)}
LDD19_KL_NX_ALL = {('unpooler', 8)}   # (layer, tokr) pairs compiled into the NX extension
kl__NX_EXT = None
def kl__nx_source(nx_all=None):
  nx_all = LDD19_KL_NX_ALL if nx_all is None else nx_all   # LDD19: Q16B passes its own set
  lines = []
  for ci, cfg in enumerate(kl_bfp_SK2_CONFIGS):
    name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
    for tokr in (8, 16, 32):
      if (layer, tokr) not in nx_all or kl_bfp_SK2_PICK.get((layer, tokr)) != name or layer == 'router' or CL != 1:
        continue
      hl, nstage = kl_bfp__sk2_geom(cfg, tokr)
      mt = 2 if tokr == 32 else 1
      nacc = 1 if (mt == 2 and nstage >= 4 and NT == 4) else NACC
      lines.append(f'if (cfg == {ci} && tokr == {tokr}) {{ launch_sk2<{mt}, {tokr}, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, '
                   f'{nacc}, 1, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, dg, dbg); return; }}')
  source = kl_bfp_SK2_SRC.replace('@@SK2_DISPATCH@@', '\n  '.join(lines)).replace('@@SK2F_DISPATCH@@', '')
  source = source.replace('(NSTAGE >= 4 ? 4 : NSTAGE)', '4').replace('NW + (COMP ? NESC : 0)', 'NW')
  source = source.replace('  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");', '')
  a = source.index('      int pe = 1;'); b = source.index('\n    }\n    mbar_arrive(cbar);', a)
  source = source[:a] + r"""
      for(int pe=1+ew;pe<=cnt;pe+=NESC) {
        const int2 e=ptab_s[pe];const int kk=e.x&65535;
        if(lane<M && kk>=gs0*KST && kk<(gs0+nst)*KST)
          cr[lane*CP+(e.x>>16)] += __int_as_float(e.y)*__bfloat162float(x[(size_t)lane*K+kk]);
      }
""" + source[b:]
  shared = kl_bfp_CUDA_SRC.replace('@@MBAR@@', kl_bfp_MBAR_HELPERS).replace('@@BODY@@', '')
  return shared + kl_bfp_MAP_SRC + source

def kl__nx_ext():
  global kl__NX_EXT
  if kl__NX_EXT is None and LDD19_KL_NX_ALL:
    import hashlib
    source = kl__nx_source(); tag = hashlib.md5(source.encode()).hexdigest()[:12]
    source = source.replace('sk2_kernel', 'sk2_kernel_nx_' + tag).replace('launch_sk2', 'launch_sk2_nx_' + tag)
    cpp = kl_bfp_CPP_SRC[kl_bfp_CPP_SRC.index('void sk2('):kl_bfp_CPP_SRC.index('void sk2f(')] + 'void sk2_set_pdl(int64_t n);'
    kl__NX_EXT = kl_load_inline('proj_dl_nx_' + hashlib.md5(source.encode()).hexdigest()[:12], '#include <torch/extension.h>\n' + cpp,
      cuda_sources=source, functions=['sk2', 'sk2_set_pdl'], no_implicit_headers=True,
      extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'], verbose=False)
  return kl__NX_EXT

# LDD19: Q16B (measure_integrator r4i3): NX on qkv_gate at T 16 in a SECOND extension (the NX-un8 .so stays byte-identical).
LDD19_KL_NX2_SET = {('qkv_gate', 16), ('o', 16), ('unpooler', 16)}   # set() = K1+K2 behaviour
kl__NX2_EXT = None
def kl__nx2_ext():
  global kl__NX2_EXT
  if kl__NX2_EXT is None and LDD19_KL_NX2_SET:
    import hashlib
    source = kl__nx_source(LDD19_KL_NX2_SET); tag = hashlib.md5(source.encode()).hexdigest()[:12]
    source = source.replace('sk2_kernel', 'sk2_kernel_nx_' + tag).replace('launch_sk2', 'launch_sk2_nx_' + tag)
    cpp = kl_bfp_CPP_SRC[kl_bfp_CPP_SRC.index('void sk2('):kl_bfp_CPP_SRC.index('void sk2f(')] + 'void sk2_set_pdl(int64_t n);'
    kl__NX2_EXT = kl_load_inline('proj_dl_nx_' + hashlib.md5(source.encode()).hexdigest()[:12], '#include <torch/extension.h>\n' + cpp,
      cuda_sources=source, functions=['sk2', 'sk2_set_pdl'], no_implicit_headers=True,
      extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'], verbose=False)
  return kl__NX2_EXT

kl__NX_PRIOR_RUN = kl_bfp__run_sk2
def kl_bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):   # LDD19: NX hook (after the upstream narrow qkv hook)
  tokr = kl_bfp__sk2_tokr(m)
  nx2 = (layer, tokr) in LDD19_KL_NX2_SET and m > 8   # LDD19: Q16B
  if not nx2 and ((layer, tokr) not in LDD19_KL_NX_SET or (layer == 'qkv_gate' and m <= 8)):
    return kl__NX_PRIOR_RUN(x, layer, m, n, k, state, dg, dbg)
  name = kl_bfp__sk2_name(layer, m); ci, stream, tab, pmax, b4, n_rows = state['sk2'][name]
  out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
  ext = kl__nx2_ext() if nx2 else kl__nx_ext(); ext.sk2_set_pdl(kl_bfp_SK2_PDL)
  ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, tokr, dg, state['dummy'] if dbg is None else dbg)
  return out


# LDD19: ER32 (proj_builder r4i5, PROJ.md): the T = 32 fused pooler + BF16 router sk2f in its own extension, with ER escape warps:
# on another warp's stage they hand the TMA ring slot back before walking the escape table; on an owned stage they copy the stage's
# escape x values into registers in batches of 8 and hand the slot back before the RMWs (same RMW order: bit-identical outputs).
# The main kl bfp extension is untouched. 0 = base behaviour (no extension built, T = 32 stays on the main kl bfp sk2f).
LDD19_ER32 = 1
LDD19_ER32X = {'qkv_gate', 'unpooler'}   # LDD19: ER32X layers whose T = 32 sk2 also runs on the ER extension (set() = ER32 only)
kl__ER32_OLD = '        if ((st % NESC) != ew) {\n          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;\n          __syncwarp();\n          if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n          continue;\n        }\n        const unsigned char* xs = gbase + slot * SST;\n        while (pe <= cnt) {\n          const int2 e = ptab_s[pe];\n          const int kk = e.x & 0xFFFF;\n          if (kk >= kend) break;\n          if (lane < M && kk >= kbeg) {\n            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;\n            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(\n                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);\n            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);\n          }\n          ++pe;\n        }\n        __syncwarp();\n        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n      }'
kl__ER32_NEW = '        constexpr int ERB = (FUSE && MT == 2) ? 8 : (MT == 2 && NT == 3) ? 8 : (MT == 2 && NWR == 2 && NT == 4) ? 4 : 0;   // LDD19: ER32 / ER32X\n        if constexpr (ERB > 0) {\n        if ((st % NESC) != ew) {\n          __syncwarp();\n          if (lane == 0) mbar_arrive(empty0 + 8 * slot);   // LDD19: ER early release (the walk reads only the table)\n          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;\n          continue;\n        }\n        const unsigned char* xs = gbase + slot * SST;\n        for (bool held = true; held;) {   // LDD19: ER batched gather, release after the stage\'s last x copy\n          int2 e[ERB]; float xv[ERB]; int nv = 0;\n#pragma unroll\n          for (int u = 0; u < ERB; ++u) e[u] = ptab_s[min(pe + u, cnt)];\n#pragma unroll\n          for (int u = 0; u < ERB; ++u) {\n            const int kk = e[u].x & 0xFFFF;\n            const bool in = pe + u <= cnt && kk < kend;   // sorted by k: the in-stage entries are a prefix\n            nv += in;\n            xv[u] = 0.f;\n            if (in && lane < M && kk >= kbeg) {\n              const int kl = kk - kbeg, uu = kl >> 6, kc = kl & 63;\n              xv[u] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(\n                  xs + uu * XU + lane * 128 + (((kc >> 3) ^ ((uu * Mb + lane) & 7)) << 4) + (kc & 7) * 2));\n            }\n          }\n          if (nv < ERB) {   // the stage\'s last batch: its x values must be in registers before the slot is handed back\n#pragma unroll\n            for (int u = 0; u < ERB; ++u) asm volatile("" :: "f"(xv[u]));\n            __syncwarp();\n            if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n            held = false;\n          }\n#pragma unroll\n          for (int u = 0; u < ERB; ++u)\n            if (u < nv && lane < M && (e[u].x & 0xFFFF) >= kbeg) cr[lane * CP + (e[u].x >> 16)] += __int_as_float(e[u].y) * xv[u];\n          pe += nv;\n        }\n        } else {\n        if ((st % NESC) != ew) {\n          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;\n          __syncwarp();\n          if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n          continue;\n        }\n        const unsigned char* xs = gbase + slot * SST;\n        while (pe <= cnt) {\n          const int2 e = ptab_s[pe];\n          const int kk = e.x & 0xFFFF;\n          if (kk >= kend) break;\n          if (lane < M && kk >= kbeg) {\n            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;\n            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(\n                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);\n            cr[lane * CP + (e.x >> 16)] += __int_as_float(e.y) * __bfloat162float(v);\n          }\n          ++pe;\n        }\n        __syncwarp();\n        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n        }\n      }'
kl__ER_EXT = None


LDD19_ER_TN = 1
LDD19_ER_TN16 = 1
kl__ER_TN_EDITS = (("""    named_bar(1, NC);
    __shared__ int s_rlast;
    if (tid == 0) {
""", """    if (!norc_) named_bar(1, NC);   // LDD19: TN2
    __shared__ int s_rlast;
    if (!norc_) {   // LDD19: TN2
    if (tid == 0) {
"""), ("""        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);
  }
""", """        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;
      }
    }
    named_bar(1, NC);
    }
  }
"""))


def kl__er_source():
  s = kl_bfp__sk2_src(True)
  keep = []
  for line in s.split("\n"):
    t = line.strip()
    if t.startswith("if (tokr == 8)") or (t.startswith("if (tokr == 16)") and not (LDD19_ER_TN and LDD19_ER_TN16)):
      continue   # only the sk2f tokr-32 instantiation (LDD19: TN2 + tokr 16)
    if t.startswith("if (cfg == ") and not any(t.startswith("if (cfg == %d && tokr == 32)" % ci) for ci, c in enumerate(kl_bfp_SK2_CONFIGS)
                                             if c[1] in LDD19_ER32X and kl_bfp_SK2_PICK.get((c[1], 32)) == c[0]):
      continue   # ER32X: the picked T = 32 sk2 configs of the LDD19_ER32X layers only
    keep.append(line)
  s = "\n".join(keep)
  assert s.count(kl__ER32_OLD) == 1 and s.count("if (tokr == 32)") == 1 and s.count("&& tokr == 32)") == len(LDD19_ER32X)
  s = s.replace(kl__ER32_OLD, kl__ER32_NEW)
  if LDD19_ER_TN:   # LDD19: TN2 (solo_optimizer r5 it1): no s_rlast round and no barriers around it under NORC16 (bit-identical; proj_builder's TN at T 16 / 32)
    for a, b in kl__ER_TN_EDITS:
      assert s.count(a) == 1, a[:60]
      s = s.replace(a, b)
  shared = kl_bfp_CUDA_SRC.replace("@@MBAR@@", kl_bfp_MBAR_HELPERS).replace("@@BODY@@", "")
  return shared + kl_bfp_MAP_SRC + s


def kl__er_ext():
  global kl__ER_EXT
  if kl__ER_EXT is None and LDD19_ER32:
    import hashlib
    source = kl__er_source(); tag = hashlib.md5(source.encode()).hexdigest()[:12]
    source = source.replace("sk2_kernel", "sk2_kernel_er_" + tag).replace("launch_sk2", "launch_sk2_er_" + tag)
    c = kl_bfp_CPP_SRC; b = c.index("void sk2f(")
    cpp = c[c.index("void sk2("):b] + c[b:c.index(";", b) + 1] + "\nvoid sk2_set_pdl(int64_t n);"
    kl__ER_EXT = kl_load_inline("proj_dl_er_" + tag, "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
      functions=["sk2", "sk2f", "sk2_set_pdl"], no_implicit_headers=True,
      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
  return kl__ER_EXT


kl__ER_PRIOR_RUN = kl_bfp__run_sk2
def kl_bfp__run_sk2(x, layer, m, n, k, state, dg=0, dbg=None):   # LDD19: ER32X hook (after the NX hook)
  if not (LDD19_ER32 and layer in LDD19_ER32X and kl_bfp__sk2_tokr(m) == 32):
    return kl__ER_PRIOR_RUN(x, layer, m, n, k, state, dg, dbg)
  name = kl_bfp__sk2_name(layer, m); ci, stream, tab, pmax, b4, n_rows = state['sk2'][name]
  out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
  ext = kl__er_ext(); ext.sk2_set_pdl(kl_bfp_SK2_PDL)
  ext.sk2(stream, x, out, tab, pmax, b4, m, n, k, ci, 32, dg, state['dummy'] if dbg is None else dbg)
  return out

# Fresh-thread primary-context binding, following matvey's helper and the
import threading as kl__GlueContextThreading
kl__GLUE_CONTEXT_TLS = kl__GlueContextThreading.local()
kl__GLUE_CONTEXT_API = None

def kl__glue_context_api():
  global kl__GLUE_CONTEXT_API
  if kl__GLUE_CONTEXT_API is None:
    api = kl__C.CDLL("libcuda.so.1")
    for name, args in (
        ("cuCtxGetCurrent", [kl__C.POINTER(kl__C.c_void_p)]),
        ("cuCtxGetDevice", [kl__C.POINTER(kl__C.c_int)]),
        ("cuDeviceGet", [kl__C.POINTER(kl__C.c_int), kl__C.c_int]),
        ("cuDevicePrimaryCtxRetain", [kl__C.POINTER(kl__C.c_void_p), kl__C.c_int]),
        ("cuCtxSetCurrent", [kl__C.c_void_p]),
    ):
      function = getattr(api, name)
      function.argtypes, function.restype = args, kl__C.c_int
    kl__GLUE_CONTEXT_API = api
  return kl__GLUE_CONTEXT_API

def kl__glue_context_check(status, operation):
  if status:
    raise RuntimeError(f"{operation} failed with CUDA driver status {status}")

def kl__glue_bind_context(x):
  index = x.device.index if x.device.index is not None else 0
  if getattr(kl__GLUE_CONTEXT_TLS, "device", None) == index:
    return
  api = kl__glue_context_api()
  context = kl__C.c_void_p()
  kl__glue_context_check(api.cuCtxGetCurrent(kl__C.byref(context)), "cuCtxGetCurrent")
  current_device = kl__C.c_int(-1)
  if context.value:
    kl__glue_context_check(api.cuCtxGetDevice(kl__C.byref(current_device)), "cuCtxGetDevice")
  if not context.value or current_device.value != index:
    device, primary = kl__C.c_int(), kl__C.c_void_p()
    kl__glue_context_check(api.cuDeviceGet(kl__C.byref(device), index), "cuDeviceGet")
    kl__glue_context_check(api.cuDevicePrimaryCtxRetain(kl__C.byref(primary), device), "cuDevicePrimaryCtxRetain")
    kl__glue_context_check(api.cuCtxSetCurrent(primary), "cuCtxSetCurrent")
  kl__GLUE_CONTEXT_TLS.device = index

kl__glue_context_parent_kernel = kl_kernel_fn
def kl_kernel_fn(data, state):
  kl__glue_bind_context(data["attn_input"])
  return kl__glue_context_parent_kernel(data, state)



@triton.jit
def kl__router_input_norm(S1, G, OUT):
  t = tl.program_id(0)
  offs = tl.arange(0, 8192)
  s1 = tl.load(S1 + t * 8192 + offs)
  gw = 1.0 + tl.load(G + offs).to(tl.float32)
  hh = s1 * tl.rsqrt(tl.sum(s1 * s1, 0) / 8192.0 + 1e-5) * gw
  tl.store(OUT + t * 8192 + offs, hh)


def kl__pool_router_fp32(h, hr, T, state, ppick, norc=False):   # LDD19: NORC16 kwarg
  if T <= 8:
    if state.get("bfp") is not None and ppick in ("bfp", "bfpf"):
      if ppick == "bfpf":
        kl__bfp_pdl(T, "pool")
      pooled = kl__bfp_gemm(state, "pool", h, T, kl_D_EXPERT, kl_HIDDEN)
    else:
      pooled = F.linear(h, state["pr"][:kl_D_EXPERT])
    logits = kl__router_fp32_gemv(hr, state)
    return (pooled, logits), -1
  if ppick == "bfpf" and T <= 32 and state.get("bfp") is not None:
    kl__bfp_pdl(T, "pool")
    return kl_bfp_fused_pooler_router(h, state["bfp"], norc=bool(norc and T in (16, 32))), -1   # LDD19: NORC16
  if ppick == "bfp" and state.get("bfp") is not None:
    pooled = kl__bfp_gemm(state, "pool", h, T, kl_D_EXPERT, kl_HIDDEN)
    logits = kl__bfp_gemm(state, "router", h, T, kl_N_EXPERTS, kl_HIDDEN)
    return (pooled, logits), -1
  return kl_dense(h, state["pr"], T, state, "pr")


kl__tune_with_bias_restore = kl__tune


def kl__tune(st, log=False):
  bias0 = st["bias"].clone()
  try:
    return kl__tune_with_bias_restore(st, log=log)
  finally:
    st["bias"].copy_(bias0)


kl__moe_global_cache = kl__moe


class kl__RouterExt:
  def __init__(self, module):
    self._module = module

  def __getattr__(self, name):
    return getattr(self._module, name)

  def __deepcopy__(self, memo):
    return self


def kl__moe(M=None, state=None):
  if state is None:
    return kl__moe_global_cache(M)
  sfx = tuple(state["ma_sfx"])
  if M is not None and kl_MOE_TINY_DEFINES and M <= kl_MOE_TINY_MAX_M:
    key, defines = "tiny", kl_MOE_TINY_DEFINES
  elif M is not None and kl_MOE_TINY2_DEFINES and M <= kl_MOE_TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(kl_MOE_TINY2_DEFINES)):
    key, defines = "tiny2", kl_MOE_TINY2_DEFINES
  elif M is not None and kl_MOE_MID_DEFINES and M <= kl_MOE_MID_MAX_M:
    key, defines = "mid", kl_MOE_MID_DEFINES
  elif M is not None and 16 < M <= 32 and tuple(kl__main_defines(sfx)) == tuple(kl_MOE_DEFINES):
    key, defines = "b4", tuple(kl_MOE_DEFINES) + ("-DOKC_B4FIN=1",)
  else:
    key, defines = "main", kl__main_defines(sfx)
  ext = state["ma_ext"].get(key)
  if ext is None:
    ext = kl__RouterExt(kl__ma_ext(tuple(defines) + sfx))
    if kl_MOE_HAS_PDL:
      ext.okc_set_pdl(0)
    state["ma_ext"][key] = ext
  return ext


kl__router_fp32_parent_prepare = kl_prepare


def kl_prepare(weights):
  st = kl__router_fp32_parent_prepare(weights)
  for T in ((1, 8, 16, 32) if LDD19_SLIM_KL else (1, 8, 16, 32, 128, 256)):   # LDD19: iter2 slim
    kl__moe(T, st)
  return st



kl__ROUTER_FP32_CUDA = r"""#include <cuda_runtime.h>
#include <stdint.h>
__global__ __launch_bounds__(256) void router_fp32_kernel(
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int m) {
  const int expert = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* row = w + (size_t)expert * 8192;
  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int k = tid * 4; k < 8192; k += 1024) {
    const float4 b = *(const float4*)(row + k);
#pragma unroll
    for (int t = 0; t < 8; ++t) if (t < m) {
      const float4 a = *(const float4*)(x + (size_t)t * 8192 + k);
      acc[t] = fmaf(a.x, b.x, acc[t]);
      acc[t] = fmaf(a.y, b.y, acc[t]);
      acc[t] = fmaf(a.z, b.z, acc[t]);
      acc[t] = fmaf(a.w, b.w, acc[t]);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int d = 16; d; d >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], d);
  }
  __shared__ float partial[8][8];
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < 8; ++t) partial[warp][t] = acc[t];
  }
  __syncthreads();
  if (warp < m) {
    float v = lane < 8 ? partial[lane][warp] : 0.f;
#pragma unroll
    for (int d = 4; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    if (lane == 0) out[(size_t)warp * 256 + expert] = v;
  }
}
int router_fp32_launch(int64_t x, int64_t w, int64_t out, int m, int64_t stream) {
  if (m < 1 || m > 8) return -1;
  router_fp32_kernel<<<256, 256, 0, (cudaStream_t)stream>>>(
      (const float*)x, (const float*)w, (float*)out, m);
  return (int)cudaGetLastError();
}
"""
kl__ROUTER_FP32_EXT = None


def kl__router_fp32_ext():
  global kl__ROUTER_FP32_EXT
  if kl__ROUTER_FP32_EXT is None:
    import hashlib
    digest = hashlib.md5(kl__ROUTER_FP32_CUDA.encode()).hexdigest()[:12]
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint router_fp32_launch(int64_t,int64_t,int64_t,int,int64_t);'
    kl__ROUTER_FP32_EXT = kl_load_inline(
        name='router_fp32_' + digest,
        cpp_sources=cpp,
        cuda_sources=kl__ROUTER_FP32_CUDA,
        functions=['router_fp32_launch'],
        no_implicit_headers=True,
        with_pytorch_error_handling=False,
        verbose=False,
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90,code=sm_90'])
  return kl__ROUTER_FP32_EXT


def kl__router_fp32_gemv(x, state):
  m = x.shape[0]
  out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
  rc = state['router_fp32_ext'].router_fp32_launch(
      x.data_ptr(), state['router_fp32'].data_ptr(), out.data_ptr(), m,
      torch.cuda.current_stream().cuda_stream)
  if rc:
    raise RuntimeError(f'FP32 router launch failed: {rc}')
  return out


kl__router_fp32_native_parent_prepare = kl_prepare


def kl_prepare(weights):
  state = kl__router_fp32_native_parent_prepare(weights)
  state['router_fp32_ext'] = kl__RouterExt(kl__router_fp32_ext())
  return state


kl__kfix_router_fp32_gemv = kl__router_fp32_gemv
def kl__router_fp32_gemv(x, state):
  if state.get('router_fp32_ext') is None:
    state['router_fp32_ext'] = kl__RouterExt(kl__router_fp32_ext())
  return kl__kfix_router_fp32_gemv(x, state)


kl_GLE_T = {1: True}
kl_GLE_FRT = True
LDD19_NORC = 1   # LDD19: NORC on/off (ramp_engineer, run 2 it5; R3'); 0 = previous behaviour
LDD19_NORC16 = 1   # LDD19: NORC16, the same for the T = 16 / 32 stock bfp fused pooler (t8_converter, run 2 it6); 0 = previous behaviour
LDD19_NORC8 = 1   # LDD19: NORC8, the same for the T = 8 sl_tile path (gle pooler + kl__sl_tile_route); 0 = previous behaviour
kl_GLE_ILV = 1
kl_GLE_CAPTURE = None
kl_GLE_RC = {1: 8}
kl__GLE_EXT = None


def kl__gle_source():
  cfg = next(c for c in kl_bfp_SK2_CONFIGS if c[0] == kl_bfp_SK2_PICK[("o", 8)])
  name, layer, n_rows, R, NT, NWR, WSK, UPW, NACC, CL = cfg
  assert (layer, R, NWR * WSK, CL) == ("o", 32, 4, 1), cfg
  _, nstage = kl_bfp__sk2_geom(cfg, 8)
  s = kl_bfp_SK2_SRC
  _nr = ("    if (tid == 0) {\n      int old;\n"   # LDD19: NORC (route sums the 4 router partials; no counter)
         "      asm volatile(\"atom.add.release.gpu.global.s32 %0, [%1], 1;\" : \"=r\"(old) : \"l\"(rcnt + rt) : \"memory\");\n")
  assert s.count(_nr) == 1
  s = s.replace(_nr, ("    if (tid == 0) {\n#if LDD19_NORC\n      if (ep.norc) { s_rlast = 0; } else\n#endif\n      {\n      int old;\n"
                     "      asm volatile(\"atom.add.release.gpu.global.s32 %0, [%1], 1;\" : \"=r\"(old) : \"l\"(rcnt + rt) : \"memory\");\n"), 1)
  _nr2 = "      if (s_rlast) asm volatile(\"fence.acq_rel.gpu;\" ::: \"memory\");\n    }\n"
  assert s.count(_nr2) == 1
  s = s.replace(_nr2, "      if (s_rlast) asm volatile(\"fence.acq_rel.gpu;\" ::: \"memory\");\n      }\n    }\n", 1)   # LDD19: NORC
  s = s.replace("#define SK2_DIAG 0", "#define SK2_DIAG 0\n"
                "struct GleEpi { const float *s, *a, *b, *a2; const __nv_bfloat16* p; __nv_bfloat16* uv; float* part; float* uvf; const float *x32, *w32; int fpr, ilv, norc; };\n"
                "static GleEpi g_gle;\n#ifndef LDD19_NORC\n#define LDD19_NORC 1\n#endif", 1)  # LDD19: NORC
  s = s.replace("           float* __restrict__ part, int* __restrict__ rcnt, int pdl) {",
                "           float* __restrict__ part, int* __restrict__ rcnt, int pdl, const GleEpi ep) {", 1)
  pre = r"""  __shared__ __align__(16) float4 gx_s[64], gx_a[64], gx_b[64], gx_a2[64];
  __shared__ __align__(16) uint2 gx_p[64];
  // the epilogue's input-side vectors (residual stream row, alpha / beta / alpha2 / pre) go to shared memory by
  // cp.async before the wait and are awaited only in the epilogue: no registers held across the main loop (2 CTAs / SM
  // next to attention) and no stall behind the pre-wait weight stages
  // ep.s is data['stream'], the layer's input residual stream: nothing in the call writes it, which is what makes this
  // pre-wait copy safe (keep it that way: any in-call writer of data['stream'] must move this copy after the wait)
  if (!FUSE && tid < 8 * M) {   // o only (this source also holds the fused pooler)
    const size_t col = (size_t)cta * R + (tid & 7) * 4;
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&gx_s[tid]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d0), "l"(ep.s + (size_t)(tid >> 3) * N + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a[tid])), "l"(ep.a + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_b[tid])), "l"(ep.b + col) : "memory");
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_a2[tid])), "l"(ep.a2 + col) : "memory");
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"((uint32_t)__cvta_generic_to_shared(&gx_p[tid])), "l"(ep.p + col) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
"""
  anchor = '  __syncthreads();\n  if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");   // every thread'
  assert s.count(anchor) == 1
  s = s.replace(anchor, pre + anchor, 1)
  epi = r"""  constexpr int R4 = R / 4;
  if constexpr (!FUSE) {   // o: the gle epilogue; the fused pooler keeps the stock epilogue loop below
  static_assert(R4 == 8 && CL == 1 && sizeof(OutT) == 2, "gle epilogue: o8h geometry, BF16 out");
    const int tt = tid >> 3, row = (tid & 7) * 4;
    const bool on = tid < R4 * M;
    float q[7] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (on) {
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
      }
      const size_t col = (size_t)cta * R + row;
      OutT* o = out + (size_t)tt * N + col;
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
      const float pf[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
      asm volatile("cp.async.wait_all;" ::: "memory");   // this thread's own copies
      const uint2 xp = gx_p[tid];
      const float4 xs = gx_s[tid], xa = gx_a[tid], xb = gx_b[tid], xa2 = gx_a2[tid];
      const float4 xas = make_float4(__fmul_rn(xa.x, xs.x), __fmul_rn(xa.y, xs.y), __fmul_rn(xa.z, xs.z), __fmul_rn(xa.w, xs.w));
      // pre = 1 + pre_ffn_norm in FP32 (the reference's (1 + w), not a BF16-rounded copy)
      const float pp[4] = {__fadd_rn(1.f, __uint_as_float(xp.x << 16)), __fadd_rn(1.f, __uint_as_float(xp.x & 0xffff0000u)),
                           __fadd_rn(1.f, __uint_as_float(xp.y << 16)), __fadd_rn(1.f, __uint_as_float(xp.y & 0xffff0000u))};
      const float ex_as[4] = {xas.x, xas.y, xas.z, xas.w}, ex_b[4] = {xb.x, xb.y, xb.z, xb.w};
      const float ex_a2[4] = {xa2.x, xa2.y, xa2.z, xa2.w};
      float uu[4], vv[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float as = ex_as[j], bp = __fmul_rn(ex_b[j], pf[j]);
        uu[j] = __fmul_rn(as, pp[j]);
        vv[j] = __fmul_rn(bp, pp[j]);
        q[0] = __fmaf_rn(pf[j], pf[j], q[0]);
        q[1] = __fmaf_rn(as, as, q[1]);
        q[2] = __fmaf_rn(as, bp, q[2]);
        q[3] = __fmaf_rn(bp, bp, q[3]);
        if (col >= 1024) {
          const float x = __fmul_rn(ex_a2[j], as), y = __fmul_rn(ex_a2[j], bp);
          q[4] = __fmaf_rn(x, x, q[4]);
          q[5] = __fmaf_rn(x, y, q[5]);
          q[6] = __fmaf_rn(y, y, q[6]);
        }
      }
      // u, v: FP32 rows for the router (uvf) and BF16 rows for the pooler (uv); rows tt (u) and M + tt (v)
      auto put = [&](const float* f, size_t r) {
        __nv_bfloat162 h0 = __floats2bfloat162_rn(f[0], f[1]), h1 = __floats2bfloat162_rn(f[2], f[3]);
        uint2 hk;
        hk.x = *reinterpret_cast<uint32_t*>(&h0); hk.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(ep.uv + r * N + col) = hk;
        *reinterpret_cast<float4*>(ep.uvf + r * N + col) = make_float4(f[0], f[1], f[2], f[3]);
      };
      put(uu, (size_t)tt);
      put(vv, (size_t)(M + tt));
    }
    // the 8 lanes of one token (one aligned lane group) sum their columns: one partial row per (CTA, token)
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 4);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 2);
      q[k] += __shfl_xor_sync(0xffffffffu, q[k], 1);
    }
    if (on && (tid & 7) == 0) {
      float4* pq = reinterpret_cast<float4*>(ep.part + ((size_t)cta * M + tt) * 8);
      pq[0] = make_float4(q[0], q[1], q[2], q[3]);
      pq[1] = make_float4(q[4], q[5], q[6], 0.f);
    }
  }
  else
  for (int idx = tid; idx < R4 * M; idx += NC) {
"""
  anchor = '  constexpr int R4 = R / 4;\n  for (int idx = tid; idx < R4 * M; idx += NC) {'
  assert s.count(anchor) == 1
  s = s.replace(anchor, epi, 1)
  fpr_stage = (
      "        } else if (ep.fpr) {   // FP32 router stage j = st - nst: columns rq * 2048 + 512 j .. + 511\n"
      "          const int kb = rq * 2048 + (st - nst) * 512;\n"
      "          mbar_expect_tx(full0 + 8 * slot, 8 * 2048 + M * 2048);\n"
      "          for (int r = 0; r < 8; ++r)\n"
      "            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);\n"
      "          for (int t = 0; t < M; ++t)\n"
      "            bulk_g2s(s0 + 16384 + t * 2048, reinterpret_cast<const uint8_t*>(ep.x32 + (size_t)t * 8192 + kb), 2048, full0 + 8 * slot);\n"
      "        } else {   // router piece: global router stage rq * 4 + j of tile rt")
  fpr_cons = (
      "  if constexpr (FUSE) {\n    if (ep.fpr) {\n"
      "      static_assert(!FUSE || (NW == 8 && SST >= 32768), \"fp32 router stages: 8 consumer warps, 32 KB slots\");\n"
      "      // thread: column group cg = tid & 127 (4 of the stage's 512 columns) x all 8 router rows x tokens 4h .. 4h + 3\n"
      "      // (h = tid >> 7): every shared-memory float4 is read by 2 threads, not 8 warps; FP32 FMAs\n"
      "      const int cg = tid & 127, th = tid >> 7;\n"
      "      float af[32];\n"
      "#pragma unroll\n"
      "      for (int i = 0; i < 32; ++i) af[i] = 0.f;\n"
      "      for (int st = nst; st < nstt; ++st) {\n"
      "        const int slot = st % NSTAGE;\n"
      "        mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
      "        const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + cg * 4;\n"
      "        const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384) + cg * 4;\n"
      "        float4 xa[4];\n"
      "#pragma unroll\n"
      "        for (int u = 0; u < 4; ++u)\n"
      "          xa[u] = (th * 4 + u < M) ? *reinterpret_cast<const float4*>(xs + (th * 4 + u) * 512) : make_float4(0.f, 0.f, 0.f, 0.f);\n"
      "#pragma unroll\n"
      "        for (int r = 0; r < 8; ++r) {\n"
      "          const float4 bw = *reinterpret_cast<const float4*>(ws + r * 512);\n"
      "#pragma unroll\n"
      "          for (int u = 0; u < 4; ++u) {\n"
      "            float v = af[r * 4 + u];\n"
      "            v = fmaf(xa[u].x, bw.x, v); v = fmaf(xa[u].y, bw.y, v); v = fmaf(xa[u].z, bw.z, v); v = fmaf(xa[u].w, bw.w, v);\n"
      "            af[r * 4 + u] = v;\n"
      "          }\n"
      "        }\n"
      "        __syncwarp();\n"
      "        if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
      "      }\n"
      "      // reduction over the 128 column groups through the (now idle) ring: red[th][cg][33] (static indices only)\n"
      "      if constexpr (COMP) mbar_wait(cbar, 0);   // as the BF16 router: the escape warps are done with the ring\n"
      "      named_bar(1, NC);                         // and every consumer is done reading its last stage\n"
      "      float* red32 = reinterpret_cast<float*>(gbase);\n"
      "#pragma unroll\n"
      "      for (int i = 0; i < 32; ++i) red32[(th * 128 + cg) * 33 + i] = af[i];\n"
      "      named_bar(1, NC);\n"
      "      float* red2 = red32 + 256 * 33;\n"
      "      {\n"
      "        const int o = tid & 63, pq = tid >> 6, h = o >> 5, e = o & 31;\n"
      "        float v = 0.f;\n"
      "        for (int c = pq * 32; c < pq * 32 + 32; ++c) v += red32[(h * 128 + c) * 33 + e];\n"
      "        red2[pq * 64 + o] = v;\n"
      "      }\n"
      "      named_bar(1, NC);\n"
      "      if (tid < 64) {\n"
      "        const int h = tid >> 5, e = tid & 31, r = e >> 2, t = h * 4 + (e & 3);\n"
      "        const float v = red2[tid] + red2[64 + tid] + red2[128 + tid] + red2[192 + tid];\n"
      "        if (t < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + t * 8 + r, v);\n"
      "      }\n"
      "      named_bar(1, NC);\n"
      "    } else {\n    // router piece:")
  ilv_edits = (
      ("  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n",
       "  const int P = (CL == 1 && pdl > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n"
       "  const bool ilv = FUSE && ep.fpr && ep.ilv && (nstt % 4) == 0 && nstt / 4 > P;\n"
       "  const int PER = ilv ? nstt / 4 : (1 << 30);\n"
       "  const int OFF = ilv ? (ep.ilv == 2 ? P : PER - 1) : 0;   // router stage slot inside each period\n"),
      ("        if (!FUSE || st < nst) {\n"
       "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
       "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);\n"
       "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n",
       "        const bool isr = ilv ? (st % PER == OFF) : (FUSE && st >= nst);\n"
       "        const int ps = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
       "        if (!isr) {\n"
       "          mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);\n"
       "          bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + ps) * wstride, WBY, full0 + 8 * slot);\n"
       "          if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + ps) * UPS);\n"),
      ("          const int kb = rq * 2048 + (st - nst) * 512;\n",
       "          const int kb = rq * 2048 + (ilv ? st / PER : st - nst) * 512;\n"),
      ("        if (FUSE && st >= nst) {   // router stages:",
       "        if (ilv ? (st % PER == OFF) : (FUSE && st >= nst)) {   // router stages:"),
      ("        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;\n",
       "        const int psq = ilv ? st - (st / PER + (st % PER > OFF ? 1 : 0)) : st;\n"
       "        const int kbeg = (gs0 + psq) * KST, kend = kbeg + KST;\n"),
      ("        if ((st % NESC) != ew) {", "        if ((psq % NESC) != ew) {"),
      ("  for (int st = 0; st < nst; ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n",
       "  float afi[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};   // ilv: FP32 router accumulators (warp = router row)\n"
       "  for (int st = 0; st < (ilv ? nstt : nst); ++st) {\n    const int slot = st % NSTAGE;\n    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);\n"
       "    if constexpr (FUSE) if (ilv && st % PER == OFF) {   // an interleaved FP32 router stage\n"
       "      const float* ws = reinterpret_cast<const float*>(gbase + slot * SST) + warp * 512;\n"
       "      const float* xs = reinterpret_cast<const float*>(gbase + slot * SST + 16384);\n"
       "#pragma unroll\n"
       "      for (int i = 0; i < 4; ++i) {\n"
       "        const int k = lane * 4 + i * 128;\n"
       "        const float4 bw = *reinterpret_cast<const float4*>(ws + k);\n"
       "#pragma unroll\n"
       "        for (int t = 0; t < 8; ++t) if (t < M) {\n"
       "          const float4 a = *reinterpret_cast<const float4*>(xs + t * 512 + k);\n"
       "          afi[t] = fmaf(a.x, bw.x, afi[t]); afi[t] = fmaf(a.y, bw.y, afi[t]);\n"
       "          afi[t] = fmaf(a.z, bw.z, afi[t]); afi[t] = fmaf(a.w, bw.w, afi[t]);\n"
       "        }\n"
       "      }\n"
       "      __syncwarp();\n"
       "      if (lane == 0) mbar_arrive(empty0 + 8 * slot);\n"
       "      continue;\n"
       "    }\n"),
      ("  if constexpr (FUSE) {\n    if (ep.fpr) {\n",
       "  if constexpr (FUSE) {\n    if (ilv) {   // interleaved FP32 router: warp butterflies of afi, then the 4-quarter reduction below\n"
       "#pragma unroll\n"
       "      for (int t = 0; t < 8; ++t)\n"
       "#pragma unroll\n"
       "        for (int d = 16; d; d >>= 1) afi[t] += __shfl_xor_sync(0xffffffffu, afi[t], d);\n"
       "      float mine = afi[0];\n"
       "#pragma unroll\n"
       "      for (int t = 1; t < 8; ++t) mine = lane == t ? afi[t] : mine;\n"
       "      if (lane < M) __stcg(part + ((size_t)rt * 4 + rq) * 32 * 8 + lane * 8 + warp, mine);\n"
       "      if constexpr (COMP) mbar_wait(cbar, 0);\n"
       "      named_bar(1, NC);\n"
       "    } else if (ep.fpr) {\n"))
  for old, new in (
          ("        } else {   // router piece: global router stage rq * 4 + j of tile rt", fpr_stage),
          ("  if constexpr (FUSE) {\n    // router piece:", fpr_cons),
          ("    named_bar(1, NC);\n    __shared__ int s_rlast;",
           "    named_bar(1, NC);\n    }\n    __shared__ int s_rlast;")):
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new, 1)
  for old, new in ilv_edits:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new, 1)
  s = s.replace("wsr, outr, part, rcnt, pdl));", "wsr, outr, part, rcnt, pdl, g_gle));", 1)
  s = s.replace("wsr, outr, part, rcnt, 0);", "wsr, outr, part, rcnt, 0, g_gle);", 1)
  host = (f"void sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,\n"
          f"         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {{")
  assert s.count(host) == 1
  disp = (f"if (tokr == 8 && N == 8192 && K == 1024 && M <= 8) {{ launch_sk2<1, 8, {NT}, {NWR}, {WSK}, {UPW}, {nstage}, true, false, "
          f"{NACC}, {CL}, __nv_bfloat16>(Ws, x, out, ptab, pmax, base4, M, N, K, 0, nullptr); return; }}")
  pcfg = next(c for c in kl_bfp_SK2_CONFIGS if c[0] == "pooler")
  flines = []
  for tokr in (8,):
    _, pns = kl_bfp__sk2_geom(pcfg, tokr)
    _, _, _, pR, pNT, pNWR, pWSK, pUPW, pNACC, pCL = pcfg
    flines.append(f"if (tokr == {tokr}) {{ launch_sk2<1, {tokr}, {pNT}, {pNWR}, {pWSK}, {pUPW}, {pns}, true, false, "
                  f"{pNACC}, 1, __nv_bfloat16, true>(Ws, x, out, ptab, pmax, base4, M, 4096, 8192, 0, nullptr, "
                  f"reinterpret_cast<const uint8_t*>(Wsr.data_ptr()), outr.data_ptr<float>(), part.data_ptr<float>(), "
                  f"rcnt.data_ptr<int>()); return; }}")
  s = s.replace("@@SK2_DISPATCH@@", disp).replace("@@SK2F_DISPATCH@@", "\n  ".join(flines))
  s += r"""
void gle_o(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4, int64_t M,
           at::Tensor s, at::Tensor a, at::Tensor b, at::Tensor a2, at::Tensor p, at::Tensor uv, at::Tensor part, at::Tensor uvf) {
  TORCH_CHECK(M >= 1 && M <= 8 && s.size(0) == M && uv.size(0) == 2 * M && uvf.size(0) == 2 * M && part.numel() >= 256 * M * 8, "gle_o: shapes");
  g_gle = GleEpi{s.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), a2.data_ptr<float>(),
                 reinterpret_cast<const __nv_bfloat16*>(p.data_ptr()), reinterpret_cast<__nv_bfloat16*>(uv.data_ptr()),
                 part.data_ptr<float>(), uvf.data_ptr<float>()};
  sk2(Ws, x, out, ptab, pmax, base4, M, 8192, 1024, 0, 8, 0, ptab);
}
void gle_pr(at::Tensor Ws, at::Tensor Wsr, at::Tensor x, at::Tensor out, at::Tensor outr, at::Tensor ptab, int64_t pmax,
            int64_t base4, at::Tensor part, at::Tensor rcnt, int64_t M, at::Tensor x32, at::Tensor w32, int64_t wpf) {
  // fused pooler (BF16 x rows) + FP32 router (FP32 x32 rows, FP32 weights w32) at M <= 8
  TORCH_CHECK(M >= 1 && M <= 8 && x.size(0) == M && x32.size(0) == M && x32.scalar_type() == at::kFloat &&
              w32.scalar_type() == at::kFloat && w32.size(0) == 256 && w32.size(1) == 8192, "gle_pr: shapes");
  g_gle = GleEpi{};
  g_gle.x32 = x32.data_ptr<float>(); g_gle.w32 = w32.data_ptr<float>(); g_gle.fpr = 1; g_gle.ilv = (int)(wpf & 255); g_gle.norc = (int)(wpf >> 8);   // LDD19: NORC in wpf bit 8
  sk2f(Ws, Wsr, x, out, outr, ptab, pmax, base4, part, rcnt, M, 8);
  g_gle = GleEpi{};
}
"""
  shared = kl_bfp_CUDA_SRC.replace("@@MBAR@@", kl_bfp_MBAR_HELPERS).replace("@@BODY@@", "")
  return shared + kl_bfp_MAP_SRC + s



# LDD19: TN (proj_builder r4 it6, PROJ.md s6): gle_pr's tail under NORC without the s_rlast round and its barriers (1 = on; 0 = the previous gle source).
LDD19_GLE_TN = 1
kl__TN_EDITS = (('      if constexpr (COMP) mbar_wait(cbar, 0);\n      named_bar(1, NC);\n    } else if (ep.fpr) {\n', '      if constexpr (COMP) mbar_wait(cbar, 0);\n      if (!ep.norc) named_bar(1, NC);   // LDD19: TN\n    } else if (ep.fpr) {\n'), ('    __shared__ int s_rlast;\n    if (tid == 0) {\n', '    __shared__ int s_rlast;\n    if (!ep.norc) {   // LDD19: TN\n    if (tid == 0) {\n'), ('        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;\n      }\n    }\n    named_bar(1, NC);\n  }\n', '        outr[(size_t)(i >> 3) * 256 + rt * 8 + (i & 7)] = v;\n      }\n    }\n    named_bar(1, NC);\n    }\n  }\n'))
kl__TN_PRIOR_SRC = kl__gle_source
def kl__gle_source():
  s = kl__TN_PRIOR_SRC()
  if LDD19_GLE_TN:
    for a, b in kl__TN_EDITS:
      assert s.count(a) == 1
      s = s.replace(a, b)
  return s


def kl__gle_ext():
  global kl__GLE_EXT
  if kl__GLE_EXT is None:
    source = kl__gle_source()
    tag = _hashlib.md5(source.encode()).hexdigest()[:12]
    source = (source.replace("sk2_kernel", "sk2_kernel_gle" + tag).replace("launch_sk2", "launch_sk2_gle" + tag)
              .replace("void sk2(", "void sk2_gle" + tag + "(").replace("  sk2(Ws,", "  sk2_gle" + tag + "(Ws,")
              .replace("void sk2f(", "void sk2f_gle" + tag + "(").replace("  sk2f(Ws,", "  sk2f_gle" + tag + "(Ws,"))
    cpp = ("void gle_o(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab, int64_t pmax, int64_t base4,"
           " int64_t M, torch::Tensor s, torch::Tensor a, torch::Tensor b, torch::Tensor a2, torch::Tensor p,"
           " torch::Tensor uv, torch::Tensor part, torch::Tensor uvf);\n"
           "void gle_pr(torch::Tensor Ws, torch::Tensor Wsr, torch::Tensor x, torch::Tensor out, torch::Tensor outr,"
           " torch::Tensor ptab, int64_t pmax, int64_t base4, torch::Tensor part, torch::Tensor rcnt, int64_t M,"
           " torch::Tensor x32, torch::Tensor w32, int64_t wpf);\nvoid sk2_set_pdl(int64_t n);\n")
    kl__GLE_EXT = kl_load_inline("gle_o_" + tag, "#include <torch/extension.h>\n" + cpp, cuda_sources=source,
                           functions=["gle_o", "gle_pr", "sk2_set_pdl"], no_implicit_headers=True,
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"], verbose=False)
  return kl__GLE_EXT


@triton.jit
def kl__gle_carry(PO, S, ALPHA, BETA, ALPHA2, t, cols, r1, ROWS1: tl.constexpr):
  p = tl.load(PO + t * 8192 + cols).to(tl.float32)
  s = tl.load(S + t * 8192 + cols)
  a = tl.load(ALPHA + cols)
  b = tl.load(BETA + cols)
  bn = (p * r1).to(tl.bfloat16).to(tl.float32)
  if ROWS1:
    ns = tl.fma(b, bn, a * s)
  else:
    ns = tl.where(cols < 4096, tl.fma(a, s, b * bn), tl.fma(b, bn, a * s))
  return tl.load(ALPHA2 + cols) * ns


@triton.jit
def kl__gle_route(POOLED, LOGITS, PART, T, BIAS, GPRE, IDS, WTS, XOUT, LS, LIDS, RT,
               PO, S, ALPHA, BETA, ALPHA2, S1,
               MODE: tl.constexpr, PARTS: tl.constexpr, NCH: tl.constexpr, TT: tl.constexpr, NORC: tl.constexpr,
               CEL: tl.constexpr):   # LDD19 (MP/CEL)
  t = tl.program_id(0)
  kind = tl.program_id(1)
  k8 = tl.arange(0, 8)
  cta = tl.arange(0, 256)
  q = tl.sum(tl.load(PART + (cta[:, None] * T + t) * 8 + k8[None, :]), 0)
  q0 = tl.sum(tl.where(k8 == 0, q, 0.0), 0)
  q1 = tl.sum(tl.where(k8 == 1, q, 0.0), 0)
  q2 = tl.sum(tl.where(k8 == 2, q, 0.0), 0)
  q3 = tl.sum(tl.where(k8 == 3, q, 0.0), 0)
  r1 = tl.rsqrt(q0 / 8192.0 + 1e-8)
  r2 = tl.rsqrt((q1 + 2.0 * r1 * q2 + r1 * r1 * q3) / 8192.0 + 1e-5)
  if kind > PARTS:
    cc = ((kind - PARTS - 1) * TT + t) * (8192 // (NCH * TT)) + tl.arange(0, 8192 // (NCH * TT))
    for tok in tl.static_range(TT):
      r1t = tl.rsqrt(tl.sum(tl.load(PART + (cta * TT + tok) * 8), 0) / 8192.0 + 1e-8)
      if CEL:   # LDD19 (MP/CEL): the carry is re-read by the finish kernel: keep it in L2
        tl.store(S1 + tok * 8192 + cc, kl__gle_carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1), eviction_policy="evict_last")
      else:
        tl.store(S1 + tok * 8192 + cc, kl__gle_carry(PO, S, ALPHA, BETA, ALPHA2, tok, cc, r1t, TT == 1))
  elif kind < PARTS:
    c = tl.arange(0, 4096)
    oc = kind * (4096 // PARTS) + tl.arange(0, 4096 // PARTS)
    gpre = 1.0 + tl.load(GPRE + oc).to(tl.float32)
    _gdc_wait()
    _gdc_launch()
    pooled = (tl.load(POOLED + t * 4096 + c).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + c).to(tl.float32) * r1) * r2
    inv = tl.rsqrt(tl.sum(pooled * pooled, 0) / 4096.0 + 1e-5)
    if PARTS == 1:
      vals = pooled
    else:
      vals = (tl.load(POOLED + t * 4096 + oc).to(tl.float32) + tl.load(POOLED + (T + t) * 4096 + oc).to(tl.float32) * r1) * r2
    tl.store(XOUT + t * 4096 + oc, (vals * inv * gpre).to(tl.bfloat16))
  else:
    e = tl.arange(0, 256)
    bias = tl.load(BIAS + e)
    ls = tl.load(LS)
    q4 = tl.sum(tl.where(k8 == 4, q, 0.0), 0)
    q5 = tl.sum(tl.where(k8 == 5, q, 0.0), 0)
    q6 = tl.sum(tl.where(k8 == 6, q, 0.0), 0)
    tail = q4 + 2.0 * r1 * q5 + r1 * r1 * q6
    _gdc_wait()
    _gdc_launch()
    tl.store(RT + t, r1)
    tl.store(RT + T + t, tail)
    if NORC:   # LDD19: NORC: LOGITS is the pooler's fused_part; same summation order as its last-CTA block
      eb = (e // 8) * 1024 + (e % 8)
      l0 = tl.load(LOGITS + eb + t * 8) + tl.load(LOGITS + eb + 256 + t * 8) + tl.load(LOGITS + eb + 512 + t * 8) + tl.load(LOGITS + eb + 768 + t * 8)
      tq = (T + t) * 8
      l1 = tl.load(LOGITS + eb + tq) + tl.load(LOGITS + eb + 256 + tq) + tl.load(LOGITS + eb + 512 + tq) + tl.load(LOGITS + eb + 768 + tq)
      logit = (l0 + l1 * r1) * r2
    else:
      logit = (tl.load(LOGITS + t * 256 + e) + tl.load(LOGITS + (T + t) * 256 + e) * r1) * r2
    sc = tl.sigmoid(logit * 0.1767766952966369)
    sel = sc + bias
    if MODE != 2:
      bits = sel.to(tl.uint32, bitcast=True)
      ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
      packed = (ordered.to(tl.uint64) << 32) | (255 - e).to(tl.uint64)
      if MODE == 0:
        picked = tl.topk(packed, 8)
      else:
        sorted_values = tl.sort(packed, descending=True)
        picked = tl.gather(sorted_values, k8, 0)
      ids = (255 - (picked & 255)).to(tl.int32)
      wv = tl.gather(sc, ids, 0)
    else:
      ids = tl.zeros([8], dtype=tl.int32)
      wv = tl.zeros([8], dtype=tl.float32)
      for k in tl.static_range(8):
        idx = tl.argmax(sel, 0)
        val = tl.sum(tl.where(e == idx, sc, 0.0), 0)
        ids = tl.where(k8 == k, idx, ids)
        wv = tl.where(k8 == k, val, wv)
        sel = tl.where(e == idx, float("-inf"), sel)
    wv = wv / (tl.sum(wv, 0) + 1e-15)
    tl.store(IDS + t * 8 + k8, ids)
    lid = (ids - ls + 256) % 256
    tl.store(LIDS + t * 8 + k8, tl.where(lid < 32, lid, -1))
    tl.store(WTS + t * 8 + k8, wv)


def kl__gle_frt(xb, x32, state, norc=False):   # LDD19: NORC kwarg
  bst = state["bfp"]
  m = xb.shape[0]
  dev = xb.device
  if "fused_part" not in bst:
    bst["fused_part"] = torch.empty(32 * 4 * 32 * 8, dtype=torch.float32, device=dev)
    bst["fused_cnt"] = torch.zeros(32, dtype=torch.int32, device=dev)
  pci, pstream, ptab_, ppmax, pb4, _ = bst["sk2"]["pooler"]
  pooled = torch.empty((m, 4096), device=dev, dtype=torch.bfloat16)
  logits = torch.empty((m, 256), device=dev, dtype=torch.float32)
  kl__bfp_pdl(m, "pool")
  ext = kl__gle_ext()
  ext.sk2_set_pdl(kl_bfp_SK2_PDL)
  ext.gle_pr(pstream, bst["sk2"]["router"][1], xb, pooled, logits, ptab_, ppmax, pb4, bst["fused_part"], bst["fused_cnt"],
             m, x32, state["router_fp32"], int(kl_GLE_ILV) | (256 if norc else 0))   # LDD19: NORC
  return pooled, (bst["fused_part"] if norc else logits)   # LDD19: NORC (1-D partials)


kl__gle_prior_pool_router_fp32 = kl__pool_router_fp32
def kl__pool_router_fp32(h, hr, T, state, ppick, norc=False):   # LDD19: NORC8 kwarg
  if (kl_GLE_FRT and T <= 8 and hr is not None and ppick in ("bfp", "bfpf") and state.get("bfp") is not None
          and state.get("router_fp32") is not None and hr.is_contiguous() and h.is_contiguous()):
    return kl__gle_frt(h, hr, state, norc=norc), -1   # LDD19: NORC8
  if norc and T in (16, 32):   # LDD19: NORC16
    return kl__gle_prior_pool_router_fp32(h, hr, T, state, ppick, norc=True)
  return kl__gle_prior_pool_router_fp32(h, hr, T, state, ppick)


def kl__gle_o_choice(state, T):
  bst = state.get("bfp")
  if bst is None or not bst.get("sm90"):
    return False
  ch = bst.get("choice", {}).get(("o", T)) or kl_bfp__nearby_choice("o", T, bst) or kl_bfp__static_choice("o", T, bst)
  return ch[0] == "sk2" and kl_bfp__sk2_name("o", T) == kl_bfp_SK2_PICK[("o", 8)]


def kl__gle_after_attention(data, state, o_in, gate, T):
  dev = o_in.device
  f32 = dict(device=dev, dtype=torch.float32)
  g = state['graft']
  bst = state["bfp"]
  ci, stream, tab, pmax, b4, n_rows = bst["sk2"][kl_bfp__sk2_name("o", T)]
  P = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  uv = torch.empty((2 * T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  uvf = torch.empty((2 * T, kl_HIDDEN), **f32)
  part = torch.empty((256 * T * 8,), **f32)
  rt = torch.empty((2, T), **f32)
  nch = kl_GLE_RC.get(T, kl_GLE_RC.get(str(T), 0))
  s1 = torch.empty((T, kl_HIDDEN), **f32)
  kl__bfp_pdl(T, "o")
  ext = kl__gle_ext()
  ext.sk2_set_pdl(kl_bfp_SK2_PDL)
  ext.gle_o(stream, o_in, P, tab, pmax, b4, T, data['stream'], g['alpha1'], g['beta1'], g['alpha2'], state['g_preffn'], uv, part,
            uvf)
  _norc = bool(LDD19_NORC and kl_GLE_FRT)   # LDD19: NORC (route sums the router partials)
  if kl_GLE_FRT:
    pooled, logits = kl__gle_frt(uv, uvf, state, norc=_norc)
  else:
    pooled = kl__bfp_gemm(state, "pool", uv, 2 * T, kl_D_EXPERT, kl_HIDDEN)
    logits = kl__router_fp32_gemv(uvf, state)
  if kl_GLE_CAPTURE is not None:
    kl_GLE_CAPTURE.update(logits=logits, pooled=pooled, part=part, P=P)
  ids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  lids = torch.empty((T, 8), device=dev, dtype=torch.int32)
  wts = torch.empty((T, 8), **f32)
  xm = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
  kl__graft_al(pooled, logits, xm, ids, wts, lids, part, rt)
  kl__gle_route[(T, kl_SL_TILE_CONFIG[T][1] + 1 + nch)](pooled, bst["fused_part"] if _norc else logits, part, T, state["bias"], state["g_preexp"], ids, wts, xm,
                                                  data["local_start"], lids, rt, P, data['stream'], g['alpha1'], g['beta1'],
                                                  g['alpha2'], s1, MODE=kl_SL_ROUTE_CONFIG[T][1],
                                                  PARTS=kl_SL_TILE_CONFIG[T][1], NCH=nch, TT=T, NORC=int(_norc), CEL=int(LDD19_MP_CEL),
                                                  num_warps=kl_SL_TILE_CONFIG[T][0], launch_pdl=True)
  m = torch.empty((T, kl_D_EXPERT), device=dev, dtype=torch.bfloat16)
  mx = kl__moe(T, state)
  if kl_MOE_HAS_PDL:
    mx.okc_set_pdl(1 if kl__pdl(T) else 0)
  for m0 in range(0, T, kl__MA_MMAX):
    m1 = min(T, m0 + kl__MA_MMAX)
    mx.okc_run(state["ma_w"], state["ma_scratch"], xm[m0:m1], lids[m0:m1], wts[m0:m1], m[m0:m1], 0, 32, state["nsm"])
  U, S = kl_dense(m, state["unpool_slice"], T, state, "un")
  if S != 1 or U.dtype != torch.bfloat16:
    raise RuntimeError('gle: BF16 sliced unpooler expected')
  s2 = torch.empty((T, kl_HIDDEN), **f32)
  out = torch.empty((T, kl_HIDDEN), device=dev, dtype=torch.bfloat16)
  nw, bc = kl_SL_CLUSTER_CONFIG[T]
  kl__sl_cluster_finish[(T, 8192 // bc)](U, T, s1, rt[1], g['beta2'], g['post2'], g['pre2'], s2, out, BC=bc, num_warps=nw,
                                      launch_pdl=True)
  return out, s2, ids


kl__gle_prior_after = kl__after_attention
def kl__after_attention(data, state, o_in, gate, T):
  if (kl_GLE_T.get(T, kl_GLE_T.get(str(T), False)) and kl_GLE_RC.get(T, 0) > 0 and gate is None and kl__pdl(T) and 'graft' in state
          and state.get('compose_graft_enabled', {}).get(T, True) and T in kl_SL_TILE_CONFIG and T in kl_SL_CLUSTER_CONFIG):
    mmp = state.get("mm_pick", {})
    ppick = mmp.get((T, "pr"), mmp.get((kl__bucket(T), "pr"), False))
    opick = mmp.get((T, "o"), mmp.get((kl__bucket(T), "o"), False))
    moe = kl__pick_expert_backend(state["moe_pick"].get(T, state["moe_pick"].get(kl__bucket(T), "agent")),
                               marlin_available=state["bank"] is not None, agent_available=state["ma_ok"])
    gw = state.get('g_preffn')
    if (ppick in ("bfp", "bfpf") and opick == "bfp" and moe == "agent" and state.get("bfp") is not None
            and state.get("router_fp32") is not None and state.get("router_fp32_ext") is not None
            and gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == kl_HIDDEN and gw.is_contiguous()
            and kl__gle_o_choice(state, T)):
      return kl__gle_after_attention(data, state, o_in, gate, T)
  return kl__gle_prior_after(data, state, o_in, gate, T)


kl__gle_prior_prepare = kl_prepare
def kl_prepare(weights):
  kl__gle_ext()
  return kl__gle_prior_prepare(weights)


kl_GLE_B1X = True
kl__GLE_W_CUR = None


def kl__gle_b1x_source():
  s = kl__SL_GLUE_CUDA
  for old, new in (
          ("float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];\n  __shared__ float partials[3][NW];",
           "float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N],nsx[N];\n  __shared__ float partials[4][NW];"),
          ("    if constexpr(POST) glb<V>(a.post+col,post+r*V);\n",
           "    if constexpr(POST || ROUTER) glb<V>(a.post+col,post+r*V);   // ROUTER: post = w (pre_ffn_norm)\n"),
          ("        float bn=gb(gm(b[k],r1));\n",
           "        float bn=gb(gm(b[k],r1));\n"
           "        if constexpr(ROUTER) {   // the router's FP32 copy: no BF16 rounding of r1 * P\n"
           "          float bx=gm(b[k],r1);\n"
           "          nsx[k]=(SLICED || (PRELOADED_BASE && col<4096)) ? gf(alpha[k],s[k],gm(beta[k],bx)) : gf(beta[k],bx,gm(alpha[k],s[k]));\n"
           "        }\n"),
          ("  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n",
           "  float r2, r2x=0.f;\n"
           "  if constexpr(ROUTER) {   // the router copy's sum of squares rides the same cluster exchange\n"
           "    float zs;\n"
           "    r2=gr(cluster_squares_triple<NW>(ns,masked,nsx,partials[1],partials[2],partials[3],cluster,rank,tail,zs),1.e-5f);\n"
           "    r2x=gr(zs,1.e-5f);\n"
           "  } else {\n"
           "    r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);\n"
           "  }\n"),
          ("template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n",
           "template<int NW,int N>\n"
           "__device__ __forceinline__ float cluster_squares_triple(const float (&x)[N], const float (&y)[N], const float (&z)[N],\n"
           "    float* sm, float* smy, float* smz, cg::cluster_group cluster, int rank, float& tail, float& zsum) {\n"
           "  constexpr int warps_per_cta=NW/8;\n"
           "  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;\n"
           "  float sx=butterfly(local_squares(x),16);\n"
           "  float sy=butterfly(local_squares(y),16);\n"
           "  float sz=butterfly(local_squares(z),16);\n"
           "  if(lane==0 && warp<warps_per_cta) {\n"
           "    #pragma unroll\n"
           "    for(int dst=0;dst<8;dst++) {\n"
           "      volatile float* px=cluster.map_shared_rank(sm,dst);\n"
           "      volatile float* py=cluster.map_shared_rank(smy,dst);\n"
           "      volatile float* pz=cluster.map_shared_rank(smz,dst);\n"
           "      px[rank*warps_per_cta+warp]=sx;\n"
           "      py[rank*warps_per_cta+warp]=sy;\n"
           "      pz[rank*warps_per_cta+warp]=sz;\n"
           "    }\n"
           "  }\n"
           "  cluster.sync();\n"
           "  tail=butterfly(smy[lane&(NW-1)],NW/2);\n"
           "  zsum=butterfly(smz[lane&(NW-1)],NW/2);\n"
           "  return butterfly(sm[lane&(NW-1)],NW/2);\n"
           "}\n\n"
           "template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>\n"),
          ("      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);\n",
           "      if constexpr(ROUTER) {\n"
           "        float hx[V];\n"
           "        #pragma unroll\n"
           "        for(int j=0;j<V;j++) hx[j]=gm(gm(nsx[r*V+j],r2x),ga(1.f,post[r*V+j]));\n"
           "        gsf<V>(a.router+row*8192+col,hx);\n"
           "      }\n"),
          ("void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {\n"
           "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),",
           "void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router,TV w) {\n"
           "  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)w.data_ptr(),(const BF*)pre.data_ptr(),")):
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new, 1)
  return s


@kl__lru_cache(None)
def kl__gle_b1x_module():
  from tvm_ffi.cpp import load_inline as _tvm_load_inline
  src = kl__gle_b1x_source()
  return _tvm_load_inline(name='gle_b1x_' + _hashlib.md5(src.encode()).hexdigest()[:12], cuda_sources=src,
                          functions=['b1'], extra_cuda_cflags=['-O3', '-std=c++17', '--use_fast_math'])


class kl__GleB1x:
  def __init__(self, base):
    self.base = base

  def b1(self, branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router):
    if write_router and kl_GLE_B1X and kl__GLE_W_CUR is not None:
      return kl__gle_b1x_module().b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, True, kl__GLE_W_CUR)
    return self.base.b1(branch, stream, pre, alpha, beta, out, nxt, alpha2, tail, router, write_router)


kl__gle_b1x_prior_module = kl__sl_cluster_module
def kl__sl_cluster_module():
  return kl__GleB1x(kl__gle_b1x_prior_module())


kl__gle_b1x_prior_after = kl__after_attention
def kl__after_attention(data, state, o_in, gate, T):
  global kl__GLE_W_CUR
  gw = state.get('g_preffn')
  kl__GLE_W_CUR = gw if (gw is not None and gw.dtype == torch.bfloat16 and gw.numel() == kl_HIDDEN and gw.is_contiguous()) else None
  return kl__gle_b1x_prior_after(data, state, o_in, gate, T)


kl__gle_b1x_prior_prepare = kl_prepare
def kl_prepare(weights):
  kl__gle_b1x_module()
  return kl__gle_b1x_prior_prepare(weights)


LDD19_KL_PIN1632 = 1   # LDD19: PIN1632 flag (0: keep the compose policy's timed pick at T = 16 / 32)
kl__pin_parent_prepare = kl_prepare
def kl_prepare(weights):
  st = kl__pin_parent_prepare(weights)
  cge = st.setdefault('compose_graft_enabled', {})
  for T in (1, 8):
    cge[T] = True
  if LDD19_KL_PIN1632:   # LDD19: PIN1632 (kernel_diff_auditor r4i2): graft glue at T = 16 / 32, not the prepare-time timing pick
    for T in (16, 32):
      cge[T] = True
  return st
# ---------------------------------------------------------------------------------------------- rs128 (chenwainuo)
kl__rq_prior_prepare = kl_prepare


# LDD19 HYBRID dispatch. HYBRID = False is exactly v24 (the Keell pipeline is neither prepared nor called).
HYBRID = True
HY_ROUTES = {('decode', 'sliding'): ((1, 1), (8, 8), (16, 16), (32, 32)), ('decode', 'full'): ((1, 1), (8, 8), (16, 16), (32, 32)), ('verify', 'sliding'): ((8, 8),), ('verify', 'full'): ((8, 8),)}

from torch.utils.cpp_extension import load_inline as _hy_raw_inline


def kl_load_inline(*args, **kw):
  if args:
    args = ('k' + args[0],) + tuple(args[1:])
  else:
    kw['name'] = 'k' + kw['name']
  return _hy_raw_inline(*args, **kw)


assert not kl_PCX_CFG and not kl_bfp__SK2_DELAY


def _hy_keell_route(cfg):
  kind = 'decode' if cfg['q_len'] == 1 else 'verify'
  layer = 'sliding' if cfg['window'] else 'full'
  T = cfg['tokens']
  return any(lo <= T <= hi for lo, hi in HY_ROUTES.get((kind, layer), ()))


def _hy_keell_prepare(weights):
  st = kl__rq_prior_prepare(weights)
  st["chen_rq"] = {}
  torch.cuda.synchronize()
  return st


HY_PREBUILD = True
import threading as _hy_threading


def _hy_keell_prebuild(weights):
  global kl__MA_SFX, kl__MA_SC6W, kl__MA_SC6M
  kl__MA_SFX = kl__MA_SFX_IF_PRMT if (kl__MA_PRMT_SCALES and kl__ma_scales_prmt_exact(weights["w13_sf"], weights["w2_sf"])) else ()
  kl__MA_SC6W = bool(kl_SC6_WGS and kl__MA_SFX and kl__wgs_on() and kl__sc6_ok(weights["w13_sf"], weights["w2_sf"]))
  kl__MA_SC6M = bool(LDD19_KL_SC6M and kl__MA_SFX and kl__sc6_ok(weights["w13_sf"], weights["w2_sf"]))   # LDD19: SC6M
  fns = (kl__att7, kl__att7q) + (() if LDD19_SLIM_KL else (kl__moe,)) + (kl__moe_tiny, kl__moe_tiny2, kl__moe_mid, kl_bfp__get_ext, kl__graft_glue_module,
         kl__graft_b2_module, kl__sl_cluster_module, lambda: kl__moe(32), kl__de_ext, kl__nx_ext, kl__nx2_ext, kl__er_ext, kl__router_fp32_ext, kl__gle_ext,
         kl__gle_b1x_module)

  def build(fn):
    try:
      fn()
    except Exception as ex:
      print(f"[hybrid] Keell prebuild failed: {ex!r}"[:300], flush=True)
  ths = [_hy_threading.Thread(target=build, args=(f,), daemon=True) for f in fns]
  for th in ths:
    th.start()
  return ths


_hy_v24_prepare = prepare
_hy_v24_kernel_fn = kernel_fn


def prepare(weights):
  on = HYBRID and bool(HY_ROUTES)
  ths = _hy_keell_prebuild(weights) if on and HY_PREBUILD else ()
  st = _hy_v24_prepare(weights)
  for th in ths:
    th.join()
  if on:
    st['hy_keell'] = _hy_keell_prepare(weights)
  return st


def kernel_fn(data, state):
  if HYBRID and 'hy_keell' in state and _hy_keell_route(data['config']):
    return kl_kernel_fn(data, state['hy_keell'])
  return _hy_v24_kernel_fn(data, state)


def _hy_check_guards():
  _check_guards()
  kl__check_guards()
