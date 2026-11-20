# Reviewer notes: team LDD19, submission `FINAL_3r.diff` (md5 6e575ac0ae00631d3fa202f19ad4a9cf)

File: `FINAL_3r.diff`, 407,657 bytes, md5 `6e575ac0ae00631d3fa202f19ad4a9cf`, against stock SGLang 297e54fd.
Every name in this file is the readable name: no identifier is renamed or shortened, and there is no name map. Compared with its
fully commented reference merge (md5 `4f28c8a329881042f3b77c9d6acea53b`, 426,137 bytes), comments and optional whitespace are removed
from Python rows the patch adds (CUDA sources untouched) and the diff carries 1 row of context; section 7 says exactly what was removed and gives the proof that both copies compile to the same code.
Nothing is dropped: the content is FINAL_2's, with the owner's combine of section 6.3 run as two kernels. These notes are not part of the patch.

## 1. Summary
**This submission = the previous one (scored run 7b445fef8f3942af, 3.0221x, diff md5 82378913) + two changes: (1) bitwise: layer 0's prefill experts run stock's own Triton kernel with a large-row tile config (section 6.2); (2) order-only, with today's rounding points: in the fast layers, on prefill chunks of 8,192..16,384 rows, the expert kernel writes each output row straight into the inbox of the rank that owns the token, and a combine on the owner (two kernels: per-source partials, then the fixed-ring sum) replaces the per-rank combine and the experts' NCCL reduce-scatter (section 6.3). It is delivered with comments and optional whitespace removed from its fully commented merge; no name is changed (section 7). Section 6.1 repeats the change that run already carried against the run before it (3.0097x, md5 28a40f21).** The text of sections 1-5 is the previous submission's (plus one bullet in section 1 and two sentences in section 4 for change (2)); every function it cites carries the same name in this file.

The patch speeds up the Muse Spark target model on 8 x H200 (TP8 = EP8) with a DFlash drafter. No precision is lowered
and no needed work is skipped: each change is bitwise, the same products summed in another order, FA3 with another
split plan, or (last layer) work on rows nobody reads left out. Most of it serves the long first turn (prefill):

- **Sequence-parallel prefill** on the 60 Hopper-eligible MoE layers and on stock layer 0: after attention, each rank
  works on T/8 rows (o projection reduce-scatter, norms and gates, router, pooler, experts, experts reduce-scatter,
  unpooler), with a Hopper NVFP4 expert kernel in the W4A16 form stock uses on Hopper.
- **Transports on the copy engine** between ranks, over CUDA IPC of plain allocations: the next layer's qkv input, the
  streamed attention output into the o projection, and the next layer's o weight. Each has a load-time self-test.
- **Per-forward copies of hot experts:** for chunks of >= 4,096 rows, a few rows of the busiest experts are computed by a
  lightly loaded rank on a byte-identical copy of that expert's weights, pulled per forward.
- **Experts' reduce-scatter into the owner's inbox** (section 6.3): on prefill chunks of 8,192..16,384 rows the expert
  kernel stores each output row into the token owner's inbox over CUDA IPC; the owner sums the 8 per-rank partials in a
  fixed ring order with one BF16 rounding per add, as NCCL's ring reduce-scatter does. Load-time self-test, bitwise.
- **Last layer:** only the rows the logits processor reads go through the o projection / FFN / final norm.
- **Verify step:** an FA3-exact attention kernel with another split plan, plus pinned GEMM configs and fused kernels.
- **Drafter side:** the draft fc runs on each rank's own rows of the target's tap layers; drafter GEMM / top-k kernels.

Launch flags added to the judge's command (stock options): `--swa-full-tokens-ratio 0.25` (sliding-window layers get
a quarter of the full layers' KV tokens instead of stock's 0.8: more full-attention KV room in the same memory);
`--hicache-size 120` (host KV cache pool of 120 GB); `--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.375
--prefill-delayer-max-delay-passes 6` (stock's prefill delayer: under load a prefill may wait a few passes so more
prefills share a batch); `--prefill-max-requests 32` (at most 32 requests per prefill batch). No other flag is set.

## 2. Parts, in the order a first turn meets them

Each part names its switch (an environment variable, default on, or a code constant) where it has one. Every load-time
transport decision is a MIN over the 8 ranks on the CPU group, so all ranks take the same path; per forward, the path
is chosen from the batch's shape (rows, mode), which is the same on every rank.

### 2.1 Request intake on the host
- **What:** `decode_TokenIds`, after one type check of the whole list (`set(map(type, value)) <= {int}`), returns it
  as-is instead of validating it element by element; the
  scheduler's broadcast of requests (`broadcast_pyobj`) and the radix cache's small integer MIN/MAX all-reduce go
  through a shared-memory channel between the 8 processes of one host.
- **Files:** `api/v1/api_types.py` (`decode_TokenIds`); `srt/distributed/host_shm_coll.py` (`_Chan.make`,
  `all_reduce`, `broadcast`); callers in `srt/utils/common.py` (`broadcast_pyobj`) and `srt/mem_cache/unified_radix_cache.py`.
- **Class: bitwise.** The same bytes reach every rank; MIN/MAX of integers is exact; anything outside the channel's
  limits (non-int dtype, > 64 elements, > 8 MiB, other ops) falls through to `torch.distributed` on that call.
- **Setup / fallback:** off for a single process or a non-x86_64 host; otherwise `_Chan.make` maps the file on every
  rank and takes a MIN over ranks of "mapped, same host"; any failure -> `torch.distributed` everywhere. Waits yield and time out after 1,800 s. Switches: `SGLANG_FAST_IDS_CHECK=0`,
  `SGLANG_HOST_SHM_COLL=0`.

### 2.2 Stock layer 0 on the sequence-parallel path
- **What:** layer 0 (BF16 experts) takes the same post-attention path as the 60 fast layers when that path applies
  (T >= 2,048, T divisible by 8, and stock would all-reduce through NCCL). Its experts are stock's own call
  (`mlp.experts(mlp.pre_expert_norm(p), topw, ids, ...)`) on the gathered pooled rows; for chunks of 8,192 rows or more that call
  runs inside stock's `override_config` with one fixed Triton tile config (section 6.2).
- **Files:** `srt/layers/muse_spark/muse_spark_hopper_prefill.py`: `hopper_prefill_layer_supported`, `_stock_sp_setup`,
  `MuseSparkHopperPrefill._seq_par_post_attention` (branch `if self.stock:`).
- **Class: same products, another summation order.** The o all-reduce and the experts all-reduce become reduce-scatters;
  row-wise ops (norms, gates, router, pooler, unpooler) run on this rank's T/8 rows with the same per-row arithmetic; a
  GEMM on T/8 rows may pick other tiles than on T rows (FP32 accumulation either way).
- **Setup / fallback:** `_stock_sp_setup` takes layer 0 on the path only if every rank has layer 0 and layer 1 ready
  (MIN over ranks); else stock's layer 0. Marker per rank: `[muse-comm] stsp=1 rank R layer 0`. Switch: `STOCK_SP = 0`.

### 2.3 Fast layers 1-60: attention, then the sequence-parallel post-attention
- **What:** attention stays stock's FA3 call on the FP8 KV cache. After it, each rank computes T/8 rows: o projection
  reduce-scatter, the attention-boundary norm and residual gate, the router (BF16 hi/lo split weights, FP32 logits) and
  top-k, the pooler, the routed experts, the experts reduce-scatter, the unpooler and the FFN boundary. Routes and pooled
  rows are all-gathered so every rank computes its local experts for all T tokens. From 4,096 rows the o projection
  multiplies each rank's T/8 rows by the whole o weight (all heads; inputs and weight arrive by 2.5 / 2.6) instead of
  a per-rank partial + reduce-scatter; the next layer's qkv GEMM runs on blocks of T/8 rows (2.8).
- **Files:** `muse_spark_hopper_prefill.py`: `MuseSparkHopperPrefill.applies`, `.forward`, `._seq_par_post_attention`;
  entry from `MuseSpark1xDecoderLayer.forward` in `srt/models/muse_spark.py`.
- **Class: same products, another summation order** (the sequence-parallel reduce-scatter; row-wise ops per row; GEMMs
  on T/8-row blocks or over all heads at once accumulate the same products in FP32 with another tile / split order).
- **Guards:** `applies()` per forward: EXTEND, inside the breakable prefill graph, FA3 backend with FP8 cache and page
  size 1, no SP / CP / DP attention, no encoder; `_seq_par()` as in 2.2. Otherwise the stock layer runs. Switches:
  `SGLANG_MUSE_SPARK_HOPPER_PREFILL`, `..._SEQ_PAR`, `..._TP_SLICE`, `..._LT_GEMM`, `..._ROUTE_ROWS` (=0 to disable).
- **Long single request:** for one request with >= 8,192 new rows, `_gathered_attention` copies the K/V rows the
  attention reads into a contiguous buffer (`_kv_gather_kernel`, byte copies) and calls the same FA3 kernel with
  stock's `num_splits`; on sliding-window layers the copy starts at most 160 rows below the window, so only keys the
  window masks are left out. Class: the same FA3 call on the same K/V bytes; the key range starts at another offset,
  so FA3 may group key blocks differently (summation order only). Switch: `SGLANG_MUSE_SPARK_HOPPER_PREFILL_KV_GATHER=0`.

### 2.4 Hopper NVFP4 expert kernel
- **What:** one persistent grouped-GEMM launch for the routed experts. Enabled only where stock would take its Marlin
  path on sm_90 (`use_hopper_moe` requires `use_marlin` and capability (9, 0)); weights are repacked at load.
- **Files:** `kernels/ops/moe/muse_spark_hopper_moe.py` (`hopper_moe_module`, `fast_scales_ok`,
  `repack_nvfp4_experts`), `kernels/jit/csrc/muse_spark/hopper_prefill_moe.cuh`; `srt/layers/muse_spark/
  muse_spark_nvfp4_moe.py` (`_process_weights_for_hopper`), `muse_spark_fp4_moe.py`.
- **Class: stock's W4A16 class.** Hopper has no FP4 MMA: NVFP4 weights are dequantised to BF16 and multiplied with BF16
  MMA, FP32 accumulation, as Marlin does; the gate scale serves both w1 and w3 "as Marlin" (warning in the code if they
  differ). Built with `--use_fast_math` (fast intrinsics: organisers 2026-11-03).
- **Run-to-run:** routes are placed inside an expert by atomic counters, so the last bit of a row's rounding can change
  between runs (<= 1 bf16 ulp); within one forward every rank receives the same bytes.
- **Fallback:** `fast_scales_ok` checks at load that every E4M3 scale byte is in 0x08..0x7E (the range the fast scale
  decode covers); otherwise the exact-scale build is used. Switch: `SGLANG_MUSE_SPARK_HOPPER_MOE=0`.
- **Attribution:** kernel based on team vjeux's `muse_prefill_layer.py` (okc-2026, Apache-2.0); bit layout adapted from
  ZhimaoL/sglang okc_patch (Apache-2.0). Headers in the files.

### 2.5 Streamed attention output into the o projection (copy-engine pushes)
- **What:** instead of NCCL `all_to_all_single` + a relayout copy, each rank pushes block p of its attention output into
  rank p's IPC-mapped input buffer with `cuMemcpy2DAsync` on a copy stream; an 8-element all-reduce on the layer's
  NCCL communicator orders the pushes before the o GEMM. The o GEMM (same weight, same shape) reads the same bytes.
- **Files:** `muse_spark_hopper_prefill.py`: `_OCE.push`, `_OCE.setup`; used in `_seq_par_post_attention`.
- **Class: bitwise** against the NCCL all-to-all path it replaces (the o GEMM itself is in 2.3's class).
- **Self-test at load:** eager push and a push captured in a CUDA graph and replayed, compared bit for bit with a CPU
  reference built by an all-gather on the CPU group, MIN over ranks. Fallback: the NCCL all-to-all path
  (`_o_exchange` / `_o_from_exchange`). Marker: `[muse-comm] o2ce=1 rank R`.
  Switch: `SGLANG_MUSE_SPARK_HOPPER_PREFILL_O_STREAM=0` (also turns off 2.6).

### 2.6 The next layer's o weight by copy-engine copies
- **What:** the next layer's o-projection weight shards are copied from the peers' existing weights (IPC-mapped at
  load) into one [8192, 8192] buffer by 2D copies on a copy stream, instead of an NCCL all-gather + layout copy.
- **Files:** `_OWeightCE.gather`, `_OWeightCE.setup`, `prepare_o_stream`.
- **Class: bitwise** (the same weight bytes land in the same place).
- **Self-test at load:** first and last shard eagerly, and inside a CUDA graph replay, compared with an all-gather on
  the CPU group; MIN over ranks. Fallback: NCCL all-gather. Marker: `[muse-comm] owce=1 rank R`.

### 2.7 Per-forward copies of hot experts (row split)
- **What:** for chunks of >= 4,096 rows (`SPLIT_MIN_T = 1 << 12`) on the fast layers, every rank computes the same plan
  from the chunk's exact routing counts (all ranks hold the same gathered top-8 ids): up to 8 moves, each = the last q
  rows (token order, q a multiple of 256, at least 768) of one hot expert, handed to a lightly loaded rank (at most 3
  moves received and at most 4 sent per rank: `NB`, `CAP` in `_split_plan_kernel`'s call). That
  rank pulls a byte copy of the expert's NVFP4 weights, scales and global scales from the owner over CUDA IPC into one
  of 3 staging slots, computes those rows with the same kernel, and adds them into its partial with the
  same combine (per-term RMS norm x routing weight, FP32 sum).
- **Files:** `_SplitPull.routes` / `.pull` / `.borrowed` / `.selftest`, `_split_plan_kernel`, `_split_pull_kernel`,
  `_split_add_kernel`, `_split_setup`.
- **Class: same products, another summation order.** Every (token, expert) term is computed exactly once, on one rank,
  at stock's precision; router, routing weights and selected experts are untouched. A moved term joins another rank's
  BF16 partial before the experts reduce-scatter: the same terms of the same sum, grouped differently (a row holding
  local and moved terms gets one more BF16 rounding, the BF16-partial class of 2026-11-03).
- **Ruling:** "a GPU may hold an identical copy of a few of another GPU's expert weights, inside the +2 GB limit, and
  compute some of that expert's rows; the server may pick those experts from the routing counts of the same run"
  (organisers, relayed to the team 2026-11-16 23:43 EST).
- **Self-test at load:** checks the expert layout and scale range; opens every peer's expert weights; pulls experts of
  every peer (first and last split layer) eagerly and in a CUDA graph replay and compares checksums with the owners'
  own; runs the planner on a fixed 4,096-row routing on every rank and requires the same plan everywhere, with the
  number of terms taken off their home ranks equal to the number the borrowing ranks take. MIN over ranks; any
  failure -> no split on any rank. Marker per rank:
  `[muse-comm] split=1 rank R nb=3 layers=60 move=1`. Switch: `SGLANG_MUSE_SPARK_LIVE_SPLIT=0` (2 = control: plan and
  pulls run, no row moves).

### 2.8 Copy-engine qkv gather into the next layer
- **What:** the FFN boundary's T/8 output rows are gathered to all ranks by copy-engine copies from IPC-mapped staging
  buffers; the next layer's qkv GEMM runs per landed block, each block waiting on a CUDA event recorded after its copy.
- **Files:** `_CopyEngineGather.gather`, `.setup`.
- **Class: bitwise** against the NCCL all-gather it replaces (same bytes in the same buffer; the consumer runs the same
  op on the same values; the per-block qkv GEMM is in 2.3's class).
- **Self-test at load:** eager with a consumer, and captured in a CUDA graph and replayed, compared with the expected
  gather; MIN over ranks. Fallback: NCCL all-gather. Marker: `[muse-comm] ipc_qkv=1 rank R ev`.
  Switch: `SGLANG_MUSE_SPARK_HOPPER_PREFILL_CE_QKV=0`.

### 2.9 Drafter taps of the target
- **What:** at the tap layers each rank keeps its own T/8 rows of each tap; after the last layer the draft fc runs on
  those rows (`_tap_fc`) and an all-gather returns the projected rows. The drafter's `project_target_hidden` then reads
  the projected rows instead of projecting again; the target hands over the first of its identical tap items instead
  of a concatenation of copies.
- **Files:** `srt/models/muse_spark.py`: `_tap_start`, `_tap_put`, `_tap_finish`, `_tap_fc`,
  `MuseSparkTextForCausalLM.forward`; `srt/speculative/dflash_worker_v2.py`: `_bind_tap_fc`; `srt/models/dflash.py`:
  `project_target_hidden`, `fc_row_split`.
- **Class:** each row's fc product computed once on one rank (the same split as the drafter's own `fc_row_split`); the
  hand-over is bitwise.
- **Guards:** binds only if the drafter uses the stock `project_target_hidden` and row-split fc; `project_target_hidden`
  raises if the projected row count differs from the target's. Marker: `[muse-comm] tapfc=1 rank R`.
  Switch: `SGLANG_MUSE_SPARK_DFLASH_TAP_FC=0`.

### 2.10 Last layer: read rows only
- **What:** in layer 61 Q/K/V, the K/V cache write and attention run on every row (later tokens read that K/V); the o
  projection, FFN and final norm run only on the rows `LogitsProcessor._get_pruned_states` reads (each request's last
  row; with input logprobs, the requested ranges), in slices of at most 1,024 rows (`SGLANG_MUSE_SPARK_READ_ROWS_SLICE`,
  default 1024); other rows of the result are zero. A slice of <= 8 rows is padded to 9 when the chunk is larger, so the
  router keeps the GEMM stock uses above `ROUTER_GEMM_MAX_TOKENS` = 8. When at most 9 rows are read, the last fast layer skips its full residual all-gather and the read rows are fetched
  from their owner rank; otherwise the full all-gather runs.
- **Files:** `srt/models/muse_spark.py`: `_read_rows`, `forward_read_rows`, `_read_rows_tail`, `_tail_rows`,
  `_exit_rows`, `_exit_full`; the gate in `MuseSpark1xModel.forward`.
- **Class:** skipped work only on rows nobody reads (organisers 2026-11-16: allowed "in a general way that can prove
  that you're not skipping computation that's needed"); the tail at T = read rows is the same per-row arithmetic with
  another tile choice. The row fetch is bitwise.
- **Guards:** EXTEND only; off for scoring / pooling, FULL hidden capture without taps, aux hidden capture, routed-expert
  capture, a tapped last layer, full prefill graph, torch.compile, SP / DP attention.
  Switches: `SGLANG_MUSE_SPARK_LAST_LAYER_READ_ROWS=0`, `SGLANG_MUSE_SPARK_EXIT_READ_ROWS=0`.
- Also here: the final norm runs inside the last FFN boundary on the stream rounded to BF16, as the unfused tail does
  (`SGLANG_MUSE_SPARK_FINAL_NORM_FUSION=0`); stock layers' pooler / unpooler GEMMs from 1,024 rows run on T/8 rows per
  rank + all-gather (`_stock_dense`, `SGLANG_MUSE_SPARK_STOCK_TP_SLICE=0`).

### 2.11 After the turn: host cache write-back
- `kernels/jit/csrc/kvcacheio/staged_write_back.cuh`: runs of contiguous pages are copied with one `cudaMemcpyAsync`;
  `mem_cache/pool_host/base.py`: staging chunk 64 -> 1,024 pages. Bitwise (same bytes). Idea from ZhimaoL/sglang (Apache-2.0).

### 2.12 Verify steps (every later token)
- **Attention:** `MuseSpark1xAttention._verify_fa3_attention` calls `verify_attn`
  (`kernels/jit/csrc/muse_spark/l2_prefetch_verify.cuh`) for TARGET_VERIFY calls with <= 8 query rows per request, 16
  query heads, no softcap, an e4m3 cache [slots, 1, 2, 64]; full-attention layers at batch 1 and every other shape use
  stock's `flash_attn_with_kvcache`. **Class: FA3 with another split plan** (organisers 2026-11-16: "a different FA3
  split count is fine"): FP8 Q/K/V, FP32 scores, FA3's ex2 / rcp / lg2 approximations, P rounded to e4m3 where FA3
  rounds it, FP32 per-split partials and combine, every key attended; only the split boundaries differ.
- **Other verify-path items:** pinned cuBLASLt configs for the verify-shape GEMMs (`hopper_bf16_lt_gemm.py` `LtGemm`;
  BF16 in, FP32 accumulate; the FP32 router GEMM at a few rows uses TF32, which stock enables for this model by
  default, `arg_groups/overrides.py`); the router's hi and lo GEMMs as one GEMM with the hi + lo add fused into routing
  (`routing.cuh` `kSplit2`); an L2 prefetch of the next weight (reads only); the Hopper MoE engine for small batches
  (`hopper_moe_engine.cuh`, W4A16 as 2.4; switch `SGLANG_MUSE_SPARK_HOPPER_MOE_ENGINE=0`); a fused q/k norm + rope +
  FP8 KV store (`qk_norm_rope_store_verify.cuh`); stock's slice GEMM + all-gather (each rank multiplies by its 1/8 of
  the output columns, the gather returns the same columns) allowed on sm_90 for the pooler and unpooler of small batches.
- **Drafter:** TP-split draft fc + all-gather (port of sgl-project/sglang#45895, Apache-2.0; `SGLANG_OPT_DFLASH_SPLIT_FC`),
  row-split fc for >= 512 rows (`SGLANG_MUSE_SPARK_DFLASH_FC_ROW_SPLIT`), skinny / head GEMM, shard top-k, conv + norm
  kernels under `kernels/*/speculative/`.

## 3. What is NOT changed
- The router, its weights and bias, the routing weights and the selected experts of every token.
- Every model weight and its precision: NVFP4 experts, BF16 dense weights, FP8 KV cache, FP32 router logits.
- Sampling and the logits processor. The drafter's weights and model are stock's; it only proposes tokens, and every
  committed token is decided by the target's verify step, whose acceptance code the patch does not change.
- Expert placement: every rank holds the experts stock gives it; the copies of 2.7 are per forward, in staging slots.
- The judge's launch settings other than the flags in section 1. The one environment default the patch sets:
  `NCCL_MAX_P2P_NCHANNELS=8` by `os.environ.setdefault` in `srt/entrypoints/engine.py` (only when 2.5 is on; a value
  already in the environment wins).
- The copy-engine transports of 2.5, 2.6, 2.7 and 2.8 use plain allocations shared by CUDA IPC: no torch symmetric
  memory, no multicast. The only multicast use is stock's own `CustomAllReduceV2` slice-GEMM + all-gather, which stock
  runs on Hopper at <= 8 rows; the patch lets it run up to 256 rows (pooler, draft fc) and 136 rows (unpooler) on
  sm_90 (`muse_spark_all_gather.py` `_large_m_limit`).

## 4. Checks a reviewer can repeat
**The judge figures quoted in this section were measured on the previous submission (b33a5d478bd3449c); this submission has no judge reading of its own yet.**

**The judge's own figures for this run (b33a5d478bd3449c):** every gate passed; prefill KL 0.048, decode KL 0.012,
later-turn agreement 0.97, GPU memory 2.3 GB under stock; 3.0097x.

**The startup line** (rank 0, `srt/managers/boot_warmup.py` `warm_batch_size_variants`):

    [muse-comm] form A split=1 on 8/8 ranks nb=3 layers=60 move=1; o2ce=1 ipc_qkv=1 owce=1 tapfc=1

| field | meaning | set in |
|---|---|---|
| `form A` | a label in the log text only: the hot-expert row split of 2.7 | `boot_warmup.py` |
| `split=1` | 2.7 passed its self-test on every rank (0 = off; then `; first failing check: rank R <msg>` follows) | `_SplitPull.setup` |
| `on 8/8 ranks` | ranks with the split on (all or none: MIN over ranks) | `_SplitPull.setup` |
| `nb=3` | staging slots per rank (`_SplitPull.NB`) | `_SplitPull` |
| `layers=60` | fast layers that may split (all 60 Hopper-path layers of this model) | `_split_setup` |
| `move=1` | rows are moved (`SGLANG_MUSE_SPARK_LIVE_SPLIT=1`; 0 = control mode 2) | `_split_setup` |
| `o2ce=1` | 2.5 passed its self-test | `_OCE.setup` |
| `ipc_qkv=1` | 2.8 passed its self-test | `_CopyEngineGather.setup` |
| `owce=1` | 2.6 passed its self-test | `_OWeightCE.setup` |
| `tapfc=1` | 2.9 is bound | `DFlashWorkerV2._bind_tap_fc` |

Each rank also prints its own lines (`[muse-comm] owce=1 rank R`, `o2ce=1 rank R`, `ipc_qkv=1 rank R ev`,
`split=1 rank R nb=3 layers=60 move=1 <msg>`, `stsp=1 rank R layer 0`, `tapfc=1 rank R`); owce, o2ce, ipc_qkv and
split lines end with the first error text of that rank (empty when its test passed). A self-test field reading 0 means that part runs stock's path (for 2.5, 2.6, 2.8: the NCCL collective it
replaces) on every rank. Change (2) (section 6.3) prints its own per-rank line, `[muse-comm] yrow=1 rank R` (or `yrow=0 rank R`
+ the first failing step); it has no field in the rank-0 line.

**Kill switches** (environment, default on; set to 0 to fall back): `SGLANG_MUSE_SPARK_HOPPER_PREFILL`,
`..._HOPPER_PREFILL_SEQ_PAR`, `..._HOPPER_PREFILL_TP_SLICE`, `..._HOPPER_PREFILL_LT_GEMM`, `..._HOPPER_PREFILL_ROUTE_ROWS`,
`..._HOPPER_PREFILL_KV_GATHER`, `..._HOPPER_PREFILL_O_STREAM`, `..._HOPPER_PREFILL_CE_QKV`, `SGLANG_MUSE_SPARK_LIVE_SPLIT`,
`SGLANG_MUSE_SPARK_HOPPER_MOE`, `..._HOPPER_MOE_ENGINE`, `..._LAST_LAYER_READ_ROWS`, `..._EXIT_READ_ROWS`,
`..._FINAL_NORM_FUSION`, `..._STOCK_TP_SLICE`, `..._DFLASH_TAP_FC`, `..._DFLASH_FC_ROW_SPLIT`, `SGLANG_OPT_DFLASH_SPLIT_FC`,
`SGLANG_HOST_SHM_COLL`, `SGLANG_FAST_IDS_CHECK` (all in `srt/environ.py` except the last two); code constant `STOCK_SP`.
Change (2) has no switch of its own: it is off whenever `..._HOPPER_PREFILL_SEQ_PAR` or `..._HOPPER_PREFILL_ROUTE_ROWS` is 0.

## 5. Size
407,657 bytes for this diff (comments and optional whitespace removed, section 7); its fully commented reference merge (md5
4f28c8a329881042f3b77c9d6acea53b) is 426,137 bytes.

## 6. What this submission changes against the previous ones
### 6.1 Already in the previous submission (scored 3.0221x, diff md5 82378913) against the one before it (scored 3.0097x, diff md5 28a40f21)
One file differs: `python/sglang/srt/models/muse_spark.py` (+2,996 bytes in the readable diff). Nothing else.

**Layer 0's input: the fp32 residual cast and the input RMSNorm in two passes (class: bitwise).**
- Where: `MuseSpark1xModel.forward` calls `_l0_stream_norm(self, hidden_states)` right after the embedding (and after the
  sequence-parallel scatter when that path is active). Its two Triton kernels are `_l0_up_sq_kernel` and
  `_l0_scale_kernel`, and `_mul_rn` is the multiply they use.
- What stock does: `stream = hidden_states.to(fp32)`, then layer 0's `input_layernorm` on that fp32 stream with the bf16 weight,
  and the caller casts the result to bf16 (`.to(operator_dtype)`). With an fp32 input and a bf16 weight, torch does not use its
  fused RMSNorm kernel (it warns "Mismatch dtype between input and weight ... Cannot dispatch to fused") and runs the composite
  version: pow, mean, add eps, rsqrt, multiply by rsqrt, multiply by weight, which with the cast makes seven passes over the rows.
- What the patch does: pass 1 reads the bf16 rows once and writes the fp32 stream and the fp32 squares. The mean, the eps add and
  the rsqrt then stay torch's own ops on the same fp32 squares. Pass 2 computes `x * rsqrt * weight` in stock's order and casts
  to bf16 with round-to-nearest-even, the same cast stock uses.
- Why bitwise: bf16 -> fp32 is exact (a 16-bit shift). The squares and both products are single IEEE-754 fp32 multiplies,
  round-to-nearest-even with subnormals kept (`_mul_rn` is the plain `mul.rn.f32` instruction, written inline only so that the
  compiler cannot flush subnormal inputs to zero). They are the same products, in the same order, that stock's torch kernels
  compute. The reduction (`mean`) is torch's own call on the same tensor, so its summation order is stock's.
- Fallback: `_l0_stream_norm` returns `None`, and stock's two lines run unchanged, unless the norm is the model's own RMSNorm
  class with a contiguous bf16 / fp32 weight of the hidden size, the input is a non-empty, contiguous 2-D bf16 CUDA tensor, the
  hidden size is a power of two, and the residual dtype is fp32 with a bf16 operator dtype.
- Checked: a one-GPU probe in the serving image compared this function with stock's two lines (the composite `F.rms_norm`) in
  217 cases, eager and inside CUDA graphs, including rows with subnormal values: all 217 bitwise equal. The probe's negative
  control (the same products in another association, `x * (rsqrt * weight)`) differed in 616 values, which shows that the
  comparison sees a one-ulp change. An earlier version
  that used Triton's own multiply failed exactly the subnormal rows, which is why `_mul_rn` exists.
- Scope: the function sits in the model's forward with no forward-mode condition, so prefill, decode and target-verify forwards
  all take it when the conditions above hold (with the same bytes as stock in each). Nothing else in those forwards changes, and
  the embedding, the weights and every other layer are untouched.

### 6.2 New in this submission against the previous one (diff md5 82378913), change 1 of 2: Layer 0's prefill experts: a large-row tile config for stock's own Triton kernel (class: bitwise)
- **What:** in `MuseSparkHopperPrefill._seq_par_post_attention`, branch `if self.stock:` (layer 0 only), stock's experts call is wrapped
  in `override_config(_L0_CFG if p.shape[0] >= 8192 else None)`. `override_config` and `get_config` are stock's own
  (`srt/layers/moe/moe_runner/triton_utils/__init__.py`); stock's `try_get_optimal_moe_config` returns the override when one is set.
  `_L0_CFG = BLOCK_SIZE_M 128, BLOCK_SIZE_N 256, BLOCK_SIZE_K 64, GROUP_SIZE_M 8, num_warps 8, num_stages 4`, for both expert GEMMs.
  `p` is the all-gathered pooled tensor (T rows, T = the prefill chunk), so a 16,384-row chunk and a last chunk of 8,192 rows or more
  take it. Layer-0 forwards of 2,048 to 8,191 rows on this path pass `None` and keep stock's config file entry exactly as before;
  decode, verify, short extends and the last layer never reach this branch.
- **What stock does today:** stock looks up `triton_3_8_0/E=32,N=4096,device_name=NVIDIA_H200[_down].json`, finds no file, falls
  back to the `triton_3_7_1` files (the boot log says so on every rank), and takes the entry nearest to M; the largest key there is
  M=4096, so a 16,384-row chunk runs `BLOCK_SIZE_M 64, BLOCK_SIZE_N 128, BLOCK_SIZE_K 64, num_warps 4, num_stages 3`.
- **Files:** `srt/layers/muse_spark/muse_spark_hopper_prefill.py` (module constants `_L0_CFG`, `_L0_LOG`; the `if self.stock:` branch).
- **Class: bitwise.** The kernel is stock's `fused_moe_kernel`, unchanged, with the same weights, inputs, routes and expert
  order. `BLOCK_SIZE_K` stays 64 as in stock's entry, so every output element accumulates the same K steps in the same order in
  FP32 and is rounded once to BF16 the same way; only the output tiling (which program computes which block of rows and columns),
  the warps per program and the pipeline depth change. A one-GPU probe in the serving image (layer 0's per-rank shapes, 16,384
  tokens, top-8 of 256 experts, a skewed routing profile; random BF16 weights) compared both GEMM outputs with stock's config:
  `torch.equal` on every valid row, for the hottest and the median rank, for this config and 8 others (20 of 20 cells equal,
  max difference 0.0).
- **Ruling:** none needed (the same kernel and the same arithmetic per element; a tile choice, as stock's own config files make).
- **Self-test / fallback / marker:** none (no new transport, no new kernel source). The first call of 8,192 rows or more prints one
  line per rank: `[muse-comm] l0cfg=1 rank R T N {...}` with the config dict that stock's lookup sees. Switch: none (remove the `with`).
- **Startup line:** unchanged (the rank-0 `[muse-comm] form A ...` line); the new per-rank line above is added once at the first
  large prefill (during the prefill graph capture at boot).
- **Scope:** layer 0 only. Fast layers 1-60, the last layer, decode and verify are untouched. No memory, KV or flag change.

### 6.3 New in this submission against the previous one (diff md5 82378913), change 2 of 2: the experts' reduce-scatter written by the expert kernel into the token owner's inbox (route (b); class: order-only, with today's rounding points)
- **What:** in the fast sequence-parallel layers (1-60) on prefill chunks of 8,192..16,384 rows, the expert kernel's down-projection
  store warps write each output row (bf16, 4,096 wide, 8 KB: the same bytes the kernel writes today) straight into the inbox of the
  rank that owns the token (slot `(token - owner*m)*8 + k`, `m = T/8`), over NVLink through CUDA IPC, instead of into the local
  output buffer. Rows that form A's per-forward hot-expert split computes on a helper rank go to the same owner slot. The owner then
  runs a combine over its 8 x m slots (two kernels, `_rb_part_kernel` then `_rb_ring_kernel`; see "The combine runs as two kernels" below) that replaces three steps of today's path: `_combine_kernel` (per-rank weighted sum
  over k), `_split_add_kernel` (form A's borrowed rows), and the NCCL reduce-scatter of the experts' output. The transfer now runs
  under the expert kernel's compute instead of after it (timeline row `rs_experts` transfer: 18.28 rig ms exposed per turn, mean of 8 ranks).
- **What stock / the previous submission does today:** each rank sums its local k terms in FP32 and rounds to bf16 (`_combine_kernel`),
  adds form A's borrowed rows (FP32 add, bf16 round, `_split_add`), then `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` sums the 8
  per-rank partials, one bf16 round per hop along NCCL's ring.
- **Files:** `kernels/jit/csrc/muse_spark/hopper_prefill_moe.cuh` (template constant `RB`: `RB = false` is the previous binary; with
  `RB = true` the kind-1 store goes to `rdst[row] + ntile*256`; padding rows go to a local 8 KB scratch row);
  `kernels/ops/moe/muse_spark_hopper_moe.py` (`experts_rb`, two host wrappers); `srt/layers/muse_spark/muse_spark_hopper_prefill.py`
  (`_RBox`, `_rb_sync_kernel`, `_rb_part_kernel`, `_rb_ring_kernel`, path choice in `_seq_par_post_attention`, `borrowed(..., rb)`).
- **Path choice (the same on every rank):** `rb = _RB.get(device) if route_rows and not stock and 8192 <= T <= 16384 else None`.
  `T` is the chunk's row count (the same on all 8 ranks); `_RB[device]` is set once at load from a MIN over ranks. No clock, no timing.
- **Class: order-only, with today's rounding points. Exactly:**
  1. Per computing rank s, the partial for token t is `P_s = bf16(FP32 sum over s's k in k order of rmsnorm(y*(1+w))*topw)`, with
     `_combine_kernel`'s own expressions; if s also computed moved rows of t, `P_s = bf16(P_s + FP32 sum of those)`, as `_split_add`.
     These are **bitwise today's per-rank partials** (same products, same order, same rounding points).
  2. The 8 partials are summed `acc = P_{me+1}; acc = bf16(acc + P_s)` for s = me+2 .. me (indices mod 8): one bf16 round per add,
     as each hop of NCCL's ring reduce-scatter. This IS NCCL's arithmetic if NCCL's ring on that channel is the identity order
     0 -> 1 -> ... -> 7; NCCL chooses its rings at init from its topology search and stock does not define the order. **We do not
     claim bitwise against the NCCL output** (it cannot be checked at load without an eager NCCL call before KV sizing, which our rules
     forbid); we claim bitwise against today's partials summed in this fixed ring order, which the self-test proves on all 8 ranks.
  No lower precision, no new rounding point, no skipped term. A one-round FP32 variant was considered and NOT used (it would move the
  rounding point).
- **The combine runs as two kernels (`_rb_part_kernel`, `_rb_ring_kernel`); bitwise equal to a one-kernel combine.** A one-kernel
  form (`_rb_combine_kernel`, one program per token: 8 ring steps, each over the token's 8 k slots; file md5 53003b1b) was measured in our tests and is not
  in this file. It ran one program per token, so the 64 slot checks of a token ran one after another: 0.29-0.32 ms per 16,384-row chunk, about
  3x the base path's `_combine_kernel` (0.087-0.096 ms) and 10x its memory floor (rig timelines 20261119_060747, control tree with
  the same combine, and 20261119_080149, file 53003b1b itself). The same arithmetic is now split into two launches on the
  same stream, one after the other:
  1. **pass 1, `_rb_part_kernel`, grid `(m, 8)`** (program `(i, q)` = owned token `i`, ring step `q`, source rank `s = (me + 1 + q) % 8`):
     the one-kernel form's loop body for step `q`, word for word: for k = 0..7 in k order, if slot `i*8 + k` came from source `s`
     (`rid // 32`, or the form-A plan for a moved row), load it, `y*(1+w)`, RMS norm, times the routing weight, add in FP32 to `al`
     (own rows) or `ab` (moved rows). Then `p = bf16(al)`, and if moved rows were added `p = bf16(p + ab)`, as `_combine_kernel` /
     `_split_add` do. The program stores `p` (bf16) into the inbox slot of the first k it used, and writes that slot index into
     `ix[i, q]` (int32 `[m, 8]`), or -1 when source `s` sent none of token `i`'s rows (or the row is padding).
  2. **pass 2, `_rb_ring_kernel`, grid `(m,)`**: for q = 0..7, `p` = slot `ix[i, q]` loaded and widened to FP32, or +0.0 when `ix[i, q]`
     is -1; `acc = p` at q = 0, else `acc = bf16(acc + p)`; store `acc` as bf16, or NaN unless `fl[8] == 8` (the NaN rule below).
- **Why the result is bitwise the same as the one-kernel form's:** (a) pass 1 uses the same loads, products, FP32 additions in the
  same k order and the same reductions over the same 4,096-wide vector (block = the whole row, `num_warps=8`), so each partial
  `P_s` has the same bits; (b) `P_s` is rounded to bf16 before it is stored, and the one-kernel form rounded it at the same point
  (`al.to(bf16).to(f32)`), so pass 2's widening `bf16 -> f32` gives exactly the value the one-kernel form added; (c) a source that sent
  no row adds +0.0, and the one-kernel form added `bf16(zeros) = +0.0` there; (d) the ring order (`me+1 .. me`), the
  `acc = p if q == 0 else bf16(acc + p)` rule and the NaN rule are the same text. So there is no new rounding point, no new order and
  no lower precision. The class and section 6.3's exact arithmetic are the same.
- **Why the in-place store cannot corrupt a partial:** slot `i*8 + k` holds the row of token `i`'s k-th expert, and that k has
  exactly one source rank, so only program `(i, q)` of that source reads it. The program reads that slot before its first reduction
  and stores `P_s` into it only after its last one, and the stored value depends on every load it made. No other program and no later
  pass reads the slot's original row. Pass 2 runs after pass 1 on the same stream. A peer writes this inbox again only for the next
  layer, after that layer's pooled all-gather, which this rank issues after both passes (the same reuse argument as before).
  Pass 1's store is complete at the kernel boundary, before this rank enters the next pooled all-gather, so a peer's next-layer row
  always lands after it.
  Every `ix` cell is written (the grid covers all `(i, q)`), so `torch.empty` is safe.
- **Evidence:** CPU interpreter stub A2 (`stub_combine_v2.py`): the two-pass combine vs today's kernels + the emulated ring
  `me+1 .. me`, 4 cases (form A moving 0 / 76 / 120 / 129 rows, padding rows): **bitwise 4 / 4**. Another ring order differs in
  4,089-7,632 of 16,384 cells, so the comparison is sensitive. A forced timeout gives all-NaN. 8 / 8 non-equivalent mutants are
  caught (on this file: `stub_A2_final3r.out`). The load-time self-test (unchanged text, below) calls the same `combine()` and so runs both passes,
  eager and in a CUDA graph, bitwise against the CPU reference on all 8 ranks.
  One-GPU 8-process probe v5 (landing checked before the combine, since pass 1 overwrites first slots; the real kernels, one plan
  per cell), on this file (screen 20261119_135002_final3r) and on the same code in our earlier test copy (screen 20261119_070744_final2v2): T 16,384 (zipf pad 0 / 211, uniform pad 211), 8,704 (zipf pad 80), 8,192 (zipf pad 37),
  eager + two CUDA-graph replays on each of the 8 ranks: **0 rows differ bitwise, 0 NaN rows, 0 landing errors**; no hang, 8 / 8 rc 0.
- **Unchanged:** the expert kernel and its inbox stores, the epoch protocol, the bounded wait, the fence, the self-test code and its
  steps, the marker, the fallback, the path choice, the scope (fast layers 1-60, chunks of 8,192..16,384 rows). Only the owner's
  combine changes. Memory: + `ix`, 64 KB at 16,384 rows (prefill graph pool). The inbox is the same 128 MiB. KV: no change.
- **Load-time self-test (`_RBox.selftest`, all 8 ranks, before KV sizing, no NCCL), at T = 8,192.** Other shapes: a one-GPU 8-process
  probe on the real kernels (v3 screen 20261119_031400_rbs, v4 = v3 + T 8,704 in the gate screens of this tree) proves landing +
  combine bitwise at T 16,384 (zipf pad 0 / 211, uniform pad 211) and 8,192 (probe v3 / v4) and at T 8,704 (the 74K turn's last chunk
  bucket, zipf pad 80; probe v4); the other buckets between 8,192 and 16,384: rank agreement only; form A's borrowed rows under route
  (b): the TP8 self-test at T 8,192 only (the probe runs without form A). Probes v3 / v4 ran on the one-kernel combine (md5 53003b1b); on this file's two-kernel combine, probe v5 (screen 20261119_135002_final3r) proves T 16,384 (zipf pad 0 / 211, uniform pad 211), 8,704 (zipf pad 80) and 8,192 (zipf pad 37) bitwise (0 rows differ on 8 ranks, eager and two graph replays). Self-test: seeded inputs (the same on every rank;
  experts 0-3 biased so form A moves rows; 37 padding rows). Steps, each followed by a MIN over ranks on the CPU group:
  `warm` (compiles the new Triton kernels `_rb_sync_kernel`, `_rb_part_kernel`, `_rb_ring_kernel` everywhere before any cross-rank wait; a forced timeout must give an all-NaN output) ->
  `base` (today's kernels: `experts_from_routes` + form A pull + `borrowed` = this rank's partial before the reduce-scatter) ->
  `ref` (CPU-group all-gather of the 8 partials; the ring sum above in torch on CPU, round to nearest even) -> fill the inbox with NaN ->
  route (b) eager: **bitwise** (`torch.equal` on the int16 view) -> capture in a CUDA graph -> fill NaN -> replay: bitwise -> fill NaN ->
  replay again: bitwise (epoch reuse). Then `graph.reset()`, `synchronize()`, `empty_cache()`; nothing the test made stays allocated.
  **Same partials on both sides:** `base` computes the routes once (`lid`, `pos`, counts; form A's moved-row routes too) and the eager
  run and both graph replays of route (b) reuse exactly that route set (only the expert kernel's work-queue counters are re-zeroed
  before each run). The gather plan and the expert kernel are pure functions of the route set, so the self-test proves landing +
  combine **bitwise against the fixed-ring reduce of today's partials on that route set**: deterministic by construction.
- **The base path's own run-to-run noise (one honest line):** today's routing places a token inside its expert's row block in the order
  thread blocks reach a shared atomic counter, and the row's position decides which tile (and tile shape) computes it, so the SAME
  input run twice through today's path (previous submission included) differs after the reduce in 0-837 of an owner's 2,048 rows
  (screen 20261119_025308_rb); a differing row differs in a median of 1-7 of its 4,096 elements, 99.7-99.98 % of those by <= 1 bf16
  ulp (the rest +0 / -0 or tiny sign-crossing values) (one-GPU 8-process probe, screens 20261119_025308_rb and _031400_rbs: same
  route set twice = 0 partial rows differ in every cell; routes recomputed = 0-890 partial rows differ). This is a property of the
  Hopper expert kernel of the previous submissions (in every scored file since K3), not of stock's kernel. Route (b) is order-only
  because it sums the 8 partials in a FIXED ring (me+1 .. me), where NCCL's reduce-scatter sums them in the ring order NCCL chose at
  init; it adds no new rounding point, no lower precision and no skipped term. Separately, the base path itself is not run-to-run
  bitwise (the line above); that noise is independent of route (b) and is in today's file.
- **Fallback:** any exception or mismatch on any rank -> MIN = 0 on all 8 ranks -> `_RB[device] = None` -> every layer takes the
  previous submission's path (today's kernels + NCCL reduce-scatter), unchanged. Setup cannot strand a rank: every rank reaches every
  CPU-group collective (the handle all-gather, each MIN) whatever failed locally; the build, the handle exchange, the IPC open
  and every self-test step run inside a try (a raise = a failed step), and the cleanup after the steps (`graph.reset()`, dropping the cached route set with `self.__dict__.pop("rt", None)`)
  cannot raise even when no route set was ever made (a failure in `warm` or before `base` stored it). Checked by CPU stub C2
  (`stub_setup_real.py`: the tree's own setup + self-test + MIN code, 8 ranks, a fault on rank 5 at warm / base / eager / capture /
  replay, as a mismatch or a raise: every rank falls back, 0 stranded, 0 exceptions escape; no fault: all 8 armed). Not covered by
  a try: building the self-test's seeded inputs (~0.2 GB, before the first step); an allocation failure there raises on that rank
  and its load fails visibly (the server does not start), as any load-time out-of-memory does; it cannot arm a wrong path.
- **Marker:** one line per rank: `[muse-comm] yrow=1 rank R` (armed) or `[muse-comm] yrow=0 rank R <first failing step: message>`.
  Declared marker regex `\[muse-comm\] yrow=1 rank \d+`.
- **Epoch protocol (no reset node):** a device counter `ep` (int32, one per rank). Per rb layer-forward the 1-program `_rb_sync_kernel`
  reads `e = ep + 1`, stores it, writes `e` into slot `me` of each of the 8 ranks' flag rows (atomic exchange, release, system scope),
  then waits until its own 8 flags all equal `e`. All ranks take the same path every forward, so their counters stay equal. The value
  is read from memory at replay time, so it is not baked into the CUDA graph. Before the flags, `cuStreamWriteValue32` (default flags)
  puts a stream-scoped system memory fence after the expert kernels' NVLink stores. Why epochs: with a constant flag + reset, a
  writer that is late past a timeout leaves a stale flag that a LATER replay accepts before the new rows land (wrong rows, no NaN; CPU
  stub B); with epochs the late replay reads NaN and the next one is correct again.
- **Bounded wait:** up to 2^18 polls, each = 8 volatile loads + `nanosleep.u32 200` (PTX: 0..400 ns) -> an estimated 40-260 ms (not timed), then it
  stops. The wait is bounded by peers' local GPU work: every rank passed this layer's pooled all-gather (a lockstep NCCL collective)
  before its expert kernel, and from there to its flag is GPU work in one graph segment (the hot-rank tail measured on the rig is ~0.5 ms per 16K chunk; the timeout is ~100x that). The
  timeout is a fault detector, never a hang. Inbox reuse is safe for the same reason: a peer can write layer L+1's rows only after
  its pooled all-gather of L+1, which needs this rank's, issued after this rank's combine of L.
- **NaN poisoning:** `_rb_sync_kernel` stores the count of matched flags in `fl[8]`; the combine's ring kernel writes NaN to every output row unless
  `fl[8] == 8`. A timed-out wait therefore gives NaN rows, never stale rows and never a hang; NaN then spreads through the next layers,
  so the request's output is visibly wrong (the judge's correctness check). There is NO runtime counter or log line for a timeout
  (no host read inside a forward); a per-rank sticky counter (`fl[9]`, ~120 B) printed by a test hook was considered and not built.
  In a healthy run the timeout never fires.
- **Memory / KV:** +128 MiB inbox + 4 KB flags + 8 KB scratch per GPU, allocated at load before KV sizing (+ the combine's 64 KB index `ix` per 16,384-row chunk, prefill graph pool); the output buffer of the
  expert kernel and the per-rank partial are no longer in the prefill graph's pool. Net KV: measured per boot (TP0 capture line).
- **Scope:** fast layers 1-60, prefill chunks of 8,192..16,384 rows only, including packed multi-request chunks (several users' turns
  in one 8,192..16,384-row chunk). Layer 0, the last layer, extends below 8,192 rows, decode and verify are untouched. No flag change.
- **Proofs we ran (our test files and rig sessions; not part of the patch, available on request):** CPU interpreter stubs `stub_combine.py` (bitwise vs today's kernels + the emulated ring, 4 cases,
  timeout -> all NaN, 8 / 8 mutants caught; one-kernel form), `stub_combine_v2.py` (stub A2: the same on this file's two-kernel combine: 4 / 4 bitwise, NaN, 8 / 8), `stub_proto.py` (epoch protocol, late writers), `stub_setup_real.py` (stub C2 above, on this tree: PASS; the same stub on the tree before the cleanup fix fails at `warm`: the error escapes setup),
  `stub_ctl.py` (control arm), `stub_sameplan.py` (the tree's own self-test code, 8 processes on a real gloo group, a mock expert path
  whose route positions change every call: same-plan test 48 / 48 rank-trials armed, the previous design 0 / 24, a corrupted landed
  row and a missing counter re-zero both caught); on our 8-GPU rig, rank agreement 0 mismatches over 145 tensors per boot.
- **Proofs a reviewer can repeat (the patch itself does them):** the load-time self-test above, with its `[muse-comm] yrow=1 rank R` line on
  all 8 ranks (otherwise `yrow=0 rank R` on all 8 ranks; the rank where a step failed adds that step and its error). Rank agreement
  (the same output bytes on all 8 ranks of a forward) is not done by the patch: a reviewer repeats it with their own checksum of each
  forward's outputs on every rank (our rig figure is in the bullet above).

## 7. What was removed from the reference merge (Python comments and optional whitespace; 1 row of diff context; no renamed name)
This diff (md5 6e575ac0ae00631d3fa202f19ad4a9cf, 407,657 bytes) is produced mechanically from the fully commented diff
**md5 4f28c8a329881042f3b77c9d6acea53b (426,137 bytes)**, kept beside it as the reference copy. **No identifier is renamed or
shortened anywhere: every function, class, attribute, module global and local keeps its readable name, so there is no name map.**
No code, check, self-test, guard or fallback is removed: the content is FINAL_2's (section 6.3's combine as two kernels). **The CUDA sources (.cuh) are untouched:**
all 11 are byte-identical to the reference merge's. Three steps:
1. **Comments removed from Python (C, -2,641 bytes in the diff):** `#` comments on rows the patch adds. A comment-only row becomes
   an empty row, so every line number is kept, and a trailing comment (`code  # ...`) is cut with the spaces before it. Never
   removed: licence, copyright, source and attribution comments, `# noqa` / `# type:` / `# fmt:` / `# pragma`, docstrings (all
   kept), and every comment on a protected row (`@triton.jit` / `@torch.compile` definitions, whose source text is a compile-cache
   key) or on a row our lossless checks match by its text.
2. **Optional whitespace removed from Python (H, -6,889 bytes):** on rows the patch adds, the spaces between tokens that Python's
   tokenizer does not need (`a, b = f(x, y)` -> `a,b=f(x,y)`). Indentation, the row count, stock rows, rows with comments,
   f-strings and multi-row strings are untouched. Checked per file: the token stream and the AST are equal.
3. **One row of context in the diff (`git diff -U1`, -8,950 bytes against the 2-row context of our earlier files):** the tree the
   diff produces is unchanged. `git apply` of this file on a clean 297e54fd checkout gives a tree byte-equal, file for file
   (42 / 42), to the tree the 2-row-context diff of the same source gives.
**Proof (Python):** the diff changes 31 Python files. For every one of them, every code object compiled from this copy equals the
reference merge's code object (equal `co_code`, constants (frozensets compared as sets), names, variable names, free and cell
variables, flags, argument counts, stack size, exception table, first line and qualified name; only line / column tables are
not compared): **1,734 code objects, 0 differences** (`CODE_EQ OK: 31 files, 1734 code objects, 0 files differing`). A negative
control (one constant changed in one file) makes the same check fail on that file. Every non-Python file is byte-identical.
**To check:** `git apply` both diffs on two clean 297e54fd checkouts (this one applies with its 1 row of context); compile each
changed Python file in both and compare code objects as above; `cmp` the CUDA sources.
