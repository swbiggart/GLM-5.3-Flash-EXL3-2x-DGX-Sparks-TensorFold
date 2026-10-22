# Full GLM-5.3 TP4 versus DeepSeek 4.1 and native GLM Flash transport

Read-only source comparison, 2026-10-22. No live access, GPU execution or implementation was performed for this comparison. Initially written in the isolated research clone, then copied into this experiment by the integrator.

## Frozen sources

- Public recipe: `bertholomus/glm-5.3-tensorfold-tp4-4xgb10` at **`9a5c1100c68cd8ea832803c120ee63313049fc21`**, local `/tmp/bertholomus-glm53-tp4-research`.
- Referenced engine branch `bertholomus/TensorFold:glm-dsa-tp4` at **`ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595`**, local `/tmp/bertholomus-glm53-tp4-research/engine`.
- Accepted DeepSeek neighbor implementation: **`b8b115e7ed6c86d81b7575caa6f3d4ade52b323c`**. The GLM import records exact source hashes in `/tmp/glm-tp4-rdma-candidates-20261022/glm-neighbor-source.json`.
- Our native GLM Flash production candidate: **`c8b5f94b24524fbf776181efc6ba4937c62c1c05`**. Probe-only HEAD **`b733c37688c44772332f99515b26e7fe21f20ff7`** leaves production Comm, transport and CUDA kernel source unchanged.

Requested assignment: Astra High. Runtime model, effort and Fast-setting evidence was not exposed to this child; these settings are not independently verified here.

## Main conclusion

The public recipe targets **full 753B GLM-5.3 (`glm_moe_dsa`)**, using Python/PyTorch orchestration and custom CUDA kernels. It is neither the Flash model nor our native Zig serving path. Its MLA/DSA architecture has no Flash KDA layers, hyper-connections or k-pool compression; it uses its own MTP layer rather than our DFlash2 setup. Its measured performance and model-specific kernels cannot be substituted for Flash qualification. [Recipe](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10/blob/9a5c1100c68cd8ea832803c120ee63313049fc21/README.md#L3-L15), [architecture](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/families/glm_moe_dsa/config.py#L8-L12)

Its strongest portable lessons are graph warm-up coverage, per-device completion ownership, and the possibility of combining collection with a model-specific consumer. Its all-peer transport is not suitable for our neighbor-only physical ring unchanged.

## Transport comparison

| Property | Full GLM repository | Accepted DS41 / our Flash neighbor adapter |
|---|---|---|
| Topology | Every selected device opens QPs to every other rank; striped direct writes to every peer | Two neighbor matchings, XOR1 then XOR3, with explicit per-host FIRST/SECOND rails |
| Host memory | `cudaHostAllocMapped | cudaHostAllocPortable`, then verbs registration | mmap plus portable mapped CUDA HostRegister; our existing GLM relay also exposes allocated/registered modes |
| GPU path | Stage into host slots, CPU RDMA proxy, collect rank-major output | Same broad staged-host design, with two exchanges and dedicated intermediate scratch |
| Completion | RDMA WRITE_WITH_IMM; CPU publishes a sequence after every device's stripe completes | Imported DeepSeek sequence protocol; linked failure state across the adapter's two local stages |
| Eligibility | Contiguous FP32, 16-byte-multiple payload, configured capacity | Positive aligned payload through 512 KiB, with strict pre-enqueue fallback |
| Graphs | Captured staging/collection kernels, no host graph nodes | Captured native kernels; our qualification configuration disables PDL |

These are **host-staged transports**, not direct GPU-memory RDMA. The full GLM implementation requires every peer to be reachable on corresponding device indices. Its default GID is 5; our fleet requires explicit GID3. Changing only HCA names does not turn its all-peer QPs into a neighbor-ring transport. [Allocation and QPs](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/cuda/rdma_gather.cu#L520-L580), [peer mapping](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/cuda/rdma_gather.cu#L604-L644), [adapter and defaults](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/cuda/rdma.py#L54-L89)

Our Flash forward gathers each rank's FP32 `[rows,4096]` partial into global rank order, then applies HC post, twice per layer. The 512 KiB cutoff covers 32 such rows. Full-rank gather is the reusable boundary. DS pair-only, quarter-placement, Markov and diagonal helpers implement different subgroup or layout operations; they cannot replace this boundary without changing its consumers. Frozen local Flash callsites (`/tmp/glm-tp4-rdma-candidates-20261022/zig/src/families/glm_cuda/cuda_forward.zig:264`; frozen local source), HC post (`/tmp/glm-tp4-rdma-candidates-20261022/zig/src/families/glm_cuda/cuda_hc.zig:30`; frozen local source)

## Portable lessons and limits

### Graph warm-up

The full-model branch pre-captures concurrent round and MTP-step shapes during warm-up, with every rank capturing in identical order. The recipe reports 50 graphs in 3 seconds and removal of a first-burst penalty. This supports measuring first-request capture cost separately. Before proposing a native change, inspect whether our current warm-up already covers actual benchmark shapes. `TF_GLM_PRECAPTURE` is their Python control, not an existing native Flash control. [Implementation](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/families/glm_moe_dsa/cuda/multi.py#L932-L969), [reported result](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10/blob/9a5c1100c68cd8ea832803c120ee63313049fc21/README.md#L30-L41)

### Completion ownership

Their October fix scopes QP-number lookup to the NIC owning the CQ, because different NICs can assign the same QP number. Our imported neighbor already uses `peerOf(nic, qp_num)`. This is an applicable invariant already satisfied, not a new finding against our candidate. [Their fix](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/cuda/rdma_gather.cu#L397-L408), our lookup (`/tmp/glm-tp4-rdma-candidates-20261022/zig/src/families/glm_cuda/neighbor_rdma.zig:381`; frozen local source)

### Fused collection and row-owner reduction

Their fused collect/residual avoids materializing and rereading the full `[world,rows,hidden]` result. For larger windows, they scatter owned rows, sum in rank order and gather BF16 branches. Defaults are `TF_GLM_FUSED_RESIDUAL=1`, `TF_GLM_REDUCE_ROWS=9`, and `TF_RDMA_STAGE_BLOCKS=8`.

These are promising designs, not portable Flash kernels. Full GLM explicitly computes `bf16(x + bf16(rank_sum))`; Flash uses HC state and different post-processing. A Flash fusion must preserve its own rank order, rounding points, alias rules and signed zeros. Earlier Flash arithmetic experiments demonstrated that apparently harmless padding/reassociation can fail strict signed-zero equality. Do not fold a fusion into the transport comparison. [Dispatch](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/families/glm_moe_dsa/cuda/forward.py#L35-L109), [explicit arithmetic](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/cuda/rdma_gather.cu#L199-L240)

### Failure policy

The full-model adapter coordinates creation/connection failures and falls back to NCCL. Runtime waits observe an abort word; its watchdog marks the lane broken and aborts RDMA/NCCL on peer loss or step timeout. Our explicit custom profiles retain stricter startup rejection and fatal post-selection failures. Preserve those semantics so a benchmark expected to exercise custom transport cannot silently become an NCCL result. [Fallback](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/families/glm_moe_dsa/cuda/engine.py#L335-L351), [watchdog](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/families/glm_moe_dsa/cuda/multi.py#L314-L353)

### Model-specific controls

Q5 MLA cache, DCP4, embedding splitting, MTP selection reuse and prompt `mma3` are model-specific changes. The source explicitly says prompt `mma3` produces different bits from decode kernels. They are not environment settings to copy into native Flash, and their memory savings or quality results do not establish Flash behavior. [Forward controls](https://github.com/bertholomus/TensorFold/blob/ddfad356dc16f4d6bed9cfcc9d60cd0ebc0a3595/src/tensorfold/families/glm_moe_dsa/cuda/forward.py#L49-L65), [recipe controls](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10/blob/9a5c1100c68cd8ea832803c120ee63313049fc21/README.md#L74-L80)

## Measurement limits

The recipe reports about 82 aggregate tok/s for four code streams at 64K, and about 64 tok/s for its 1M DCP configuration. These are author-reported results on different weights, model arithmetic, drafting and concurrency. The release numbers were measured on engine `3eb35dd`, while the pinned branch head is `ddfad356…`. [Reported configurations and results](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10/blob/9a5c1100c68cd8ea832803c120ee63313049fc21/README.md#L45-L72)

The public greedy harness requests **up to 300 tokens**, measures client HTTP elapsed time including prompt processing, and saves output text rather than token-ID arrays. It does not enforce exactly 300 generated tokens, nor independently substantiate exact token identity from its saved text alone. Its numbers cannot serve as our C1/exact-400 baseline. [Actual harness](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10/blob/9a5c1100c68cd8ea832803c120ee63313049fc21/bench/tf_greedy.py#L1-L22)

Its separate `tf_bench.py` streaming speed calculation increments once per nonempty SSE piece, then labels that rate tok/s. A piece can contain multiple tokens, especially with drafting or stream smoothing; that metric cannot replace usage-based token throughput. Its simple cases and needle test are useful smoke fixtures, but the repository does not ship Grill or the unpublished 428-case reference suite. [Streaming harness](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10/blob/9a5c1100c68cd8ea832803c120ee63313049fc21/bench/tf_bench.py#L17-L46)

The code comments' microsecond improvements and the README's speed/quality claims are useful hypotheses and reported evidence; this research did not reproduce them. Historical DeepSeek improvements similarly do not predict GLM Flash serving gains.

## Bounded experiment priorities

1. **Our own native Zig baseline first:** Grill 0.3.0 default C1/exact-400, retaining the exact harness command/config, model/source/AOT fingerprints, actual output lengths and raw results. The unavailable 17/4/428/56 reference scripts remain unavailable; do not reconstruct their methodology from headline numbers.
2. **Complete correctness gates:** production-dispatch byte/guard tests, changing-input graph replay, slot reuse, alias/cutoff cases and actual transport counters. At integration, the corrected probe03 rerun passed all four profiles on all four ranks; the prior masked failure was not proof of memory incoherence.
3. **Transport matrix:** NCCL, relay-allocated, relay-registered and neighbor, holding weights, arithmetic, L2, SIDE, prefill rows, DFlash2, cache, context and concurrency fixed. Require deterministic token-ID equality before timing; matched A/B/B/A repeats for survivors.
4. **Existing effective native controls:** a bounded L2/SIDE 2×2 screen, followed by prefill rows 1024/2048/4096. Separate cold TTFT, warm decode and peak memory. Inspect actual graph captures before changing warm-up.
5. **Draft/cache experiments separately:** compare no-drafts versus DFlash2 at matched greedy output, then cache-off versus modest cache budgets with cold and prefix-reuse cases kept separate.
6. **Only after these gates, propose source changes:** explicit neighbor block/cutoff tuning or Flash-specific fused collect/HC. Each needs an independent review and byte/rounding/signed-zero qualification before model testing. Full-model DCP/quantization/MTP changes require a separate architecture and quality track.

## Findings boundary

This bounded comparison found **no new source-backed P1/P2 in our frozen candidate**. It is not a complete correctness audit, a new deployment acceptance, or evidence of performance improvement. The root integrator owns live access, GPU resources, review and any implementation decision.
