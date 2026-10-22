# Source-backed exploration matrix

Source: Mia recipe a6c7e08, production adapter c8b5f94, probe-only correction b733c376. Native Astra High requested for analysis; runtime model/effort/Fast evidence was unavailable to the child. No transport performance prediction is inferred from DeepSeek results.

GLM gathers FP32 `[rows,4096]` shards into `[4,rows,4096]` after attention and MoE, then performs HC post-processing with WORLD=4/ROUND=0. The 512KiB cutoff covers32rows. Native serving drafts use16tokens; workspace capacity64 is not a serving-window control. DeepSeek's 2D pair/quarters/diagonal operations cannot replace this full rank-major gather. The physical rails and complete neighbor transport can be reused while preserving GLM arithmetic.

| Order | Candidates | Gate |
|---|---|---|
| 1 | NCCL; allocated relay; registered relay; two-match neighbor | Actual four-rank production dispatch: eager and changing-input graphs, bytes, guards, cutoff/alias cases and execution counters |
| 2 | Same transports, L2=1/SIDE=0/prefill2048/DFlash2/context131072/parallel8/cache0 | Exact serial/drafted/concurrent token IDs, then matched timing; finalists A/B/B/A |
| 3 | L2/SIDE `(1,0),(0,0),(1,1),(0,1)` | Exact tokens before decode timing |
| 4 | PREFILL_ROWS1024/2048/4096 | Cold TTFT, memory, short boundaries,8K and prefix continuation |
| 5 | DFlash2 versus no drafts at1/4/8streams | Exact greedy sequence, acceptance and end-to-end throughput |
| 6 | CLI prompt-cache-gib0/2/4, memory permitting | Cold and warm prefix separately, eviction/continuation/cancellation |

Secondary screens only for promising finalists: L2 modes0/bulk/lines/touch at8MiB, then4/8/16MiB or individual a/f/o sites. Confirm logs: invalid settings disable prefetch. Relay cutoff128/256/512KiB is configurable; neighbor512KiB and blocks/slots/grain are fixed in this adapter. Preserve explicit per-host FIRST/SECOND rails and GID3.

Inactive at TP4: HC split, prefill overlap and own-front specializations. Native MULTI_WINDOW/EXL3_LOADS/DEC_ORDER/MULTIPREFILL have no readers. MTP is a separate parallel1 experiment; MTP minmax/mse quantization does not tune DFlash2. TF_GLM_EAGER is presence-based: unset it for graphs. TF_GLM_MTP accepts0/1/auto; `off` is invalid. Model and drafter basenames must match across all ranks. CLI prompt-cache-gib overrides the cache budget; TF_GLM_CACHE_GIB also affects pool planning.

Arithmetic changes have a separate acceptance gate. Shared rotation off (`TF_GLM_SHARED_SUH=0`) is a candidate for bitwise qualification. HC reduction order, FMA grouping, quantizers, rounding and NCH padding require source/AOT candidates and guarded numeric oracles. October16 fixed-five-chunk merge failed12 strict signed-zero cases; exact-NCH preserved626688 negative-zero BF16 elements. HC-v5 evidence does not establish HC signed-zero coverage. Token equality alone cannot qualify arithmetic changes.

Retain complete operations, exact inputs/output IDs, effective source/config fingerprints, cache/draft state and per-run results. A fastest-tested profile is not a global optimum or proof of full-context quality/reboot recovery.
