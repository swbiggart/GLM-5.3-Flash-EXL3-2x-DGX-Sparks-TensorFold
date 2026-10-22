# GLM Flash TP4: transport research and retained receipts

The selected profile is **GLM's existing neighbor-relay all-gather with registered host memory, 512 KiB per-rank slots, L2 prefetch and the shared-expert side stream**. On our four-GB10 ring, the final same-binary A/B/A/B comparison measured **+16.57% and +16.18%** over matched NCCL controls using the public Grill C1/exact400 protocol. This is the fastest profile observed in this study; the short sequential captures do not establish a global optimum or causal attribution.

This branch preserves the experiments, including alternatives and failures. The smaller intended submission is [`feat/tp4-registered-relay`](https://github.com/swbiggart/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/feat/tp4-registered-relay), with a single opt-in installer setting. Its cleaned source has separate CPU/semantic checks and still needs GPU qualification. No PR to MiaAI is authorized by publishing these branches.

## What was reproduced from upstream

We used **Grill 0.3.0's unmodified default C1/exact400 fixture, request encoder, acquisition schedule and statistical comparison**, as reported in [MiaAI PR #200](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/200). Each complete capture contains eight acquisitions, each with one warmup and three measured requests: 32 requests and 12,800 reported completion tokens. The rate includes TTFT and request settlement; it is not decode-only speed.

Our stock upstream Zig/NCCL baseline measured **155.64 tokens/s** (median of acquisition medians). PR #200 reports **+0.68% Zig versus Python**, with a nominal model-based 95% interval of +0.08% to +1.29%, but supplies no absolute Grill rate or raw capture directories. We therefore cannot establish a numerical match to the author's absolute speed. Their hardware was three DGX Sparks and one Dell GB10, with two rails per cable; our ring used two DGX Sparks and two MSI GB10 systems, also with two rails per cable. We did not repeat their Python-versus-Zig experiment.

Their **17 parity / 4 resume / 428 sweep / 56 main-set** fixture scripts are excluded from the public recipe and were unavailable to us. Those exact suites remain **unreproduced**. Our exact-token, mixed-workload and recall checks are supplemental. The TP2 90.5/75.1 TPS numbers in patch 0013 are not TP4 rates.

[Protocol, commands and interpretation](receipts/grill-protocol.md) · [Grill source/release pins](receipts/grill-source-pins.json) · [stock baseline](receipts/grill-stock-nccl-01/report.json)

## A/B/C/D transport comparison

All transport candidates use the same native Flash math and rank-major gather layout. Each initial comparison below uses the fresh original ported-NCCL baseline, not a comparison capture as a new baseline.

| Initial profile | Median acquisition tokens/s | Official change versus ported NCCL | Nominal model-based 95% interval |
| --- | ---: | ---: | ---: |
| Ported NCCL | 155.22 | — | — |
| A: allocated-memory GLM relay, 512 KiB | 178.74 | +14.99% | +14.49% to +15.49% |
| B: registered-memory GLM relay, 512 KiB | **180.88** | **+16.45%** | **+16.00% to +16.91%** |
| C: DeepSeek-style two-stage neighbor, 512 KiB | 178.41 | +14.92% | +14.46% to +15.38% |
| D: registered-memory GLM relay, 1024 KiB | 180.40 | +16.12% | +15.57% to +16.68% |

Raw captures and official reports: [NCCL](receipts/grill-ported-nccl-baseline-03/), [allocated relay](receipts/grill-relay-allocated-01/), [registered relay](receipts/grill-relay-registered-01/), [neighbor](receipts/grill-neighbor-01/), [1024 KiB relay](receipts/grill-relay-registered-1024-01/). The earlier stock-to-ported NCCL runtime comparison measured −0.96% [−1.32%, −0.59%]; the source change is not a controlled explanation of that sequential difference. Its [raw capture](receipts/grill-ported-nccl-01/) is retained.

### Final same-configuration A/B/A/B

Weights, binary/image content, context, parallelism, cache budgets, L2 prefetch and side-stream settings match. Only transport selection changes. Every comparison uses the original A1 root, as required by Grill's baseline lineage.

| Capture | Transport | Median acquisition tokens/s | Official change versus A1 | Nominal model-based 95% interval |
| --- | --- | ---: | ---: | ---: |
| [A1](receipts/grill-final-control-nccl-01/) | NCCL | 156.27 | — | — |
| [B1](receipts/grill-final-relay-registered-512-01/) | Registered relay512 | **182.16** | **+16.57%** | **+15.59% to +17.56%** |
| [A2](receipts/grill-final-control-nccl-02/) | NCCL unchanged control | 156.14 | −0.24% | −0.88% to +0.41% |
| [B2](receipts/grill-final-relay-registered-512-02/) | Registered relay512 | 181.71 | +16.18% | +15.57% to +16.78% |

All 128 requests report 400 completion tokens and zero cached prompt tokens; the output text hashes match stock. Grill does not request full token IDs; separate qualification supplies those. A2 is directionally inconclusive, not proof of equivalence. The official percentage uses differences of log means, so it need not equal the ratio of the displayed medians. Intervals assume stable, independent acquisition-level variation and apply to these capture periods only. [Independent raw/hash-chain audit](receipts/final-abab-offline-audit-v2.json).

## Configuration exploration and selection

| Registered relay512 setting | 1 / 4 / 8-stream aggregate tokens/s | 30-second four-stream admission |
| --- | ---: | ---: |
| L2PF1 / SIDE0 | 87.53 / 134.30 / 174.93 | 152.79 |
| L2PF0 / SIDE0 | 83.29 / 130.37 / 172.37 | 149.84 |
| L2PF1 / SIDE1 | **87.70 / 135.70 / 175.63** | **153.52** |
| L2PF0 / SIDE1 | 83.36 / 131.54 / 173.11 | 149.93 |

These two-repetition screens are descriptive; the small SIDE margin is not an established independent gain. The 1024 KiB cutoff showed no material advantage in the short serving screen, so 512 KiB was retained. [Knob screen](receipts/knob-screen-analysis.json), [transport screen](receipts/transport-screen-analysis.json), [all effective controls](receipts/supported-knobs.md).

The final configuration completed **57 requests at 157.02 aggregate tokens/s**, zero worker errors, in a separate 90-second four-stream admission test. Median/p95 TTFT was 0.182/0.333 s. [Raw load evidence](receipts/bench-final-90s-02/), [analysis](receipts/final02-steady-analysis.json). This is a different workload and metric from Grill; do not compare their rate numbers directly.

Cache exploration retained `TF_GLM_CACHE_GIB=32` with **5,619,712 KV rows**, while `--prompt-cache-gib 3` limits kept states separately. An earlier env3/CLI3 trial shrank the KV pool and is confounded. The clean trial's 32,752-token prompt reused 32,704 tokens with identical 128-token output arrays. Cold serial TTFT was 19.740 s and warm drafted/repeated median TTFT 0.166 s; draft mode also changed, so this is not an isolated cache speedup. [Clean analysis](receipts/cache3-pool32-evidence-analysis.json), [confounded trial](receipts/cache3-evidence-analysis.json), [final supplemental probes](receipts/accept-final-02/).

## Why the DeepSeek work helps GLM

The useful transfer was **registered host memory** (`mmap` plus CUDA host registration), applied to GLM's existing relay. Both use GPU staging through host memory and CPU RDMA proxies; neither is direct GPU-memory RDMA. We also tested DeepSeek's two-stage XOR1/XOR3 neighbor schedule with explicit global-rank placement and sticky error handling. GLM's existing relay avoids that adapter's two serial GPU-staged rounds, consistent with its measured lead; we did not measure a decomposition of those costs.

DeepSeek TP4 has a 2-D split and specialized pair/quarter/Markov/diagonal consumers. Flash uses a 1-D TP4 full gather before HC post, so those arithmetic shortcuts do not transfer directly. Full GLM-5.3 uses different MLA/DSA/MTP math and Python orchestration. Its all-peer QPs are unsuitable for a neighbor-only physical ring unchanged, and its SSE-piece throughput counter is unsuitable for comparing drafted token rates. [Pinned full-model comparison and lessons](receipts/bertholomus-comparison.md).

## Topology, build and scope

The logical topology remains a cycle of four hosts, each connected to its two neighbors. The relay discovers devices and subnets; it supports one or two rails per neighbor without hardcoding our NIC names, addresses, GID index or ports. **GPU measurements cover our two-rail/dual-PCI-domain layout only**. The focused source adds CPU route tests for one- and two-rail layouts, shuffled NIC order and non-/24 masks. The research-only DeepSeek neighbor adapter currently requires two rails per edge and GID3; its launcher rejects unsupported layouts. **Switched TP4 is refused by these Zig launcher profiles.** The registered-host-memory approach could be adapted to the existing direct all-peer RoCE path for switched networks, but needs a separate startup/profile implementation and byte/graph/exact-token/performance qualification. Ring relay is not assumed to be the best switched schedule.

Build recipe base: MiaAI `zig-engine` **a6c7e08ae864239125d45fc15146d26ec74c7f6d**, TensorFold **e7c523ebfa16b8094cf9e35e3773a7d25befcc5b** plus patches0001–0013. Patch 0014 preserves the tested production source **c8b5f94b24524fbf776181efc6ba4937c62c1c05**, with the probe-only fixes through **44cd03144b1701be22449bea3d8f99bf44046056**. [Exact provenance](../../zig/TRANSPORT_SOURCE.json). CPU reconstruction verifies the tested blobs, accounting for any explicitly recorded publication-only attribution headers, with six documented pre-existing executable-mode differences; original upstream modes are preserved.

The accepted historical engine SHA-256 is `a2c4b936bab2da7b4cfa4318ea4bb0619621340152e0182f4634550b128d6f7f`. Model revision: `078455ffe6472f9a52fbc1139f58b9db2881b25c`; DFlash2: `bf582e4eacc1810f76656d1811693ff6c6737d2a`. Effective arena: context 131072, parallel 8, max output 32768, FP8 KV, 24 GiB reserve. The production engine binary and model weights are not redistributed.

**Working NCCL remains a prerequisite**, since larger/ineligible gathers still use it. Our dual-PCI fleet used patched NCCL 2.30.7, SHA-256 `1cdf5f3e2d617d5c07a87903eb8849d5bd7758f6c900455d613a2e0eb289d63e`. Public patch source: [sparkring's NCCL directory](https://github.com/FujitsuPolycom/sparkring/tree/4b9b6a0a213456b96400c5e1a1cf59c20ff892c8/spark_transport/nccl). This is a dependency record, not a claim that this checkout alone reproduces that binary. The recipe's stock NCCL wheel is unchanged and is **not qualified for our dual-domain setup** by these results. Neither launcher selector configures wiring, addressing or that library.

For research launcher syntax and invalid-setting gates, see [Zig engine documentation](../zig-engine.md). Patch reconstruction/host checks and their proof limits are retained in [packaging-checks](packaging-checks/). The focused cleanup has no NVCC/GPU/inference run; historical results apply to the source and binaries identified by their receipts, not automatically to a rebuilt focused binary.

## Evidence index and failures retained

[The manifest](receipts-manifest.json) hashes every unmodified exported receipt. [Offline verifier and retained clients](../../tools/tp4-rdma-bench/README.md) explain integrity checking, matched request identities and reproduction commands. Original reports retain producer-local paths; excluded private lifecycle files referenced by them are not required for offline benchmark analysis.

| Evidence | Location / interpretation |
| --- | --- |
| Exact-token qualification | `receipts/*-qualification.jsonl`, `receipts/*-parity.json`: stock/ported NCCL, registered512/1024, neighbor, eager/graph, tuning/cache/final profiles. 40 matched cases where stated. Allocated relay used a different nonce: **18 exact shared cases, 22 unmatched**, despite 40 internal passes. |
| GPU byte/guard/rank/NCCL-oracle and graph replay | `receipts/probe*` and `profile-probe*.json`: all four ranks, actual execution counters. Registered512/1024 pass 64 changing-input replays. Probe04 extends cutoff boundaries; production kernels unchanged. |
| Failed registered probe and fix | Original `probe-relay-registered-*`, successful `probe03-relay-registered-*`, [probe03 review](receipts/review-10-probe03.md): counter snapshot raced peer replay; a barrier fixed the probe. Do not label the old masked error a demonstrated coherence failure. |
| Invalid setup refusal | `receipts/negative-*`: bad memory setting and missing rail. |
| Incomplete Grill roots | `grill-ported-nccl-baseline-01` and other retained attempts: incomplete/failed captures are excluded from estimates, not repaired or silently retried into success. |
| Fault handling | `peer-abort-attempt1-*` finished before the intended kill and is **not fault proof**. `peer-loss-model-*` records the actual owned model rank 3 loss: peers exited with control-link loss within 4.46 s; the pending stream had no finish/DONE. This is one model-level fault case, not proof for every profile or fault. |
| Recall/cache/cancellation/serving | `accept-*`, `bench-*`, their analyses and final serving consoles. Code output was not executed or graded; cancellation closure does not independently measure retirement time. |
| Source/build/review | Source patches, build receipts/logs and source-only review reports under `receipts/`; new packaging checks separately above. |

No full-131K quality, BF16/Python reference equivalence, vision, MTP, long-duration reliability, eight-way sustained capacity or reboot recovery is established. The historical claim of exact stock-native outputs does not supersede upstream dense-upload/FP8 parity limits.

## Credit

**BertholomusAI (Albert Lee, [bertholomus](https://github.com/bertholomus))** supplied the DeepSeek RDMA implementation, the registered-memory approach and the public mixed-workload client that made this exploration possible. [CREDITS](../../CREDITS.md) and [NOTICE](../../NOTICE) retain his attribution, the pinned imports and the original TensorFold/MiaAI/Hector Isaza/Jürgen Schmied/b12x credits. PlotArmor's unmodified Grill provides the official comparisons. Integration, experiments and evidence publication: Scott Biggart (`swbiggart`).

### Packaging and compatibility clarification

Default backend selection is retained, but this is a new binary/image and startup protocol. Strict pre-NCCL agreement now rejects mixed rank settings and strengthens failure checking; mixed old/new binaries are unsupported. The focused branch also rejects retired research environment selectors, so an existing research deployment needs an explicit recipe migration and fresh binary qualification. No deployed binary or recipe is replaced by this publication. Neither a CPU route test nor a semantic build qualifies an additional physical topology or the stock NCCL wheel.
