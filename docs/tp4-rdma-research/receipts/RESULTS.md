# GLM Flash TP4: benchmark and RDMA results

The fastest tested profile uses GLM's neighbor-relay all-gather with registered host memory, a 512 KiB per-rank cutoff, L2 prefetch and the shared-expert side stream. The DeepSeek neighbor adapter also works, but was slower on the measured Flash workloads. All profiles preserve the native Flash gather layout and arithmetic.

## Published benchmark reproduced

We used the repository's published **Grill 0.3.0 default C1/exact400 protocol**, with its unmodified fixture, encoder, acquisition schedule and comparison code. Every valid capture completed eight acquisitions and 32 requests, each with exactly 400 reported output tokens. This metric includes TTFT and request settlement. It is not decode-only speed.

| Same-binary transport comparison | Median acquisition rate, tok/s | Official change | Nominal model-based 95% interval |
|---|---:|---:|---:|
| Initial ported NCCL baseline | 155.22 | — | — |
| GLM allocated-memory relay, 512 KiB | 178.74 | +14.99% | +14.49% to +15.49% |
| GLM registered-memory relay, 512 KiB | 180.88 | +16.45% | +16.00% to +16.91% |
| DS-style two-stage neighbor, 512 KiB | 178.41 | +14.92% | +14.46% to +15.38% |
| GLM registered-memory relay, 1024 KiB | 180.40 | +16.12% | +15.57% to +16.68% |
| Final A1 NCCL control: SIDE1, cache3 | 156.27 | — | — |
| Final B1 registered relay: SIDE1, cache3 | **182.16** | **+16.57%** | **+15.59% to +17.56%** |
| Final A2 NCCL unchanged control | 156.14 | −0.24% | −0.88% to +0.41% |
| Final B2 registered relay | 181.71 | +16.18% | +15.57% to +16.78% |

The final A1/B1/A2/B2 sequence uses identical weights, binary, image content, context, parallelism, cache budgets, prefetch and side-stream settings. Only the transport selection differs. All 128 requests report zero cached prompt tokens. Every official comparison in this sequence uses the original A1 baseline; a check capture cannot become a new baseline. A2 is an unchanged control with an inconclusive direction, not proof of equivalence. The official change uses the difference of log means of acquisition medians, so it need not equal the ratio of the displayed medians. These are sequential capture comparisons; their intervals assume independent, stable acquisition-level variation and do not establish causality or a global optimum.

The separate stock-versus-ported NCCL runtime check measured −0.96% [−1.32%, −0.59%]. We do not attribute that small sequential difference to a specific source change. Two incomplete baseline attempts are excluded and retained: one was stopped prematurely by the integrator; another never reached health after a strict device-user gate refusal.

[Official B1 comparison](grill-final-comparison.json), [official B2 comparison](grill-final02-comparison.json), [unchanged A2 control](grill-final-control-nccl-02/report.json), [raw final control](grill-final-control-nccl-01/), [raw final relay](grill-final-relay-registered-512-01/), [protocol and source pins](grill-protocol.md).

The unpublished **17 parity / 4 resume / 428 sweep / 56 main-set** fixtures are excluded from the public recipe, and the user has no separate checkout. Those exact suites remain **unreproduced**. Our qualification, recall and mixed-workload cases are supplemental. We reproduced the public Grill protocol on our fleet, not the author's Python-versus-Zig headline on its different hardware.

## Configuration exploration

Registered relay at 512 KiB was retained after 512/1024 KiB and transport screens. At 1024 KiB, 33–64-row verification windows use custom transport too; this did not yield a material advantage in our short serving screen. Neighbor remains qualified at 512 KiB only.

| Registered relay512 | 1-stream / 4-stream / 8-stream aggregate tok/s | 30-second four-stream admission |
|---|---:|---:|
| L2PF1 / SIDE0 | 87.53 / 134.30 / 174.93 | 152.79 |
| L2PF0 / SIDE0 | 83.29 / 130.37 / 172.37 | 149.84 |
| L2PF1 / SIDE1 | **87.70 / 135.70 / 175.63** | **153.52** |
| L2PF0 / SIDE1 | 83.36 / 131.54 / 173.11 | 149.93 |

These two-repetition screens are descriptive. The small SIDE margin is not a statistically established gain. All four settings passed the 40-job internal qualification. [Screen analysis](knob-screen-analysis.json), [transport screen](transport-screen-analysis.json).

The retained final02 profile completed **57 requests at 157.02 aggregate tok/s** during a separate 90-second four-stream admission test, with zero worker errors. Median/p95 TTFT was 0.182/0.333 seconds. The earlier identical final01 profile completed57requests at156.46tok/s. Every generation retained actual token arrays and verified draft metadata. This duration does not qualify long-running reliability, eight-way sustained capacity or full-window operation. [Raw final02 load evidence](bench-final-90s-02/), [validation](final02-steady-analysis.json), [earlier final01 load](bench-final-90s-01/).

`TF_GLM_CACHE_GIB` controls surplus KV capacity as well as the default kept-state budget. An initial env3/CLI3 trial reduced the KV pool and is explicitly confounded. The final setting retains **env32 and 5,619,712 shared KV rows**, with **CLI `--prompt-cache-gib 3`** controlling kept states separately. The clean trial's 32,752-token prompt resumed 32,704 tokens and preserved all 128 output token IDs. Cold serial TTFT was 19.740 s; the median warm drafted/repeated TTFT was 0.166 s. Draft mode also differs between those observations, so this is a descriptive cache observation, not an isolated causal speedup. [Independent raw-byte analysis](cache3-pool32-evidence-analysis.json).

Several apparent tuning switches are inactive for native TP4: HC split, prefill overlap and the TP2 microbatch controls. Full-model Python Q5/DCP/MTP controls are not Flash switches. We kept the native arithmetic unchanged; a fused collect/HC or different prompt MMA path would need separate rounding, rank-order and signed-zero qualification. [Effective control inventory](supported-knobs.md).

## What transferred from DeepSeek

The DS host-registration approach was useful in GLM's existing relay. We also imported its adjacent XOR1/XOR3 two-stage schedule, with explicit global-rank placement, graph sequencing, device-scoped completion lookup, sticky failure propagation and PDL disabled. Both paths use GPU staging through host memory and CPU RDMA proxies; neither is direct GPU-memory RDMA.

Flash requires a complete rank-major gather before HC post. DS pair/quarter/Markov/diagonal specializations reflect its 2-D tensor split and different consumers, so they cannot simply replace Flash's 1-D gather. Existing GLM relay forwarding avoids the DS adapter's two serial GPU-staged rounds, which is consistent with its measured lead here; this is an architectural explanation, not a measured decomposition of costs.

The additional [full GLM-5.3 repository comparison](bertholomus-comparison.md) targets a different 753B MLA/DSA model with Python orchestration and MTP. Its all-peer QPs do not fit this neighbor-only ring unchanged. Useful lessons include NIC-scoped completion ownership, graph warm-up coverage and fused collect/consumer designs. The first two are already covered in our source. Fused full-model BF16 residual math is not Flash HC math. Its published speed figures and SSE-piece counter cannot substitute for our usage-based matched benchmark.

## Correctness and failure evidence

- NCCL, allocated relay, registered relay and neighbor passed four-rank eager byte/rank/guard/oracle and changing-input CUDA graph replay gates. Registered relay passed both 512 and 1024 KiB cutoff plans, with execution counters proving the custom paths ran.
- Ported NCCL, registered relay, neighbor, 1024 KiB and eager-versus-graph qualification retained exact stock token arrays for all 40 matched fixtures. The allocated-relay trial used a different nonce: only 18 cross-profile fixtures are identical and exact; 22 remain unmatched.
- The clean cache trial and final serving candidate passed serial/drafted/overlapping-client code, prose, structured and nonce checks, cancellation recovery, synthetic 32K recall and repeated prefix reuse. Generated code was not executed or graded. Cancellation closure does not independently prove backend retirement time.
- A reviewed-controller stop of exact owned model rank3 during active private SSE caused the other ranks to exit3 with explicit control-link loss within 4.46 seconds of rank3's exit. The pending stream produced no completed response. This qualifies that peer-loss case, not every network fault or reboot.
- Four-host exact runtime/image/binary/argv/environment, exclusive compute ownership and stopped-baseline checks passed. All 1,068 protected ports were inaccessible from the external LAN client on each host. Client ports, firewall, fabric addresses, GID3, rails and keys were unchanged.

Native dense-upload clipping and FP8 loading retain upstream parity limits against ATen/Python. Strict identity against our stock native baseline does not establish BF16 reference quality. No full-131K quality, vision, long-duration reliability, reboot recovery, tailnet inference or private-assistant role acceptance is claimed.
