# GLM Zig TP4: published Grill comparison protocol

Prepared 2026-10-22 from public source and release metadata. This report author performed no SSH, GPU, model traffic, service changes or binary execution. The integrator owns live captures and has separately reported installing the verified Grill binary. SparkDash adaptation remains paused; it is not this protocol.

## Exact source and artifact identities

Grill release **v0.3.0** resolves to commit **`755859a084471a17c6f153088fc78ffadc04606e`**, source tree `998605207ab44845baa8017efa8661816fffdb73`. The read-only source checkout is `/tmp/thegrill-v030-glm-research`; its HEAD and clean status were checked. The release is a prerelease. [Pinned source](https://github.com/plotarmordev/thegrill/tree/755859a084471a17c6f153088fc78ffadc04606e), [release](https://github.com/plotarmordev/thegrill/releases/tag/v0.3.0).

Published aarch64 release metadata identifies:

| Item | SHA-256 |
| --- | --- |
| `grill-perf-0.3.0-aarch64-unknown-linux-gnu.tar.gz` | `0dae75a72bee501622dafe6ea0c26e838bfb6060f3651f6cd9bb1eee5f2c557a` |
| `bin/grill-perf` | `2b48429f5b97cc949a221b8fcef20cbcbb583432ceb766a18d8b1bf3a10b9310` |
| `Cargo.lock` | `d78477995c81d34e9fcd9d8dc7194c39b036bba37dcd04b7d73f601ecfdb7552` |
| Embedded `baseline-v1.json` | `a2eb10e38e7c351c91a3dc78452af3e99b505d718342c3344517375f8567c959` |

The last two hashes were also computed from the pinned local source and match the published receipt. This report independently inspected the published receipt and checksum sidecar, not the installed head binary. [Receipt](https://github.com/plotarmordev/thegrill/releases/download/v0.3.0/grill-perf-0.3.0-aarch64-unknown-linux-gnu.receipt.json), [checksum](https://github.com/plotarmordev/thegrill/releases/download/v0.3.0/grill-perf-0.3.0-aarch64-unknown-linux-gnu.tar.gz.sha256).

Source build recorded in that receipt:

```sh
cargo +1.98.0 build --locked --release \
  --target aarch64-unknown-linux-gnu -p grill-perf --bin grill-perf
```

The release runtime baseline is native Linux aarch64, Ubuntu 24.04/glibc 2.39, with a readable system CA store even for loopback HTTP. Source builds additionally require Rust/Cargo 1.98 and a native C/C++ toolchain and CMake. No Python helper or backend hook is needed for the default serving comparison. The pinned installation guide still names v0.2.0; use the explicit v0.3.0 release/receipt above rather than its stale version download example. [Pinned installation guide](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/docs/performance/INSTALL.md).

## Published Zig TP4 comparison and its scope

GLM recipe [PR #200](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/200) reports a default structured C1/exact400 Grill comparison using `--change runtime`: **IMPROVED, +0.68%, model-based 95% interval +0.08% to +1.29%**. Python run `20261020T042355-65a691` precedes Zig run `20261020T042859-61a3fe`. Both Grill runs use a 24 GiB reserve, DFlash2, parallel 8 and context 131072. The tested physical ring comprised three DGX Sparks and one Dell Pro Max GB10; the current four-GB10 fleet is a different hardware configuration.

That result describes those two capture periods on one short structured workload. It does not establish causality, general serving improvement, concurrent or long-prompt performance, equivalence or noninferiority. The original raw Grill capture directories were not obtained in this analysis, so their published result has not been independently replayed here.

PR #200 was observed merged, with head `f8bfe7ce116df26de8555835365409efb2a5e42f` and merge commit `fa619b5819223af1b884374384012d27ded7dc45`. Its body is mutable; the UTF-8 body fetched from the GitHub API on 2026-10-22 hashed to `8f33cc2e7ae7d39e59502f83283a36b31a45b6e91ee27a8888cf2f643e7f5dc3`. Use [its immutable head](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/commit/f8bfe7ce116df26de8555835365409efb2a5e42f) for source provenance; the body hash identifies the report text inspected here.

The present recipe snapshot is `a6c7e08ae864239125d45fc15146d26ec74c7f6d`. Its [TP4 documentation](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/a6c7e08ae864239125d45fc15146d26ec74c7f6d/docs/zig-engine.md#tp4-experimental) specifies a switchless ring, `ENGINE=zig SPLIT=0 COMM=nccl DRAFTER=dflash2 PARALLEL=8 CONTEXT=131072 MEMORY_RESERVE_GIB=24`. This outer recipe commit is distinct from the native port commit: [PORT_REF](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/a6c7e08ae864239125d45fc15146d26ec74c7f6d/zig/PORT_REF) records `a10c8647af52c98a7024092d1d9f1824843e3693`, branch `r107-int`, and the first five patch headers name that native commit. The recipe builds TensorFold base `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` plus its patch series. Do not identify a new run merely by the outer recipe revision; retain actual image, binary, model, drafter, effective arguments and transport evidence.

## Exact default workload and acquisition schedule

The embedded [baseline-v1 workload](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/examples/baseline-v1.json) has one user message, shown below. It uses `vllm-fixed-v1`, streaming, exact 400 reported completion tokens, cache mode `observe`, temperature zero, top-p one, no seed and thinking false. The actual [request encoder](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/wire.rs#L115) sends:

```json
{
  "model": "GLM-5.3-Flash-EXL3",
  "messages": [{"role": "user", "content": "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."}],
  "stream": true,
  "max_tokens": 400,
  "temperature": 0.0,
  "top_p": 1.0,
  "chat_template_kwargs": {"thinking": false},
  "stream_options": {"include_usage": true},
  "min_tokens": 400,
  "ignore_eos": true
}
```

The model selector is supplied by the operator; all other fields above follow the stock fixture. No `draft`, `return_token_ids`, `cache_salt` or alternate `enable_thinking` field is added. Drafter behavior therefore follows the serving configuration. Backend support for the requested controls must be established separately; successful response handling and a requested thinking flag alone do not prove template behavior. Rejected controls are not stripped and retried. Observed cache mode does not force a cold or prefix-hit state.

[study.rs](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/study.rs#L6) fixes **eight acquisitions** with a fresh HTTP client per acquisition. Every acquisition has one warmup followed by three measured C1 waves, each containing one request. Each capture therefore permits **8 warmup + 24 measured = 32 requests**, at most **12,800 reported output tokens**, within **300 seconds** including collection overhead. Per-request total/idle limits are 60/30 seconds; response limit is 1 MiB and wave buffer limit is 64 MiB. `--seconds` changes only the prospectively declared whole-capture time allowance, not request count, and is not inherited by `check`. No replacement acquisitions or automatic retries repair an incomplete capture. [Detailed protocol](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/docs/performance/README.md#default-scope-and-budget).

Do not pass `--selection` to reproduce the default statistical comparison. Even explicitly selecting the same C1 fixture changes the capture identity to descriptive-only scope.

## Deployment declarations and commands

Every declaration must contain truthful nonempty strings for `model_revision`, `runtime`, `hardware` and `settings`. These are operator declarations, not server attestation. Store actual checkpoint revision, actual engine/image/binary identity, physical fleet/layout identity and effective configuration fingerprint. Avoid credentials in declarations.

The [change validator](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/study.rs#L309) requires exactly the selected field to differ:

| Change | Required declaration relationship |
| --- | --- |
| `runtime` | Engine/build identity changes; weights, hardware and settings match. Appropriate for the published Python-to-Zig comparison only when those other facts actually match. |
| `settings` | Configuration changes; weights, runtime and hardware match. Appropriate for same-binary transport or configuration changes. |
| `none` | Every field matches; unchanged-deployment control. A directional control result still requires investigation. |
| `model_revision` / `hardware` | Only that selected identity changes. |

If both runtime and configuration change, do not fabricate matching declarations to obtain a verdict. The simple baseline/check path does not admit a multi-field migration. A new current-fleet runtime comparison also cannot reuse the historical mixed-hardware baseline. A stock-versus-tuned Zig comparison is a settings comparison, not a fresh reproduction of Zig-versus-Python.

Only the integrator performs live work. The following are command templates, not evidence of execution by this author. `PERF` must name the same verified binary for both captures; each output destination must be fresh, with an existing private parent. The full loopback resource URL avoids credentials at the native backend.

```sh
"$PERF" baseline \
  --endpoint http://127.0.0.1:40053/v1/chat/completions \
  --local-http --model GLM-5.3-Flash-EXL3 \
  --deployment serving-before.json --out results/before

# Integrator separately performs the authorized serving change.

"$PERF" check results/before \
  --deployment serving-after.json --change runtime \
  --out results/after

"$PERF" compare results/before results/after --json
```

For the same-binary stock-versus-tuned comparison, substitute `--change settings` and the actual configuration declarations. `check` inherits the baseline's endpoint, model selector, loopback permission, authentication-variable name, workload and collector identity; it does not accept replacement endpoint/workload options. Authentication, if separately needed, is supplied with `baseline --auth-env NAME`; never put the credential value on the command line or in a declaration.

**A check result cannot become the baseline of another check.** [study.rs](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/study.rs#L1108) rejects a baseline whose `baseline_sha256` is already populated. Therefore, after `check STOCK --change runtime --out PORTED`, `check PORTED --change settings --out RELAY` is invalid. Capture a fresh standalone `baseline` while ported NCCL is serving, then use that new original root for `check NEW_PORTED_BASELINE --change settings --out RELAY`. This is another complete, prospectively budgeted 32-request capture; retaining PORTED as historical runtime-comparison evidence does not make it an eligible baseline. Do not copy or edit manifests to bypass lineage.

## What the default metric means

For each eligible C1 wave, throughput is the server-reported **400 completion tokens divided by dispatch-to-settlement wall time**, including TTFT and terminal collection. At higher concurrency the general wave formula sums reported completions across streams and divides by first-dispatch-to-last-settlement time. It is achieved fixed-wave throughput, not decode-only throughput, maximum capacity or a sustained-load claim. The acquisition observation is the median of its three measured wave throughputs; warmup is retained but excluded. [Contract](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/docs/performance/CONTRACT.md#L223), [arithmetic](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/evidence.rs#L129).

Let `a` and `b` contain the eight acquisition medians for baseline and candidate. The estimate is `100 * (exp(mean(log(b)) - mean(log(a))) - 1)`. The log-scale standard error is `sqrt(var(log(a))/8 + var(log(b))/8)`, using sample variances. The model-based 95% interval uses critical value `2.364624251` and conservative df 7. All eight acquisitions on each side must complete. Zero estimated variation leaves the observed percentage visible but withholds interval/confidence. Sequential captures can carry time, load and cache effects; fresh clients do not prove independence. [Assessment implementation](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/study.rs#L1310).

## Official comparison output and extraction

Use the original executable's **`compare BASELINE CANDIDATE --json`** on the capture roots, not their `acquisition-*` children. It performs offline evidence verification and recomputation, with no model traffic. It does **not** rewrite stored `report.json`; if retaining stdout, choose a fresh file outside the immutable captures. Do not pass `--reference`: capture comparisons reject that raw-run option. [CLI dispatch](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/main.rs#L255).

The JSON object has `version: 1`, `kind: "performance-capture-comparison-v1"`; fields are defined in [Report](https://github.com/plotarmordev/thegrill/blob/755859a084471a17c6f153088fc78ffadc04606e/crates/grill-perf/src/study.rs#L199). Extract from a newly retained `compare.json`:

```sh
jq '{version, kind, model, result, result_scope, scope, declared_change,
     baseline_complete_acquisitions, candidate_complete_acquisitions,
     baseline_accounting, candidate_accounting,
     baseline_acquisition_medians, candidate_acquisition_medians,
     observed_change_percent, model_based_interval_percent,
     model_based_confidence_level, baseline_capture_sha256,
     candidate_capture_sha256, reasons, stop_reason, first_failure}' compare.json
```

Read `result` and coverage together. `baseline_ready` is a baseline-capture readiness flag; it is not the comparison success flag and is normally false on a compare report. A ready baseline can have `result: INCONCLUSIVE` because no candidate exists yet. For complete stock default captures, expect eight acquisition medians per side and accounting of 32 dispatched/complete requests, 12,800 reported completion tokens, and zero missing-completion-usage requests per side. Missing usage is unknown, not zero. On a failed load, null accounting means unavailable; zero complete-acquisition placeholders do not prove observed zero coverage.

| `result` | Display/meaning | Comparison exit |
| --- | --- | --- |
| `IMPROVED` | Measured faster in the capture periods; interval wholly above zero | 0 |
| `REGRESSED` | Measured slower in the capture periods; interval wholly below zero | 2 |
| `INCONCLUSIVE` | No supported direction or incomplete coverage; not equivalence | 2 |
| `INVALID` | Evidence, responses or declarations fail validation | 1 |
| `DESCRIPTIVE` | Explicit selection complete; no default-C1 directional verdict | 0 |

Retain exit 2 and its JSON rather than treating it as a transport failure or retrying until favorable. Baseline readiness has distinct exits: ready 0, incomplete 2, invalid 1. The official report's `observed_change_percent` is authoritative for the comparison estimate. The median arrays provide per-acquisition achieved TPS; computing a separate median across those arrays is only an additional descriptive summary and does not replace the official log-mean estimate.

## Missing reference suites and proof limits

The GLM source documentation reports 17 parity cases, four chat-resume cases, a 428-case sweep and a 56-request main set with eight concurrent requests and draft off/on. These are separate qualification suites, not additional Grill default cells. The exact harnesses and fixture manifests were not located in the recipe snapshot or the inspected Grill source. References to `zig-notes/r107-int.md` and `work/zig-notes/r107-tpn.md` do not provide those absent files. [make-zig-patches.sh](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/a6c7e08ae864239125d45fc15146d26ec74c7f6d/tools/make-zig-patches.sh#L13) explicitly excludes source-port notes and capture/reference tools. Do not invent replacements and label them reproduced upstream suites.

The reported 428-case workload took 974 s on Zig versus 85 s on Python; that comparison had capture-hook/reserve differences. It is not the short Grill result. The 90.5 drafted / 75.1 serial TPS numbers in patch 0013 explicitly concern TP2 and must not be assigned to TP4. Exact output-token IDs, serial/draft parity, required/named tools, long-context quality, sustained concurrent capacity and reboot recovery require separate evidence. The default request does not ask for token IDs and its statistical report proves none of those gates.

## Additional source hashes inspected

All Grill paths below are relative to commit `755859a084471a17c6f153088fc78ffadc04606e`.

| Path | SHA-256 |
| --- | --- |
| `crates/grill-perf/src/study.rs` | `1f69a00bd8e88597a599ac3471dcefa5be9e3fcde82a759dcdad8b8dd1dc7f69` |
| `crates/grill-perf/src/main.rs` | `3decaffea51b50cc7c92d507316d130d0ff88fcfee036cce6662db4eb43b02fe` |
| `crates/grill-perf/src/wire.rs` | `ab3e9d10552d9467487e18655d9d613551283b455f4f90bcbb0a6390bfe6165a` |
| `crates/grill-perf/src/evidence.rs` | `711e8ed0f78c9770b4c25800987d8ca424947c8c7b18d63479122974c09c4a9f` |
| `crates/grill-perf/src/schedule.rs` | `7f0ea5c0ff1db581c5b5433182d5d34e5332aa662b53dde99e7b34828c778b84` |
| `docs/performance/README.md` | `1eae3bac762e9517c74f55fc8057635da1cb40038b4df46b566b40d3571b681e` |
| `docs/performance/CONTRACT.md` | `d157b8d7574e14d2d528d4262842fda20fbda6ec965f8a35b3a13ffebccabe57` |

For the local recipe snapshot identified as `a6c7e08ae864239125d45fc15146d26ec74c7f6d`:

| Path | SHA-256 |
| --- | --- |
| `docs/zig-engine.md` | `8b66255984d76a0f1de7b2b99f011065f223d58798caeb0807dec1e0e2ff69fa` |
| `tools/make-zig-patches.sh` | `b2d53fd7af3264b344fa310480b31f2f9c57bc5d3de689603b90701da63a3f4c` |
| `zig/PORT_REF` | `9a42bd247196ebe5603bf772a85c4b867759820a90268bd19fe5088298b597e9` |
| `zig/patches/0013-zig-glm-round-batching.patch` | `7be776fa76a73d9159d867f4e75f62653a35c749925c18cf94c57311e1eb8f96` |
