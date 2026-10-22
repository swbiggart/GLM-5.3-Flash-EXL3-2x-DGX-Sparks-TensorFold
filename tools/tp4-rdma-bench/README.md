# Retained benchmark clients

These tools preserve the synthetic workloads used in the [TP4 RDMA study](../../docs/tp4-rdma-research/README.md). They do not reproduce MiaAI's unpublished 17/4/428/56-request reference suites. The primary speed comparison uses the unmodified public Grill 0.3.0 binary and its default C1/exact400 workload; see the [pinned protocol and commands](../../docs/tp4-rdma-research/receipts/grill-protocol.md).

## Offline receipt verification

From the repository root, with Python 3.10 or later:

```sh
python3 tools/tp4-rdma-bench/verify_receipts.py
python3 -m unittest discover -s tools/tp4-rdma-bench -p 'test_*.py'
python3 tools/tp4-rdma-bench/analyze_glm_runs.py \
  --run nccl=docs/tp4-rdma-research/receipts/bench-upstream-nccl-screen01 \
  --run relay=docs/tp4-rdma-research/receipts/bench-relay-registered-512-screen01 \
  --run neighbor=docs/tp4-rdma-research/receipts/bench-neighbor-screen01 \
  --output /tmp/glm-screen-analysis.json
```

The integrity verifier checks every exported file, length and SHA-256 against the manifest. It establishes archive integrity relative to that manifest, not truth of the original measurement. The analyzer validates raw SSE/token/usage metadata and summary hashes. Read each coverage status: exit 0 can include incomplete or unmatched request coverage. Warmups contribute to parity only, never measured throughput.

## Run the supplemental workloads

The portable adapter targets an **already running, isolated** GLM backend at `http://127.0.0.1:40053`, model `GLM-5.3-Flash-EXL3`. Run it on the head host. It does not start, stop or configure the model. Do not expose this backend publicly. These runs generate model traffic and need an idle fleet:

```sh
GLM_BENCH_SCREEN=1 python3 tools/tp4-rdma-bench/bench_glm.py --stage all --out /tmp/new-glm-screen
python3 tools/tp4-rdma-bench/bench_glm.py --stage all --out /tmp/new-glm-full
```

Screen mode uses two repetitions at 1/4/8 streams and a 30-second four-stream admission window; full mode uses six repetitions and 90 seconds. Both include a warmup and cold 8K prefill probes. Completion counts come from server usage, and the adapter requires exact output token IDs. This is the public DeepSeek mixed-workload client adapted to GLM, **not** Grill's statistical protocol. Reusing an output filename is refused.

`glm_accept.py` supplies the additional synthetic serial/draft/concurrent, cancellation/recovery and long-needle/cache probes; inspect `--help` before issuing traffic. Their reported throughput is descriptive. None executes or grades generated code.

## Provenance and licensing

- `vendor/kit_bench.py` is retained byte-for-byte from **BertholomusAI (Albert Lee)**, [`bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/tree/9af1f4567ee49ec2a31f68975982093277e7090d), `tools/kit_bench.py`, Apache-2.0. Its upstream license is retained as `vendor/LICENSE.bertholomus`.
- `vendor/benchmark.py` records raw requests, responses and SSE metadata around that client. `vendor/original_glm_adapter.py` is the original study adapter. `bench_glm.py` changes only the import path to use this local vendor directory. [SOURCE.json](SOURCE.json) records hashes and the adaptation.
- `vendor/qualify_glm_native.py` is the **unchanged, site-specific original producer**, retained for auditability. It expects existing mode-0600 proxy key files at its original deployment paths and never contains key values. Do not run it as an offline verifier or copy its private gateway layout into an installer. Qualification JSONL records fixture identities and output IDs/text; it does not retain every full HTTP request body.
- Grill is an external tool by PlotArmor. No Grill binary is redistributed; its pinned source, release license and verification hashes are linked in the protocol.

No credentials, model weights, serving binaries or deployment controllers are shipped here. Raw receipts retain nonsecret original producer paths and hardware settings to preserve their hash chains.
