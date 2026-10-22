# probe04 review: 512 and 1024 KiB relay cutoff plans

**Verdict:** Cleared. No P1 or P2. Only the probe changed; the production engine and kernels are untouched. I ran no SSH, build, tests or GPU work, and wrote no files. I did not read the receipt (`/tmp/glm-probe04-receipt.json`) or your report (`/tmp/glm-probe04-review.md`).

**Hashes and scope:**
- The incremental patch (`23eceb7b…24b7`) and the combined 03+04 patch (`b0737ad9…3142`) both match.
- Commit `44cd0314` on top of `b733c376` changes only `zig/tests/cuda/glm_transport_probe.zig`, and its `+`/`-` lines match the incremental patch exactly.

**What I verified:**
- **The cutoff comes from the live communicator.** `activeCutoff` reads the relay's `slot_bytes`, which every rank agreed on during the handshake; mismatched slot sizes make `required` relay fail in `Comm.init`. Neighbor uses its fixed 512 KiB. NCCL returns 0, including when the extra struct exists only for L2 prefetch.
- **Unsupported setups are refused.** `SizePlan.init` refuses NCCL with a non-zero cutoff, neighbor at anything but 512 KiB, and relay at anything but 512 or 1024 KiB. Profile parsing is exact, so `"ROCE"`, `" roce"`, `"auto"` and similar are rejected.
- **All ranks must agree the plan.** The plan text (profile, cutoff and the three graph sizes) is passed through `agreeRanks` before any gather. A rank with a different plan dies.
- **The 512 KiB plans are preserved.** NCCL, neighbor and 512 KiB relay keep the original graph sizes (16384, 524304, 32768) and the original 10 eager cases.
- **The 1024 KiB graph is right.** Its sizes are 540672 (33 FP32 rows), 1048592 and 1048576 (64 rows).
  - Relay `fits` is `bytes ≤ slot_bytes` and a multiple of 4, so the plan is custom, NCCL, custom.
  - That keeps exactly two custom calls per replay, so the `2 × replays` counter targets and the relay stripe math are unchanged.
  - The new check requires both outer calls to use the requested backend, which also tightens the NCCL profile.
  - At this cutoff, the eager cases 524288, 524304 and 1048576 now take the relay path, and 540672 plus 1048592 are added. That covers both sides of the 1 MiB boundary.
- **Buffer sizes are bounded and overflow-checked.** `caseSizes` caps bytes at 1 MiB + 16 and allows only offsets 0 or 4. Send is `bytes + 2·guard + offset`, receive is `4·bytes + 2·guard + offset`, and the oracle is `4·bytes`, the same layout as before. The last rank's in-place source ends exactly at the payload end, before the trailing guard.
- **Mixed versions are kept apart.** The `v3` agreement and pass strings stop probe03 and probe04 from being mixed.

**Proof limits:**
- I didn't run the claimed host build and 6 tests, or the aarch64 semantic build. There is no NVCC build and no GPU run yet.
- **The probe still runs with `TF_GLM_L2PF=0`.** In production with prefetch on, the relay launch adds up to 64 prefetch blocks beside its 16 gather blocks. That path, and nonempty prefetch at 1 MiB shards, is not exercised.
- **The probe proves transport bytes only.** A 1024 KiB cutoff in production sends every decode gather up to 64 rows over the relay. It therefore needs its own exact-token, eager-versus-graph and matched-performance gates before acceptance.
- GB10 coherence for registered memory is still qualified only through probe passes, which still need to run at both cutoffs.
- No claim of a healthy deployment.
