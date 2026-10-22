# probe03 replay-barrier fix: review

**Verdict:** Cleared. No P1 or P2. The change is limited to the probe, and production Comm, transport and kernel code are unchanged. I ran no SSH, tests or GPU work and wrote no files.

**Hashes match.** The patch is `dcdaac8a…98a7`, and commit `b733c376` changes only `zig/tests/cuda/glm_transport_probe.zig`. Its `+`/`-` lines match the incremental patch exactly, on top of `c8b5f94b`.

**Your race hypothesis is consistent with the code:**
- In ring mode, `relayed_stripes` counts forwards this rank's proxy posts for its neighbours' stripes. A fast peer that starts replaying can make a slow rank's proxy relay before that rank samples `before_replay`.
- That would produce a false `CaptureExecutedTransport`.
- `roce_ops` counts only this rank's own posts, and the neighbor counters only local writes, so this race only affects the relay counter. That fits a failure seen only in the relay ring.

**The fix is correct.** Each rank now samples `before_replay` and compares it with `drained`, and only then joins a new all-rank barrier ("capture snapshots complete").
- No rank can replay until every rank has taken its sample.
- A rank whose counters don't match dies before the barrier. Its peers then fail on the closed link with their own real error, instead of being released.
- The eager barrier before draining, and the replay barrier plus drain afterwards, are unchanged. The new test covers both outcomes, including that a mismatched rank never releases its peers.

**Errors are no longer masked.** Every `errdefer die(error.ProbeOperationFailed)` is gone.
- After `Comm.init`, each fallible call goes through `checked(rank, …)`, which dies at the call site with the actual error.
- It calls `_exit`, so no CUDA or transport cleanup runs behind a failed peer.
- The checks that already reported their real error are unchanged: byte checks, `waitCounters` and `validateReplayCounts`.
- `captureReady` takes `link` as `anytype`, which matches `Link.agreeRanks(gpa, rank, text)`.
- The `v2` agreement string stops v1 and v2 probes from being mixed.

**What the earlier rank-3 failure can and can't mean.** The masked `ProbeOperationFailed` must have come from one of the old `try` or `return` paths reached after the `GRAPH` line. Those are the barrier, capture, instantiate, launch and prepare calls, plus the `CaptureExecutedTransport` return. Byte checks already reported explicit errors such as `WrongRankBytes`, so no byte mismatch occurred in any check that ran. But the registered relay's graph replay is still unqualified, and only a rerun with the unmasked error and snapshots can confirm the cause.

**P3 items:**
- `cuda.Stream.init` still uses `try` after `Comm.init`. If it failed, the unwind would run `comm.deinit()`, which could block behind peers. It's unlikely to matter.
- `agreeRanks` has no timeout, so the external supervisor's deadline is still required.

**Proof limits:**
- I didn't run the claimed 4 host tests, the executable build or the Linux semantic build.
- I didn't see the real-GPU PASS logs for NCCL, the allocated relay and neighbor, or the registered relay's failure log; I've taken them as stated.
- Registered memory on GB10 remains unqualified until probe03 passes on all four ranks.
- No claim of a healthy deployment.
