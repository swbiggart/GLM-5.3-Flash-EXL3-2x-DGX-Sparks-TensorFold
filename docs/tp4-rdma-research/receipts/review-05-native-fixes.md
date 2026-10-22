# Re-review: native fixes for the two neighbor P2s (read-only)

**Verdict:** Both prior neighbor P2s are fixed, and I found no new P1 or P2 in the changed code. I ran no tests, builds, GPU work or live commands, and wrote no files.

**Hashes checked:**
- Incremental patch `40f77db4…60e1` and combined patch `ff835d43…27a6` both match.
- The frozen tree `c8b5f94b…` differs from the previously reviewed `a8dcb54e…` snapshot in exactly the six files in the incremental patch, line for line.

## The two P2 fixes

**1. PDL is now off.**
- `cuda_neighbor.zig` builds the ring settings with `.pdl = false`. Both kernel launches then go without the PDL attribute, and the kernels skip their wait/early-release step.
- Every neighbor kernel is therefore ordered by the normal stream, and the early-release hazard is gone.
- The startup log now prints `pdl=false`.

**2. Replay counters now prove the graph used the custom transport.** I checked each count against how the counters actually work.
- **Neighbor:**
  - Each ring's sequence advances by one per gather, starting from zeroed scalars.
  - `completedWrites` is the highest fully completed sequence. It only moves forward, because completions on each queue pair arrive in order.
  - So the eager target (equal to `neighbor_enqueues` before capture) and the replay target (exactly `before + 2 × replays` per stage) are correct.
  - The exact equality check catches both missing and duplicated replays.
- **Relay:**
  - `tf_roce_ops` goes up by one per posted sequence (`roce_proxy.c:641`), so one per gather.
  - In ring mode each rank posts `n_relay = n_flag` forwards per gather (`tf_roce_set_ring`). The `roce_enqueues × n_flag` and `calls × n_flag` targets are therefore right.
  - The replay drain makes the full relay target a hard requirement, although `validateReplayCounts` itself only checks that relays increased.
- **Ordering:**
  - Capture is checked not to advance any execution counter, and host enqueue counters must stay unchanged through replay.
  - Each drain is preceded by an all-rank agreement, so a rank can't time out waiting on a slower peer. Each drain is bounded at 5 s with 1 ms polls and checks for transport failure on every poll.

## Sticky-failure additions

- **`Gather.run` latches every refusal.** Its `errdefer` abort now sits at the top, so buffer refusal, a failed check and a second stream all set both stages' failure. The new test covers both the buffer and stream cases.
- **Failures after neighbor is selected latch too.** `allGatherThen` has a conditional `errdefer` that also covers a failed prefetch. It only applies once neighbor has been chosen, so NCCL and relay calls are unaffected.
- **Snapshots and spill check for transport failure.**
  - `takeSnap` checks before copying and again after synchronizing. Because every earlier gather has finished by then, no aborted bytes can be kept.
  - `spillOut` and `spillWrite` check before starting, after each chunk is copied off the device, and before the final write.
  - These checks run on the spill thread, which is safe: `check()` only reads atomics.
  - `Engine.comm` is set in production (`cuda_native.zig:115`).
- **No regression for other profiles.** With NCCL only, or with no communicator, `check()` returns without error.

## Minor notes (P3)

- **Relay op count is a lower bound only.** `roce_ops` uses `>=`, so a duplicated relay op would go unnoticed. That only matters for diagnostics; the byte checks still guard correctness.
- **Spill skips silently.** `spillOut` returns on a transport failure without logging; a one-line warning would help.
- **Completion counters only prove local sends.** They show local NIC writes completed, not that peers received the data. Receipt is proven by the per-replay byte and oracle checks, which are still in place.

## Proof limits

- I did not run the claimed 47 transport tests, 3 probe tests, or the Linux semantic build.
- There is no NVCC build and no GPU run. That still leaves unproven:
  - GB10 coherence for registered memory.
  - Whether the counter drains complete within 5 s on real hardware.
  - Peer-loss handling under the fleet watchdog.
  - Exact tokens with the model.
- Neighbor still needs its own GPU qualification, and the lifecycle hardware-env allowlist must be extended for its per-host FIRST/SECOND rail lists before it can be launched.
