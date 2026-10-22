Showing 1 of 143 activities (limited to 1)

# Re-review: combined GLM native transport candidate (read-only)

**Verdict:** No P1. Two new P2 findings, both in the neighbor path. They should be resolved, or explicitly gated, before the neighbor profile runs on GPUs. The NCCL and relay profiles (allocated and registered) look clear to proceed to the weight-free GPU gate. I ran no tests, no build, no SSH and no GPU work, and wrote no files this round.

**Hashes and sources checked:**
- The patch hash matches (`16d87a7d…39e4b8`), and its files match the tree diff against the unchanged baseline.
- The imported neighbor files match the hashes pinned in `glm-neighbor-source.json`:
  - The CUDA kernel is byte-identical to upstream.
  - The sequence helper is byte-identical apart from the stated import rename.
  - `rdma.zig` differs only in the stated changes: the import rename, the unregister-failure cleanup, and the two new read-only counters.
- One caveat: the local upstream checkout I compared against is at `51726210…`, not `b8b115e7…`. So I matched content hashes only; I did not confirm the commit.

## Earlier stage-1 P2s

- **Rank disagreement now fails together.** Settings are parsed and swapped with every rank before NCCL opens, and an invalid row makes all ranks fail together (`Startup.read`, then `gatherToAll`, then `validateStartupRows`).
  - All fields, including prefetch, ring and legacy-enable, must match. The tighter behaviour is documented.
  - Mixed old/new binaries are now explicitly unsupported. They would hang on the new startup frame, so the lifecycle must enforce one hash per fleet and an external deadline.
- **Driver-table host registration is fixed.** It uses the driver table's register/unregister functions and only unmaps memory if unregistering succeeded, on every path.
- **Counters are real.** They come from the actual dispatch path and the proxies, and the relay counter is now atomic.

## What I verified in the new neighbor path

- **Global rank layout is correct.** Stage one (xor1, ring rank `g%2`) and stage two (xor3, ring rank `g/2`) produce `[0,1,2,3]` for every rank. The peer info in `connect` is indexed correctly.
  - The pairs are 0–1 and 2–3, then 0–3 and 1–2. All four are physical neighbours on spark → spark-peer → b9dc → c21f, given ranks 0–3 in that host order.
- **Sizes are correct.** Stage two moves `bytes/8` float4s, i.e. twice the stage-one shard. The 1 MiB slot holds twice the 512 KiB cutoff.
- **Shape and alias selection happens before anything is enqueued**, and every rank makes the same choice:
  - It depends only on the byte count, plus pointer properties that are the same on every rank in practice: allocations are 256-byte aligned, and in-place gathers overlap on every rank.
  - The 8-byte startup gather and in-place gathers go to NCCL.
  - Buffer-size arithmetic is overflow-checked.
- **All gathers run on one stream.** Every gather call site resolves to the engine stream (`e.stream`/`tri.s`/`th`). The side stream only runs the shared expert, and graph capture uses the same stream. A stream bound to a different handle is refused before enqueue.
- **Kernel ordering is safe internally.** Each neighbor kernel waits for its predecessor before touching anything. So scratch can't be rewritten by the next call while stage two is still reading it.
- **Failures are sticky with no NCCL fallback:**
  - An enqueue error aborts both local rings.
  - Every later dispatch and the in-place gather check the failure first.
  - The sampler checks after its synchronize, and every sampling path goes through `Sampler.gather`.
  - Any error after the ranks have agreed a call is fatal: the gate's `fatalFn` exits the process (`_exit(3)`).
- **Teardown order is correct.** The engine (graphs, then stream) is torn down before the communicator. Neighbor teardown stops the proxy threads, then destroys the queue pairs and memory registrations, then unregisters the memory. The probe's defers are in the same order.
- **The probe's setup is properly ordered.** `upload` synchronizes the NULL stream after the sentinel memsets, so there is no race with the probe's non-blocking stream.

## P2 findings (new)

1. **Kernel launch ordering (PDL) can let a following GLM kernel read the output early.**
   - The neighbor stage and collect kernels release dependent kernels at their very first instruction (`griddepcontrol.launch_dependents`), and the adapter fixes PDL on.
   - The next kernel launched with PDL can therefore start while `collect` is still waiting for peers. It is only correct if it waits (`griddepcontrol.wait`) before reading the gathered data.
   - GLM's `lane_gemv.cu` and `qmm_group.cu` do wait before reading inputs; I checked both.
   - I could not check the ahead-of-time Triton kernels: their PDL flag comes from kernel-set JSON that is not in this tree (`aot.zig:107`).
   - NCCL and the relay never release dependents early, so this exposure is new to the neighbor path. The weight-free probe runs no GLM kernels, so it can't catch it.
   - Fix: qualify neighbor first with `pdl = false` in the adapter's settings (the kernels honour that flag). Alternatively, audit every PDL-enabled kernel that can follow a gather, then compare exact tokens with PDL on and off.

2. **The probe's counters can't prove that graph replays used the custom transport.**
   - The eager phase already makes the neighbor completed-write counters, relay ops and relayed stripes non-zero before capture, and the final check only requires them to be greater than zero.
   - NCCL would also produce correct bytes, and the backend label comes from a host-side query, not from what the graph contains.
   - Fix:
     - Take a `stats()` snapshot before and after the replay loop.
     - Require the neighbor first- and second-stage completed-write counts each to rise by exactly `2 × replays`, and relay `roce_ops` to rise by at least that much.
     - Require the relay stripe count to rise on all ranks when the ring is on.
     - Require enqueue counters to stay unchanged during replay.

## P3 findings

- **Errors that don't latch.** In `run()`, the early returns (buffer refusal, the up-front failure check, and a second-stream refusal) don't set the sticky failure. Today that is safe only because every caller propagates the error and the gate exits. If a future caller caught the error and continued, the ranks' ring sequences would drift apart and mix data between calls silently. Make every failure after neighbor has been chosen latch.
- **No device-side timeout.** Neighbor kernels wait without a timeout, unlike the relay's trap after `TF_ROCE_WAIT_S`. Recovery depends on the watchdog: control-link loss, or `TF_TP_TIMEOUT_S` (300 s, and only on calls with a deadline set). The probe has no watchdog at all, so it needs an external supervisor (already documented).
- **No startup check that the rails are wired as listed.** A wrong or swapped order in the FIRST/SECOND lists still connects, then fails on the first gather when retries are exhausted. Run the probe after any rail-mapping change.
- **Snapshots and spill don't check transport failure.** `takeSnap` and the disk spill tier, which survives restarts, store state without calling `Comm.check()`. The exit-on-error gate makes a poisoned spill unlikely; adding a check before snapshot and spill writes would close it fully.
- **Resource notes.** Each rank runs two busy-polling proxy threads, about two cores. The optional `glm_neighbor_rdma` fatbin must be confirmed present in the deployed binary.

## Proof limits

- I did not run the claimed 46 + 6 focused tests or the full suite of 215; this was source review only.
- There is no evidence yet from a real Linux NVCC build or from the GPUs. That includes:
  - GB10 coherence for registered memory.
  - The ordering interaction between the neighbor kernels and real GLM kernels.
  - Peer-loss handling under fleet failure.
  - Exact tokens with the model.
- I am not claiming a healthy deployment.