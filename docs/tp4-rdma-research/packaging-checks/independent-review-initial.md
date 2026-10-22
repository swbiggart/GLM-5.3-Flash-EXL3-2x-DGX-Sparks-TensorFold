Review complete. I stayed read-only throughout: git show/diff/ls-tree, file reads and hashing only. I didn't build, run tests, use SSH or the GPU, delegate, or write any file.

# Independent review: GLM TP4 registered relay (focused `7a73080` / `c0b1568`) and research (`e58a1ec`)

**Pins confirmed:**
- Focused patch0014 sha256 is `6113205a…cf68f`, in both the worktree and the commit.
- Research patch0014 sha256 is `c10e9b29…b739`.
- The `TRANSPORT_SOURCE.json` file hashes match the focused result receipt.
- `c8b5f94`→`44cd031` changes only `zig/tests/cuda/glm_transport_probe.zig`, so "production unchanged, probe → probe04" holds.
- The research packaging checks ran in `/tmp/glm-tp4-research-packaging-20261022` at HEAD `02de5d6`, not `e58a1ec`. Its `start.sh`, `nodes.sh`, verifier, launcher test, `TRANSPORT_SOURCE.json` and patch are byte-identical to `e58a1ec`. I didn't compare `docs/zig-engine.md`.

## P1: fix before pushing to the public fork

1. **The focused branch links to a doc that doesn't exist.** `docs/zig-engine.md:43` and `:167` link `tp4-registered-relay.md`, which isn't in `7a73080`. The engine guide `GLM_TP4_REGISTERED_RELAY.md` is also left out of the patch, on the grounds that it is "maintained as recipe docs instead" (`verify_tp4_native_patch.py`, `TRANSPORT_SOURCE.json`). No recipe doc exists, so the published branch loses:
   - the qualification limits
   - the probe invocation and NCCL-reference instructions
   - the attribution paragraph
2. **The attribution changes aren't in either frozen commit.**
   - `CREDITS.md`, `NOTICE` and `README.md` are modified but uncommitted in the research worktree. The new NOTICE text says it covers the "research and selected registered-relay branches", but the focused repo is clean and carries none of it.
   - `e58a1ec` publishes imported Apache-2.0 code: `glm_neighbor_rdma.cu`, `neighbor_rdma.zig` and `neighbor_rdma_seq.zig`. None of these files carries a header saying where it came from or that it was modified. The provenance lives only in `glm-neighbor-source.json`.
   - **Fix:** commit a reviewed NOTICE/CREDITS change to each branch before pushing, and add "adapted from bertholomus/TensorFold b8b115e7, modified" headers to the three imported files.

## P2: correctness and accuracy of claims

3. **"Default parity" holds for which backend gets selected, not for behaviour or the binary.** With the flag unset or 0, upstream code paths now run modified code:
   - **Startup:**
     - Comm.init adds a pre-NCCL startup exchange over the link (`roce_tag`). That changes the link protocol, so a fleet mixing upstream and new binaries is incompatible.
     - Mixed per-rank `TF_GLM_ROCE`, `TF_GLM_L2PF` or `TF_ROCE_RING` now fails hard and collectively. Upstream fell back to NCCL for mixed RoCE, and L2PF was a local setting.
     - TP1 now rejects the retired selectors.
   - **Runtime (Zig):**
     - `Comm.check()` fail-fast now runs in the sampler gather, `takeSnap`, `spillOut`/`spillWrite` and `allGatherInPlace` whenever optional allocated RoCE is active (the TP2/TP3 defaults).
     - `Region.alloc` reorders `device()` and `memset`, and this also applies to Flash Next's RoCE.
     - The proxy's `fwd_posted` counter is now atomic.
   - **Launcher:**
     - It exports `TF_GLM_TP4_REGISTERED_RELAY=0` to every rank container, including the Python engine.
     - `test_upstream_config_matches_pinned_recipe` compares only 12 keys, not the forwarded env.
     - The image tag changes for default users.
   - **Verification gap:** none of these default paths (TP2/TP3 allocated RoCE, TP4 NCCL, Flash Next) was NVCC-built or GPU-run for `c0b1568`.
   - **Wording:** say "selection semantics retained; startup validation and failure checks strengthened" and list the changes.

4. **The focused binary can't simply replace the accepted production binary.**
   - **Recipe incompatibility:** `spark/recipes/glm53-native-tp4.json` sets `TF_GLM_TRANSPORT=roce`, `TF_GLM_ROCE_MEMORY=registered` and `TF_GLM_NEIGHBOR_FIRST`/`_SECOND` on every rank, and `c0b1568` rejects all of them (`RetiredGlmTransportSelector`). Moving production onto it needs a recipe migration to `TF_GLM_TP4_REGISTERED_RELAY=1` with those variables removed.
   - **Does the c8b5f94 evidence carry over? Only partly:**
     - *What matches:* the selected path's handshake, Region and proxy code is the same as `c8b5f94` running with `TRANSPORT=roce` and `MEMORY=registered`.
     - *What changed since:* the build graph (`glm_neighbor_rdma` kernel and `fabric` import removed) and the startup code (rewritten).
     - *Status:* the GPU results from `c8b5f94` are supporting evidence, not qualification of `c0b1568`.

5. **The patched-NCCL dependency is under-disclosed.**
   - Production loads `/fleet-nccl` (a patched NCCL 2.30.7) through `LD_LIBRARY_PATH`. The registered relay still sends gathers over 512 KiB, in-place gathers and broadcasts through NCCL.
   - The focused docs disclose this, but nothing warns at runtime.
   - The research docs (`e58a1ec` `zig-engine.md`: "existing pins and NCCL launcher defaults are unchanged", pip wheel listed) never say the receipts used patched NCCL.

6. **Config validation can block `./stop.sh`.**
   - `stop.sh` (and `prepare.sh`, `publish-image.sh`, `test-*.sh`) source `config.sh`, which `exit 1`s on profile errors.
   - Example: `.env` has `TP4_NETWORK=registered-relay`, but `FABRIC_EXPECT` comes only from the switchless wrapper, or a retired selector is left in `.env`. Then `./stop.sh` exits before stopping anything.
   - **Fix:** run the validation only for start/prepare.

7. **Following the research selection table gives 1024 KiB, not 512.**
   - The "Registered relay" row omits `TF_ROCE_MAX_KB=512`, so `config.sh` raises it to 1024 at PARALLEL>4. Only the example command pins 512, which is the accepted profile.
   - Native neighbor accepts only GID 3, but the launcher only checks that the value is numeric. A GID other than 3 therefore fails after the existing containers have been stopped. That contradicts the claim that errors are caught before the stop.

## P3: what the evidence does and doesn't prove

8. **Collective startup and teardown.** I traced every failure after the exchange: startup rows, `sameMemoryProfile`, `planRoutes`, `setRoutes`/`setRing` (via `verdict`), connect/start (via `verdict`), and `RequiredRoceUnavailable`. Each is decided either from identical gathered rows or by a verdict exchange, so every rank errors together and then runs `errdefer ncclCommDestroy`. `local.?` is safe because every rank's row byte is checked to be 1 first. A registered region is freed only after `tf_roce_close`.
   - **Not proven:** NCCL 2.30.7's destroy-without-finalize behaviour. Also, `gatherToAll` on rank 0 returns `BadRoceRow` without replying, so followers wait for the link timeout (pre-existing).
   - None of the failure paths (mixed profile, missing HCA, required setup failing) has been run on hardware.
9. **Probe gaps.**
   - It passes an empty `Prefetch`, so it never exercises L2 prefetch, which production runs with `TF_GLM_L2PF=1` inside the relay launch.
   - It doesn't exercise side-stream concurrency (`TF_GLM_SIDE=1`) or inject faults.
   - The ring check in `validateReplayCounts` (`>` only) is weak, but `waitCounters` enforces the exact lower bound.
   - On failure, `die()` leaves peers blocked until the external deadline (documented).
10. **Host tests.**
    - They cover `Startup.parse` but not `Startup.read`, so detection of retired selectors from the environment is untested.
    - They don't cover the registered-mode forcing of 512 in `openRoce`.
    - The routing tests check route planning only. One-rail-per-neighbour relay and the dynamic and fixed GID paths have never been run live; production fixed `NCCL_IB_GID_INDEX=3`.
11. **Research neighbour path.**
    - It is bound to a single stream (`NeighborConcurrentStream`, fatal and sticky) and accepts only GID 3. Both are documented.
    - Whether it works with `TF_GLM_SIDE=1` hasn't been established.
12. **Reconstruction.**
    - The six mode differences mean `a628d29` reproduces the source bytes, but not the file modes, of base plus 13 patches. That's documented.
13. **PR200.** Neither committed branch has any absolute-TPS or reproduced-performance claim (I grepped both diffs). Keep it that way in the PR body.

**Proof limits of this review:** reading source and diffs only. I didn't re-run the claimed 39+6, 47+6+7 or 2889-file checks. I didn't inspect the untracked root docs/tools or anything in the research `docs/tp4-rdma-research/` directory, and I didn't examine the GPU-side relay kernel or the imported CUDA code in depth.
