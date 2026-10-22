Showing 1 of 66 activities (limited to 1)

# Read-only review: GLM TP4 stage-1 RDMA patch and lifecycle controller

**Verdict:** Stage 1 has no P1. It has two P2s, and registered mode must not be used until a byte gate proves it. The lifecycle files have one P1 and several P2s. **Do not use the helper to gate, stop or launch yet.** I ran nothing: no tests, no build, no SSH, no GPU, so there is no evidence of a healthy deployment.

**Hashes checked.**
- Patch: `310eadf1…a1f9b`.
- `glm-tp4-host.py`: `021f2b27…e732`.
- `glm-tp4-trial.py`: `83d5bba7…fc37`.
- `test_glm_tp4_lifecycle.py`: `db035a5b…c747`.
- The tree diff between the baseline and the candidate is exactly the patch: `roce.zig`, `cuda_comm.zig` and the new `roce_region.zig`.

**Disclosure:** I accidentally wrote one scratch file, `/tmp/.x_review_diff_ignore`. It is a diff dump outside both reviewed trees and holds no secrets. I left it in place rather than run another write; it is safe to delete.

## Stage 1 (RDMA memory option)

**The default arithmetic and transport are unchanged.** I confirmed this from the source:
- The kernel `fn_roce.cu`, `roce_proxy.c` and the region layout are untouched.
- In the default ("allocated") mode the memory is allocated, zeroed and sized exactly as before. Only the order of two setup calls changed, which has no effect.
- Clean-up order is still correct: the proxy closes and releases its RDMA registrations before the region is freed. The error paths unwind in the right order too.
- The flashnext family still uses the default mode, so it is unaffected.

**P2-1. Settings that differ between ranks now stop startup instead of falling back to NCCL.**
- The handshake row now carries the "RoCE on" setting (`TF_GLM_ROCE`) and the ring setting (`TF_ROCE_RING`), and `sameMemoryProfile` fails startup if any rank differs.
- Before the patch, one rank with `TF_GLM_ROCE=0` made every rank fall back to NCCL, as the comments at `cuda_comm.zig:18-20` and `:97` describe. That conflicts with the design README line 28: "Keep GLM transport failure/fallback semantics".
- Checking the ring setting is an improvement, but it is still a behaviour change.
- Mixed binaries also behave unevenly. An older binary sends zeros in those bytes and does not check them, so new ranks fail while old ranks wait.
- Fix: either drop the "RoCE on" bit, or accept and document the change and enforce one image hash on all ranks (see L-P2-3).

**P2-2. A bad setting on one rank can leave the other ranks hung.**
- These checks run on each rank separately, after NCCL has started and before the handshake: the new memory variable, the new required variable, and the "RoCE off but registered/required" combination.
- A rank that fails returns an error without sending its row. The other ranks then block in `link.expect`, which has no timeout (`cuda_link.zig:247`).
- They only recover if the failing process exits and its socket closes. Its error clean-up runs `ncclCommDestroy` first; I can't rule out that this blocks.
- Fix: check these settings before NCCL starts, or put the error into the row so every rank fails together after the gather.

**Registered mode needs a byte gate before any use (proof gate, not a code bug).**
- The DeepSeek source this was adapted from says that on GB10 registered pages use the GPU's cached, coherent path, while `cuMemHostAlloc` memory goes through an uncached one (`dsv41/rdma.zig:86-87`).
- The GLM protocol has only been proven on the uncached path. Its sends are ordinary stores followed by a system fence and a doorbell. Its receives are a flag read with acquire, then relaxed system-scope payload reads.
- The memory-model ordering is legal, so correctness then rests on GB10 hardware keeping GPU caches coherent with NIC reads and writes. If it does not, a fresh flag could arrive with a stale payload, which is silent corruption.
- DeepSeek's neighbour RDMA working on these hosts is indirect evidence only, because it uses a different kernel.
- Required gate: a weight-free test with a payload stamped per sequence, running many iterations across both slots on all four ranks with the ring relay. It must check every byte, run before inference, and set `TF_GLM_ROCE_REQUIRED=1`.

**P3 items (stage 1):**
- The log says `backend=roce-relay` whenever `TF_ROCE_RING=1`, even with fewer than four ranks, where the ring setting is ignored.
- That log line shows what was negotiated, not which transport each gather actually used. The proxy's operation and relay counters are never surfaced for GLM; use them as GPU evidence.
- Registered mode leaves the region out of the host-memory accounting (about 5 MiB).
- `HostReg` opens `libcuda` a second time even though the driver table already has both register functions (`abi.zig:164-165`). DeepSeek's reason for doing this does not apply here. The flags are also written as raw `0x01 | 0x02` rather than the named constants.
- On its error path, registered mode unmaps the memory even if unregistering failed, which contradicts the deliberate "leak instead" rule in `free`.
- The third new test never runs the allocation; it only proves that both paths compile.

## Lifecycle helper and controller

What already holds:
- Nothing restores, removes or restarts a model automatically.
- When input ends (EOF), the helper only releases its locks.
- Keys are read only from 0600 files and never printed.
- IDs and images are immutable, restart policy is `no`, and the head is checked to bind loopback 40053.
- The MSI gateways are never written.

**P1. Stale receipts block containment and the repair loop.**
- Every `gate`, `stop-candidate` and `launch` call re-checks the gate receipt, which expires after 900 s and requires both keys to pass discovery and inference. A test confirms this behaviour (`test_stale_gate_blocks_stop`).
- Once the baseline is stopped and the gateway is gated, no truthful inference smoke is possible until the candidate works.
- So about 15 minutes after the receipt is issued (likely during model load), the helper can no longer stop the candidate or relaunch it for repair. Relaunching would need a fabricated receipt, which conflicts with the October 19 policy to keep repairing behind the gate.
- Fix:
  - Require the pre-change receipt only for the first gate, stopping the baseline, and the first launch.
  - Make `stop-candidate` and re-gating need only host identity and the lease.
  - Define a truthful in-trial relaunch gate: SSH and sudo checks, plus both keys returning 503 on the LAN and 401 without a key.

**P2 findings (lifecycle):**

1. **Stopping the candidate depends on a healthy fleet.**
   - The controller's `stop-candidate` first runs `status()` on all four hosts. That full check requires a single exact process command line, no foreign GPU users, the listener check, and gateway HTTP probes.
   - Any anomaly, which is exactly when you need to stop, blocks the stop on every host, even with `host` set.
   - The host-side stop also requires the gateway probe, so an nginx outage blocks the stop.
   - Fix: containment should check identity and ownership only, and be allowed whenever the gateway is not open.

2. **No way to take back a candidate after EOF or a poisoned connection.**
   - A new session with the same operation name fails because the backup directory already exists.
   - Under a new name, the still-running old candidate counts as a foreign GPU user, so the session refuses to start.
   - That forces an unreviewed manual `docker stop`. Fix: add an `adopt` action that verifies the recorded ID, label, image, command line and environment.

3. **Nothing enforces that all ranks start from the same hash.**
   - The controller does not require the same `image_id`, engine binary hash, environment, or command line apart from the rank and head-only flags across the four hosts.
   - Per-host `candidates` overrides are also unchecked, and the weights are checked by path only, with no checkpoint hash (AGENTS requires exact checkpoint evidence).
   - Combined with stage-1 P2-1, mixed ranks fail unevenly.

4. **A failed nginx config test or reload leaves the file and the live config disagreeing.**
   - `proxy_write` does not restore the old bytes if `nginx -t` or the reload fails.
   - In `reopen`, that error escapes before the restore `try` block. The file then says open while live nginx is still gated, and the next reload by anyone exposes the candidate.
   - After that, every gateway check reports "drift", which also blocks containment.
   - Fix: rewrite the expected bytes, re-test and reload on failure, and add a test for it.

5. **The GPU-idle check may prove nothing.** It relies only on `nvidia-smi --query-compute-apps`, with no positive control.
   - Its behaviour on GB10 is unverified. Require the running baseline's PIDs to appear in that output, and add a `fuser`/`lsof` check of `/dev/nvidia*`.
   - Non-container GPU users, such as the systemd services on c21f, are only caught through `nvidia-smi`.

6. **`devices: []` is accepted, allowing a silent transport downgrade.** Without the InfiniBand devices, NCCL quietly falls back to sockets, and GLM RoCE falls back to NCCL unless `TF_GLM_ROCE_REQUIRED=1`. For this ring, require `['/dev/infiniband']` plus log evidence of the transport.

7. **Possible runtime blockers to verify before launch.**
   - `--cap-drop ALL` removes root's override of file permissions, so the weights and the bind-mounted NCCL library must be world-readable; check this.
   - With a read-only root filesystem, a quantization cache could only go in the 4 GiB `/tmp` tmpfs. That uses GB10's shared memory and is lost on every start (the root agent's open question).

**P3 items (lifecycle):**
- `lan_url` is not tied to the host: any of the three IPs is accepted on any rank.
- After creating the container, the helper does not re-check runtime, capabilities, security options, devices, IPC, shared memory, tmpfs, ulimits or GPU device requests.
- Clock skew: a host clock behind the Mac rejects freshly issued receipts.
- Only port 40053 is checked; listeners on 40060–40062 and NCCL's ephemeral ports are not.
- nginx reloads are asynchronous, so probes can fail spuriously (in the safe direction).
- A launcher script as `argv[0]` fails closed.
- A `docker create` that times out can leave an untracked container; this also fails closed, through the name conflict.

## Proof limits

- I reviewed source only. I did not run the claimed 6, 204 or 30 tests; running them would write a Zig cache into the reviewed trees.
- Nothing was compiled for the GPU target and nothing ran on the live hosts.
- How `nvidia-smi` behaves on GB10, the MSI maintenance-mode syntax (401 before 503), file permissions and the firewall all need live, read-only evidence.

**Next step:** once the P1 and P2 fixes land, send updated hashes for a fresh review in this lane.