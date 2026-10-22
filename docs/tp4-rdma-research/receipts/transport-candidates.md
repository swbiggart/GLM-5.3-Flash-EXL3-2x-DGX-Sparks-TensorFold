# GLM transport candidates — unqualified on GPUs

Source reconstruction: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold
`zig-engine` at `a6c7e08ae864239125d45fc15146d26ec74c7f6d`, underlying
TensorFold `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`, recipe patches 0001–0013.
The imported neighbor host protocol, sequence helpers and CUDA kernel are pinned
in `glm-neighbor-source.json` to TensorFold PR30 merge `b8b115e7…`. Host adaptations rename the
sequence-helper import, make unregister-failure cleanup safe, and expose read-only
completed-write/registered-byte diagnostics. The separate GLM adapter supplies bootstrap,
topology, dispatch, scratch, lifecycle and result-boundary checks.

No model arithmetic, HC accumulation, rank order, tensor layout, KV code or
inference defaults change. No live deployment, NVCC build, GPU execution,
performance win or exact-token qualification is claimed by this patch.

## Profiles and startup

All ranks must run the same build and profile. Set `TF_GLM_TRANSPORT` explicitly:

| Profile | Additional environment | Behavior |
|---|---|---|
| `nccl` | `TF_GLM_ROCE=0`, unset memory/required overrides | Existing NCCL all-gathers |
| `roce` | `TF_GLM_ROCE=1`, `TF_ROCE_RING=1`, `TF_GLM_ROCE_MEMORY=allocated` | Existing GLM relay, required on all ranks |
| `roce` | Same, but `TF_GLM_ROCE_MEMORY=registered` | Same relay/protocol, RoCE region alone uses mmap + portable mapped HostRegister |
| `neighbor` | `TF_GLM_NEIGHBOR_GID_INDEX=3`, explicit per-host FIRST/SECOND lists below; memory unset or `registered` | Required two-match gather; documented shape fallback remains NCCL |

Keep existing per-host NCCL HCA/graph/GID3 and physical ring mapping unchanged.
For neighbor, `TF_GLM_NEIGHBOR_FIRST` and `TF_GLM_NEIGHBOR_SECOND` must each name
exactly two distinct local devices, with no overlap between lists. FIRST reaches
rank `rank xor 1`, SECOND reaches `rank xor 3`. This is the existing deployed DS
FIRST/SECOND mapping, not a derivation from the printed host order. Both devices
in each list are used to stripe that match. Missing/incorrect lists or non-3 GID
fail setup collectively. No default rail choice or NCCL-only success is allowed
for an explicitly selected custom profile.

Startup logs actual backend, memory mode and cutoff. Configuration validation
and validity/profile exchange run BEFORE opening any NCCL handle. A malformed
setting is sent as an invalid row, so every rank rejects it together.
This intentionally tightens earlier fallback semantics: mixed enable/ring/memory/
prefetch/required settings fail startup instead of negotiating optional NCCL.
All four ranks MUST use the identical image/binary fingerprint; mixed old/new
binaries are unsupported, including their different startup frame sequence.
Uniform legacy optional profiles still fall back together when hardware setup
fails; explicit custom profiles fail closed. All ranks agree backend;
relay additionally agrees memory/required/ring/legacy-enable and slot size.
Neighbor open/connect/proxy-start each have all-rank success agreement. It uses
fixed qualified-protocol defaults: four slots, stage blocks 16, collect blocks
48, grain 4, PDL/own-early/unroll/fence enabled. Existing `TF_RDMA_*` and historic
DS specialized switches are not consumed by this adapter. Their use would need
a separate explicit, agreed option implementation and qualification.

Neighbor eligibility is positive, at most 512 KiB per rank, size divisible by
16, 16-byte-aligned non-null pointers and fully disjoint send/output/scratch.
The full rank gather is xor1 into dedicated two-rank scratch, then xor3 into
`[rank0,rank1,rank2,rank3]`. Unsupported sizes/layouts select NCCL *before* any
custom enqueue. The eight-byte GLM startup proposal and in-place gathers thus
remain supported. Actual model callsites must provide identical layout classes
on every rank, as existing GLM does; asymmetric pointers/alias classes across
ranks are unsupported and are not silently reconciled.

The imported protocol is ordered by one CUDA execution stream. GLM's logical
serving lanes share the engine stream. Adapter binds to its first stream and
rejects another before enqueue. Concurrent execution across multiple CUDA
streams is unsupported. Graphs must be destroyed and streams finished before
Comm teardown; existing engine ownership already provides this order. No
captured graph survives a backend/profile change.

Any custom enqueue error sets both local abort words; future dispatch and
result-producing sampler synchronization reject the sticky failure. No NCCL
retry occurs after partial enqueue. The copied kernel abort behavior can return
partial bytes, so the added `Comm.check()` after synchronization is essential.
Production peer-loss watchdog still exits the process. Failure propagation and
watchdog bounds require separate fleet tests before acceptance.

## Weight-free production probe

Build on the Linux CUDA image with Zig 0.17 and the normal complete fatbin set:

```
# Add the new optional kernel to a COPY of the normal fatbin directory.
nvcc -std=c++20 -O3 -fatbin -arch=sm_121 zig/kernels/cuda/glm_neighbor_rdma.cu -o /candidate-fatbins/glm_neighbor_rdma.fatbin
zig build glm-transport-probe -Dfatbins=/candidate-fatbins -Dstrip=true
```

Or use the ordinary `-Dnvcc=... -Dsm=121` source build. Relay/NCCL builds may omit
the optional new kernel; selecting neighbor then fails explicitly. This build
also needs normal `/usr/include/infiniband/verbs.h` for the GLM relay C proxy.
Cross-compiling on a host without that header selects the existing proxy stub,
so it is a semantic check only and must not be deployed.

For the initial relay gate use `TF_ROCE_MAX_KB=512`; the probe requires its
32 KiB graph call to be custom and its 512 KiB+16 call to be NCCL. Different
cutoffs outside that interval need a separately adjusted graph case.

One process on each of the four hosts, with the selected profile and pinned
network environment, under an external deadline that terminates all peers on
any failure:

```
TF_GLM_L2PF=0 ./zig-out/bin/tf-glm-transport-probe RANK MASTER_FLEET_IP 40060 64
```

It opens the real GLM Link and Comm, with exact master-interface binding, no
borrowed/externally mocked NCCL communicator, no model, and no Torch. It checks:

- Guards and unmodified source, rank-contiguous bytes versus both a CPU pattern
  and direct NCCL oracle, including signed-zero/NaN payload/subnormal words.
- Changing eager payloads: 8 B, 16 B, 68 B, 4 KiB, 1/6/16/32 FP32 GLM rows,
  512 KiB + 16 B, and 1 MiB per rank.
- Four-byte-offset alignment, in-place alias through general production dispatch,
  and explicit production `allGatherInPlace`.
- One captured sequence of 16 KiB / 512 KiB+16 / 32 KiB gathers, covering mixed
  custom/NCCL transport, 64 changing-input replays and ring-slot reuse.
- Production dispatch backend printed for every eager/layout case; post-sync
  transport failure check before inspecting or accepting bytes.
- Actual production successful-enqueue counts by backend, relay proxy operation/
  stripe counts and both neighbor stages' completed-write sequence counts.
  Enqueue counts include graph capture, not replay; proxy counts include replay.
  Required custom profiles fail the probe if their execution counters stay zero.

Exit 0 and PASS on *all four ranks* is required. Its elapsed time includes CPU
patterns, guard checks and oracle transfers; it is NOT a performance benchmark.
It does not qualify peer-loss handling, nonempty prefetch, concurrent CUDA
streams, model output, sustained serving, or speed. Run separate failure and
matched inference gates. Probe deliberately exits without CUDA teardown on
operation failure to avoid hanging cleanup behind a failed peer; the external
supervisor must stop the remaining ranks.

Host checks: `zig build test-glm-transport-host glm-transport-probe-host-build`.
The host executable has no CUDA blobs and is not a deployable GPU probe. The
full GLM host suite has nine pre-existing macOS SocketFailed failures also seen
on the untouched baseline; report those separately from focused tests.

## Required acceptance sequence

1. Review exact patch and imported-source hashes; real Linux/NVCC build.
2. Weight-free probe in NCCL, allocated relay, registered relay, neighbor order;
   all four ranks, no unintended fallback. Capture logs and image/binary hashes.
3. Kill/abort a peer under a controlled deadline; verify no sampled output can
   be accepted after transport failure and all processes are stopped before reuse.
4. Frozen model/checkpoint/config, deterministic repeated exact-token corpus
   against matched NCCL; graph/eager, prompt/decode boundaries and DFlash2.
5. Matched timing repeats only after correctness. Any numerical/math-path
   changes are separate candidates with their own quality acceptance.

Registration-only stage changes no global HostBuffer behavior. Memory cleanup
never unmaps pages if CUDA unregister fails, including setup failure; it leaks
until process exit in that exceptional case. It does not promise a new address
alignment beyond mmap's system-page guarantee.

Registered allocations are not included in the pre-existing global HostBuffer
usage tracker. `Comm.stats().registered_host_bytes` and probe COUNTERS report
these separately (neighbor owns two ring sets, roughly 24 MiB plus metadata,
and a separate 1 MiB device scratch). No global HostBuffer accounting or
allocation behavior changes. The relay diagnostic stripe counter is now atomic
so reading it while its proxy runs is defined; transport payload/protocol is unchanged.

Negative startup gate: use one invalid memory/backend/required value, one
missing neighbor rail/GID, and one mismatched valid profile on separate controlled
four-process launches. Require collective failure before any NCCL initialization
log; no peer may remain blocked past the supervisor deadline. These fault launches
have not been run by this source-only builder.
