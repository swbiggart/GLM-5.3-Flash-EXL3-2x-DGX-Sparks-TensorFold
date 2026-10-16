# Dense prompt merge coverage (patch 0008)

Patch `0008-zig-glm-dense-merge-bucket.patch` makes prompt prefill use the shipped
five-chunk `_merge` bucket. It is a candidate awaiting GPU qualification; no native
GPU execution, served parity, performance or recovery result is claimed here.

## Failure and scope

Recipe `f1ee79a6bf5319775079412c727f5f6b0ebb1e19` uses TensorFold
`e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` plus seven Zig patches. Its AOT archive
SHA-256 is `7637723dc62816e82704016e5f9dec799d594d5736f9e5a96220c486423aa301`.
Both ranks of a TP=2 GB10 deployment loaded, passed health and short tool requests,
then failed an uncached prompt with `MissingTritonVariant` for `_merge`, `R: i32 512`.

The Python baseline counted 1,193 prompt tokens. If native tokenization produces
the same count, the native 64-token grid first fills 1,152 tokens, then 41. The old
prefill code asks for `NCH=ceil(1152/512)=3`, even for its first 512-row attention
block. There is no such captured variant. The native request failed before usage
was reported, and the diagnostic does not print constexprs, so that token count
and NCH value are inferred, not observed native measurements.

The shipped H=32, LW=512 dense merge buckets are 1, 2, 4, 5, 6 and 8. Buckets 1,
5 and 6 cover all three R classes (constant 1, runtime divisible by 16, and other
runtime values); 2 and 4 cover aligned R only, and 8 covers unaligned R only.
Thus the gap belongs to this recipe's shipped launch/capture combination, and
does not depend on this deployment's network, vision setting or prompt content.
It is separate from the `_scores_rows` repair in [PR #170](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/170)
and transport repair in [PR #174](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/174).

## Why five chunks

Dense prompt rows stop at position 2,050; position 2,051 starts sparse attention.
Five 512-key chunks therefore cover every key any dense row can see. A prompt
chunk can extend beyond that boundary, but only its dense rows reach this merge.

The Python reference at `tensorfold/families/glm5_next/cuda/latent.py`, reconstructed
from v0.6.0 plus this recipe's Python patches, has these properties:

- `_dense_chunks` does not load queries or cache keys when `chunk * 512 > pos + row`.
  It still writes every partial: PO=0, PM=-infinity, PL=0.
- `_merge` visits chunks in increasing order. For a zero-weight partial, its
  `active` condition is false, giving `a=1`, `b=0`, and leaving the maximum unchanged.
  For finite accumulators the update is the numerical identity `o=o*1+0*0`,
  `l=l*1+0*0`. The active chunks and their order remain unchanged.
- Both launches use the same NCH and R, so their partial-buffer strides agree.
  Prompt scratch already reserves at least five chunks: its minimum is
  `ceil((2048 + 1)/512)=5`. Allocation sizes are unchanged.

This is a source-level argument, not proof of compiled GPU byte parity. Compiler
optimization, zero signs and device floating-point behavior still need the device
gate. Extra empty chunks also write partials, so short-prompt timing needs measurement.

The change extracts the existing dense launch block into a private helper so its
actual launches can be recorded in a host test. It does not change sparse attention,
the reduction kernel, the AOT matcher or archive. MTP's separate dense-origin path
in `cuda_forward.zig` is outside this patch; this is not a claim of complete AOT
coverage or support for other geometries, arenas or contexts.

## Offline checks

Apply patches 0001–0008 in filename order to the pinned TensorFold checkout, using
`git apply --check` and `git apply`, then use the pinned Zig 0.17.0 compiler:

```sh
zig build test-glm-cuda-host
zig build test-cuda-host
```

The added Zig test records the real dense prefill helper and calls the real
`cuda.aot.matches` against the existing captured-signature fixture. The fixture's
14 merge records and seven dense-chunk records match the shipped archive's hashes,
argument specializations and constexprs. The archive has one additional NCH=1,
R=1 merge record; the test does not need it after the change.

The test covers 33 valid prefix/tail chunks: the 1,152+41 case, the 512/1,024/1,536
boundaries, row classes 1/15/16/63/64/512, the dense/sparse seam, mixed chunks whose
old NCH was 6/7/8, and the maximum configured 16,384-row chunk. Scratch is limited
to five chunks in the test. A fully sparse chunk emits no dense launches.

With only the old NCH formula restored, the regression fails on `_merge` at
pos=0, rows=1152, grid={512,32,1}. With the fix it passes. The focused Zig invocation
passes 30 tests (one regression plus imported compile checks). On the macOS host,
the unchanged GLM suite has 192 passes, one skip and nine local-socket failures;
the patched suite has 193 passes, one skip and the same nine failures.

## Device and model gates still required

`tests/gpu/test_dense_merge_bucket_gpu.py` follows the recipe's existing pytest
GPU style. In a qualified Python recipe image, run:

```sh
ENGINE=python scripts/test-gpu.sh -k five_chunks_preserve_dense_attention_bits
```

Its 44 cases compare the reference's original NCH to five, with both head tile
sizes, R=512 and short tails, and old buckets 1–9 and 37. It poisons scratch before
each launch, checks that invisible chunks were initialized to neutral partials,
and requires identical BF16 output bytes. This tests JIT kernels, not shipped cubins.
It has not been run as part of the offline work.

Build the candidate native image with the same pinned inputs and unchanged AOT
archive. The coordinator must qualify the actual cubins and binary on TP=2 GB10:
uncached native-token prompts around 1,024/1,536 and the dense/sparse seam; the
original failing request; greedy drafted/plain token parity; solo/concurrent
parity; tools, health and recovery afterward. Record actual native token counts,
both-rank logs, image/source/archive hashes and before/after timings. Health alone
and the host matcher test do not establish numerical or served correctness.

The recipe's `tools/make-zig-patches.sh` omits the original capture/reference tools
and packages an externally captured archive. This patch needs no recapture, but
the archive cannot be reproduced from this checkout alone. TensorFold declined
the earlier capture-based GLM path in [PR #546](https://github.com/ashhart/TensorFold/pull/546);
this repair therefore belongs to the recipe's `zig-engine` branch.

Credits: MiaAI-Lab's Zig recipe; Ash Hart and TensorFold contributors' Python
attention and native runtime; jschmied's CUDA launch infrastructure. No code from
PR #170 or #174 is reused; their changes remain applied.
