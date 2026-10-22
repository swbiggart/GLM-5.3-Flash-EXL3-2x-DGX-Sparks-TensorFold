Cleared to push both branches to the swbiggart fork: no P1s or P2s remain. This review was read-only like the earlier ones; I didn't run the verifier or tests.

## The two open P2s are fixed

**1. Research PR body wording.** `/tmp/glm-research-pr-body.md` (06:23) now says no model weights, serving binaries, credential values, gateway configurations or deployment controllers are published. It also discloses the nonsecret provenance that is published: key labels and file paths, local producer paths, hardware addresses and hostnames, and the source-review notes on lifecycle and gateway design. That matches what my grep found in the receipts. The same disclosure now appears consistently in:
- the research README (evidence index)
- `receipts-manifest.json`, where only `excluded_categories` and `preservation_note` changed; the receipt hashes, file count (2,423) and total bytes are unchanged
- `tools/tp4-rdma-bench/README.md`

**2. Focused PR draft (`71660a4`).** It now reads: "The installer refuses switched TP4; the native opt-in requires logical ring-relay mode (`TF_ROCE_RING=1`) but does not itself detect physical wiring". That matches what `Startup.parse` and `planRoutes(ring=true)` actually do. This one-line doc change is the only difference from `31ce9e7`.

## The rest of the wording delta checks out
- **No code changed.** The diff `744d1b3..5947fb5` has zero lines under zig, scripts, `start.sh`, tests or `tools/verify-zig-transport.py`.
- **Hashes match.** `SOURCE.json` now records the tools README hash as `df351262…`, which matches the committed file. All 45 hashes in `packaging-checks/manifest.json` match their committed files.
- **The earlier P3s are resolved:**
  - `packaging-checks/README.md` marks the old `research-*` receipts (patch `c10e9b29`, `production_byte_identical=true`) as historical and `research-reviewfix-*` as the current proof (`9e57a723…`). It labels `independent-review-initial.md` as superseded, not current clearance.
  - The README now says "ring-relay" instead of "neighbor-relay", and the initial table's A–D became Candidates 1–4. The PR body uses the same Candidate naming.
- **The failed Bash 3 integrator run is recorded.** Bash 4+ is an existing prerequisite, and the 8+8 launcher tests passed under Bash 5. I'm relying on that receipt rather than having re-run it.

## Remaining limits
- No NVCC or GPU run exists for either the annotated research source or the focused source. The docs state this.
- The PR200 hardware description and the "Grill as used in PR #200" attribution still need an exact citation. I couldn't check them offline; that's a P3, not a blocker.
- I sampled the 75 MB of receipts by pattern grep, not exhaustively.

This clears only the fork push and the research PR within the fork. Opening any MiaAI PR still needs the user's explicit approval.
