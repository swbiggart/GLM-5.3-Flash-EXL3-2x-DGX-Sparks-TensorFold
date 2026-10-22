# Source packaging evidence

These receipts describe local source reconstruction, compiler/host and launcher checks. They do not qualify a new GPU binary.

- `research-*` records the original pre-publication-annotation source packaging. In particular its old patch hash and `production_byte_identical=true` describe that historical stage only.
- **`research-reviewfix-*` is the current research source proof**: patch SHA-256 `9e57a723f1377609ef8e3b1d9b965c3e833afb4b8ff095cacd15c88a1f346eae`. It reports `production_byte_identical=false` and `production_matches_after_declared_prefix_removal=true`, with exact before/after/prefix hashes. The only production-file differences are three declared attribution comments; the probe has the separate probe03/04 changes.
- `glm-focused-*` records the focused native/outer cleanup checks. `glm-focused-stop-review-*` records the later stop-path fix, with native source unchanged.
- `glm-publication-*` and `integrator-checks.json` record the integrator's final launcher checks. The first invocation used unsupported macOS Bash3 and failed on the upstream associative arrays; rerunning under the required Bash5 passed. This is recorded rather than silently discarded.
- `independent-review-initial.md` is the initial review and contains findings since addressed. The final review/resolution is recorded separately; do not treat the initial report as current clearance.

Historical real-GPU benchmark/probe files remain unmodified under `../receipts/`, with their own manifest. Build/source checks here are separate from those captures. Producer-local paths and nonsecret review details remain as provenance.
