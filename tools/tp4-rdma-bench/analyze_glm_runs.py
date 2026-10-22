#!/usr/bin/env python3
"""Compare TensorFold benchmark artifacts without third-party dependencies.

Example: python3 analyze_glm_runs.py --run nccl=DIR --run rdma=DIR --output report.json
Only --ignore-draft removes request.draft from matching. All other request
fields, endpoint, workload filename, prompt token count, observed cache state,
and server thinking metadata remain part of request identity. Warmup is included
in token parity when present, but never mixed into measured throughput.
Exit 1 means an artifact/worker/metadata error or actual token divergence; exit 0
can still mean incomplete or absent comparison coverage. Inspect each coverage
status. Repeated observations need not have equal counts for request-level parity.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def median(values):
    clean = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return statistics.median(clean) if clean else None


def is_int(value):
    return type(value) is int and value >= 0


def issue(errors, source, message):
    errors.append({"source": source, "message": message})


def observation(row, workload, source, ignore_draft, errors):
    """Validate a generation record, then return a strict comparison identity."""
    if row.get("kind") != "stream":
        return None
    request, result = row.get("request"), row.get("result")
    before = len(errors)
    if not isinstance(request, dict) or not isinstance(result, dict):
        issue(errors, source, "generation requires request and result objects")
        return None
    if not isinstance(row.get("path"), str) or not row["path"]:
        issue(errors, source, "missing endpoint path")
    ids = result.get("token_ids")
    if not isinstance(ids, list) or not all(is_int(x) for x in ids):
        issue(errors, source, "missing or invalid output token_ids")
        ids = []
    usage = result.get("usage", {})
    if not isinstance(usage, dict):
        issue(errors, source, "invalid usage object")
        usage = {}
    for field, value in [("result.tokens", result.get("tokens")),
                         ("usage.completion_tokens", usage.get("completion_tokens"))]:
        if not is_int(value) or value != len(ids):
            issue(errors, source, f"{field} does not match output token_ids count {len(ids)}")
    requested_draft, actual_draft = request.get("draft"), result.get("drafts")
    if type(requested_draft) is not bool or type(actual_draft) is not bool:
        issue(errors, source, "request draft and result drafts must be explicit booleans")
    elif requested_draft != actual_draft:
        issue(errors, source, "server drafts differs from requested draft")
    prompt_tokens = usage.get("prompt_tokens")
    if not is_int(prompt_tokens) or prompt_tokens != result.get("prompt_tokens"):
        issue(errors, source, "prompt token metadata missing or inconsistent")
    details = usage.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if not is_int(cached) or cached != result.get("cached_tokens"):
        issue(errors, source, "cached token metadata missing or inconsistent")
    elif is_int(prompt_tokens) and cached > prompt_tokens:
        issue(errors, source, "cached token count exceeds prompt token count")
    server_thinking = {}
    events = result.get("sse_events")
    if not isinstance(events, list):
        issue(errors, source, "missing captured SSE metadata for independent verification")
        events = []
    if any(isinstance(event, dict) and event.get("error") for event in events):
        issue(errors, source, "captured SSE contains a server error")
    metadata = [e["tensorfold"] for e in events if isinstance(e, dict)
                and isinstance(e.get("tensorfold"), dict)]
    if not metadata:
        issue(errors, source, "missing server tensorfold metadata")
    else:
        final = metadata[-1]
        if final.get("token_ids") != ids:
            issue(errors, source, "captured server token_ids differs from result token_ids")
        if type(final.get("drafts")) is not bool or final.get("drafts") != actual_draft:
            issue(errors, source, "captured server drafts does not verify result drafts")
        server_thinking = {k: final[k] for k in ("enable_thinking", "reasoning_effort") if k in final}
    if len(errors) != before:
        return None
    normalized = dict(request)
    if ignore_draft:
        normalized.pop("draft", None)
    identity = {"workload": workload, "path": row["path"], "request": normalized,
                "prompt_tokens": prompt_tokens, "cached_tokens": cached,
                "server_thinking": server_thinking}
    return {"identity_sha256": digest(identity), "identity": identity,
            "ids": tuple(ids), "source": source, "draft": actual_draft,
            "ttft_s": result.get("ttft_s"), "decode_tps": result.get("decode_tps")}


def summarize(summaries, observations):
    concurrent = {}
    for count in (1, 4, 8):
        rows = [r for r in summaries.get("concurrent", {}).get("result", [])
                if r.get("streams") == count]
        concurrent[str(count)] = {
            "repetitions": len(rows),
            "median_aggregate_tps": median(r.get("aggregate_tps") for r in rows),
            "median_per_request_ttft_s": median(t for r in rows for t in r.get("ttft_s", [])),
        }
    steady = summaries.get("steady", {}).get("result", [])
    prefill = [o for o in observations if o["identity"]["workload"] == "prefill8k"
               and o["identity"]["prompt_tokens"] == 8192
               and o["identity"]["cached_tokens"] == 0]
    return {"concurrent": concurrent,
            "steady": {"summary_rows": len(steady),
                       "median_aggregate_tps": median(r.get("aggregate_tps") for r in steady),
                       "reported_requests": sum(r.get("requests", 0) for r in steady)},
            "cold_8k": {"validated_requests": len(prefill),
                        "median_ttft_s": median(o["ttft_s"] for o in prefill),
                        "median_prefill_tps": median(8192 / o["ttft_s"] for o in prefill
                            if type(o["ttft_s"]) in (int, float) and o["ttft_s"] > 0)}}


def load_run(name, directory, ignore_draft=False):
    directory = Path(directory).resolve()
    errors, observations, sources, summaries, worker_errors = [], [], [], {}, []
    raw_paths = sorted(directory.glob("*-raw.jsonl"))
    if not raw_paths:
        issue(errors, str(directory), "no *-raw.jsonl artifacts found")
    for raw_path in raw_paths:
        workload = raw_path.name.removesuffix("-raw.jsonl")
        raw_bytes = raw_path.read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        sources.append({"path": str(raw_path), "sha256": raw_sha})
        summary_path = directory / (workload + ".json")
        try:
            summary_bytes = summary_path.read_bytes()
            summary = json.loads(summary_bytes)
            if not isinstance(summary, dict) or not isinstance(summary.get("result"), list):
                raise ValueError("summary requires result array")
            summaries[workload] = summary
            sources.append({"path": str(summary_path),
                            "sha256": hashlib.sha256(summary_bytes).hexdigest()})
            if summary.get("raw_sha256") != raw_sha:
                issue(errors, str(summary_path), "raw_sha256 does not verify the raw artifact")
            if not isinstance(summary.get("worker_errors"), list):
                issue(errors, str(summary_path), "missing worker_errors array")
            elif summary["worker_errors"]:
                worker_errors.append({"source": str(summary_path), "errors": summary["worker_errors"]})
        except (OSError, ValueError, TypeError) as exc:
            issue(errors, str(summary_path), f"invalid/missing summary: {exc}")
        for line_no, line in enumerate(raw_bytes.decode().splitlines(), 1):
            source = f"{raw_path}:{line_no}"
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row must be an object")
                if row.get("error") or row.get("status") in ("error", "failed"):
                    issue(errors, source, "raw worker error: " + canonical(row))
                item = observation(row, workload, source, ignore_draft, errors)
                if item is not None:
                    observations.append(item)
            except (ValueError, TypeError, AttributeError) as exc:
                issue(errors, source, f"invalid raw record: {exc}")
    groups = {}
    for item in observations:
        groups.setdefault(item["identity_sha256"], []).append(item)
    divergences = []
    for key, items in sorted(groups.items()):
        variants = {}
        for item in items:
            variants.setdefault(item["ids"], []).append(item["source"])
        if len(variants) > 1:
            divergences.append({"identity_sha256": key, "workload": items[0]["identity"]["workload"],
                                "observations": len(items),
                                "variants": [{"token_ids_sha256": digest(ids), "count": len(refs),
                                              "sources": sorted(refs)}
                                             for ids, refs in sorted(variants.items())]})
    report = {"name": name, "directory": str(directory), "sources": sources,
              "benchmark_settings": {workload: {k: summary[k] for k in
                    ("cmd", "parameters", "client_sha256") if k in summary}
                    for workload, summary in sorted(summaries.items())},
              "metrics": summarize(summaries, observations), "worker_errors": worker_errors,
              "validation_errors": errors, "within_run_divergences": divergences,
              "validated_generation_observations": len(observations),
              "unique_request_identities": len(groups),
              "draft_verified_observations": {"true": sum(o["draft"] for o in observations),
                                              "false": sum(not o["draft"] for o in observations)},
              "ok": not (errors or worker_errors or divergences)}
    return report, groups


def unmatched(groups, keys):
    return {"request_identities": len(keys), "observations": sum(len(groups[k]) for k in keys),
            "by_workload": {w: {"request_identities": sum(groups[k][0]["identity"]["workload"] == w for k in keys),
                                "observations": sum(len(groups[k]) for k in keys
                                    if groups[k][0]["identity"]["workload"] == w)}
                            for w in sorted({groups[k][0]["identity"]["workload"] for k in keys})},
            "identity_sha256": sorted(keys)}


def compare(left_name, left, right_name, right):
    shared = set(left) & set(right)
    mismatches, matched = [], []
    for key in sorted(shared):
        a = {o["ids"] for o in left[key]}
        b = {o["ids"] for o in right[key]}
        if len(a) == len(b) == 1 and a == b:
            matched.append(key)
        else:
            mismatches.append({"identity_sha256": key, "workload": left[key][0]["identity"]["workload"],
                               "left_token_ids_sha256": sorted(digest(ids) for ids in a),
                               "right_token_ids_sha256": sorted(digest(ids) for ids in b),
                               "left_sources": sorted(o["source"] for o in left[key]),
                               "right_sources": sorted(o["source"] for o in right[key])})
    left_only, right_only = set(left) - shared, set(right) - shared
    status = ("token_divergence" if mismatches else "no_shared_requests" if not shared
              else "partial_request_coverage" if left_only or right_only else "complete_request_coverage")
    return {"left": left_name, "right": right_name, "coverage_status": status,
            "shared_request_identities": len(shared), "exact_matches": len(matched),
            "matched_identity_sha256": matched, "token_mismatches": mismatches,
            "unmatched_left": unmatched(left, left_only),
            "unmatched_right": unmatched(right, right_only),
            "shared_observations_left": sum(len(left[k]) for k in shared),
            "shared_observations_right": sum(len(right[k]) for k in shared),
            "repetition_count_differences": [
                {"identity_sha256": key, "left": len(left[key]), "right": len(right[key])}
                for key in sorted(shared) if len(left[key]) != len(right[key])],
            "ok": not mismatches}


def analyze(runs, ignore_draft=False):
    loaded = [load_run(name, path, ignore_draft) for name, path in runs]
    comparisons = [compare(a[0]["name"], a[1], b[0]["name"], b[1])
                   for a, b in itertools.combinations(loaded, 2)]
    return {"schema_version": 1, "matching_mode": "ignore_request_draft_only" if ignore_draft else "strict",
            "limits": ["Descriptive screening medians; no statistical significance or global optimum claim.",
                       "Token parity covers shared strict request identities only; unmatched requests are unverified.",
                       "Complete request coverage does not require equal repetition counts; these are listed separately.",
                       "Qualification fixture reports are not benchmark raw records and are not compared."],
            "runs": [r for r, _ in loaded], "comparisons": comparisons,
            "ok": all(r["ok"] for r, _ in loaded) and all(c["ok"] for c in comparisons)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIRECTORY")
    parser.add_argument("--output", type=Path, help="JSON destination (default stdout)")
    parser.add_argument("--ignore-draft", action="store_true",
                        help="explicitly compare draft/serial requests; remove ONLY request.draft from identity")
    args = parser.parse_args(argv)
    runs = []
    for spec in args.run:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or any(name == existing for existing, _ in runs):
            parser.error("--run requires unique NAME=DIRECTORY values")
        runs.append((name, path))
    try:
        report = analyze(runs, args.ignore_draft)
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(output)
        else:
            sys.stdout.write(output)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"artifact/report error: {exc}\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
