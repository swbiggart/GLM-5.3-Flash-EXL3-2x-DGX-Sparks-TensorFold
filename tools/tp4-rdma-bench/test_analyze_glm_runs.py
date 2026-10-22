import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import analyze_glm_runs as analyzer


def record(prompt="same", ids=None, cached=0, draft=True, thinking=False):
    ids = ids if ids is not None else [11, 22]
    return {"kind": "stream", "path": "/v1/chat/completions",
            "request": {"messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 2, "temperature": 0, "draft": draft,
                        "return_token_ids": True,
                        "chat_template_kwargs": {"enable_thinking": thinking}},
            "result": {"tokens": len(ids), "token_ids": ids, "drafts": draft,
                       "prompt_tokens": 8192, "cached_tokens": cached,
                       "ttft_s": 2.0, "decode_tps": 4.0,
                       "usage": {"completion_tokens": len(ids), "prompt_tokens": 8192,
                                 "prompt_tokens_details": {"cached_tokens": cached}},
                       "sse_events": [{"tensorfold": {"drafts": draft, "token_ids": ids,
                                                       "enable_thinking": thinking}}]}}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_run(self, name, rows, workload="concurrent", worker_errors=None, summary_rows=None):
        path = self.root / name
        path.mkdir(exist_ok=True)
        raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
        (path / (workload + "-raw.jsonl")).write_bytes(raw)
        summary = {"result": summary_rows or [], "worker_errors": worker_errors or [],
                   "raw_sha256": hashlib.sha256(raw).hexdigest()}
        (path / (workload + ".json")).write_text(json.dumps(summary))
        return path

    def compare(self, a, b, ignore_draft=False):
        return analyzer.analyze([("a", a), ("b", b)], ignore_draft)

    def test_request_changes_remain_unmatched_not_parity(self):
        a = self.write_run("a", [record("first")])
        b = self.write_run("b", [record("second")])
        report = self.compare(a, b)
        c = report["comparisons"][0]
        self.assertTrue(report["ok"])
        self.assertEqual(c["coverage_status"], "no_shared_requests")
        self.assertEqual(c["exact_matches"], 0)
        self.assertEqual(c["unmatched_left"]["observations"], 1)
        self.assertEqual(c["unmatched_right"]["request_identities"], 1)

    def test_token_mismatch_fails(self):
        a = self.write_run("a", [record(ids=[1, 2])])
        b = self.write_run("b", [record(ids=[1, 3])])
        report = self.compare(a, b)
        self.assertFalse(report["ok"])
        self.assertEqual(report["comparisons"][0]["coverage_status"], "token_divergence")
        self.assertEqual(len(report["comparisons"][0]["token_mismatches"]), 1)

    def test_repeated_same_request_divergence_detected_within_one_run(self):
        a = self.write_run("a", [record(ids=[1, 2]), record(ids=[1, 3])])
        report = analyzer.analyze([("a", a)])
        self.assertFalse(report["ok"])
        self.assertEqual(len(report["runs"][0]["within_run_divergences"]), 1)

    def test_equal_duplicates_and_order_independence(self):
        a = self.write_run("a", [record("one"), record("two"), record("one")])
        b = self.write_run("b", [record("two"), record("one")])
        report = self.compare(a, b)
        c = report["comparisons"][0]
        self.assertTrue(report["ok"])
        self.assertEqual(c["coverage_status"], "complete_request_coverage")
        self.assertEqual(c["exact_matches"], 2)
        self.assertEqual(len(c["repetition_count_differences"]), 1)
        reverse = self.compare(b, a)["comparisons"][0]
        self.assertEqual(c["matched_identity_sha256"], reverse["matched_identity_sha256"])

    def test_cache_state_separates_requests_even_when_draft_ignored(self):
        a = self.write_run("a", [record(cached=0, draft=False)])
        b = self.write_run("b", [record(cached=4096)])
        c = self.compare(a, b, ignore_draft=True)["comparisons"][0]
        self.assertEqual(c["shared_request_identities"], 0)
        self.assertEqual(c["unmatched_left"]["request_identities"], 1)

    def test_only_explicit_mode_ignores_draft(self):
        a = self.write_run("a", [record(draft=False)])
        b = self.write_run("b", [record()])
        self.assertEqual(self.compare(a, b)["comparisons"][0]["shared_request_identities"], 0)
        self.assertEqual(self.compare(a, b, True)["comparisons"][0]["exact_matches"], 1)

    def test_thinking_endpoint_and_workload_remain_separate(self):
        baseline = self.write_run("base", [record()])
        for kind in ("thinking", "endpoint", "workload", "max_tokens"):
            with self.subTest(kind=kind):
                row = record(draft=False)
                workload = "concurrent"
                if kind == "thinking":
                    row = record(draft=False, thinking=True)
                if kind == "endpoint":
                    row["path"] = "/v1/completions"
                if kind == "workload":
                    workload = "steady"
                if kind == "max_tokens":
                    row["request"]["max_tokens"] = 3
                other = self.write_run(kind, [row], workload)
                self.assertEqual(self.compare(baseline, other, True)["comparisons"][0]["exact_matches"], 0)

    def test_invalid_ids_counts_draft_and_sse_fail(self):
        for field in ("ids", "count", "draft", "sse_ids", "sse_draft", "sse_missing", "sse_error", "cache"):
            with self.subTest(field=field):
                row = record()
                if field == "ids":
                    row["result"].pop("token_ids")
                if field == "count":
                    row["result"]["usage"]["completion_tokens"] = 3
                if field == "draft":
                    row["result"]["drafts"] = False
                if field == "sse_ids":
                    row["result"]["sse_events"] = [{"tensorfold": {"token_ids": [999], "drafts": True}}]
                if field == "sse_draft":
                    row["result"]["sse_events"][0]["tensorfold"]["drafts"] = False
                if field == "sse_missing":
                    row["result"].pop("sse_events")
                if field == "sse_error":
                    row["result"]["sse_events"].append({"error": {"message": "worker crashed"}})
                if field == "cache":
                    row["result"]["cached_tokens"] = 10
                path = self.write_run(field, [row])
                report = analyzer.analyze([(field, path)])
                self.assertFalse(report["ok"])
                self.assertGreater(len(report["runs"][0]["validation_errors"]), 0)
                self.assertEqual(report["runs"][0]["validated_generation_observations"], 0)

    def test_partial_coverage_is_explicit(self):
        a = self.write_run("a", [record(), record("extra"), record("extra")])
        b = self.write_run("b", [record()])
        c = self.compare(a, b)["comparisons"][0]
        self.assertEqual(c["coverage_status"], "partial_request_coverage")
        self.assertEqual(c["exact_matches"], 1)
        self.assertEqual(c["unmatched_left"]["request_identities"], 1)
        self.assertEqual(c["unmatched_left"]["observations"], 2)

    def test_worker_errors_and_hash_tampering_fail(self):
        path = self.write_run("a", [record()], worker_errors=["timeout"])
        self.assertFalse(analyzer.analyze([("a", path)])["ok"])
        self.write_run("a", [record()])
        with (path / "concurrent-raw.jsonl").open("a") as raw:
            raw.write(json.dumps(record()) + "\n")
        report = analyzer.analyze([("a", path)])
        self.assertFalse(report["ok"])
        self.assertIn("raw_sha256", report["runs"][0]["validation_errors"][0]["message"])

    def test_metrics_exclude_cached_prefill_and_use_medians(self):
        rows = [{"streams": 1, "aggregate_tps": 10, "ttft_s": [1]},
                {"streams": 1, "aggregate_tps": 20, "ttft_s": [3]}]
        path = self.write_run("a", [record()], summary_rows=rows)
        self.write_run("a", [record("cold"), record("cached", cached=8192)], "prefill8k")
        self.write_run("a", [record()], "steady", summary_rows=[{"aggregate_tps": 99, "requests": 1}])
        metrics = analyzer.analyze([("a", path)])["runs"][0]["metrics"]
        self.assertEqual(metrics["concurrent"]["1"]["median_aggregate_tps"], 15)
        self.assertEqual(metrics["concurrent"]["1"]["median_per_request_ttft_s"], 2)
        self.assertEqual(metrics["steady"]["median_aggregate_tps"], 99)
        self.assertEqual(metrics["cold_8k"]["validated_requests"], 1)
        self.assertEqual(metrics["cold_8k"]["median_prefill_tps"], 4096)

    def test_cli_json_and_failure_exit_code(self):
        a = self.write_run("a", [record(ids=[1])])
        b = self.write_run("b", [record(ids=[2])])
        output = self.root / "report.json"
        result = subprocess.run([sys.executable, str(Path(analyzer.__file__)),
                                 "--run", f"a={a}", "--run", f"b={b}",
                                 "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(output.read_text())["ok"])


if __name__ == "__main__":
    unittest.main()
