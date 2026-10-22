#!/usr/bin/env python3
"""Supplemental GLM checks; not a reproduction of upstream 17/4/428/56 suites.

Run on the head: python3 glm_accept.py --out /root/NEW_PRIVATE_DIRECTORY
Use --long-tokens 0 to skip the bounded needle; --cache-reuse adds two identical
long requests. No credentials, settings changes, generated-code execution or
service controls. The route and model are deliberately fixed.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

MODEL = "GLM-5.3-Flash-EXL3"
HOST, PORT, PATH = "127.0.0.1", 40053, "/v1/chat/completions"
MAX_BYTES, MAX_LINE = 8 * 1024 * 1024, 65536
NEEDLE = "GLM_ACCEPT_NEEDLE_7D26A94C"
LIMITS = [
    "Supplemental cases, not the unpublished upstream 17/4/428/56 fixtures.",
    "No code is executed and no quality, full-context, sustained-capacity or reboot qualification is established.",
    "Client request intervals can overlap without proving simultaneous backend decoding.",
    "Closing an SSE connection does not prove when the backend retired its work; subsequent requests test usability only.",
    "Cache counts are provider observations; a zero count is not proof of universally cold state.",
    "Needle sizing tokenizes raw text; actual chat-template prompt tokens are retained separately.",
]


class CheckError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def count(value):
    return type(value) is int and value >= 0


class Evidence:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(mode=0o700)  # Exclusive; never reuse an existing run.
        os.chmod(self.path, 0o700)

    def put(self, name, data):
        require(Path(name).name == name, "unsafe evidence filename")
        fd = os.open(str(self.path / name), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())

    def json(self, name, value):
        self.put(name, encoded(value) + b"\n")


class Budget:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds
        self.lock, self.connections = threading.Lock(), {}

    def remaining(self):
        seconds = self.end - time.monotonic()
        require(seconds > 0, "whole-run deadline exceeded")
        return seconds

    def add(self, connection):
        with self.lock:
            self.connections[connection] = None

    def connected(self, connection):
        with self.lock:
            self.connections[connection] = connection.sock

    def remove(self, connection):
        with self.lock:
            self.connections.pop(connection, None)

    def close(self):
        with self.lock:
            connections = list(self.connections.items())
        for connection, active_socket in connections:
            try:
                active_socket = active_socket or connection.sock
                if active_socket is not None:
                    active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass


def events(response, raw, budget):
    """Bounded SSE parser, preserving received bytes including CRLF framing."""
    data = []
    while True:
        budget.remaining()
        line = response.readline(MAX_LINE + 1)
        raw.extend(line)
        require(len(raw) <= MAX_BYTES, "response exceeds byte bound")
        require(len(line) <= MAX_LINE, "SSE line exceeds byte bound")
        if not line:
            require(not data, "unterminated SSE event")
            return
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                value = b"\n".join(data).decode("utf-8")
                data = []
                if value == "[DONE]":
                    yield None
                else:
                    event = json.loads(value)
                    require(isinstance(event, dict), "SSE data must be an object")
                    require(not event.get("error"), "server returned an SSE error")
                    yield event
        elif line.startswith(b"data:"):
            data.append(line[5:].removeprefix(b" "))


def validated(events_seen, body, started, first, ended):
    require(events_seen, "empty generation stream")
    usage = None
    metadata = None
    models = set()
    text, reasoning, finish = [], [], None
    for event in events_seen:
        require(event.get("model") in (None, MODEL), "response model differs from fixed model")
        if event.get("model") is not None:
            models.add(event["model"])
        if event.get("usage") is not None:
            usage = event["usage"]
        if event.get("tensorfold") is not None:
            require(isinstance(event["tensorfold"], dict), "invalid tensorfold metadata")
            metadata = event["tensorfold"]
        choices = event.get("choices", [])
        require(isinstance(choices, list) and len(choices) <= 1, "unexpected generation choices")
        for choice in choices:
            require(isinstance(choice, dict) and choice.get("index", 0) == 0, "unexpected choice index")
            delta = choice.get("delta", {})
            require(isinstance(delta, dict), "invalid generation delta")
            for field, target in (("content", text), ("reasoning_content", reasoning)):
                value = delta.get(field)
                if value is not None:
                    require(isinstance(value, str), "nontext generation delta")
                    target.append(value)
            if choice.get("finish_reason") is not None:
                require(finish in (None, choice["finish_reason"]), "conflicting finish reasons")
                finish = choice["finish_reason"]
    require(isinstance(usage, dict), "server usage missing")
    require(models == {MODEL}, "response model identity missing")
    require(isinstance(metadata, dict), "server tensorfold metadata missing")
    ids = metadata.get("token_ids")
    require(isinstance(ids, list) and bool(ids) and all(count(x) for x in ids), "exact token_ids missing/invalid")
    require(count(usage.get("completion_tokens")) and usage["completion_tokens"] == len(ids),
            "usage completion count differs from exact token_ids")
    require(len(ids) <= body["max_tokens"], "output exceeds bounded token cap")
    require(count(usage.get("prompt_tokens")) and usage["prompt_tokens"] > 0, "prompt usage missing")
    if "total_tokens" in usage:
        require(usage["total_tokens"] == usage["prompt_tokens"] + len(ids), "total usage count mismatch")
    require(type(metadata.get("drafts")) is bool and metadata["drafts"] == body["draft"],
            "server draft metadata does not verify requested mode")
    require(metadata.get("enable_thinking") is False, "server thinking=false metadata not verified")
    require(not "".join(reasoning), "unexpected reasoning with thinking disabled")
    require(finish in ("stop", "length"), "missing/unsupported finish reason")
    require(bool("".join(text)) and first is not None, "no answer content observed")
    details = usage.get("prompt_tokens_details", {})
    require(isinstance(details, dict), "invalid prompt usage details")
    cached = details.get("cached_tokens")
    if cached is not None:
        require(count(cached) and cached <= usage["prompt_tokens"], "invalid cached token count")
    return {"text": "".join(text), "reasoning": "".join(reasoning), "finish_reason": finish,
            "usage": usage, "tensorfold": metadata, "token_ids": ids,
            "token_ids_sha256": sha(encoded(ids)), "draft_verified": True,
            "cached_tokens": cached, "started_monotonic_s": started,
            "ended_monotonic_s": ended, "ttft_s": first - started, "elapsed_s": ended - started}


class Client:
    def __init__(self, evidence, budget):
        self.evidence, self.budget = evidence, budget
        self.rows, self.lock = [], threading.Lock()

    def request(self, label, path, body, cancel=False, stream=True):
        require(path in (PATH, "/tokenize"), "unapproved request route")
        raw, result, response = bytearray(), None, None
        started, first = time.monotonic(), None
        row = {"label": label, "path": path, "request": body,
               "request_sha256": sha(encoded({"path": path, "body": body})),
               "started_monotonic_s": started, "ok": False}
        self.evidence.json(label + "-request.json", row)
        connection = http.client.HTTPConnection(HOST, PORT, timeout=min(180, self.budget.remaining()))
        self.budget.add(connection)
        try:
            # http.client uses this literal address directly: no environment
            # proxy discovery, redirects, authentication or credential reads.
            connection.request("POST", path, body=encoded(body),
                               headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"})
            self.budget.connected(connection)
            response = connection.getresponse()
            row["http_status"] = response.status
            row["content_type"] = response.getheader("Content-Type", "")
            if response.status != 200:
                while len(raw) <= MAX_BYTES:
                    self.budget.remaining()
                    chunk = response.read1(min(65536, MAX_BYTES + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                require(len(raw) <= MAX_BYTES, "error response exceeds byte bound")
                raise CheckError("backend returned a non-200 status")
            if not stream:
                while True:
                    self.budget.remaining()
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    raw.extend(chunk)
                    require(len(raw) <= MAX_BYTES, "JSON response exceeds byte bound")
                result = json.loads(raw)
                require(isinstance(result, dict) and not result.get("error"), "invalid tokenizer result")
            else:
                require(row["content_type"].split(";", 1)[0].strip().lower() == "text/event-stream",
                        "backend did not return SSE")
                seen, done = [], False
                for event in events(response, raw, self.budget):
                    if event is None:
                        done = True
                        break
                    seen.append(event)
                    has_content = any(isinstance(c, dict) and isinstance(c.get("delta"), dict)
                                      and bool(c["delta"].get("content")) for c in event.get("choices", []))
                    if has_content and first is None:
                        first = time.monotonic()
                        if cancel:
                            result = {"closed_after_first_content": True, "complete_stream": False,
                                      "events_before_close": seen, "started_monotonic_s": started,
                                      "ended_monotonic_s": first, "ttft_s": first - started}
                            break
                if cancel:
                    require(result is not None, "cancellation stream completed before any content")
                else:
                    require(done, "generation ended without [DONE]")
                    result = validated(seen, body, started, first, time.monotonic())
                    result["sse_events"] = seen
            row["ok"], row["result"] = True, result
            return row
        except BaseException as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            if response is not None:
                response.close()  # Required: close makefile's socket reference too.
            connection.close()
            self.budget.remove(connection)
            row["settled_monotonic_s"] = time.monotonic()
            row["response_sha256"] = sha(raw)
            self.evidence.put(label + "-response.bin", raw)
            self.evidence.json(label + "-result.json", row)
            with self.lock:
                self.rows.append(row)

    def generate(self, label, prompt, draft, tokens=128, cancel=False):
        require(type(tokens) is int and 1 <= tokens <= 128 and type(draft) is bool,
                "generation exceeds bounded request parameters")
        return self.request(label, PATH, {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0, "top_p": 1, "max_tokens": tokens, "ignore_eos": True,
                            "stream": True, "stream_options": {"include_usage": True},
                            "return_token_ids": True, "draft": draft,
                            "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}, cancel=cancel)

    def tokenize(self, label, prompt):
        row = self.request(label, "/tokenize", {"model": MODEL, "prompt": prompt, "add_special_tokens": False}, stream=False)
        value = row["result"]
        require(count(value.get("count")) and isinstance(value.get("tokens"), list)
                and value["count"] == len(value["tokens"]) and all(count(t) for t in value["tokens"]),
                "tokenizer count/IDs missing or inconsistent")
        return value["count"]


def parity(left, right, ignore_draft=False):
    a, b = dict(left["request"]), dict(right["request"])
    if ignore_draft:
        a.pop("draft", None)
        b.pop("draft", None)
    require(left["path"] == right["path"] and a == b, "comparison requests differ beyond permitted draft field")
    require(left["result"]["token_ids"] == right["result"]["token_ids"],
            "exact output token mismatch: " + left["label"] + " vs " + right["label"])
    require(left["result"]["usage"]["prompt_tokens"] == right["result"]["usage"]["prompt_tokens"],
            "identical request has inconsistent prompt token counts")
    for field in ("text", "reasoning", "finish_reason"):
        require(left["result"][field] == right["result"][field], "output " + field + " mismatch")
    return {"left": left["label"], "right": right["label"], "ignore_draft_only": ignore_draft,
            "comparison_request_sha256": sha(encoded({"path": left["path"], "body": a})),
            "token_ids_sha256": left["result"]["token_ids_sha256"], "exact_match": True,
            "cached_tokens": [left["result"]["cached_tokens"], right["result"]["cached_tokens"]]}


def overlap(client, cases):
    ready = threading.Barrier(len(cases) + 1)
    rows, errors = {}, []
    lock = threading.Lock()

    def worker(name, prompt):
        try:
            ready.wait(timeout=client.budget.remaining())
            row = client.generate("overlap-" + name, prompt, True)
            with lock:
                rows[name] = row
        except BaseException as exc:
            with lock:
                errors.append(type(exc).__name__ + ": " + str(exc))

    threads = [threading.Thread(target=worker, args=case, daemon=True) for case in cases]
    for thread in threads:
        thread.start()
    ready.wait(timeout=client.budget.remaining())
    while any(thread.is_alive() for thread in threads):
        client.budget.remaining()
        for thread in threads:
            thread.join(timeout=min(0.05, client.budget.remaining()))
    require(not errors, "overlapping request failure: " + "; ".join(errors))
    require(len(rows) == len(cases), "overlapping request results missing")
    common = min(r["result"]["ended_monotonic_s"] for r in rows.values()) - max(
        r["result"]["started_monotonic_s"] for r in rows.values())
    require(common > 0, "four client request intervals did not overlap")
    return rows, common


def long_prompt(records):
    lines = ["Read the following inventory. Remember the unique NEEDLE_VALUE and return its exact value when asked.\n"]
    for index in range(records + 1):
        if index == records // 2:
            lines.append("NEEDLE_VALUE=" + NEEDLE + "\n")
        if index < records:
            lines.append(f"Record {index:05d}: the harbor warehouse stores blue bolts, green nuts, silver washers and copper rings.\n")
    lines.append("Question: What is the exact NEEDLE_VALUE? Return that value first, without changing any character.\n")
    return "".join(lines)


def sized_needle(client, target):
    low, high, best = 0, 4096, None
    for attempt in range(14):
        middle = (low + high) // 2
        prompt = long_prompt(middle)
        actual = client.tokenize(f"needle-size-{attempt:02d}", prompt)
        if actual <= target:
            best = (prompt, actual, middle)
            low = middle + 1
        else:
            high = middle - 1
        if low > high:
            break
    require(best is not None and 0 <= target - best[1] <= 128, "could not size deterministic needle within 128 raw-text tokens below target")
    return best


def run(client, args, report):
    cases = [
        ("code", "Write a complete Python thread-safe LRU cache with per-entry TTL. Include invariants and unittest examples. Do not stop early."),
        ("prose", "Write a detailed practical guide to maintaining a bicycle: cleaning, brakes, gears and puncture repair. Use explanatory paragraphs."),
        ("structured", "Write JSON records for fictional inventory. Every object has id, name, quantity and location. Include the literal nonce GLM_STRUCT_62F1 in the first record."),
        ("nonce", "The archive label is GLM_NONCE_B79D26. Begin by copying that exact label, then explain how to verify a file checksum with Python's standard library."),
    ]
    solo = {}
    for name, prompt in cases:
        off = client.generate("solo-" + name + "-serial", prompt, False)
        on = client.generate("solo-" + name + "-draft", prompt, True)
        report["comparisons"].append(parity(off, on, ignore_draft=True))
        solo[name] = on
    concurrent, span = overlap(client, cases)
    report["four_client_intervals_overlap_s"] = span
    for name, _ in cases:
        report["comparisons"].append(parity(solo[name], concurrent[name]))
    client.generate("cancel-essay", "Write a very long essay explaining the history and practice of ocean navigation. Continue in detailed paragraphs.", True, cancel=True)
    short = "Begin with the exact label GLM_AFTER_CANCEL_924F. Then describe a checksum in one short sentence."
    off = client.generate("after-cancel-serial", short, False, tokens=32)
    on = client.generate("after-cancel-draft", short, True, tokens=32)
    report["comparisons"].append(parity(off, on, ignore_draft=True))
    report["cancellation"] = {"closed_after_first_content": True, "following_short_pair_exact": True,
                              "backend_retirement_independently_verified": False}
    if not args.long_tokens:
        report["needle"] = {"status": "skipped", "reason": "explicit --long-tokens 0"}
        return
    prompt, actual, records = sized_needle(client, args.long_tokens)
    report["needle"] = {"status": "running", "target_raw_tokens": args.long_tokens,
                        "actual_raw_tokens": actual, "records": records, "prompt_sha256": sha(prompt.encode()),
                        "required_value": NEEDLE, "tokenization_excludes_chat_template": True}
    off = client.generate("needle-serial", prompt, False)
    on = client.generate("needle-draft", prompt, True)
    require(NEEDLE in off["result"]["text"] and NEEDLE in on["result"]["text"], "needle value not recalled in both modes")
    report["comparisons"].append(parity(off, on, ignore_draft=True))
    report["needle"].update(status="passed", prompt_tokens=[off["result"]["usage"]["prompt_tokens"], on["result"]["usage"]["prompt_tokens"]])
    if args.cache_reuse:
        reuse = [client.generate("needle-reuse-" + str(i), prompt, True) for i in (1, 2)]
        for item in reuse:
            require(item["result"]["cached_tokens"] is not None, "cache reuse requires present cached token count")
            report["comparisons"].append(parity(on, item))
        counts = [item["result"]["cached_tokens"] for item in reuse]
        report["cache_reuse"] = {"cached_tokens": counts, "observed_hit": any(n > 0 for n in counts),
                                  "zero_hits_are_accepted": True, "settings_changed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new private evidence directory")
    parser.add_argument("--seconds", type=int, default=600, help="whole-run deadline, 1..600 seconds")
    parser.add_argument("--long-tokens", type=int, default=32768, help="raw-text needle target, 4096..32768; 0 skips")
    parser.add_argument("--cache-reuse", action="store_true", help="two additional identical long requests; zero cache hits are accepted")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 600:
        parser.error("--seconds must be 1..600")
    if args.long_tokens != 0 and not 4096 <= args.long_tokens <= 32768:
        parser.error("--long-tokens must be 0 or 4096..32768")
    if args.cache_reuse and not args.long_tokens:
        parser.error("--cache-reuse requires the long needle")
    os.umask(0o077)
    evidence = Evidence(args.out)
    budget = Budget(args.seconds)
    client = Client(evidence, budget)
    report = {"schema_version": 1, "kind": "supplemental-glm-acceptance", "ok": False,
              "route": f"http://{HOST}:{PORT}", "model": MODEL, "proof_limits": LIMITS,
              "parameters": {"seconds": args.seconds, "long_tokens": args.long_tokens, "cache_reuse": args.cache_reuse},
              "client_sha256": sha(Path(__file__).read_bytes()), "comparisons": [],
              "started_unix_s": time.time()}
    evidence.json("commission.json", report)

    def expired(_signum, _frame):
        raise CheckError("whole-run deadline/interruption")

    previous_alarm = signal.signal(signal.SIGALRM, expired)
    previous_term = signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, budget.remaining())
    try:
        run(client, args, report)
        budget.remaining()
        report["ok"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        budget.close()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
        report["ended_unix_s"] = time.time()
        report["settled_requests"] = [{"label": row["label"], "ok": row["ok"],
                                        "request_sha256": row["request_sha256"],
                                        "response_sha256": row["response_sha256"]} for row in sorted(client.rows, key=lambda row: row["label"])]
        evidence.json("report.json", report)
    print(json.dumps({"ok": report["ok"], "evidence_dir": str(evidence.path), "error": report.get("error")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
