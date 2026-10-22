#!/usr/bin/env python3
"""Bounded localhost qualification; no service controls or credential arguments.

Run the identical `--suite timings` fixtures on each backend for client timing
comparisons. Full qualification requires reported token parity and draft state.
The separate public-source/nonce, tools and vision gates remain necessary.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import stat
import time
import urllib.error
import urllib.request


BASE = 'http://127.0.0.1/v1'
MODEL = 'GLM-5.3-Flash-EXL3'
KEYS = {name: '/home/spark/.config/spark-api-proxy/' + name + '-api-key'
        for name in ('hermes', 'omp')}
BOUNDARIES = (1023, 1024, 1025, 1535, 1536, 1537, 2050, 2051, 2052)
TIMING_PROMPTS = (64, 128, 256, 512)
MAX_BYTES = 2 * 1024 * 1024


class QualificationError(RuntimeError):
    """Only fixed, credential-safe messages may be used here."""


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_runtime_key(path):
    """The readiness.py checked-descriptor pattern; no symlinks or FIFOs."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd) as runtime:
        metadata = os.fstat(runtime.fileno())
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise QualificationError('Key must be a mode-0600 regular runtime file')
        key = runtime.read(8193).strip()
    if not key or len(key) > 8192 or any(c.isspace() for c in key):
        raise QualificationError('Runtime key empty, oversized or contains whitespace')
    return key


def local_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def safe_error(error):
    result = {'error_type': type(error).__name__}
    if isinstance(error, QualificationError):
        result['reason'] = str(error)
    elif isinstance(error, urllib.error.HTTPError):
        result['http_status'] = error.code
    # Never persist arbitrary exception messages, HTTP bodies or headers.
    return result


def object_value(value):
    if not isinstance(value, dict):
        raise QualificationError('Expected JSON object')
    if 'error' in value:
        raise QualificationError('Server returned an error envelope')
    return value


def positive_int(value):
    return type(value) is int and value > 0


def request_body(prompt, limit, draft=True, stream=True):
    if type(limit) is not int or not 1 <= limit <= 512:
        raise QualificationError('Output limit must be between 1 and 512')
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'top_p': 1.0, 'max_tokens': limit,
            'draft': draft, 'return_token_ids': True, 'stream': stream,
            'stream_options': {'include_usage': True},
            'chat_template_kwargs': {'thinking': False, 'enable_thinking': False},
            'thinking': {'type': 'disabled'}}


def validate_tokenizer(value):
    value = object_value(value)
    count, tokens = value.get('count'), value.get('tokens')
    if not positive_int(count) or not isinstance(tokens, list) or len(tokens) != count:
        raise QualificationError('Tokenizer count and tokens disagree')
    if any(type(t) is not int or t < 0 for t in tokens):
        raise QualificationError('Invalid tokenizer token IDs')
    return {'count': count}


def parse_stream(response, body, start, clock=time.monotonic):
    """Validate SSE completion, usage, visible output and exact output evidence."""
    text, reasoning, models = [], [], set()
    usage = metadata = None
    first = finish = None
    done = False
    consumed = 0
    for raw in iter(lambda: response.readline(MAX_BYTES + 1), b''):
        consumed += len(raw)
        if consumed > MAX_BYTES:
            raise QualificationError('Stream exceeds response byte limit')
        line = raw.strip()
        if not line or line.startswith(b':') or line.startswith(b'event:'):
            continue
        if not line.startswith(b'data:'):
            raise QualificationError('Unexpected SSE line')
        payload = line[5:].strip()
        if payload == b'[DONE]':
            done = True
            break
        item = object_value(json.loads(payload))
        if item.get('model') is not None:
            models.add(item['model'])
        if item.get('usage') is not None:
            usage = object_value(item['usage'])
        if item.get('tensorfold') is not None:
            metadata = object_value(item['tensorfold'])
        choices = item.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise QualificationError('Expected at most one streaming choice')
        for choice in choices:
            choice = object_value(choice)
            if choice.get('index', 0) != 0:
                raise QualificationError('Unexpected choice index')
            delta = object_value(choice.get('delta', {}))
            if delta.get('tool_calls'):
                raise QualificationError('Unexpected tool call in text fixture')
            for field, destination in (('content', text), ('reasoning_content', reasoning),
                                       ('reasoning', reasoning)):
                content = delta.get(field)
                if content is not None and not isinstance(content, str):
                    raise QualificationError('Invalid stream text field')
                if content:
                    if first is None:
                        first = clock()
                    destination.append(content)
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
    end = clock()
    if not done or finish not in ('stop', 'length') or models != {MODEL}:
        raise QualificationError('Incomplete stream, finish reason or model mismatch')
    if not usage or any(not positive_int(usage.get(k))
                        for k in ('prompt_tokens', 'completion_tokens')):
        raise QualificationError('Missing positive prompt/completion usage')
    if usage['completion_tokens'] > body['max_tokens']:
        raise QualificationError('Server exceeded requested output limit')
    details = object_value(usage.get('completion_tokens_details', {}))
    if reasoning or details.get('reasoning_tokens', 0) != 0:
        raise QualificationError('Thinking was not disabled')
    visible = ''.join(text)
    if not visible.strip() or first is None or end <= first or first < start:
        raise QualificationError('Missing visible text or invalid client timings')
    metadata = metadata or {}
    if 'enable_thinking' in metadata and metadata['enable_thinking'] is not False:
        raise QualificationError('Reported thinking state was not disabled')
    if 'drafts' in metadata and type(metadata['drafts']) is not bool:
        raise QualificationError('Invalid reported draft state')
    ids = metadata.get('token_ids')
    if ids is not None and (not isinstance(ids, list) or
                           len(ids) != usage['completion_tokens'] or
                           any(type(t) is not int or t < 0 for t in ids)):
        raise QualificationError('Returned token IDs disagree with completion usage')
    token_sha = metadata.get('token_sha')
    if token_sha is not None and (not isinstance(token_sha, str) or
                                 not re.fullmatch(r'[a-fA-F0-9]{12,64}', token_sha)):
        raise QualificationError('Invalid reported token hash')
    # Preserve only source-defined scalar/list metadata, never arbitrary fields.
    engine = {k: metadata[k] for k in ('drafts', 'enable_thinking', 'token_ids', 'token_sha')
              if k in metadata}
    for field in ('prefill_widths', 'prefill_raised'):
        value = metadata.get(field)
        if value is not None:
            expected_type = int if field == 'prefill_widths' else bool
            if not isinstance(value, list) or len(value) > 4096 or any(type(v) is not expected_type for v in value):
                raise QualificationError('Invalid reported prefill metadata')
            engine[field] = value
    count = usage['completion_tokens']
    safe_usage = {k: usage[k] for k in ('prompt_tokens', 'completion_tokens')}
    prompt_details = object_value(usage.get('prompt_tokens_details', {}))
    if 'cached_tokens' in prompt_details:
        cached = prompt_details['cached_tokens']
        if type(cached) is not int or not 0 <= cached <= usage['prompt_tokens']:
            raise QualificationError('Invalid cached prompt usage')
        safe_usage['prompt_tokens_details'] = {'cached_tokens': cached}
    return {'text': visible, 'text_sha256': digest(visible), 'usage': safe_usage,
            'tensorfold': engine, 'finish_reason': finish,
            'client_ttft_s': first - start, 'client_elapsed_s': end - start,
            'client_decode_tok_s': (count - 1) / (end - first),
            'client_end_to_end_tok_s': count / (end - start),
            'start_monotonic_s': start, 'end_monotonic_s': end,
            'draft_enabled_verified': metadata.get('drafts') is True}


def http_worker(connection, job, timeout):
    """Killed by the coordinator on absolute deadlines, including trickle reads."""
    try:
        key = read_runtime_key(KEYS[job['key']])
        body = job.get('body')
        request = urllib.request.Request(
            BASE + job['path'], data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        start = time.monotonic()
        with local_opener().open(request, timeout=timeout) as response:
            if job['kind'] == 'completion':
                result = parse_stream(response, body, start)
            else:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise QualificationError('JSON response exceeds byte limit')
                value = object_value(json.loads(raw))
                if job['kind'] == 'tokenizer':
                    result = validate_tokenizer(value)
                else:
                    data = value.get('data')
                    if not isinstance(data, list) or sum(isinstance(m, dict) and
                            m.get('id') == MODEL for m in data) != 1:
                        raise QualificationError('Model discovery mismatch')
                    result = {'model': MODEL}
        connection.send({'status': 'passed', 'started_at': job['started_at'],
                         'finished_at': utc(), **result})
    except Exception as error:
        connection.send({'status': 'failed', 'finished_at': utc(), **safe_error(error)})
    finally:
        connection.close()


def run_jobs(jobs, deadline, request_seconds, context=None):
    """At most eight HTTP workers; return immediately on any failure and reap all."""
    if not 1 <= len(jobs) <= 8 or not 0 < request_seconds <= 120:
        raise QualificationError('Invalid concurrency or request deadline')
    context = context or multiprocessing.get_context('spawn')
    workers = []
    results = [None] * len(jobs)
    try:
        for job in jobs:
            if time.monotonic() >= deadline:
                raise QualificationError('Overall deadline exceeded')
            receive, send = context.Pipe(duplex=False)
            job = {**job, 'started_at': utc()}
            process = context.Process(target=http_worker, args=(send, job, request_seconds))
            process.start()
            send.close()
            workers.append((process, receive, min(deadline, time.monotonic() + request_seconds)))
        pending = set(range(len(workers)))
        while pending:
            for index in list(pending):
                process, receive, expires = workers[index]
                now = time.monotonic()
                if now >= expires:
                    results[index] = {'status': 'failed', 'error_type': 'DeadlineExceeded',
                                      'reason': 'Absolute request or overall deadline exceeded'}
                elif receive.poll():
                    try:
                        results[index] = receive.recv()
                    except EOFError:
                        results[index] = {'status': 'failed', 'error_type': 'WorkerExited'}
                elif not process.is_alive():
                    results[index] = {'status': 'failed', 'error_type': 'WorkerExited'}
                else:
                    continue
                pending.remove(index)
                if results[index]['status'] != 'passed':
                    return [result or {'status': 'failed', 'error_type': 'CancelledAfterFailure'}
                            for result in results]
            if pending:
                time.sleep(min(0.01, max(0, deadline - time.monotonic())))
        return results
    finally:
        for process, receive, _ in workers:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
            receive.close()


def compare_outputs(serial, drafted):
    for field in ('prompt_tokens', 'completion_tokens'):
        if serial['usage'][field] != drafted['usage'][field]:
            raise QualificationError('Greedy token counts diverged')
    if serial['text'] != drafted['text'] or serial['finish_reason'] != drafted['finish_reason']:
        raise QualificationError('Greedy output text or finish reason diverged')
    a, b = serial['tensorfold'], drafted['tensorfold']
    if 'token_ids' in a and 'token_ids' in b:
        if a['token_ids'] != b['token_ids']:
            raise QualificationError('Greedy output token IDs diverged')
        return 'returned-token-ids'
    if a.get('token_sha') and b.get('token_sha'):
        if a['token_sha'] != b['token_sha']:
            raise QualificationError('Greedy output token hashes diverged')
        return 'returned-token-hashes'
    raise QualificationError('Exact output token parity is unverified')


def compare_parity(serial, drafted):
    if serial['tensorfold'].get('drafts') is not False or drafted['tensorfold'].get('drafts') is not True:
        raise QualificationError('Draft on/off state unverified or mismatched')
    return compare_outputs(serial, drafted)


def overlap(rows):
    intersection = min(r['end_monotonic_s'] for r in rows) - max(
        r['start_monotonic_s'] for r in rows)
    if len(rows) != 8 or intersection <= 0:
        raise QualificationError('Eight client request intervals did not overlap')
    return intersection


def boundary_prompt(repeats, nonce='glm-native-qualification-20261016'):
    if type(repeats) is not int or not 0 <= repeats <= 4096:
        raise QualificationError('Boundary fixture exceeds bounded input size')
    return ('Fixture ' + nonce + '. Read the padding, then write integers 1 through 60 separated by spaces.\n'
            'Padding:' + ' x' * repeats + '\nIntegers:')


def timing_prompt(repeats, nonce, target):
    if type(repeats) is not int or not 0 <= repeats <= 4096 or target not in TIMING_PROMPTS:
        raise QualificationError('Timing fixture exceeds bounded input size')
    # A distinct nonce at the front prevents earlier timing/boundary fixtures
    # from supplying a long cached prefix. Reuse the nonce across backend runs.
    return (str(target) + '_' + nonce + '_timing\nPadding:' + ' x' * repeats +
            '\nWrite integers 1 through 60 separated by spaces.')


def timing_cache_evidence(row):
    cached = row['usage'].get('prompt_tokens_details', {}).get('cached_tokens')
    if cached is None:
        return 'uncached_unverified'
    if cached != 0:
        raise QualificationError('Timing request reported cached prompt tokens')
    return 'reported_zero_cached_tokens'


class Journal:
    def __init__(self, path, backend):
        self.file = Path(path).open('x', encoding='utf-8')
        self.backend = backend
        self.request_counts = {'models': 0, 'tokenizer': 0, 'completion': 0}

    def write(self, row):
        if row.get('kind') in self.request_counts:
            self.request_counts[row['kind']] += 1
        self.file.write(json.dumps({'timestamp': utc(), 'backend': self.backend, **row}) + '\n')
        self.file.flush()


def qualify(journal, suite, total_seconds, request_seconds, nonce):
    # Reserve five seconds for bounded worker termination/reaping and the journal.
    deadline = time.monotonic() + total_seconds - 5
    request_counts = {'models': 0, 'tokenizer': 0, 'completion': 0}

    def jobs(specs, allow_unsupported=False):
        for spec in specs:
            request_counts[spec['kind']] += 1
        rows = run_jobs(specs, deadline, request_seconds)
        unsupported = (allow_unsupported and len(rows) == 1 and
                       rows[0].get('http_status') in (404, 405))
        for spec, row in zip(specs, rows):
            public = {k: v for k, v in spec.items() if k != 'body'}
            if 'body' in spec:
                public['prompt_sha256'] = digest(spec['body']['messages'][0]['content'])
                public['requested_max_tokens'] = spec['body'].get('max_tokens')
                public['requested_draft'] = spec['body'].get('draft')
            journal.write({**public, **row, **({'status': 'unsupported',
                           'fallback': 'Inference usage will measure every exact boundary'} if unsupported else {})})
        failed = [r for r in rows if r['status'] != 'passed']
        if failed:
            if unsupported:
                return None
            raise QualificationError('HTTP workload failed; inspect preceding safe diagnostic rows')
        return rows

    def completion(case, prompt, limit=32, draft=True, key='hermes'):
        return {'case': case, 'key': key, 'kind': 'completion', 'path': '/chat/completions',
                'body': request_body(prompt, limit, draft)}

    # Fail before requests if either existing runtime credential is unavailable.
    for path in KEYS.values():
        read_runtime_key(path)
    for name in KEYS:
        jobs([{'case': 'discovery', 'key': name, 'kind': 'models', 'path': '/models'}])
        jobs([completion('both-key-inference', 'Write the integers 1 through 12 separated by spaces.', key=name)])

    def token_count(prompt, optional=False):
        body = request_body(prompt, 32)
        result = jobs([{'case': 'fixture-tokenizer', 'key': 'hermes', 'kind': 'tokenizer',
                        'path': '/tokenize', 'body': body}], allow_unsupported=optional)
        return result[0]['count'] if result else None

    base_count = token_count(boundary_prompt(0, nonce), optional=True)
    if suite == 'qualification':
        for target in BOUNDARIES:
            repeats = max(0, target - (base_count or 40))
            measured = None
            for attempt in range(4):
                prompt = boundary_prompt(repeats, nonce)
                measured = token_count(prompt) if base_count is not None else None
                if measured is not None and measured != target:
                    repeats += target - measured
                    continue
                serial = jobs([completion('boundary-%d-serial-%d' % (target, attempt), prompt, draft=False)])[0]
                actual = serial['usage']['prompt_tokens']
                if measured is not None and actual != measured:
                    raise QualificationError('Tokenizer and inference prompt counts disagree')
                if actual != target:
                    repeats += target - actual
                    continue
                drafted = jobs([completion('boundary-%d-draft' % target, prompt)])[0]
                proof = compare_parity(serial, drafted)
                journal.write({'case': 'boundary-parity', 'status': 'passed',
                               'actual_prompt_tokens': actual, 'target_prompt_tokens': target,
                               'prompt_sha256': digest(prompt), 'token_parity_evidence': proof,
                               'serial_cached_tokens': serial['usage'].get('prompt_tokens_details', {}).get('cached_tokens'),
                               'scope': 'rendered prompt boundary; effective prefill widths only if reported'})
                break
            else:
                raise QualificationError('Unable to demonstrate exact boundary prompt count')

        prompts = ['Request %d. Write integers 1 through 60 separated by spaces.' % i for i in range(8)]
        solo = [jobs([completion('solo-%d' % i, prompt)])[0] for i, prompt in enumerate(prompts)]
        concurrent = jobs([completion('overlap-%d' % i, prompt) for i, prompt in enumerate(prompts)])
        seconds = overlap(concurrent)
        for a, b in zip(solo, concurrent):
            if any(r['tensorfold'].get('drafts') is not True for r in (a, b)):
                raise QualificationError('Solo or overlapping draft enabled state unverified')
            compare_outputs(a, b)
        journal.write({'case': 'concurrency', 'status': 'passed', 'requests': 8,
                       'client_interval_intersection_s': seconds,
                       'scope': 'overlapping client HTTP intervals; backend execution overlap unverified'})

    for target in TIMING_PROMPTS:
        timing_base = token_count(timing_prompt(0, nonce, target)) if base_count is not None else None
        repeats = max(0, target - (timing_base or 40))
        for attempt in range(4):
            prompt = timing_prompt(repeats, nonce, target)
            measured = token_count(prompt) if base_count is not None else None
            if measured is not None and measured != target:
                repeats += target - measured
                continue
            row = jobs([completion('short-prompt-timing-%d-attempt-%d' % (target, attempt), prompt, 64)])[0]
            cache_evidence = timing_cache_evidence(row)
            actual = row['usage']['prompt_tokens']
            if measured is not None and actual != measured:
                raise QualificationError('Tokenizer and timing prompt counts disagree')
            if actual == target:
                journal.write({'case': 'short-prompt-timing', 'status': 'passed',
                               'target_prompt_tokens': target, 'actual_prompt_tokens': actual,
                               'cache_evidence': cache_evidence,
                               'prompt_sha256': digest(prompt), 'requested_max_tokens': 64,
                               **{k: v for k, v in row.items() if k.startswith('client_')},
                               'usage': row['usage']})
                break
            repeats += target - actual
        else:
            raise QualificationError('Unable to demonstrate exact short timing prompt count')
    journal.write({'case': 'summary', 'status': 'passed', 'suite': suite,
                   'attempted_http_jobs': request_counts,
                   'request_count_scope': 'client attempts; compare server counters separately for isolation',
                   'scope': 'client timings; no cached-prefill rate or service identity claim',
                   'limits': ['Public-source/nonce, tools and vision are separate gates',
                              'MTP parallel=1 without external drafter is a separate gate',
                              'No full-context or unattended-recovery proof']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New JSONL file; existing files are refused')
    parser.add_argument('--backend-label', required=True, help='Operator label, not verified backend identity')
    parser.add_argument('--suite', choices=('qualification', 'timings'), default='qualification')
    parser.add_argument('--fixture-nonce', default='glm-native-qualification-20261016',
                        help='Reuse the identical nonce for matched backend runs')
    parser.add_argument('--total-seconds', type=float, default=600)
    parser.add_argument('--request-seconds', type=float, default=90)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.fixture_nonce):
        parser.error('Fixture nonce must be 1-64 letters, numbers, underscores or hyphens')
    if not 5 < args.total_seconds <= 600 or not 0 < args.request_seconds <= 120:
        parser.error('Total deadline must be >5s and <=600s; request deadline must be <=120s')
    journal = Journal(args.output, args.backend_label)
    try:
        journal.write({'case': 'start', 'route': BASE, 'model': MODEL, 'suite': args.suite,
                       'total_seconds': args.total_seconds, 'request_seconds': args.request_seconds,
                       'measurement': 'client monotonic clock; fixed replayable prompts',
                       'fixture_nonce': args.fixture_nonce,
                       'backend_label_evidence': 'operator supplied; independently verify deployment pins'})
        qualify(journal, args.suite, args.total_seconds, args.request_seconds, args.fixture_nonce)
    except Exception as error:
        journal.write({'case': 'summary', 'status': 'failed',
                       'recorded_http_jobs': journal.request_counts, **safe_error(error)})
        print('Qualification failed; safe diagnostics saved in the new output file.', flush=True)
        return 1
    finally:
        journal.file.close()
    print('Qualification passed; evidence saved in the new output file.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
