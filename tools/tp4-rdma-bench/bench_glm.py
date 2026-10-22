"""Reuse the retained public DeepSeek workloads against GLM, with exact-token capture."""
import os
from pathlib import Path
import sys
import threading

# Publication adaptation: resolve the retained clients beside this file.
SOURCE = Path(__file__).resolve().parent / 'vendor'
sys.path.insert(0, str(SOURCE))
import benchmark

Original = benchmark.RecordingClient


class GLMClient(Original):
    def __init__(self, out):
        benchmark.upstream.Client.__init__(self, 'http://127.0.0.1:40053', 'GLM-5.3-Flash-EXL3', None)
        self.out, self.lock, self.errors, self.local = out, threading.Lock(), [], threading.local()

    def _req(self, path, body=None, timeout=180):
        return super()._req(path, body, min(timeout, 180))

    def stream(self, body, path):
        body = dict(body, return_token_ids=True, draft=os.environ.get('GLM_BENCH_DRAFT', '1') == '1')
        if 'messages' in body:
            body['chat_template_kwargs'] = {'enable_thinking': False, 'thinking': False}
        return super().stream(body, path)

    def record(self, row):
        if row['kind'] == 'stream':
            result = row['result']
            metadata = next((x['tensorfold'] for x in reversed(result['sse_events']) if x.get('tensorfold')), {})
            ids = metadata.get('token_ids')
            if not isinstance(ids, list) or len(ids) != result['tokens']:
                raise RuntimeError('Exact output token IDs missing')
            result['token_ids'] = ids
            result['drafts'] = metadata.get('drafts')
        return super().record(row)


benchmark.RecordingClient = GLMClient
screen = os.environ.get('GLM_BENCH_SCREEN', '0') == '1'
reps = 2 if screen else 6
benchmark.CASES = {
    'warmup': ('concurrent', dict(set='a', streams='1,4,8', tokens=256, reps=1)),
    'concurrent': ('concurrent', dict(set='a', streams='1,4,8', tokens=256, reps=reps)),
    'steady': ('sustained', dict(set='a', streams='4', tokens=256, seconds=30 if screen else 90)),
    'prefill8k': ('prefill', dict(lengths='8192', reps=2, fixed_words=False)),
}
if __name__ == '__main__':
    benchmark.main()
