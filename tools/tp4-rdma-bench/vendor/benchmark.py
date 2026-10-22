"""Run the pinned upstream measurements and retain raw public requests/responses."""
import argparse,hashlib,json,pathlib,threading,time,types
import kit_bench as upstream

CASES={
 'decode':('decode',dict(set='b',tokens=384,reps=3)),
 'concurrent':('concurrent',dict(set='a',streams='2,4',tokens=256,reps=6)),
 'sustained':('sustained',dict(set='b',streams='4',tokens=256,seconds=90)),
 'prefill8k':('prefill',dict(lengths='8192',reps=3,fixed_words=False)),
 'prefill128k':('prefill',dict(lengths='131072',reps=2,fixed_words=False)),
 'depth128k':('depth',dict(lengths='131072',tokens=256)),
}

class ResponseTape:
 def __init__(self,response):self.response=response;self.lines=[]
 def __enter__(self):self.response.__enter__();return self
 def __exit__(self,*args):return self.response.__exit__(*args)
 def read(self,*args):return self.response.read(*args)
 def __iter__(self):
  for raw in self.response:self.lines.append(raw);yield raw

class RecordingClient(upstream.Client):
 def __init__(self, out):
  super().__init__('http://127.0.0.1:40053','deepseek-v4.1-flash',None)
  self.out=out;self.lock=threading.Lock();self.errors=[];self.local=threading.local()
 def _req(self,path,body=None,timeout=3600):
  response=ResponseTape(super()._req(path,body,timeout));self.local.response=response;return response
 def record(self,row):
  with self.lock:
   with self.out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 def stream(self,body,path):
  try:
   result=super().stream(body,path)
   events=[]
   for line in self.local.response.lines:
    raw=line.decode().strip()
    if raw.startswith('data:') and raw!='data: [DONE]':events.append(json.loads(raw[5:]))
   usage=next((x['usage'] for x in reversed(events) if x.get('usage')),None)
   result['usage']=usage
   result['cached_tokens']=(usage or {}).get('prompt_tokens_details',{}).get('cached_tokens')
   result['sse_events']=events
   if result['tokens'] is None or result['tokens']<=0 or not result['text']:
    raise RuntimeError('Missing usage, generated tokens, or content')
   self.record({'kind':'stream','path':path,'request':body,'result':result})
   return result
  except Exception as e:
   with self.lock:self.errors.append(type(e).__name__+': '+str(e))
   raise
 def post(self,path,body,timeout=3600):
  result=super().post(path,body,timeout)
  self.record({'kind':'post','path':path,'request':body,'result':result})
  return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=[*CASES,'all'],default='all');p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 for name in CASES if a.stage=='all' else [a.stage]:
  target=a.out/(name+'.json');raw=a.out/(name+'-raw.jsonl')
  if target.exists() or raw.exists():raise RuntimeError('Use a new output folder; never overwrite a trial')
  c=RecordingClient(raw);cmd,kwargs=CASES[name];started=time.time()
  result=getattr(upstream,'cmd_'+cmd)(c,types.SimpleNamespace(**kwargs))
  if c.errors:raise RuntimeError('Worker failed: '+str(c.errors))
  if cmd=='prefill':
   recorded=[json.loads(line) for line in raw.read_text().splitlines()]
   if not all(x['result']['cached_tokens']==0 for x in recorded if x['kind']=='stream'):raise RuntimeError('Cold prefill cache count is missing or nonzero')
  if cmd=='decode' and not all(r['identical_texts'] for r in result):raise RuntimeError('Repeated greedy output differs')
  receipt={'cmd':cmd,'parameters':kwargs,'started':started,'ended':time.time(),'result':result,'worker_errors':c.errors,
           'client_sha256':hashlib.sha256(pathlib.Path(upstream.__file__).read_bytes()).hexdigest(),
           'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
  target.write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps({'stage':name,'complete':True,'out':str(target)}),flush=True)
if __name__=='__main__':main()
