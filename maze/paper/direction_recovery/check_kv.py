import sys,json
from pathlib import Path
import torch,numpy as np
from types import SimpleNamespace
from experiment import load_model,greedy
from decode_reference import greedy_kv
S=np.load('/workspace/sets/g6.npz');torch.set_num_threads(2);dev='cuda';results={}
for name in ['g6-s0-fixed','g6-s0-trainable','g6-s0-baseline']:
 m,W=load_model(Path('/workspace/models')/name,dev);p=torch.tensor(S['prompts'][:4],device=dev);c=torch.tensor(S['conn_a'][:4],device=dev);end=torch.tensor(S['end_pos'][:4],device=dev)
 delta=(c.float()*2-1)@W if W is not None else torch.zeros(4,256,device=dev);inj=SimpleNamespace(W_inj=W,scale=1) if W is not None else None
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
  ours=greedy(m,p,end,delta,'injection',torch.zeros_like(delta),12,int(S['path_end_id']),int(S['pad_id']))
  ref=greedy_kv(m,inj,0,p,c,end,12,int(S['path_end_id']),int(S['pad_id']))
 assert torch.equal(ours,ref),name
 results[name]={'sequences':4,'decode_steps':12,'exact_reference_match':True}
Path('/workspace/output/kv_validation.json').write_text(json.dumps(results,indent=2));print(results)
