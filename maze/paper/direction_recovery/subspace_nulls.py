"""Conservative best-of-dictionary nulls in the label-predictable activation subspace."""
from pathlib import Path
import json,re
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'input_manifest.json').read_text());lookup={f"g{m['grid']}-s{m['seed']}-{m['condition']}":m for m in manifest}
cache={};out={};rng=np.random.default_rng(51091)
for p in sorted((ROOT/'results').glob('g*/g*-s*-*/*-geometry.json')):
 if int(p.parent.parent.name[1:])>10:continue
 d=json.loads(p.read_text());tag=p.parent.name
 if d['method'] not in ['relu','batchtopk','matryoshka','mp'] or tag not in lookup:continue
 site=d['site'];key=(tag,site)
 if key not in cache:
  q=torch.load(p.parent/f'linear-{site}.pt',weights_only=True,map_location='cpu')['generative'].numpy()
  _,s,v=np.linalg.svd(q,full_matrices=False);basis=v[s>s.max()*1e-5]
  mm=lookup[tag];w=torch.load(Path(mm['source'])/'injector.final.pt',weights_only=True,map_location='cpu')['W_inj'].numpy();g=mm['grid'];f=np.arange(g*g);mask=np.r_[f//g<g-1,f%g<g-1];w=w[mask];w-=w.mean(0);w/=np.linalg.norm(w,axis=1,keepdims=True)
  cache[key]=(basis,w)
 basis,w=cache[key];size=1024-d['training'][-1]['dead'];means=[]
 for _ in range(3):
  D=rng.normal(size=(size,len(basis))).astype(np.float32)@basis;D/=np.linalg.norm(D,axis=1,keepdims=True)
  means.append(float(np.abs(w@D.T).max(1).mean()))
 out['/'.join([tag,site,d['method']])]=dict(nearest_mean=float(np.mean(means)),rank=len(basis),active_atoms=size)
(ROOT/'subspace_nulls.json').write_text(json.dumps(out,indent=2));print('subspace nulls',len(out))
