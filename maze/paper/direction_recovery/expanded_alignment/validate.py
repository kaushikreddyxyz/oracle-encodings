"""Validate archived fits and reconstruct every exported cosine independently."""
import argparse,csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from methods import SAE,unit
METHODS={'relu','batchtopk','matryoshka','mp','difference_means','ridge','logistic','generative_ols'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 torch.set_num_threads(4);rows=[];checks=[];sources=[];all_keys=set()
 manifest=json.loads((ROOT/'manifest.json').read_text())
 for n in a.grids:
  root=ROOT/'results'/f'g{n}';backup=json.loads((root/'BACKUP_VERIFIED.json').read_text());assert sha(root/'results.tar.gz')==backup['sha256']
  data=np.load(ROOT/f'g{n}.npz');C=data['conn'];mask=data['mask'];edges=np.flatnonzero(mask)
  assert C.shape[0]==20000 and len(edges)==2*n*(n-1)
  train={np.packbits(c).tobytes() for c in C[:16000]};val={np.packbits(c).tobytes() for c in C[16000:]};assert not train&val
  for item in manifest:
   if item['grid']!=n:continue
   tag=item['run'];src=Path(item['source'])/'injector.final.pt';assert sha(src)==item['files']['injector.final.pt']
   sources.append({'run':tag,'injector_sha256':sha(src),'corpus_sha256':sha(ROOT/f'g{n}.npz')})
   q=torch.load(src,map_location='cpu',weights_only=True);w=q['W_inj'].float()*float(q.get('scale',1));gt=unit(w[mask]);v=(torch.tensor(C[16000:]).float()*2-1)@w
   for seed in range(301,306):
    dest=root/'output'/f'{tag}-fit{seed}';assert (dest/'DONE').exists()
    meta=json.loads((dest/'metadata.json').read_text());assert meta['steps']==2000 and meta['fit_seed']==seed and meta['data_seed']==29000+seed
    idx=np.random.default_rng(29000+seed).integers(0,16000,16000);assert hashlib.sha256(idx.tobytes()).hexdigest()==meta['bootstrap_sha256']
    assert {m['method'] for m in meta['methods']}==METHODS
    rr=list(csv.DictReader((dest/'values.csv').open()));linear=torch.load(dest/'linear.pt',map_location='cpu',weights_only=True)
    for method in sorted(METHODS):
     key=(tag,seed,method);assert key not in all_keys;all_keys.add(key)
     mm=next(m for m in meta['methods'] if m['method']==method)
     if method in linear:d=linear[method];ids=torch.arange(len(d))
     else:
      fit=torch.load(dest/f'{method}.pt',map_location='cpu',weights_only=True);ids=fit['active_ids'];net=SAE(kind=method,k=16);net.load_state_dict(fit['state']);net.eval()
      with torch.no_grad():activity=sum(net.encode((b-fit['mean'])/fit['scale']).abs().sum(0) for b in v.split(512))
      assert torch.equal(torch.where(activity>0)[0],ids),(key,'active IDs')
      assert mm['training'][-1]['step']==2000
      d=fit['state']['D'][ids]*fit['scale']
     sim=(gt@unit(d).T).abs().numpy();ri,ci=linear_sum_assignment(-sim)
     matched=[r for r in rr if r['method']==method and r['metric']=='matched'];near=[r for r in rr if r['method']==method and r['metric']=='all_directions_nearest']
     assert len(matched)==len(edges)==len(ri) and len(near)==len(d)
     assert len({r['direction'] for r in matched})==len(edges) and {int(r['edge']) for r in matched}==set(edges.tolist())
     idmap={int(x):i for i,x in enumerate(ids)};edgemap={int(x):i for i,x in enumerate(edges)}
     errs=[]
     for r in matched+near:
      assert int(r['fit_seed'])==seed and int(r['data_seed'])==29000+seed and r['run']==tag
      val=float(r['cosine']);assert np.isfinite(val) and 0<=val<=1.000002
      errs.append(abs(val-float(sim[edgemap[int(r['edge'])],idmap[int(r['direction'])]])))
     assert max(errs)<3e-5,(key,max(errs))
     actual=np.mean([float(r['cosine']) for r in matched]);assert abs(actual-float(sim[ri,ci].mean()))<3e-5
     for r in near:assert abs(float(r['cosine'])-float(sim[:,idmap[int(r['direction'])]].max()))<3e-5
     checks.append({'run':tag,'fit_seed':seed,'method':method,'matched':len(matched),'active':len(near),'max_cosine_error':max(errs)})
    rows.extend(rr)
    print('VERIFIED',tag,seed,flush=True)
 assert len(all_keys)==len(a.grids)*4*5*8
 result={'status':'passed','grids':a.grids,'underlying_models':len(a.grids)*4,'new_fits':len(all_keys),'rows':len(rows),'checks':checks,'sources':sources}
 path=ROOT/('validation.json' if len(a.grids)==6 else f'validation-g{a.grids[0]}.json');path.write_text(json.dumps(result,indent=2))
 if len(a.grids)==6:
  with (ROOT/'alignment_distribution_values.csv').open('w') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 print('PASS',len(all_keys),'fits',len(rows),'rows',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--grids',type=int,nargs='+',default=list(range(5,11)));main(p.parse_args())
