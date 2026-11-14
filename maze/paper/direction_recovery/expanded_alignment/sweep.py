"""Additional bootstrap fits; only original-graft geometry at immediate injection."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from methods import fit_sae,unit
from expanded_methods import linear_methods


def main(a):
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 inp=Path(a.input);manifest=json.loads((inp/'manifest.json').read_text())
 for item in manifest:
  n=item['grid'];tag=item['run']
  if n!=a.grid:continue
  C=np.load(inp/f'g{n}.npz')['conn'];mask=np.load(inp/f'g{n}.npz')['mask']
  q=torch.load(inp/f'{tag}.pt',map_location='cuda',weights_only=True)
  w=q['W_inj'].float()*float(q.get('scale',1));gt=w[mask];edge_ids=np.flatnonzero(mask)
  # Constant embedding and positional terms disappear under fit_sae/OLS centering.
  xx=(torch.tensor(C,device='cuda').float()*2-1)@w
  yy=torch.tensor(C[:,mask],device='cuda').float()*2-1
  for seed in a.seeds:
   data_seed=29000+seed
   dest=out/f'{tag}-fit{seed}';dest.mkdir(exist_ok=True)
   if (dest/'DONE').exists():continue
   idx=np.random.default_rng(data_seed).integers(0,16000,16000)
   x,y=xx[idx],yy[idx];v,yv=xx[16000:],yy[16000:]
   start=time.time();readouts,gen,acc=linear_methods(x,y,v,yv,seed=seed)
   torch.save({k:t.cpu() for k,t in readouts.items()},dest/'linear.pt')
   methods=list(readouts)+['relu','batchtopk','matryoshka','mp']
   rows=[];metadata=[]
   for method in methods:
    t=time.time();log=[]
    if method in readouts:d=readouts[method];ids=torch.arange(len(d),device='cuda')
    else:
     net,mean,scale,log=fit_sae(x,v,method,steps=a.steps,seed=seed)
     with torch.no_grad():
      activity=sum(net.encode((b-mean)/scale).abs().sum(0) for b in v.split(512))
      ids=torch.where(activity>0)[0];d=net.D[ids].detach()*scale
     torch.save({'state':{k:v.cpu() for k,v in net.state_dict().items()},'mean':mean.cpu(),'scale':scale.cpu(),'active_ids':ids.cpu(),'k':16,'fit_seed':seed,'data_seed':data_seed},dest/f'{method}.pt')
    with torch.no_grad():sim=(unit(gt)@unit(d).T).abs().cpu().numpy()
    assert np.isfinite(sim).all() and len(d)>=len(gt)
    ri,ci=linear_sum_assignment(-sim);assert len(set(ci))==len(gt)
    for metric,pairs in [('matched',zip(ri,ci)),('all_directions_nearest',((r,c) for c,r in enumerate(sim.argmax(0))))]:
     for r,c in pairs:rows.append(dict(run=tag,method=method,metric=metric,direction=int(ids[c]),edge=int(edge_ids[r]),cosine=float(sim[r,c]),fit_seed=seed,data_seed=data_seed))
    metadata.append({'method':method,'active':len(d),'matched':len(ri),'mean':float(sim[ri,ci].mean()),'training':log,'seconds':time.time()-t})
    print('METHOD_DONE',tag,seed,method,metadata[-1]['seconds'],flush=True)
   with (dest/'values.csv').open('w') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
   (dest/'metadata.json').write_text(json.dumps({'run':tag,'fit_seed':seed,'data_seed':data_seed,'bootstrap_unique':len(np.unique(idx)),'bootstrap_sha256':hashlib.sha256(idx.tobytes()).hexdigest(),'train_n':16000,'validation_n':4000,'steps':a.steps,'methods':metadata,'accuracy':acc,'seconds':time.time()-start},indent=2))
   (dest/'DONE').write_text('complete\n');print('FIT_DONE',tag,seed,time.time()-start,flush=True)
 expected=[out/f"{item['run']}-fit{seed}"/'DONE'
           for item in manifest if item['grid']==a.grid for seed in range(301,306)]
 if len(expected)==20 and all(p.exists() for p in expected):
  (out/'DONE').write_text('complete\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',default='input');p.add_argument('--out',default='output');p.add_argument('--grid',type=int,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[301,302,303,304,305]);p.add_argument('--steps',type=int,default=2000);main(p.parse_args())
