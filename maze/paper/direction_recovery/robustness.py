import json,argparse
from pathlib import Path
import numpy as np
import torch
from methods import fit_sae,geometry,unit
from experiment import save_json
from maze_distance import fresh_dfs,free_mask

def main(a):
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(19001);n=10;mask=free_mask(n);C=np.stack([fresh_dfs(rng,n) for _ in range(20000)]);y=torch.tensor(C[:,mask],device='cuda').float()*2-1
 for cond in ['fixed','trainable']:
  q=torch.load(Path(a.models)/cond/'injector.final.pt',weights_only=True,map_location='cuda');w=q['W_inj'].float()*q.get('scale',1);gt=w[mask];g=gt-gt.mean(0)
  x=(torch.tensor(C,device='cuda').float()*2-1)@w
  for method in ['relu','batchtopk','matryoshka','mp']:
   for seed,k,l1 in [(202,16,.1),(101,64,1.)]:
    key=f'{cond}-{method}-s{seed}-k{k}-l1{l1}';p=out/f'{key}.json'
    if p.exists():continue
    net,mean,scale,log=fit_sae(x[:16000],x[16000:],method,2000,seed,k,l1)
    with torch.no_grad():
     z=torch.cat([net.encode((b-mean)/scale) for b in x[16000:].split(512)]);zc=z-z.mean(0);yc=y[16000:]-y[16000:].mean(0)
     corr=yc.T@zc/(yc.norm(dim=0)[:,None]*zc.norm(dim=0)[None]).clamp_min(1e-8);idx=corr.abs().argmax(1);dictionary=net.D[z.abs().sum(0)>0];selected=net.D[idx]
     res=geometry(dictionary,g,None);res['label_selected_mean']=float((unit(g)*unit(selected)).sum(1).abs().mean())
    save_json(p,{'condition':cond,'method':method,'sae_seed':seed,'k':k,'l1':l1,'training':log,'identifiable':res})
    torch.save({'state':net.state_dict(),'mean':mean,'scale':scale},out/f'{key}.pt');print('ROBUSTNESS',key,res['nearest_mean'],flush=True)
 (out/'DONE').write_text('complete\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--out',required=True);main(p.parse_args())
