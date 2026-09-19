"""Reproduce the PATH_START probe-coordinate steering protocol on archived edits."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from experiment import load_model,build_data,block,save_json,judge,evaluate
from methods import linear_methods,fit_sae,unit
from maze_distance import free_mask
import torch.nn.functional as F

@torch.no_grad()
def collect(m,tokens):
    out=[[] for _ in m.blocks]
    for lo in range(0,len(tokens),64):
        t=torch.as_tensor(tokens[lo:lo+64],device='cuda').long();r=m.embed(t)+m.pos_embed(torch.arange(t.shape[1],device='cuda'))[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for i,b in enumerate(m.blocks):r,_,_=block(b,r);out[i].append(r[:,-2].float())
    return [torch.cat(a) for a in out]

@torch.no_grad()
def greedy_project(m,prompt,end,dirs,targets,layer,alpha,max_new,path_end,pad):
    B,P=prompt.shape;H,dh=m.cfg.n_heads,m.cfg.d_head;dev=prompt.device
    def project(r,active=None):
        if alpha==0:return r
        if r.ndim==3:
            co=torch.einsum('bpd,bkd->bpk',r,dirs);dv=torch.einsum('bpk,bkd->bpd',targets[:,None]-co,dirs)
            return r+dv*active
        co=torch.einsum('bd,bkd->bk',r,dirs);return r+torch.einsum('bk,bkd->bd',targets-co,dirs)
    pos=torch.arange(P+max_new,device=dev);r=m.embed(prompt)+m.pos_embed(pos[:P])[None];cache=[]
    active=(pos[None,:P]>=end[:,None]).to(r.dtype)[...,None]
    for i,blk in enumerate(m.blocks):
        r,k,v=block(blk,r)
        if i==layer:r=project(r,active)
        K=torch.empty(B,H,P+max_new,dh,device=dev,dtype=k.dtype);V=torch.empty_like(K);K[:,:,:P]=k;V[:,:,:P]=v;cache.append((K,V))
    cur=m.unembed(m.ln_final(r[:,-1])).argmax(-1);out=torch.full((B,max_new),pad,device=dev,dtype=torch.long);done=torch.zeros(B,device=dev,dtype=torch.bool)
    for step in range(max_new):
        out[:,step]=torch.where(done,pad,cur);done|=cur==path_end
        if bool(done.all()) or step==max_new-1:break
        t=P+step;r=m.embed(cur)+m.pos_embed(pos[t])[None]
        for i,blk in enumerate(m.blocks):
            q,k,v=blk.attn.qkv(blk.ln1(r)).split(blk.attn.d_inner,dim=-1);K,V=cache[i];K[:,:,t]=k.view(B,H,dh);V[:,:,t]=v.view(B,H,dh)
            y=F.scaled_dot_product_attention(q.view(B,H,1,dh),K[:,:,:t+1],V[:,:,:t+1],scale=blk.attn.scale)
            r=r+blk.attn.out(y.reshape(B,-1));r=r+blk.mlp(blk.ln2(r))
            if i==layer:r=project(r)
        cur=m.unembed(m.ln_final(r)).argmax(-1)
    return out

def archive_set(path,current):
    p=torch.load(path/'pretok-cfset.pt',weights_only=True);cf=torch.load(path/'counterfactual_set.pt',weights_only=False)
    # Token ids are verified against current grid-6 coordinate mapping below.
    end=int(p['end_pos'][0]);plen=end+9
    prompts=np.stack([p['tokens'][p['offsets'][i]:p['offsets'][i]+plen].numpy() for i in range(600)])
    ids=current['coord_ids'];lut={int(t):(int(r),int(c)) for t,r,c in ids}
    origin=np.array([lut[int(row[-1])] for row in prompts]);target=np.array([lut[int(row[-4])] for row in prompts])
    conn=p['conn'].numpy().astype(bool);B=[];mi=[];families=[]
    for e in cf['edits']:
        c=conn[e['maze']].copy()
        if e.get('block') is not None:c[e['block']]=False
        if e.get('open') is not None:c[e['open']]=True
        B.append(c);mi.append(e['maze']);families.append(e['family'])
    S=dict(grid_n=np.array(6),coord_ids=ids,prompts=prompts,end_pos=p['end_pos'].numpy(),origin=origin,target=target,
           conn_a=conn,conn_b=np.stack(B),maze=np.array(mi),level=np.zeros(len(mi),dtype=int),path_end_id=current['path_end_id'],pad_id=current['pad_id'])
    for j in range(600):
        reconstructed=np.zeros(72,dtype=bool)
        for k in range(35):
            a,b=lut[int(prompts[j,2+4*k])],lut[int(prompts[j,4+4*k])]
            r,c=a;rr,cc=b;f=36+r*6+min(c,cc) if r==rr else min(r,rr)*6+c;reconstructed[f]=True
        assert np.array_equal(reconstructed,conn[j]),'archived token vocabulary mismatch'
    return S,cf['edits'],np.array(families)

@torch.no_grad()
def run_eval(m,S,edits,indices,dirs,means,layer,alpha):
    mask=free_mask(6);free=np.flatnonzero(mask);rev={f:i for i,f in enumerate(free)};sol=[];cross=[];used=[];reached=[]
    lut={int(t):(int(r),int(c)) for t,r,c in S['coord_ids']}
    for ix in np.array_split(indices,max(1,int(np.ceil(len(indices)/64)))):
        a=S['maze'][ix];D=torch.zeros(len(ix),2,256,device='cuda');tg=torch.zeros(len(ix),2,device='cuda')
        for k,j in enumerate(ix):
            e=edits[j];slot=0
            for kind,label in [('block',0),('open',1)]:
                if e.get(kind) is not None:
                    f=rev[e[kind]];D[k,slot]=dirs[f];mid=means[f].mean();tg[k,slot]=mid+alpha*(means[f,label]-mid);slot+=1
        with torch.autocast('cuda',dtype=torch.bfloat16):
            gen=greedy_project(m,torch.as_tensor(S['prompts'][a],device='cuda').long(),torch.as_tensor(S['end_pos'][a],device='cuda'),D,tg,layer,alpha,76,int(S['path_end_id']),int(S['pad_id'])).cpu().numpy()
        for k,j in enumerate(ix):
            z=int(S['maze'][j]);e=edits[j];sol.append(judge(gen[k],S['origin'][z],S['target'][z],S['conn_b'][j],6,lut,int(S['path_end_id']),int(S['pad_id'])))
            cur=tuple(S['origin'][z]);edges=[]
            for t in gen[k]:
                if int(t) not in lut:break
                nxt=lut[int(t)];r,c=cur;rr,cc=nxt
                if abs(r-rr)+abs(c-cc)==1:edges.append(36+r*6+min(c,cc) if r==rr else min(r,rr)*6+c)
                cur=nxt
            cross.append(e.get('block') in edges);used.append(e.get('open') in edges);reached.append(cur==tuple(S['target'][z]))
    return {k:np.asarray(v) for k,v in dict(solved=sol,crossed_block=cross,used_open=used,reached=reached).items()}

def main(a):
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    current=np.load(a.set);S,edits,families=archive_set(Path(a.reference),current);m,W=load_model(Path(a.run),'cuda');assert W is None
    C,T=build_data(current,6,10000,19001,out.parent/'legacy-train.npz');A=collect(m,T);Y=torch.tensor(C[:,free_mask(6)],device='cuda').float()*2-1
    rng=np.random.default_rng(8191);base=rng.permutation(600);vi=np.flatnonzero(np.isin(S['maze'],base[:30]));ti=np.flatnonzero(np.isin(S['maze'],base[30:130]))
    dirs_by={};sweep=[]
    for l,x in enumerate(A):
        lin,gen,acc=linear_methods(x[:8000],Y[:8000],x[8000:],Y[8000:]);dirs_by[l]=(lin,gen);sweep.append({'layer':l,**acc})
    save_json(out/'probe_layer_sweep.json',sweep);best=max(sweep,key=lambda d:d['logistic'])['layer'];print('BEST_LAYER',best,sweep,flush=True)
    for l in sorted(set([0,best])):
        x=A[l];lin,gen=dirs_by[l]
        for method in ['unedited']+list(lin)+['relu','batchtopk','matryoshka','mp']:
            key=f'L{l}-{method}';p=out/f'{key}.json'
            if p.exists():continue
            log=None
            if method=='unedited':dirs=torch.zeros_like(gen)
            elif method in lin:dirs=unit(lin[method])
            else:
                net,mean,scale,log=fit_sae(x[:8000],x[8000:],method,a.steps,seed=101)
                with torch.no_grad():
                    z=torch.cat([net.encode((b-mean)/scale) for b in x[8000:].split(512)]);zc=z-z.mean(0);yc=Y[8000:]-Y[8000:].mean(0)
                    corr=yc.T@zc/(yc.norm(dim=0)[:,None]*zc.norm(dim=0)[None]).clamp_min(1e-8);idx=corr.abs().argmax(1);dirs=unit(net.D[idx].detach())
                torch.save({'state':net.state_dict(),'mean':mean,'scale':scale,'selected':idx},out/f'{key}.pt')
            co=x[8000:]@dirs.T;y=Y[8000:]>0
            means=torch.stack([(co*(~y)).sum(0)/(~y).sum(0),(co*y).sum(0)/y.sum(0)],1)
            alphas=[0.] if method=='unedited' else [1.,4.,8.];vals=[]
            for alpha in alphas:vals.append(float(run_eval(m,S,edits,vi,dirs,means,l,alpha)['solved'].mean()))
            alpha=alphas[int(np.argmax(vals))];r=run_eval(m,S,edits,ti,dirs,means,l,alpha)
            np.savez_compressed(out/f'{key}.npz',ids=ti,maze=S['maze'][ti],family=families[ti],**r)
            result={'method':method,'layer':l,'best_probe_layer':best,'alpha':alpha,'validation':vals,'test_n':len(ti),'training':log,'test_solved':float(r['solved'].mean()),'families':{}}
            for family in np.unique(families):
                sel=families[ti]==family;result['families'][family]={k:float(v[sel].mean()) for k,v in r.items()}
            save_json(p,result);print('LEGACY',key,result['test_solved'],flush=True)
    (out/'DONE').write_text('complete\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--set',required=True);p.add_argument('--out',required=True);p.add_argument('--reference',required=True);p.add_argument('--steps',type=int,default=2000);main(p.parse_args())
