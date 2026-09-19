"""Run one model; outputs are atomic and resumable at each method/site."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model import MazeGPT,MazeGPTConfig
from methods import fit_sae,linear_methods,geometry,unit
from maze_distance import free_mask,fresh_dfs

def save_json(p,d):
    p=Path(p); tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def load_model(run,dev):
    c=torch.load(run/'model.final.pt',map_location='cpu',weights_only=False)
    m=MazeGPT(MazeGPTConfig.from_dict(c['model_cfg']));m.load_state_dict(c['state_dict']);m.to(dev).eval()
    q=run/'injector.final.pt'
    if q.exists():
        w=torch.load(q,map_location=dev,weights_only=True);assert w.get('layer',0)==0
        W=w['W_inj'].float()*float(w.get('scale',1))
    else: W=None
    return m,W

@torch.no_grad()
def block(m,r):
    h,dh=m.attn.n_heads if hasattr(m.attn,'n_heads') else 16,16
    B,P,D=r.shape; q,k,v=m.attn.qkv(m.ln1(r)).split(m.attn.d_inner,dim=-1)
    h=m.attn.d_inner//dh
    q,k,v=(a.view(B,P,h,dh).transpose(1,2) for a in (q,k,v))
    y=F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=m.attn.scale)
    r=r+m.attn.out(y.transpose(1,2).reshape(B,P,-1));return r+m.mlp(m.ln2(r)),k,v

def build_data(S,n,count,seed,cache):
    if cache.exists():
        q=np.load(cache);return q['conn'],q['tokens']
    rng=np.random.default_rng(seed)
    # Split by maze connectivity, not just RNG seed. Exclude all frozen evaluation graphs.
    forbidden=set(map(bytes,np.packbits(S['conn_a'],axis=1)))
    if 'conn_b' in S:forbidden.update(map(bytes,np.packbits(S['conn_b'],axis=1)))
    accepted=[]
    while len(accepted)<count:
        c=fresh_dfs(rng,n);key=np.packbits(c).tobytes()
        if key in forbidden:continue
        forbidden.add(key);accepted.append(c)
    conn=np.stack(accepted)
    toks=np.tile(S['prompts'][0],(count,1));coords={(int(r),int(c)):int(t) for t,r,c in S['coord_ids']}
    # The two leading BOS/ADJLIST_START tokens are followed by four-token chunks.
    E=n*n-1;assert int(S['end_pos'][0])==2+4*E
    for j,C in enumerate(conn):
        fs=np.flatnonzero(C);rng.shuffle(fs)
        for k,f in enumerate(fs):
            if f<n*n:r,c=divmod(f,n);a,b=(r,c),(r+1,c)
            else:r,c=divmod(f-n*n,n);a,b=(r,c),(r,c+1)
            if rng.random()<.5:a,b=b,a
            toks[j,2+4*k]=coords[a];toks[j,4+4*k]=coords[b]
        # Reuse only origin/target suffix grammar; sample independent endpoints.
        a,b=rng.choice(n*n,2,replace=False)
        toks[j,-7]=coords[divmod(int(a),n)];toks[j,-4]=coords[divmod(int(b),n)];toks[j,-1]=coords[divmod(int(a),n)]
    np.savez_compressed(cache,conn=conn,tokens=toks);return conn,toks

@torch.no_grad()
def activations(m,W,toks,conn,end,dev):
    pre=[];post=[]
    for lo in range(0,len(toks),128):
        t=torch.as_tensor(toks[lo:lo+128],device=dev).long();c=torch.as_tensor(conn[lo:lo+128],device=dev).float()*2-1
        r=m.embed(t)+m.pos_embed(torch.arange(t.shape[1],device=dev))[None]
        if W is not None:r[:,end:]+= (c@W)[:,None]
        pre.append(r[:,-2].float())
        with torch.autocast('cuda',dtype=torch.bfloat16): out,_,_=block(m.blocks[0],r)
        post.append(out[:,-2].float())
    return {'injection':torch.cat(pre),'post0':torch.cat(post)}

@torch.no_grad()
def greedy(m,prompt,end,delta,site,extra,max_new,path_end,pad):
    B,P=prompt.shape;H,dh=m.cfg.n_heads,m.cfg.d_head;dev=prompt.device
    pos=torch.arange(P+max_new,device=dev);r=m.embed(prompt)+m.pos_embed(pos[:P])[None];cache=[]
    active=(pos[None,:P]>=end[:,None]).to(r.dtype)[...,None]
    for i,blk in enumerate(m.blocks):
        if i==0:
            r=r+active*delta[:,None]
            if site=='injection':r=r+active*extra[:,None]
        r,k,v=block(blk,r)
        if i==0 and site=='post0':r=r+active*extra[:,None]
        K=torch.empty(B,H,P+max_new,dh,device=dev,dtype=k.dtype);V=torch.empty_like(K)
        K[:,:,:P]=k;V[:,:,:P]=v;cache.append((K,V))
    cur=m.unembed(m.ln_final(r[:,-1])).argmax(-1);out=torch.full((B,max_new),pad,device=dev,dtype=torch.long);done=torch.zeros(B,device=dev,dtype=torch.bool)
    for step in range(max_new):
        out[:,step]=torch.where(done,pad,cur);done|=cur==path_end
        if bool(done.all()) or step==max_new-1:break
        t=P+step;r=m.embed(cur)+m.pos_embed(pos[t])[None]
        for i,blk in enumerate(m.blocks):
            if i==0:r=r+delta+(extra if site=='injection' else 0)
            q,k,v=blk.attn.qkv(blk.ln1(r)).split(blk.attn.d_inner,dim=-1)
            K,V=cache[i];K[:,:,t]=k.view(B,H,dh);V[:,:,t]=v.view(B,H,dh)
            y=F.scaled_dot_product_attention(q.view(B,H,1,dh),K[:,:,:t+1],V[:,:,:t+1],scale=blk.attn.scale)
            r=r+blk.attn.out(y.reshape(B,-1));r=r+blk.mlp(blk.ln2(r))
            if i==0 and site=='post0':r=r+extra
        cur=m.unembed(m.ln_final(r)).argmax(-1)
    return out

def judge(gen,orig,tgt,conn,n,lut,end,pad):
    cur=tuple(orig);clean=True;steps=0
    for token in gen:
        if token in (end,pad):break
        if int(token) not in lut:return False
        nxt=lut[int(token)];r,c=cur;rr,cc=nxt
        if abs(r-rr)+abs(c-cc)!=1:clean=False
        else:
            f=n*n+r*n+min(c,cc) if r==rr else min(r,rr)*n+c
            if not conn[f]:clean=False
        cur=nxt;steps+=1
    return bool(clean and cur==tuple(tgt) and steps>0)

@torch.no_grad()
def evaluate(m,W,S,coef,site,ids,alpha,dev):
    n=int(S['grid_n']);mask=free_mask(n);sol=[];true_sol=[]
    lut={int(t):(int(r),int(c)) for t,r,c in S['coord_ids']};ep=int(S['path_end_id']);pad=int(S['pad_id'])
    for ix in np.array_split(ids,max(1,int(np.ceil(len(ids)/64)))):
        if not len(ix):continue
        a=S['maze'][ix];ca=torch.as_tensor(S['conn_a'][a],device=dev).float()*2-1;cb=torch.as_tensor(S['conn_b'][ix],device=dev).float()*2-1
        delta=ca@W if W is not None else torch.zeros(len(ix),256,device=dev)
        extra=alpha*((cb-ca)[:,mask]@coef)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            gen=greedy(m,torch.as_tensor(S['prompts'][a],device=dev).long(),torch.as_tensor(S['end_pos'][a],device=dev).long(),delta,site,extra,min(2*n*n+4,m.cfg.n_ctx-S['prompts'].shape[1]),ep,pad).cpu().numpy()
        for k,j in enumerate(ix):
            z=int(S['maze'][j]);sol.append(judge(gen[k],S['origin'][z],S['target'][z],S['conn_b'][j],n,lut,ep,pad))
            true_sol.append(judge(gen[k],S['origin'][z],S['target'][z],S['conn_a'][z],n,lut,ep,pad))
    return np.asarray(sol),np.asarray(true_sol)


def main(args):
    start=time.time();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');dev='cuda'
    run=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    S=np.load(args.set);n=int(S['grid_n']);mask=free_mask(n);m,W=load_model(run,dev)
    if args.phase=='eval':
        evaluate_saved(args,m,W,S,out,dev);return
    cache=out.parent/f'train-g{n}-n{args.count}-dedup.npz';C,T=build_data(S,n,args.count,19001,cache)
    acts=activations(m,W,T,C,int(S['end_pos'][0]),dev);train_n=int(.8*len(C))
    Y=torch.as_tensor(C[:,mask],device=dev).float()*2-1
    # Validate exact pre-injection semantics and fast decoder against original forward.
    with torch.no_grad():
        p=torch.as_tensor(S['prompts'][:2],device=dev).long();cs=torch.as_tensor(S['conn_a'][:2],device=dev).float()*2-1
        d=cs@W if W is not None else torch.zeros(2,256,device=dev);end=torch.as_tensor(S['end_pos'][:2],device=dev)
        full=((torch.arange(p.shape[1],device=dev)[None]>=end[:,None])[...,None])*d[:,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=m(p,inject_delta=full,padding_mode='right');g=greedy(m,p,end,d,'injection',torch.zeros_like(d),1,int(S['path_end_id']),int(S['pad_id']))
        assert torch.equal(g[:,0],logits[:,-1].argmax(-1)), 'KV first token mismatch'
    rng=np.random.default_rng(91001);base=rng.permutation(3000)
    valid_ids=np.flatnonzero(np.isin(S['maze'],base[:args.n_val])&np.isin(S['level'],[1,4,8,12,13]))
    test_ids=np.flatnonzero(np.isin(S['maze'],base[args.n_val:args.n_val+args.n_test])&np.isin(S['level'],[0,1,4,8,12,13]))
    summary={'run':str(run),'grid':n,'model':run.name,'train_n':train_n,'val_n':len(C)-train_n,'steps':args.steps,'methods':{},'kv_check':True}
    gt=W[mask] if W is not None else None
    for site in (['post0'] if W is None else ['injection','post0']):
        x=acts[site][:train_n];v=acts[site][train_n:];y=Y[:train_n];yv=Y[train_n:]
        lin,gen,acc=linear_methods(x,y,v,yv);print('linear',site,acc,flush=True)
        torch.save({'generative':gen.cpu(),'readouts':{k:z.cpu() for k,z in lin.items()},'probe_accuracy':acc},out/f'linear-{site}.pt')
        methods=list(lin)+['relu','batchtopk','matryoshka','mp']
        if site=='injection' or W is None:methods=['unedited']+(['graft'] if W is not None else [])+methods
        for method in methods:
            key=f'{site}-{method}';path=out/f'{key}.json'
            if path.exists():summary['methods'][key]=json.loads(path.read_text());continue
            res={'site':site,'method':method,'probe_accuracy':acc.get(method)}
            if method in lin:
                dictionary=lin[method];selected=unit(dictionary);coef=selected*(gen*selected).sum(1,keepdim=True)
            elif method in ('graft','unedited'):
                coef=gt if method=='graft' else torch.zeros(mask.sum(),256,device=dev);dictionary=coef;selected=unit(coef)
            else:
                net,mean,scale,log=fit_sae(x,v,method,args.steps,seed=101,k=args.k);res['training']=log
                with torch.no_grad():
                    z=torch.cat([net.encode((b-mean)/scale) for b in v.split(512)])
                    zc=z-z.mean(0);yc=yv-yv.mean(0)
                    corr=(yc.T@zc)/(yc.norm(dim=0)[:,None]*zc.norm(dim=0)[None,:]).clamp_min(1e-8)
                    selected_i=corr.abs().argmax(1);full_dictionary=net.D.detach()*scale
                    dictionary=full_dictionary[z.abs().sum(0)>0]
                    selected=unit(full_dictionary[selected_i])*torch.sign(corr[torch.arange(len(corr),device=dev),selected_i])[:,None]
                    coef=selected*(gen*selected).sum(1,keepdim=True)
                    res['label_correlation_mean']=float(corr.abs().max(1).values.mean())
                    res['selected_unique_atoms']=int(selected_i.unique().numel())
                torch.save({'state':net.state_dict(),'mean':mean,'scale':scale,'kind':method,'k':args.k,'selected':selected_i,'coef':coef.cpu()},out/f'{key}.pt')
            if gt is not None:
                for gauge,g in [('raw',gt),('identifiable',gt-gt.mean(0))]:
                    res[gauge]=geometry(dictionary,g,gen)
                    res[gauge]['label_selected_mean']=float((unit(g)*unit(selected)).sum(1).abs().mean())
                    res[gauge]['label_selected_per_edge']=(unit(g)*unit(selected)).sum(1).abs().cpu().tolist()
                if method not in ('graft','unedited'):
                    torch.manual_seed(700)
                    res['random_dictionary']=geometry(torch.randn_like(dictionary),gt-gt.mean(0),gen)
                    perm=torch.randperm(len(gt),device=dev)
                    res['shuffled_labels_mean']=float((unit(gt-gt.mean(0))*unit(selected[perm])).sum(1).abs().mean())
            torch.save({'coef':coef.cpu()},out/f'{key}-coef.pt')
            save_json(out/f'{key}-geometry.json',res)
            if args.phase=='fit':
                summary['methods'][key]=res;save_json(out/'summary.json',summary)
                print('GEOMETRY',key,res.get('identifiable',{}).get('nearest_mean'),flush=True)
                continue
            # Validation chooses only global strength; graft control always exactly 1.
            alphas=[1.] if method in ('graft','unedited') else [.5,1.,2.]
            vals=[]
            for alpha in alphas:
                z,_=evaluate(m,W,S,coef,site,valid_ids,alpha,dev);vals.append(float(z.mean()))
            best=alphas[int(np.argmax(vals))]
            z,za=evaluate(m,W,S,coef,site,test_ids,best,dev)
            res.update(alpha=best,validation=dict(zip(map(str,alphas),vals)),test_solved=float(z.mean()),test_true_solved=float(za.mean()),n_test_pairs=len(z),elapsed=time.time()-start)
            np.savez_compressed(out/f'{key}-eval.npz',ids=test_ids,maze=S['maze'][test_ids],level=S['level'][test_ids],eps=S['eps'][test_ids],dist=S['dist'][test_ids],solved=z,true_solved=za)
            torch.save({'coef':coef.cpu()},out/f'{key}-coef.pt');save_json(path,res);summary['methods'][key]=res
            print('RESULT',key,'recovery',res.get('identifiable',{}).get('nearest_mean'),'solved',res['test_solved'],'seconds',time.time()-start,flush=True)
            save_json(out/'summary.json',summary)
    save_json(out/'summary.json',summary);(out/('FIT_DONE' if args.phase=='fit' else 'DONE')).write_text('complete\n')

def evaluate_saved(args,m,W,S,out,dev):
    rng=np.random.default_rng(91001);base=rng.permutation(3000)
    vi=np.flatnonzero(np.isin(S['maze'],base[:args.n_val])&np.isin(S['level'],[1,4,8,12,13]))
    ti=np.flatnonzero(np.isin(S['maze'],base[args.n_val:args.n_val+args.n_test])&np.isin(S['level'],[0,1,4,8,12,13]))
    summary=json.loads((out/'summary.json').read_text())
    for path in sorted(out.glob('*-geometry.json')):
        key=path.name.removesuffix('-geometry.json');result=out/f'{key}.json'
        if result.exists():continue
        res=json.loads(path.read_text());method=res['method'];site=res['site']
        coef=torch.load(out/f'{key}-coef.pt',weights_only=True)['coef'].to(dev)
        alphas=[1.] if method in ('graft','unedited') else [.5,1.,2.]
        vals=[]
        for a in alphas:
            z,_=evaluate(m,W,S,coef,site,vi,a,dev);vals.append(float(z.mean()))
        a=alphas[int(np.argmax(vals))];z,za=evaluate(m,W,S,coef,site,ti,a,dev)
        res.update(alpha=a,validation=dict(zip(map(str,alphas),vals)),test_solved=float(z.mean()),test_true_solved=float(za.mean()),n_test_pairs=len(z))
        np.savez_compressed(out/f'{key}-eval.npz',ids=ti,maze=S['maze'][ti],level=S['level'][ti],eps=S['eps'][ti],dist=S['dist'][ti],solved=z,true_solved=za)
        save_json(result,res);summary['methods'][key]=res;save_json(out/'summary.json',summary)
        print('EVAL',out.name,key,res['test_solved'],flush=True)
    (out/'DONE').write_text('complete\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--set',required=True);p.add_argument('--out',required=True)
    p.add_argument('--phase',choices=['fit','eval','all'],default='all');p.add_argument('--count',type=int,default=20000);p.add_argument('--steps',type=int,default=2000);p.add_argument('--k',type=int,default=16)
    p.add_argument('--n-val',type=int,default=40);p.add_argument('--n-test',type=int,default=120);main(p.parse_args())
