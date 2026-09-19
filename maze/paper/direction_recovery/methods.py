"""Compact SAE and probe implementations; see README for scope and references."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SAE(nn.Module):
    def __init__(self,d=256,width=1024,kind='relu',k=16,l1=.1):
        super().__init__(); self.kind=kind; self.k=k; self.width=width; self.l1=l1
        self.D=nn.Parameter(F.normalize(torch.randn(width,d),dim=1))
        self.E=nn.Parameter(self.D.detach().T.clone()); self.b=nn.Parameter(torch.zeros(width))
        self.center=nn.Parameter(torch.zeros(d)); self.register_buffer('threshold',torch.tensor(0.))
    def encode(self,x,train_batch=False):
        x=x-self.center
        if self.kind=='mp':
            # Greedy residual correlation, tied unit dictionary, gradients through selected coefficients and atoms.
            r=x; z=x.new_zeros(len(x),self.width)
            for _ in range(self.k):
                v,i=(r@self.D.T).max(1)
                z=z.scatter_add(1,i[:,None],v[:,None]); r=r-v[:,None]*self.D[i]
            return z
        z=F.relu(x@self.E+self.b)
        if self.kind in ('batchtopk','matryoshka'):
            if train_batch:
                v,i=z.flatten().topk(min(self.k*len(x),z.numel()),sorted=False)
                with torch.no_grad(): self.threshold.lerp_(v.min(),.05)
                z=torch.zeros_like(z).flatten().scatter(0,i,v).view_as(z)
            else: z=z*(z>self.threshold)
        return z
    def forward(self,x):
        z=self.encode(x,self.training); rec=z@self.D+self.center
        loss=(rec-x).square().sum(1).mean()
        if self.kind=='relu': loss=loss+self.l1*z.abs().sum(1).mean()
        if self.kind=='matryoshka':
            # Shared batch-topk code, equally weighted nested prefixes.
            for n in [self.width//8,self.width//4,self.width//2]:
                loss=loss+((z[:,:n]@self.D[:n]+self.center-x).square().sum(1).mean())
            loss=loss/4
        return rec,z,loss

def fit_sae(x,v,kind,steps=2000,seed=0,k=16,l1=.1):
    torch.manual_seed(seed); d=x.shape[1]
    mean=x.mean(0); scale=((x-mean).square().sum(1).mean()/d).sqrt().clamp_min(1e-6)
    x=(x-mean)/scale; v=(v-mean)/scale
    net=SAE(d,4*d,kind,k,l1).to(x.device); opt=torch.optim.Adam(net.parameters(),lr=1e-3)
    log=[]
    for step in range(steps):
        net.train(); xb=x[torch.randint(len(x),(512,),device=x.device)]
        _,_,loss=net(xb); opt.zero_grad(); loss.backward()
        with torch.no_grad(): net.D.grad-=((net.D.grad*net.D).sum(1,keepdim=True))*net.D
        opt.step()
        with torch.no_grad(): net.D.copy_(F.normalize(net.D,dim=1))
        if (step+1)%500==0 or step==steps-1:
            net.eval()
            with torch.no_grad():
                zz=torch.cat([net.encode(t) for t in v.split(512)]); rr=zz@net.D+net.center
                ev=1-(rr-v).square().sum()/((v-v.mean(0)).square().sum())
                log.append(dict(step=step+1,ev=float(ev),l0=float((zz.abs()>1e-6).float().sum(1).mean()),dead=int((zz.abs().sum(0)==0).sum())))
            print(kind,log[-1],flush=True)
    return net,mean,scale,log

def unit(x): return F.normalize(x,dim=-1)

def linear_methods(x,y,v,yv):
    """y signed +/-1, generative coefficient is per unit signed feature."""
    xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym
    gen=torch.linalg.lstsq((y.double()-y.double().mean(0)).cpu(),(x.double()-x.double().mean(0)).cpu(),rcond=1e-6,driver='gelsd').solution.float().to(x.device)
    gen=gen-gen.mean(0)
    cov=yc.T@xc/len(x); var=yc.square().mean(0).clamp_min(1e-6)
    dom=cov/var[:,None]
    xx=xc.T@xc/len(x); xy=xc.T@yc/len(x)
    ridge=torch.linalg.solve(xx+.01*xx.trace()/len(xx)*torch.eye(len(xx),device=x.device),xy).T
    torch.manual_seed(0); lin=nn.Linear(x.shape[1],y.shape[1]).to(x.device)
    opt=torch.optim.Adam(lin.parameters(),lr=.01)
    scale=xc.std(0).mean().clamp_min(1e-6)
    for _ in range(600):
        idx=torch.randint(len(x),(1024,),device=x.device)
        loss=F.binary_cross_entropy_with_logits(lin(xc[idx]/scale),(y[idx]+1)/2)+1e-4*lin.weight.square().sum()
        opt.zero_grad();loss.backward();opt.step()
    logistic=lin.weight.detach()/scale
    acc={
        'ridge':float((((v-xm)@ridge.T+ym>0)==(yv>0)).float().mean()),
        'logistic':float(((lin((v-xm)/scale)>0)==(yv>0)).float().mean())}
    return {'generative_ols':gen,'difference_means':dom,'ridge':ridge,'logistic':logistic},gen,acc

@torch.no_grad()
def geometry(d,gt,gen,labels=None):
    from scipy.optimize import linear_sum_assignment
    d=unit(d); g=unit(gt); sim=(g@d.T).abs()
    r,c=linear_sum_assignment(-sim.cpu().numpy())
    return {'nearest_mean':float(sim.max(1).values.mean()),'nearest_median':float(sim.max(1).values.median()),
            'frac_above_08':float((sim.max(1).values>.8).float().mean()),
            'assignment_mean':float(sim[r,c].mean()),'per_edge_nearest':sim.max(1).values.cpu().tolist()}
