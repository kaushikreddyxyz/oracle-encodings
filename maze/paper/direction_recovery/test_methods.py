import unittest
import torch
from methods import SAE
class Methods(unittest.TestCase):
 def test_batch_budget(self):
  torch.manual_seed(7);m=SAE(d=16,width=64,kind='batchtopk',k=4)
  z=m.encode(torch.randn(20,16),True)
  self.assertEqual(int((z>0).sum()),80)
 def test_mp_residual_monotone(self):
  torch.manual_seed(7);m=SAE(d=16,width=64,kind='mp',k=1);x=torch.randn(20,16)
  errors=[]
  for k in [1,2,4,8]:
   m.k=k;z=m.encode(x);errors.append(((x-z@m.D-m.center)**2).sum(1))
  for a,b in zip(errors,errors[1:]):self.assertTrue(bool((b<=a+1e-5).all()))
 def test_gradients(self):
  for kind in ['relu','batchtopk','matryoshka','mp']:
   m=SAE(d=16,width=64,kind=kind,k=4);_,_,loss=m(torch.randn(20,16));loss.backward()
   self.assertTrue(bool(torch.isfinite(m.D.grad).all()))
 def test_constant_sum_gauge(self):
  torch.manual_seed(7);y=torch.zeros(1000,12,dtype=torch.double)
  for row in y:row[torch.randperm(12)[:5]]=1
  w=torch.randn(12,16,dtype=torch.double);x=y@w;yc=y-y.mean(0)
  fit=torch.linalg.lstsq(yc,x-x.mean(0),driver='gelsd').solution
  torch.testing.assert_close(fit,w-w.mean(0),atol=1e-10,rtol=1e-10)
if __name__=='__main__':unittest.main()
