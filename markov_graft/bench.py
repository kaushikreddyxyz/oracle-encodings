"""Step-time benchmark for one GraftGPT config on random data: python bench.py --batch 128 [--compile]"""
import argparse, time
import torch
import torch.nn.functional as F
from model import GraftGPT
from train import SIZES

ap = argparse.ArgumentParser()
ap.add_argument("--size", default="m"); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--T", type=int, default=256); ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--compile", action="store_true"); ap.add_argument("--tag", default="")
a = ap.parse_args()
torch.set_num_threads(4)
n_layer, d, n_head = SIZES[a.size]
model = GraftGPT(8, d, n_layer, n_head, 8).cuda()
fwd = torch.compile(model, mode="reduce-overhead") if a.compile else model
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 8, (a.batch, a.T + 1), device="cuda"); g = torch.randint(0, 8, (a.batch, a.T), device="cuda")
def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = fwd(x[:, :-1], g)
        loss = F.cross_entropy(logits.reshape(-1, 8).float(), x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return loss
for _ in range(10): step()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(a.steps): step()
torch.cuda.synchronize()
ms = (time.time() - t0) / a.steps * 1e3
print(f"{a.tag} size={a.size} batch={a.batch} compile={a.compile}: {ms:.1f} ms/step, {a.batch*a.T/ms*1e3/1e6:.2f} Mtok/s", flush=True)
