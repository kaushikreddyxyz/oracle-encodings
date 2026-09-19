"""Train one GraftGPT on one Markov source with one graft arm; write results/<name>.json.

Arms:  none | oracle_<b> | leak_<b>   (b = bits of graft per token)
Loss is measured only on targets whose full k-context lies inside the sequence, so
every arm and every reference loss is averaged over the same positions.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from dgp import MarkovSource
from model import GraftGPT

SIZES = {"xs": (1, 32, 2), "s": (2, 64, 4), "m": (4, 128, 4), "l": (6, 256, 4)}   # layers, d, heads
HERE = os.path.dirname(os.path.abspath(__file__))


def run_name(a):
    m = a.k if a.m is None else a.m
    return f"A{a.A}_h{a.h:g}_k{a.k}_m{m}_{a.size}_{a.arm}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")


class Pool:
    """All sequences for a run are generated up front (vectorized numpy) and kept on the GPU as uint8,
    so the training loop does no per-step CPU work."""
    def __init__(self, src, arm, n_seq, T, rng, dev, chunk=32768, dropout=0.0, null_value=0):
        xs, gs = [], []
        for start in range(0, n_seq, chunk):
            x = src.sample(min(chunk, n_seq - start), T + 1, rng)
            xs.append(x)
            if arm != "none":
                kind, b = arm.split("_")
                g = src.grafts(x, kind, int(b))[0].astype(np.uint8)
                if dropout > 0:
                    g[rng.random(len(g)) < dropout] = null_value
                gs.append(g)
        self.x = torch.from_numpy(np.concatenate(xs)).to(dev)
        self.g = torch.from_numpy(np.concatenate(gs)).to(dev) if gs else None
        self.mask = torch.zeros(T, dtype=torch.bool, device=dev)
        self.mask[src.k - 1:] = True
        self.n, self.pos = n_seq, 0

    def next(self, batch):
        if self.pos + batch > self.n:
            self.pos = 0
        sl = slice(self.pos, self.pos + batch); self.pos += batch
        x = self.x[sl].long()
        g = None if self.g is None else self.g[sl].long()
        return x[:, :-1], x[:, 1:], self.mask.expand(x.size(0), -1), g


def loss_on(model, inp, tgt, mask, g):
    logits = model(inp, g)
    l = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tgt.reshape(-1), reduction="none")
    return (l * mask.reshape(-1)).sum() / mask.sum()


@torch.no_grad()
def evaluate(model, pool, arm, a, dev, n_batches, null_value=None):
    model.eval()
    pool.pos = 0
    tot, tot_shuf, tot_null, n = 0.0, 0.0, 0.0, 0
    for _ in range(n_batches):
        inp, tgt, mask, g = pool.next(a.batch)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            tot += loss_on(model, inp, tgt, mask, g).item()
            if g is not None:
                gs = g[torch.randperm(g.size(0), device=dev)]           # decorrelate graft from sequence
                tot_shuf += loss_on(model, inp, tgt, mask, gs).item()
                if null_value is not None:
                    tot_null += loss_on(model, inp, tgt, mask, torch.full_like(g, null_value)).item()
        n += 1
    model.train()
    if arm == "none":
        return tot / n, None, None
    return tot / n, tot_shuf / n, (tot_null / n if null_value is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--h", type=float, default=2.0)
    ap.add_argument("--table-seed", type=int, default=0)
    ap.add_argument("--size", default="m", choices=list(SIZES))
    ap.add_argument("--arm", default="none")
    ap.add_argument("--graft-fixed", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--final-eval-batches", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--wandb", default="markov-graft")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--graft-dropout", type=float, default=0.0, help="per-sequence prob. the graft is replaced by a null token")
    a = ap.parse_args()

    name = run_name(a)
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, name + ".json")
    if os.path.exists(out_path):
        print("exists, skipping", out_path); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    src = MarkovSource(a.A, a.k, a.m, a.h, a.table_seed)
    torch.set_num_threads(4)
    graft_values = 0 if a.arm == "none" else (1 << int(a.arm.split("_")[1])) + (1 if a.graft_dropout > 0 else 0)
    null_value = graft_values - 1
    n_layer, d, n_head = SIZES[a.size]
    model = GraftGPT(a.A, d, n_layer, n_head, graft_values, a.graft_fixed).to(dev)
    n_params = model.n_params()
    fwd = model if (a.no_compile or dev != "cuda") else torch.compile(model, mode="reduce-overhead")

    t_gen = time.time()
    train_pool = Pool(src, a.arm, a.steps * a.batch, a.T, np.random.default_rng(1000 + a.seed), dev,
                      dropout=a.graft_dropout, null_value=null_value)
    eval_pool = Pool(src, a.arm, a.final_eval_batches * a.batch, a.T, np.random.default_rng(2000 + a.seed), dev)
    ref = src.reference_losses(src.sample(4096, a.T + 1, np.random.default_rng(3000)))
    print(f"data generated in {time.time()-t_gen:.0f}s", flush=True)
    meta = {"name": name, "A": a.A, "k": a.k, "m": src.m, "h_bits": a.h, "lags": src.lags.tolist(),
            "description_bits": src.description_bits, "entropy_nats": src.entropy_nats,
            "size": a.size, "n_layer": n_layer, "d": d, "n_params": n_params, "arm": a.arm,
            "graft_fixed": a.graft_fixed, "graft_dropout": a.graft_dropout, "tag": a.tag, "seed": a.seed, "steps": a.steps, "batch": a.batch, "T": a.T,
            "tokens": a.steps * a.batch * a.T, "ref": ref}
    print(json.dumps(meta))

    wb = None
    if not a.no_wandb:
        try:
            import wandb
            wb = wandb.init(project=a.wandb, name=name, config=meta, reinit=True)
        except Exception as e:
            print("wandb disabled:", e)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 and p.requires_grad]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))

    curve, t0 = [], time.time()
    for step in range(a.steps):
        for gparam in opt.param_groups:
            gparam["lr"] = a.lr * sched(step)
        inp, tgt, mask, g = train_pool.next(a.batch)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
            loss = loss_on(fwd, inp, tgt, mask, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 and wb:
            wb.log({"train_loss": loss.item(), "lr": a.lr * sched(step)}, step=step)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            ev, ev_shuf, _ = evaluate(model, eval_pool, a.arm, a, dev, a.eval_batches)
            curve.append({"step": step + 1, "eval_loss": ev, "eval_loss_shuffled_graft": ev_shuf})
            print(f"{name} step {step+1} train {loss.item():.4f} eval {ev:.4f} shuf {ev_shuf} "
                  f"bayes {ref['bayes']:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if wb:
                wb.log({"eval_loss": ev, "excess_over_bayes": ev - ref["bayes"],
                        **({"eval_loss_shuffled_graft": ev_shuf} if ev_shuf is not None else {})}, step=step + 1)

    final, final_shuf, final_null = evaluate(model, eval_pool, a.arm, a, dev, a.final_eval_batches,
                                             null_value if a.graft_dropout > 0 else None)
    res = {**meta, "final_loss": final, "final_loss_shuffled_graft": final_shuf, "final_loss_null_graft": final_null,
           "excess_over_bayes": final - ref["bayes"], "curve": curve, "seconds": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    if wb:
        wb.summary.update({"final_loss": final, "final_loss_shuffled_graft": final_shuf,
                           "excess_over_bayes": final - ref["bayes"]})
        wb.finish()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
