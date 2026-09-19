#!/usr/bin/env python3
"""SAE-feature steering of the maze world model, following Spies et al. (arXiv 2412.11867).

  stage train    sparse autoencoder on resid_post of a chosen block (default 0), all token positions of fresh mazes
                 (ReLU, tied-ish init, L1 penalty, dead-latent resampling), expansion factor 8 -> 2048 latents;
                 reports reconstruction, L0, and the rollout agreement when the SAE reconstruction is patched in.
  stage features per-connection features: for every latent, precision/recall/F1 of "active at the semicolon encoding f";
                 pick the best latent per connection (plus a logistic decoder on the latent vector as the paper's decision
                 tree analogue); report how many connections have a clean single-latent feature.
  stage steer    counterfactual edit set with feature toggling (paper protocol): add f -> set latent to its typical active
                 value at ALL semicolons; remove f -> set latent to 0 at f's semicolon (and at all semicolons). Implemented as
                 a delta along the decoder direction added to the residual (no reconstruction error injected), plus a
                 "reconstruction-patched" variant for fidelity with the paper.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch

from build_counterfactual_set import feat_cells
from eval_injection import greedy_rollout
from fast_data import PretokDataset
from wm_probes import (WORK as WORK_P, SET_DIR, N_FEATS, load_model_and_cfg, make_probeset, slice_ps, rollout, edit_spec, judge, summarize)

WORK = Path("/workspace/runs/wm_sae")


class SAE(torch.nn.Module):
    """ReLU SAE; with topk > 0 only the k largest pre-activations per token are kept (TopK SAE, no L1 needed)."""
    def __init__(self, d, n_lat, topk=0):
        super().__init__()
        self.enc = torch.nn.Linear(d, n_lat); self.dec = torch.nn.Linear(n_lat, d); self.topk = topk
        torch.nn.init.kaiming_uniform_(self.dec.weight); self.dec.weight.data /= self.dec.weight.data.norm(dim=0, keepdim=True)
        self.enc.weight.data = self.dec.weight.data.T.clone(); self.b_pre = torch.nn.Parameter(torch.zeros(d))

    def encode(self, x):
        a = torch.relu(self.enc(x - self.b_pre))
        if self.topk > 0:
            v, i = a.topk(self.topk, dim=-1); a = torch.zeros_like(a).scatter_(-1, i, v)
        return a
    def decode(self, a): return self.dec(a) + self.b_pre
    def forward(self, x): a = self.encode(x); return self.decode(a), a


@torch.no_grad()
def collect_all(model, ps, layer, batch=256):
    """resid_post at `layer` for every real token position (n_tokens, d) fp16, + (maze, pos) index."""
    name = f"blocks.{layer}.hook_resid_post"; X, idx = [], []
    for lo in range(0, len(ps.conn), batch):
        hi = min(len(ps.conn), lo + batch)
        _, cache = model.run_with_cache(ps.tokens[lo:hi], attention_mask=ps.attn[lo:hi], return_type=None, names_filter=lambda nm: nm == name)
        h = cache[name]; m = ps.attn[lo:hi].bool()
        X.append(h[m].half().cpu()); bi, pi = torch.nonzero(m, as_tuple=True); idx.append(torch.stack([bi + lo, pi], 1).cpu()); del cache
    return torch.cat(X), torch.cat(idx)


def stage_train(layer: int = 0, expansion: int = 8, l1: float = 3e-2, steps: int = 20000, batch: int = 4096, lr: float = 1e-3, seed: int = 0, tag: str = "", topk: int = 0):
    device = torch.device("cuda"); torch.manual_seed(seed); WORK.mkdir(parents=True, exist_ok=True)
    model, cfg, vocab = load_model_and_cfg(device); meta = json.load(open(WORK_P / "data.json")); ps = make_probeset(PretokDataset(meta["pretok"]), vocab, device)
    X, idx = collect_all(model, ps, layer); print(f"[sae] layer {layer}: {X.shape[0]} token activations", flush=True)
    d = X.shape[1]; n_lat = expansion * d; sae = SAE(d, n_lat, topk).to(device); opt = torch.optim.Adam(sae.parameters(), lr=lr)
    if topk > 0: l1 = 0.0
    Xg = X.to(device).float(); scale = Xg.norm(dim=1).mean() / np.sqrt(d); Xg = Xg / scale                      # unit-ish scale for a stable l1
    n = Xg.shape[0]; act_count = torch.zeros(n_lat, device=device); last_resample = 0
    for step in range(steps):
        xb = Xg[torch.randint(0, n, (batch,), device=device)]; rec, a = sae(xb)
        mse = ((rec - xb) ** 2).sum(1).mean(); l1_term = (a * sae.dec.weight.norm(dim=0)[None]).sum(1).mean(); loss = mse + l1 * l1_term
        opt.zero_grad(); loss.backward()
        sae.dec.weight.grad -= (sae.dec.weight.grad * sae.dec.weight).sum(0, keepdim=True) * sae.dec.weight              # remove the component that changes decoder norms
        opt.step(); sae.dec.weight.data /= sae.dec.weight.data.norm(dim=0, keepdim=True)
        act_count += (a > 0).float().sum(0)
        if step % 2000 == 1999:
            dead = (act_count == 0).nonzero().flatten()
            if len(dead) and step < steps - 3000:                                                                   # resample dead latents on high-loss inputs
                with torch.no_grad():
                    xs = Xg[torch.randint(0, n, (8192,), device=device)]; r, _ = sae(xs); err = ((r - xs) ** 2).sum(1); pick = torch.multinomial(err / err.sum(), len(dead), replacement=True)
                    v = xs[pick] - sae.b_pre; v = v / v.norm(dim=1, keepdim=True); sae.dec.weight.data[:, dead] = v.T; sae.enc.weight.data[dead] = 0.2 * v; sae.enc.bias.data[dead] = 0
            with torch.no_grad():
                xs = Xg[:20000]; r, a = sae(xs); ev = 1 - ((r - xs) ** 2).sum() / ((xs - xs.mean(0)) ** 2).sum(); l0 = (a > 0).float().sum(1).mean()
            print(f"[sae] step {step + 1}: mse {mse.item():.4f} l1 {l1_term.item():.3f} | explained var {ev.item():.4f} L0 {l0.item():.1f} dead {len(dead)}", flush=True)
            act_count.zero_()
    # rollout agreement with reconstruction patched in
    sae.eval(); name = f"blocks.{layer}.hook_resid_post"
    def patch(resid, hook):
        r, _ = sae(resid / scale); return r * scale
    sub = slice_ps(ps, torch.arange(0, 500)); g0 = rollout(model, [], sub); g1 = rollout(model, [(name, patch)], sub)
    agree = float(np.mean([a == b for a, b in zip(g0, g1)])); print(f"[sae] rollout agreement with SAE-reconstruction patched in: {agree:.3f} (500 mazes)")
    torch.save(dict(state=sae.state_dict(), d=d, n_lat=n_lat, layer=layer, scale=float(scale), l1=l1, expansion=expansion, agree=agree, topk=topk), WORK / f"sae_L{layer}{tag}.pt"); print(f"[sae] saved sae_L{layer}{tag}.pt")


def load_sae(layer, device, tag=""):
    d = torch.load(WORK / f"sae_L{layer}{tag}.pt", weights_only=True); sae = SAE(d["d"], d["n_lat"], int(d.get("topk", 0))).to(device); sae.load_state_dict(d["state"]); sae.eval(); return sae, d


@torch.no_grad()
def stage_features(layer: int = 0, site: str = "B", tag: str = ""):
    device = torch.device("cuda"); model, cfg, vocab = load_model_and_cfg(device); sae, sd = load_sae(layer, device, tag)
    A = torch.load(WORK_P / "acts.pt", weights_only=True); si, li = A["sites"].index(site), A["layers"].index(layer)
    Xs = A["A"][si, :, li].float().to(device) / sd["scale"]; y = A["y"].to(device)
    acts = sae.encode(Xs)                                                                                              # (n_chunks, n_lat) at the chosen site
    on = acts > 0; stats = {}
    best = torch.zeros(N_FEATS, dtype=torch.long); f1s = torch.zeros(N_FEATS); typ = torch.zeros(N_FEATS)
    for f in range(N_FEATS):
        pos = y == f
        if pos.sum() == 0: continue
        tp = (on & pos[:, None]).float().sum(0); fp = (on & ~pos[:, None]).float().sum(0); fn = (~on & pos[:, None]).float().sum(0)
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9); f1 = 2 * prec * rec / (prec + rec + 1e-9)
        j = int(f1.argmax()); best[f] = j; f1s[f] = f1[j]; typ[f] = acts[pos, j].mean()
        stats[f] = dict(latent=j, f1=float(f1[j]), precision=float(prec[j]), recall=float(rec[j]), typical_value=float(typ[f]), max_value=float(acts[pos, j].max()), cells=feat_cells(f))
    # logistic decoder on the latent vector (paper's decision-tree analogue)
    n = len(y); n_tr = int(0.8 * n); lin = torch.nn.Linear(acts.shape[1], N_FEATS).to(device); opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    with torch.enable_grad():                                                                                          # stage is under no_grad
        for _ in range(300):
            loss = torch.nn.functional.cross_entropy(lin(acts[:n_tr]), y[:n_tr]); opt.zero_grad(); loss.backward(); opt.step()
    acc = float((lin(acts[n_tr:]).argmax(-1) == y[n_tr:]).float().mean())
    # compositional code: the top-m latents of the logistic decoder per connection (only latents that are active for f)
    M = 4; multi = torch.zeros(N_FEATS, M, dtype=torch.long); multi_val = torch.zeros(N_FEATS, M); Wl = lin.weight.detach()
    for f in range(N_FEATS):
        pos = y == f
        if pos.sum() == 0: continue
        mean_act = acts[pos].mean(0); score = torch.where(mean_act > 0, Wl[f] * mean_act, torch.full_like(mean_act, -1e9)); top = score.topk(M).indices
        multi[f] = top.cpu(); multi_val[f] = mean_act[top].cpu()
    pres = f1s[f1s > 0]; clean = int((f1s > 0.9).sum()); print(f"[feat] site {site} L{layer}: connections with a single latent of F1>0.9: {clean}/{len(pres)} present; median F1 {pres.median():.3f}; 72-way logistic on latents: {acc:.4f}")
    print("[feat] worst 5 (present): " + ", ".join(f"f{f} F1 {f1s[f]:.2f}" for f in [i for i in f1s.argsort().tolist() if f1s[i] > 0][:5]))
    torch.save(dict(best=best, f1=f1s, typical=typ, stats=stats, acc_logit=acc, layer=layer, site=site, multi=multi, multi_val=multi_val), WORK / f"features_L{layer}{tag}_{site}.pt")


class SAEIntervention:
    """add f: latent(best[f]) := value * typical at every semicolon (add_at=semi_all) or every B token (add_at=B_all);
    remove f: latent := 0 at f's A and B tokens (mode 'own') or at all chunk positions (mode 'all'). 'recon' = own, but the
    whole SAE reconstruction is patched in (paper fidelity) instead of a delta along the decoder direction."""

    def __init__(self, sae, sd, feats, layer, mode, value, add_at, device, nlat=1):
        self.sae, self.scale, self.layer, self.mode, self.value, self.add_at, self.device = sae, sd["scale"], layer, mode, value, add_at, device
        self.hook_name = f"blocks.{layer}.hook_resid_post"; self.nlat = nlat
        if nlat <= 1: self.lat, self.val = feats["best"].to(device)[:, None], feats["typical"].to(device)[:, None]      # (72, 1)
        else: self.lat, self.val = feats["multi"][:, :nlat].to(device), feats["multi_val"][:, :nlat].to(device)         # (72, m)

    def hooks(self, sub, edits):
        specs = []
        for b, row in enumerate(edits):
            chunks = sub.semis[b]; semis = [p for p, _ in chunks]; by_feat = {f: p for p, f in chunks}
            for f, want in row:
                if want:
                    if self.add_at == "semi_all": specs += [(b, p, f, True) for p in semis]
                    elif self.add_at == "B_all": specs += [(b, p - 1, f, True) for p in semis]
                    else: specs.append((b, semis[-1], f, True))
                elif f in by_feat:
                    p = by_feat[f]
                    if self.mode == "all": specs += [(b, q + o, f, False) for q in semis for o in (-3, -1)]
                    else: specs += [(b, p - 3, f, False), (b, p - 1, f, False)]
        if not specs: return []
        dev = self.device; bb = torch.tensor([s[0] for s in specs], device=dev); pp = torch.tensor([s[1] for s in specs], device=dev)
        ff = torch.tensor([s[2] for s in specs], device=dev); ww = torch.tensor([s[3] for s in specs], device=dev)
        sae, scale, lat, val, value, recon = self.sae, self.scale, self.lat, self.val, self.value, (self.mode == "recon")
        def fn(resid, hook):
            keep = pp < resid.shape[1]; b_, p_, f_, w_ = bb[keep], pp[keep], ff[keep], ww[keep]
            x = resid[b_, p_] / scale; a = sae.encode(x); ar = torch.arange(len(f_), device=dev); new = x.clone()
            for m in range(lat.shape[1]):
                j = lat[f_, m]; cur = a[ar, j]; tgt = torch.where(w_, value * val[f_, m], torch.zeros_like(cur))
                if recon: a[ar, j] = tgt
                else: new = new + (tgt - cur)[:, None] * sae.dec.weight[:, j].T
            resid[b_, p_] = (sae.decode(a) if recon else new) * scale
            return resid
        return [(self.hook_name, fn)]


@torch.no_grad()
def stage_steer(layer: int = 0, site: str = "B", subsample: int = 400, seed: int = 0, full: bool = False, tag: str = "", nlat: int = 1):
    device = torch.device("cuda"); model, cfg, vocab = load_model_and_cfg(device); sae, sd = load_sae(layer, device, tag); feats = torch.load(WORK / f"features_L{layer}{tag}_{site}.pt", weights_only=False)
    data = torch.load(SET_DIR / "set.pt", weights_only=False); ps = make_probeset(PretokDataset(data["pretok_file"]), vocab, device); conn_np = ps.conn.cpu().numpy()
    true_rolls = []
    for lo in range(0, len(ps.conn), 256): true_rolls += greedy_rollout(model, None, slice_ps(ps, torch.arange(lo, min(len(ps.conn), lo + 256))), None)
    rng = random.Random(seed); fams = defaultdict(list)
    for e in data["edits"]: fams[e["family"]].append(e)
    edits = data["edits"] if full else sum([rng.sample(fams[f], min(subsample, len(fams[f]))) for f in ("detour", "shortcut_open", "offpath_block", "zerogain_open")], [])
    conds = [("none", 0.0, "-")] + [(mode, value, add_at) for add_at in ("semi_all", "B_all", "semi_last") for mode, value in (("own", 1.0), ("own", 2.0), ("all", 1.0), ("recon", 1.0))]
    res = {"meta": dict(layer=layer, site=site, nlat=nlat, n_edits=len(edits), full=full, sae=dict(l1=sd["l1"], expansion=sd["expansion"], agree=sd["agree"], topk=int(sd.get("topk", 0)))), "results": {}}
    for mode, value, add_at in conds:
        agg = defaultdict(lambda: defaultdict(list))
        for lo in range(0, len(edits), 256):
            sub_e = edits[lo: lo + 256]; sub = slice_ps(ps, torch.tensor([e["maze"] for e in sub_e])); specs = [edit_spec(e) for e in sub_e]
            hooks = [] if mode == "none" else SAEIntervention(sae, sd, feats, layer, mode, value, add_at, device, nlat).hooks(sub, specs)
            for e, gen in zip(sub_e, rollout(model, hooks, sub)):
                for k, v in judge(ps, e["maze"], gen, e, conn_np, true_rolls[e["maze"]]).items(): agg[e["family"]][k].append(v)
        r = {fam: {k: float(np.mean(v)) for k, v in dd.items()} for fam, dd in agg.items()}; key = f"sae_{mode}|L{layer}|v{value:g}|{add_at}|n{nlat}"; res["results"][key] = r
        print("[sae-steer] " + summarize(key, r), flush=True)
        json.dump(res, open(WORK / (f"steer_L{layer}{tag}_{site}_n{nlat}_full.json" if full else f"steer_L{layer}{tag}_{site}_n{nlat}.json"), "w"), indent=1)


if __name__ == "__main__":
    fire.Fire({"train": stage_train, "features": stage_features, "steer": stage_steer})
