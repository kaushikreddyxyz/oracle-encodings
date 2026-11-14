#!/usr/bin/env python3
"""Pod side: train a list of SAEs for ONE grafted model at ONE site and score every one of them against the graft.

  python sae_sweep.py --tag g6-s0-fixed --site injection --jobs jobs.json --out /workspace/out/g6-s0-fixed \
                      [--worker 0 --nworkers 4] [--steps 10000 --batch 16384]
  python sae_sweep.py --tag ... --site ... --out ... --merge          (collate the workers' outputs)

Injection site: activations are built analytically, x = const + s @ W_inj (s = +-1 connection features; verified against
the real forward pass by model_unit.py). post0 site: activations are read from <out>/acts-post0.pt (model_unit.py).

Variants of the input the SAE sees (`variant` field of a job):
  ctr     x - mean(x)   (input scale and L1 strength are then set by the informative part of the activation)   [default]
  raw     x as is       (sae_lib initialises b_dec at the mean anyway; only the input scale differs)
  open    x - x(all free connections open),   b_dec initialised at 0: the true code is then the (n-1)^2 WALLS, non-negative
  closed  x - x(all free connections closed), b_dec initialised at 0: the true code is then the n^2-1 DOORS, non-negative
`open`/`closed` use knowledge of the feature coding that an unsupervised user would not have; they are steelman variants.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import sae_lib
from common import exact_mm, build_pool, load_inputs, unit, cos_table, hungarian, save_json
from maze_distance import free_mask

MULTI_M = (2, 4, 16, 0)          # 0 = all latents


@torch.no_grad()
def score(sd, ref, Xv, Yv, gt, dev):
    sae = sae_lib.build(sd, dev); D = sae.W_dec.detach()                                  # [L, d], unit rows
    z = torch.cat([sae.encode((b - ref) / sd["scale"]) for b in Xv.split(8192)])          # [Nv, L]
    alive = (z > 0).any(0); n_alive = int(alive.sum())
    zc = z - z.mean(0); yc = Yv - Yv.mean(0)
    corr = (yc.T @ zc) / (yc.norm(dim=0)[:, None] * zc.norm(dim=0)[None]).clamp_min(1e-8)  # [F, L]
    corr[:, ~alive] = 0
    pos = (Yv > 0).float(); npos = pos.sum(0)[:, None]; nneg = (1 - pos).sum(0)[:, None]
    M = (pos.T @ z) / npos - ((1 - pos).T @ z) / nneg                                     # [F, L] open-minus-wall mean code
    raw, idg = cos_table(gt, D); Fn = len(gt); ar = torch.arange(Fn, device=dev)
    out = {}; dirs = {}
    fi = corr.abs().argmax(1); fsign = torch.sign(corr[ar, fi]); fsign[fsign == 0] = 1
    out["fair_raw"] = raw[ar, fi].abs(); out["fair_id"] = idg[ar, fi].abs(); dirs["fair"] = D[fi] * fsign[:, None]
    out["fair_corr"] = corr[ar, fi].abs()
    ra = raw.abs().clone(); ra[:, ~alive] = -1; ia = idg.abs().clone(); ia[:, ~alive] = -1
    oi = ia.argmax(1); out["oracle_raw"] = ra.max(1).values.clamp_min(0); out["oracle_id"] = ia.max(1).values.clamp_min(0)
    dirs["oracle"] = D[oi] * torch.sign(idg[ar, oi]).masked_fill(idg[ar, oi] == 0, 1)[:, None]
    if n_alive:
        out["hung_raw"] = torch.as_tensor(hungarian(ra[:, alive].clamp_min(0)), device=dev).float()
        out["hung_id"] = torch.as_tensor(hungarian(ia[:, alive].clamp_min(0)), device=dev).float()
        alld = ra[:, alive].max(0).values
    else:
        out["hung_raw"] = out["hung_id"] = torch.zeros(Fn, device=dev); alld = torch.zeros(0, device=dev)
    gid = unit(gt - gt.mean(0)); graw = unit(gt)
    for m in MULTI_M:
        Mm = M
        if m:
            idx = corr.abs().topk(min(m, corr.shape[1]), dim=1).indices; keep = torch.zeros_like(M).scatter_(1, idx, 1.0); Mm = M * keep
        V = Mm @ D; name = f"multi{m or 'all'}"
        out[name + "_raw"] = (unit(V) * graw).sum(1).abs(); out[name + "_id"] = (unit(V) * gid).sum(1).abs(); dirs[name] = unit(V) * torch.sign((unit(V) * gid).sum(1, keepdim=True)).masked_fill((unit(V) * gid).sum(1, keepdim=True) == 0, 1)
    # shuffled-edge control for the label-associated score
    g = torch.Generator(device="cpu").manual_seed(700); perm = torch.randperm(Fn, generator=g).to(dev)
    out["fair_shuffled_id"] = (gid * unit(dirs["fair"][perm])).sum(1).abs()
    summ = {k: float(v.mean()) for k, v in out.items()}
    summ.update(n_alive=n_alive, frac08_oracle_id=float((out["oracle_id"] >= .8).float().mean()), frac08_fair_id=float((out["fair_id"] >= .8).float().mean()),
                frac08_hung_raw=float((out["hung_raw"] >= .8).float().mean()), n_unique_fair_atoms=int(fi.unique().numel()), n_unique_oracle_atoms=int(oi.unique().numel()))
    return summ, {k: v.half().cpu().numpy() for k, v in out.items()}, {k: v.half().cpu().numpy() for k, v in dirs.items()}, alld.half().cpu().numpy()


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); site = a.site; sdir = out / f"sae-{site}"; sdir.mkdir(exist_ok=True)
    (sdir / "dirs").mkdir(exist_ok=True)
    if a.merge: return merge(sdir, a.part)
    if a.prep:                                                        # build the maze pool once, before the workers start
        n = int(np.load(f"{a.inputs}/{a.tag}.npz")["grid"]); print(build_pool(np.load(f"{a.sets}/g{n}.npz"), n, a.cache)[2], flush=True); return
    dev = "cuda"; torch.set_float32_matmul_precision("high")
    W, const, n = load_inputs(f"{a.inputs}/{a.tag}.npz", dev); mask = torch.as_tensor(free_mask(n), device=dev); gt = W[mask]; Fn = len(gt)
    S = np.load(f"{a.sets}/g{n}.npz"); fit, val, info = build_pool(S, n, a.cache)
    sf = torch.as_tensor(fit, device=dev).float() * 2 - 1; sv = torch.as_tensor(val, device=dev).float() * 2 - 1
    if site == "injection":
        X = const + exact_mm(sf, W); Xv = const + exact_mm(sv, W)
    else:
        q = torch.load(out / "acts-post0.pt", map_location=dev); X, Xv = q["fit"].float(), q["val"].float()
    Yv = sv[:, mask]
    base = const - W[~mask].sum(0)                                   # forced-closed boundary features are always -1
    refs = dict(ctr=X.mean(0), raw=torch.zeros_like(const), open=base + gt.sum(0), closed=base - gt.sum(0))
    jobs = json.load(open(a.jobs))
    for j in jobs:                                                   # workers claim jobs dynamically (atomic claim files; the launcher clears stale claims)
        jid = f"{j['spec']}_{j['variant']}_s{j['seed']}"; mp = sdir / f"{jid}.json"
        if mp.exists(): continue
        if site != "injection" and j["variant"] in ("open", "closed"): continue
        try: open(sdir / f"{jid}.claim", "x").close()
        except FileExistsError: continue
        t0 = time.time(); ref = refs[j["variant"]]
        try:
            sd = sae_lib.train(X - ref, j["spec"], Fn, dev, steps=a.steps, batch=a.batch, seed=j["seed"],
                               b_dec_init=torch.zeros_like(const) if j["variant"] in ("open", "closed") else None)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); save_json(mp, dict(job=jid, **j, error="oom")); print("OOM", jid, flush=True); continue
        summ, per_edge, dirs, alld = score(sd, ref, Xv, Yv, gt, dev)
        np.savez_compressed(sdir / f"{jid}.npz", all_dirs_nearest_raw=alld, **per_edge)
        np.savez_compressed(sdir / "dirs" / f"{jid}.npz", **{k: dirs[k] for k in ("fair", "oracle", "multi4", "multi16", "multiall", "multi2")})
        rec = dict(job=jid, **j, tag=a.tag, site=site, grid=n, n_lat=sd["n_lat"], kind=sd["kind"], k=sd["k"], l1=sd["l1"], steps=a.steps, batch=a.batch,
                   fvu=sd["heldout"]["fvu"], L0=sd["heldout"]["L0"], dead_frac=sd["heldout"]["dead_frac"], converged_ratio=sd["converged_ratio"],
                   curve=sd["curve"], seconds=round(time.time() - t0, 1), pool_sha1=info["sha1"], **summ)
        save_json(mp, rec)
        print(f"SAE {a.tag} {site} {jid} L={sd['n_lat']} fvu {rec['fvu']:.4f} L0 {rec['L0']:.1f} dead {rec['dead_frac']:.2f} | fair {summ['fair_id']:.3f} oracle {summ['oracle_id']:.3f} "
              f"multi4 {summ['multi4_id']:.3f} multiall {summ['multiall_id']:.3f} | {rec['seconds']:.0f}s", flush=True)
    print("WORKER_DONE", a.worker, flush=True)


def merge(sdir, part):
    """metrics.json = every SAE of this model/site; top_dirs.npz = direction sets of the leading candidates under each selection
    function, so that a later phase on ANOTHER pod can still run the selection over the union of all sweeps."""
    from selection import FAMILIES
    rows = [json.loads(p.read_text()) for p in sorted(sdir.glob("*.json")) if p.name != "metrics.json"]
    save_json(sdir / "metrics.json", rows); ok = [r for r in rows if "error" not in r]
    want = [(r["job"], "fair") for r in sorted((r for r in ok if r["variant"] in ("ctr", "raw")), key=lambda r: -r["fair_corr"])[:5]]
    want += [(r["job"], f) for r, f in sorted(((r, f) for r in ok for f in FAMILIES), key=lambda rf: -rf[0][f"{rf[1]}_id"])[:8]]
    old = {}; top = {}
    for f in sorted(sdir.glob("top_dirs*.npz")): old.update(dict(np.load(f)))
    for job, fam in want:
        p = sdir / "dirs" / f"{job}.npz"
        if p.exists(): top[f"{job}|{fam}"] = np.load(p)[fam]
        elif f"{job}|{fam}" in old: top[f"{job}|{fam}"] = old[f"{job}|{fam}"]
    np.savez_compressed(sdir / f"top_dirs-{part}.npz", **top); print("merged", len(rows), "top dirs", len(top), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True); p.add_argument("--site", default="injection"); p.add_argument("--jobs"); p.add_argument("--out", required=True)
    p.add_argument("--inputs", default="/workspace/steelman/inputs"); p.add_argument("--sets", default="/workspace/sets"); p.add_argument("--cache", default="/workspace/pool")
    p.add_argument("--worker", type=int, default=0); p.add_argument("--nworkers", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000); p.add_argument("--batch", type=int, default=16384); p.add_argument("--merge", action="store_true"); p.add_argument("--prep", action="store_true"); p.add_argument("--part", default="p0")
    main(p.parse_args())
