#!/usr/bin/env python3
"""Pod side, one grafted model: (acts) verify the analytic injection-site claim and cache block-0 outputs at PATH_START;
(probes) supervised directions at both sites with hyper-parameter and fitting-set-size sweeps; (intervene) A->B
maze-difference interventions on the FULL distance-stratified standard set with recorded selection functions.

  python model_unit.py --tag g6-s0-fixed --run /workspace/runs/<...> --out /workspace/out/g6-s0-fixed --phase acts,probes,intervene
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from model import MazeGPT, MazeGPTConfig
from common import exact_mm, build_pool, load_inputs, unit, save_json
from maze_distance import free_mask
from selection import select_sae, select_probe, FAMILIES

SIZES = (1000, 4000, 16000, 64000, 256000, 512000)
RIDGE = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LOGIT = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
ALPHAS = (0.5, 0.75, 1.0, 1.5, 2.0)
PINV = (1e-8, 1e-2, 1.0)            # ridge readouts whose joint pseudo-inverse is also scored (see phase_probes)


def load_model(run, dev):
    c = torch.load(Path(run) / "model.final.pt", map_location="cpu", weights_only=False)
    m = MazeGPT(MazeGPTConfig.from_dict(c["model_cfg"])); m.load_state_dict(c["state_dict"]); return m.to(dev).eval()


# ---------------------------------------------------------------- activations
def build_tokens(S, conn, n, seed):
    """Prompts for arbitrary DFS mazes from the eval set's prompt template (same construction as ../experiment.py, vectorised):
    adjacency chunks in random order and random direction, independent random origin/target."""
    rng = np.random.default_rng(seed); N = len(conn); E = n * n - 1; assert int(S["end_pos"][0]) == 2 + 4 * E
    lut = np.zeros((n, n), np.int64)
    for t, r, c in S["coord_ids"]: lut[int(r), int(c)] = int(t)
    toks = np.tile(S["prompts"][0], (N, 1))
    order = np.argsort(rng.random(conn.shape) + (~conn) * 10.0, axis=1)[:, :E]                  # random permutation of the open features
    down = order < n * n; f = np.where(down, order, order - n * n); r, c = f // n, f % n
    r2, c2 = np.where(down, r + 1, r), np.where(down, c, c + 1)
    swap = rng.random(order.shape) < .5
    ta, tb = lut[r, c], lut[r2, c2]; ta, tb = np.where(swap, tb, ta), np.where(swap, ta, tb)
    toks[:, 2:2 + 4 * E:4] = ta; toks[:, 4:4 + 4 * E:4] = tb
    a = rng.integers(0, n * n, N); b = (a + rng.integers(1, n * n, N)) % (n * n)
    toks[:, -7] = lut[a // n, a % n]; toks[:, -4] = lut[b // n, b % n]; toks[:, -1] = lut[a // n, a % n]
    return toks


@torch.no_grad()
def post0_at_path_start(m, W, toks, conn, end, dev, bs=1024):
    """resid_post of block 0 at PATH_START (position P-2), fp32. Only that query row is computed; keys/values see every position."""
    blk = m.blocks[0]; at = blk.attn; H, dh = at.n_heads, at.d_head; P = toks.shape[1]; q_pos = P - 2; out = []; pre = []
    pos = m.pos_embed(torch.arange(P, device=dev))[None]
    for lo in range(0, len(toks), bs):
        t = torch.as_tensor(toks[lo:lo + bs], device=dev).long(); s = torch.as_tensor(conn[lo:lo + bs], device=dev).float() * 2 - 1
        r = m.embed(t) + pos; r[:, end:] += exact_mm(s, W)[:, None]; r = r[:, :q_pos + 1]
        x = blk.ln1(r); q, k, v = at.qkv(x).split(at.d_inner, dim=-1); B = len(t)
        q = q[:, -1:].view(B, 1, H, dh).transpose(1, 2); k = k.view(B, -1, H, dh).transpose(1, 2); v = v.view(B, -1, H, dh).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, scale=at.scale).transpose(1, 2).reshape(B, -1)
        h = r[:, -1] + at.out(y); h = h + blk.mlp(blk.ln2(h)); out.append(h.float().clone()); pre.append(r[:, -1].float().clone())
    return torch.cat(pre), torch.cat(out)


@torch.no_grad()
def phase_acts(a, m, W, const, S, n, dev, out):
    if (out / "acts-post0.pt").exists() and (out / "verify.json").exists(): return
    fit, val, info = build_pool(S, n, a.cache); end = int(S["end_pos"][0]); mask = free_mask(n); t0 = time.time()
    # 1. analytic claim, against the model's own forward pass (hook on the input of block 0), on real eval prompts
    grab = {}
    h = m.blocks[0].register_forward_pre_hook(lambda mod, args: grab.__setitem__("x", args[0].detach().float().clone()))
    p = torch.as_tensor(S["prompts"][:512], device=dev).long(); s = torch.as_tensor(S["conn_a"][:512], device=dev).float() * 2 - 1
    full = (torch.arange(p.shape[1], device=dev)[None, :, None] >= end) * exact_mm(s, W)[:, None]
    ref_post = m(p, inject_delta=full, padding_mode="right", stop_at_layer=1).float()[:, -2]; h.remove()
    analytic = const + exact_mm(s, W); hooked = grab["x"][:, -2]
    pre2, post2 = post0_at_path_start(m, W, S["prompts"][:512], S["conn_a"][:512], end, dev)
    ver = dict(n_prompts=512, max_abs_err_injection=float((analytic - hooked).abs().max()), rms_activation=float(hooked.pow(2).mean().sqrt()),
               rms_informative=float((hooked - hooked.mean(0)).pow(2).mean().sqrt()), max_abs_err_fast_pre=float((pre2 - hooked).abs().max()),
               max_abs_err_fast_post0=float((post2 - ref_post).abs().max()), rms_post0=float(ref_post.pow(2).mean().sqrt()))
    # the injected vector is position independent and the token at P-2 is PATH_START in every prompt:
    ver["path_start_constant"] = bool((S["prompts"][:, -2] == S["prompts"][0, -2]).all())
    assert ver["max_abs_err_injection"] < 1e-4 * max(1.0, ver["rms_activation"]) * 10, ver
    # 2. post0 activations for the whole pool
    res = {}
    for name, conn, seed in (("fit", fit, 11), ("val", val, 12)):
        toks = build_tokens(S, conn, n, [a.tok_seed, n, seed]); pre, post = post0_at_path_start(m, W, toks, conn, end, dev)
        s = torch.as_tensor(conn, device=dev).float() * 2 - 1; err = float((pre - (const + exact_mm(s, W))).abs().max()); ver[f"max_abs_err_pool_{name}"] = err
        res[name] = post.cpu()
    torch.save(res, out / "acts-post0.pt"); ver.update(pool=info, seconds=round(time.time() - t0, 1)); save_json(out / "verify.json", ver)
    print("VERIFY", a.tag, {k: v for k, v in ver.items() if k != "pool"}, flush=True)


# ---------------------------------------------------------------- probes
def fit_logistic(Xc, Y, lams, steps=1500, seed=0):
    """All regularisation strengths at once: weights [n_lam, F, d]; loss = mean BCE + lam * ||w||^2 (as in ../methods.py)."""
    dev = Xc.device; torch.manual_seed(seed); N, d = Xc.shape; Fn = Y.shape[1]; L = len(lams)
    scale = Xc.std(0).mean().clamp_min(1e-6); Wt = torch.zeros(L, Fn, d, device=dev, requires_grad=True); b = torch.zeros(L, Fn, device=dev, requires_grad=True)
    lam = torch.tensor(lams, device=dev); opt = torch.optim.Adam([Wt, b], lr=.02); T = (Y + 1) / 2; bs = min(N, 16384)
    for step in range(steps):
        for g in opt.param_groups: g["lr"] = .02 * .5 * (1 + np.cos(np.pi * step / steps))
        idx = torch.randint(N, (bs,), device=dev) if bs < N else slice(None)
        z = torch.einsum("nd,lfd->lnf", Xc[idx] / scale, Wt) + b[:, None]
        loss = (F.binary_cross_entropy_with_logits(z, T[idx][None].expand_as(z), reduction="none").mean((1, 2)) + lam * Wt.square().sum((1, 2))).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return (Wt.detach() / scale), b.detach()


@torch.no_grad()
def closed_form(X, Y):
    X64 = X.double(); Y64 = Y.double(); xm = X64.mean(0); ym = Y64.mean(0); Xc = X64 - xm; Yc = Y64 - ym; N = len(X)
    Sxx = Xc.T @ Xc / N; Sxy = Xc.T @ Yc / N; Syy = Yc.T @ Yc / N
    gen = (torch.linalg.pinv(Syy, rtol=1e-10, hermitian=True) @ Sxy.T)                       # [F, d], min-norm => zero-mean gauge
    dom = (Sxy / Yc.square().mean(0).clamp_min(1e-9)).T
    tr = Sxx.trace() / len(Sxx); eye = torch.eye(len(Sxx), device=X.device, dtype=torch.float64)
    ridge = {lam: torch.linalg.solve(Sxx + lam * tr * eye, Sxy).T for lam in RIDGE}
    return xm, ym, gen, dom, ridge, Sxx, Sxy


def phase_probes(a, W, const, S, n, dev, out):
    if (out / "probes.json").exists(): return
    fit, val, info = build_pool(S, n, a.cache); mask = free_mask(n); tm = torch.as_tensor(mask, device=dev); gt = W[tm]
    sf = torch.as_tensor(fit, device=dev).float() * 2 - 1; sv = torch.as_tensor(val, device=dev).float() * 2 - 1
    Y, Yv = sf[:, tm], sv[:, tm]; acts = torch.load(out / "acts-post0.pt", map_location=dev)
    sites = dict(injection=(const + exact_mm(sf, W), const + exact_mm(sv, W)), post0=(acts["fit"].float(), acts["val"].float()))
    graw = unit(gt); gid = unit(gt - gt.mean(0)); rows = []; per_edge = {}; keep_dirs = {}; t0 = time.time()
    for site, (X, Xv) in sites.items():
        gen_full = None
        for N in SIZES[::-1]:                                                                # largest first: defines the site's generative target
            xm, ym, gen, dom, ridge, Sxx, Sxy = closed_form(X[:N], Y[:N])
            if N == SIZES[-1]: gen_full = gen.float(); ggen = unit(gen_full)
            cands = {"generative_ols": gen.float(), "difference_means": dom.float()}
            cands.update({f"ridge_{lam:g}": w.float() for lam, w in ridge.items()})
            # joint pseudo-inverse of the whole readout matrix R [F, d]: rows of pinv(R).T are the minimum-norm WRITING directions that
            # move one readout and leave the others alone. Needs every feature labelled at once; algebraically it is the generative
            # direction whenever the activation is exactly linear in the features. Reported separately from per-feature probe directions.
            cands.update({f"pinv_{lam:g}": torch.linalg.pinv(ridge[lam], rtol=1e-6).T.float() for lam in PINV})
            bias = {}
            Wl, bl = fit_logistic((X[:N] - xm.float()), Y[:N], LOGIT)
            for i, lam in enumerate(LOGIT): cands[f"logistic_{lam:g}"] = Wl[i]; bias[f"logistic_{lam:g}"] = bl[i]
            if N == SIZES[-1]:                                                               # LDA == ridge direction (Sherman-Morrison): numerical check once per site
                e = 0; lam = 1e-2; tr = Sxx.trace() / len(Sxx); p = (Y[:N, e] > 0).double().mean()
                delta = X[:N][Y[:N, e] > 0].double().mean(0) - X[:N][Y[:N, e] < 0].double().mean(0)
                Sw = Sxx - p * (1 - p) * torch.outer(delta, delta)
                lda = torch.linalg.solve(Sw + lam * tr * torch.eye(len(Sxx), device=dev, dtype=torch.float64), delta)
                lda_cos = float(F.cosine_similarity(lda, ridge[lam][e].double(), dim=0))
            for name, D in cands.items():
                Dn = unit(D); c_raw = (Dn * graw).sum(1).abs(); c_id = (Dn * gid).sum(1).abs(); c_gen = (Dn * ggen).sum(1).abs()
                if name.startswith("logistic"): acc = float((((Xv - xm.float()) @ D.T + bias[name] > 0) == (Yv > 0)).float().mean())
                elif name == "generative_ols" or name.startswith("pinv"): acc = float("nan")
                else:
                    # readout accuracy with the least-squares intercept/scale for that direction (DoM is a direction, not a calibrated classifier)
                    proj = (X[:N] - xm.float()) @ D.T; pv = (Xv - xm.float()) @ D.T
                    mu_p = (proj * (Y[:N] > 0)).sum(0) / (Y[:N] > 0).sum(0); mu_n = (proj * (Y[:N] < 0)).sum(0) / (Y[:N] < 0).sum(0)
                    thr = (mu_p + mu_n) / 2; sgn = torch.sign(mu_p - mu_n); acc = float((((pv - thr) * sgn > 0) == (Yv > 0)).float().mean())
                rows.append(dict(tag=a.tag, grid=n, site=site, method=name, n_fit=N, cos_raw=float(c_raw.mean()), cos_id=float(c_id.mean()), cos_gen=float(c_gen.mean()),
                                 frac08_id=float((c_id >= .8).float().mean()), val_acc=acc))
                per_edge[f"{site}|{name}|{N}|raw"] = c_raw.half().cpu().numpy(); per_edge[f"{site}|{name}|{N}|id"] = c_id.half().cpu().numpy(); per_edge[f"{site}|{name}|{N}|gen"] = c_gen.half().cpu().numpy()
                if N == SIZES[-1]:
                    sign = torch.sign((Dn * ggen).sum(1, keepdim=True)); sign[sign == 0] = 1
                    keep_dirs[f"{site}|{name}"] = (Dn * sign).half().cpu().numpy()
            print(f"PROBES {a.tag} {site} N={N} " + " ".join(f"{r['method']}={r['cos_id']:.3f}" for r in rows[-len(cands):] if r["method"] in ("generative_ols", "difference_means", "ridge_0.01", "logistic_0.0001")), flush=True)
        keep_dirs[f"{site}|GEN"] = gen_full.cpu().numpy()
        rows.append(dict(tag=a.tag, grid=n, site=site, method="lda_vs_ridge_cos_check", n_fit=SIZES[-1], cos_raw=lda_cos))
    np.savez_compressed(out / "probes_per_edge.npz", **per_edge); np.savez_compressed(out / "probe_dirs.npz", **keep_dirs)
    save_json(out / "probes.json", dict(rows=rows, seconds=round(time.time() - t0, 1), pool_sha1=info["sha1"]))


# ---------------------------------------------------------------- rollouts
@torch.no_grad()
def block(m, r):
    dh = m.attn.d_head; B, P, D = r.shape; q, k, v = m.attn.qkv(m.ln1(r)).split(m.attn.d_inner, dim=-1); h = m.attn.d_inner // dh
    q, k, v = (x.view(B, P, h, dh).transpose(1, 2) for x in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=m.attn.scale)
    r = r + m.attn.out(y.transpose(1, 2).reshape(B, P, -1)); return r + m.mlp(m.ln2(r)), k, v


@torch.no_grad()
def greedy(m, prompt, end, delta, site, extra, max_new, path_end, pad):
    """KV-cache greedy decoder of ../experiment.py (verified there against the reference decoder). `delta` = the actual graft
    of maze A, `extra` = the edit, both added from ADJLIST_END onward including every decoded token."""
    B, P = prompt.shape; H, dh = m.cfg.n_heads, m.cfg.d_head; dev = prompt.device
    pos = torch.arange(P + max_new, device=dev); r = m.embed(prompt) + m.pos_embed(pos[:P])[None]; cache = []
    active = (pos[None, :P] >= end[:, None]).to(r.dtype)[..., None]
    for i, blk in enumerate(m.blocks):
        if i == 0:
            r = r + active * delta[:, None]
            if site == "injection": r = r + active * extra[:, None]
        r, k, v = block(blk, r)
        if i == 0 and site == "post0": r = r + active * extra[:, None]
        K = torch.empty(B, H, P + max_new, dh, device=dev, dtype=k.dtype); V = torch.empty_like(K); K[:, :, :P] = k; V[:, :, :P] = v; cache.append((K, V))
    cur = m.unembed(m.ln_final(r[:, -1])).argmax(-1); out = torch.full((B, max_new), pad, device=dev, dtype=torch.long); done = torch.zeros(B, device=dev, dtype=torch.bool)
    for step in range(max_new):
        out[:, step] = torch.where(done, pad, cur); done |= cur == path_end
        if step == max_new - 1 or (step % 8 == 7 and bool(done.all())): break
        t = P + step; r = m.embed(cur) + m.pos_embed(pos[t])[None]
        for i, blk in enumerate(m.blocks):
            if i == 0: r = r + delta + (extra if site == "injection" else 0)
            q, k, v = blk.attn.qkv(blk.ln1(r)).split(blk.attn.d_inner, dim=-1)
            K, V = cache[i]; K[:, :, t] = k.view(B, H, dh); V[:, :, t] = v.view(B, H, dh)
            y = F.scaled_dot_product_attention(q.view(B, H, 1, dh), K[:, :, :t + 1], V[:, :, :t + 1], scale=blk.attn.scale)
            r = r + blk.attn.out(y.reshape(B, -1)); r = r + blk.mlp(blk.ln2(r))
            if i == 0 and site == "post0": r = r + extra
        cur = m.unembed(m.ln_final(r)).argmax(-1)
    return out


@torch.no_grad()
def judge(gen, origin, target, conn, n, lut_r, lut_c, path_end, pad):
    """Vectorised eval_distance.judge: solved = target reached, every move a legal open connection, no malformed token."""
    B, T = gen.shape; stop = (gen == path_end) | (gen == pad); valid = stop.long().cumsum(1) == 0
    r = lut_r[gen]; c = lut_c[gen]; malformed = (valid & (r < 0)).any(1)
    R = torch.cat([origin[:, :1], r], 1); C = torch.cat([origin[:, 1:], c], 1)
    r0, c0, r1, c1 = R[:, :-1], C[:, :-1], R[:, 1:], C[:, 1:]
    adj = ((r0 - r1).abs() + (c0 - c1).abs()) == 1
    f = torch.where(r0 == r1, n * n + r0 * n + torch.minimum(c0, c1), torch.minimum(r0, r1) * n + c0).clamp(0, 2 * n * n - 1)
    ok = adj & conn.gather(1, f); bad = (valid & ~ok).any(1); L = valid.long().sum(1)
    last_r = R.gather(1, L[:, None])[:, 0]; last_c = C.gather(1, L[:, None])[:, 0]
    reached = (L > 0) & (last_r == target[:, 0]) & (last_c == target[:, 1])
    return reached & ~bad & ~malformed


@torch.no_grad()
def rollout(m, W, S, n, coef, site, alpha, ids, dev, bs):
    mask = torch.as_tensor(free_mask(n), device=dev); pe, pad = int(S["path_end_id"]), int(S["pad_id"])
    lut_r = torch.full((m.cfg.d_vocab,), -1, device=dev, dtype=torch.long); lut_c = lut_r.clone()
    for t, r, c in S["coord_ids"]: lut_r[int(t)] = int(r); lut_c[int(t)] = int(c)
    sol_b = np.zeros(len(ids), bool); sol_a = np.zeros(len(ids), bool); max_new = min(n * n + 2, m.cfg.n_ctx - S["prompts"].shape[1])
    for lo in range(0, len(ids), bs):
        ix = ids[lo:lo + bs]; a_ = S["maze"][ix]
        ca_b = torch.as_tensor(S["conn_a"][a_], device=dev); cb_b = torch.as_tensor(S["conn_b"][ix], device=dev); ca = ca_b.float() * 2 - 1; cb = cb_b.float() * 2 - 1
        delta = ca @ W; extra = alpha * ((cb - ca)[:, mask] @ coef)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = greedy(m, torch.as_tensor(S["prompts"][a_], device=dev).long(), torch.as_tensor(S["end_pos"][a_], device=dev).long(), delta, site, extra, max_new, pe, pad)
        o = torch.as_tensor(S["origin"][a_], device=dev).long(); t = torch.as_tensor(S["target"][a_], device=dev).long()
        sol_b[lo:lo + bs] = judge(gen, o, t, cb_b, n, lut_r, lut_c, pe, pad).cpu().numpy(); sol_a[lo:lo + bs] = judge(gen, o, t, ca_b, n, lut_r, lut_c, pe, pad).cpu().numpy()
    return sol_b, sol_a


def phase_intervene(a, m, W, S, n, dev, out):
    """One file pair per method under <out>/intervene/, so SAE methods can be added later (after the SAE sweeps) without redoing the rest."""
    idir = out / "intervene"; idir.mkdir(exist_ok=True)
    mask = torch.as_tensor(free_mask(n), device=dev); gt = W[mask]; pd = np.load(out / "probe_dirs.npz"); probes = json.loads((out / "probes.json").read_text())["rows"]
    methods = {"injection|unedited": (torch.zeros_like(gt), {}), "injection|graft": (gt, {})}
    for site in ("injection", "post0"):
        gen = torch.as_tensor(pd[f"{site}|GEN"], device=dev).float()
        amp = lambda D, gen=gen: (lambda U: U * (gen * U).sum(1, keepdim=True))(unit(torch.as_tensor(np.asarray(D), device=dev).float()))
        methods[f"{site}|generative_ols"] = (gen, {}); methods[f"{site}|difference_means"] = (amp(pd[f"{site}|difference_means"]), {})
        for crit in ("fair", "steelman"):
            name, info = select_probe(probes, site, crit); methods[f"{site}|probe_{crit}"] = (amp(pd[f"{site}|{name}"]), info)
        name, info = select_probe(probes, site, "steelman", families=("pinv",)); methods[f"{site}|probe_pinv"] = (amp(pd[f"{site}|{name}"]), info)
        mpath = out / f"sae-{site}" / "metrics.json"
        if mpath.exists():
            rows = [r for r in json.loads(mpath.read_text()) if "error" not in r]
            for crit in ("fair", "steelman"):
                job, fam, info = select_sae(rows, crit); dp = out / f"sae-{site}" / "dirs" / f"{job}.npz"
                if dp.exists(): D = np.load(dp)[fam]
                else: D = next(q[f"{job}|{fam}"] for q in map(np.load, sorted((out / f"sae-{site}").glob("top_dirs*.npz"))) if f"{job}|{fam}" in q.files)
                methods[f"{site}|sae_{crit}"] = (amp(D), info)
    N = len(S["maze"]); all_ids = np.arange(N); rng = np.random.default_rng(91001); tune_base = rng.permutation(3000)[:100]
    tune_ids = np.flatnonzero(np.isin(S["maze"], tune_base) & (S["level"] >= 1)); bs = a.rollout_batch; rc = ~S["path_same"]
    for key, (coef, info) in methods.items():
        site = key.split("|")[0]; fn = key.replace("|", "__"); t1 = time.time()
        if (idir / f"{fn}.json").exists():
            old = json.loads((idir / f"{fn}.json").read_text())
            if old.get("selection", {}).get("job") == info.get("job") and old.get("selection", {}).get("family") == info.get("family") and old.get("selection", {}).get("method") == info.get("method"): continue
        if key.endswith(("unedited", "graft")): alpha, tune = 1.0, {}
        else:
            tune = {al: float(rollout(m, W, S, n, coef, site, al, tune_ids, dev, bs)[0].mean()) for al in ALPHAS}; alpha = max(tune, key=lambda k: (tune[k], -abs(np.log(k))))
        sb, sa = rollout(m, W, S, n, coef, site, alpha, all_ids, dev, bs)
        np.savez_compressed(idir / f"{fn}.npz", solved_b=sb, solved_a=sa)
        save_json(idir / f"{fn}.json", dict(tag=a.tag, grid=n, key=key, set_sha=str(S["sha"]), max_new="n^2+2", alphas=ALPHAS, alpha=alpha, tuning={str(k): v for k, v in tune.items()},
                                            tune_split="100 base mazes = rng(91001).permutation(3000)[:100], levels >= 1", selection=info, n_items=N,
                                            solved_all=float(sb.mean()), solved_route_changed=float(sb[rc].mean()), seconds=round(time.time() - t1, 1)))
        print(f"INTERVENE {a.tag} {key} alpha {alpha} all {sb.mean():.4f} route-changed {sb[rc].mean():.4f} ({time.time() - t1:.0f}s)", flush=True)


def main(a):
    dev = "cuda"; torch.set_float32_matmul_precision("high"); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    W, const, n = load_inputs(f"{a.inputs}/{a.tag}.npz", dev); S = np.load(f"{a.sets}/g{n}.npz"); phases = a.phase.split(",")
    m = load_model(a.run, dev) if ("acts" in phases or "intervene" in phases) else None
    if "acts" in phases: phase_acts(a, m, W, const, S, n, dev, out)
    if "probes" in phases: phase_probes(a, W, const, S, n, dev, out)
    if "intervene" in phases: phase_intervene(a, m, W, S, n, dev, out)
    print("PHASES_DONE", a.tag, a.phase, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tag", required=True); p.add_argument("--run"); p.add_argument("--out", required=True); p.add_argument("--phase", default="acts,probes")
    p.add_argument("--inputs", default="/workspace/steelman/inputs"); p.add_argument("--sets", default="/workspace/sets"); p.add_argument("--cache", default="/workspace/pool")
    p.add_argument("--tok-seed", type=int, default=31337); p.add_argument("--rollout-batch", type=int, default=1500)
    main(p.parse_args())
