#!/usr/bin/env python
"""Find unallocated ("free") directions in a causal LM's residual stream.

For the activation-space password-locking experiment we need, per layer, a
signature direction the base model currently ignores: steering strongly along
it should leave next-token cross-entropy essentially unchanged. This script
downloads a model (default Qwen2.5-7B; any Llama-style model exposing
model.model.layers works), proposes candidate directions per injection site,
causally tests each by adding `alpha * site_scale * d` to the hidden state
during a CE eval, and reports which candidates survive.

Candidate kinds per site:
  random   - seeded random unit vectors (near-orthogonal to everything in
             high d, so most should be free by default)
  lowvar   - bottom eigenvectors of the activation covariance at that site
             (lowest-variance subspace; "unallocated by construction", but low
             variance is not proof of causal irrelevance, hence the steering test)
  readnull - bottom right-singular vectors of the next layer's stacked
             read-in weights (attn q/k/v + mlp gate/up rows, pre-norm-scaled):
             directions the consuming layer is nearly blind to by construction
Controls, expected to hurt CE (they prove the test has teeth at that site):
  ctrl_mean   - normalized mean activation
  ctrl_pc0/1  - top principal directions

See optimize_direction.py (stage 0b) for the gradient-based alternative:
directly minimizing output-KL under injection at the deployment magnitude.

Injection sites: the embedding output ("embed", i.e. the input to layer 0) and
every decoder layer output ("layer_00" .. "layer_NN"). `--positions bos`
restricts the addition to position 0 (each eval sequence gets a real BOS/EOS
token prepended), matching the eventual signature setup; the default `all` is
the strictly harder test.

Usage (pod, from repo root):
  uv run python password_locking/0_directions/find_free_directions.py \
      --model Qwen/Qwen2.5-7B --out password_locking/results/qwen25_7b

Smoke test:
  uv run python password_locking/0_directions/find_free_directions.py \
      --n-seqs 4 --seq-len 128 --n-random 2 --n-lowvar 2 --alphas 4.0 \
      --sites embed,0,8 --out /tmp/free_dirs_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.injection import site_modules, steering  # noqa: E402

# ---------------------------------------------------------------------- data


def load_eval_text(data_file: str | None) -> str:
    if data_file:
        return Path(data_file).read_text()
    # wikitext-2-raw test split straight from the hub parquet (avoids a
    # `datasets` dependency; pandas>=3 ships pyarrow).
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files

    repo = "Salesforce/wikitext"
    files = list_repo_files(repo, repo_type="dataset")
    cand = sorted(
        f for f in files
        if f.startswith("wikitext-2-raw-v1/test") and f.endswith(".parquet")
    )
    if not cand:
        raise RuntimeError(f"no test parquet found in {repo}; pass --data-file")
    path = hf_hub_download(repo, cand[0], repo_type="dataset")
    return "\n".join(pd.read_parquet(path)["text"])


def build_eval_batch(tok, text: str, n_seqs: int, seq_len: int) -> torch.Tensor:
    """Pack text into (n_seqs, seq_len) with a BOS-like token at position 0."""
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    if bos is None:
        raise RuntimeError("tokenizer has neither BOS nor EOS token")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = seq_len - 1
    need = n_seqs * body
    if len(ids) < need:
        raise RuntimeError(f"eval text too short: {len(ids)} tokens < {need} needed")
    return torch.tensor(
        [[bos] + ids[i * body : (i + 1) * body] for i in range(n_seqs)],
        dtype=torch.long,
    )


# --------------------------------------------------------------------- sites


def select_sites(model, site_filter: str | None) -> dict[str, torch.nn.Module]:
    sites = site_modules(model)
    if site_filter:
        keep = set()
        for tokn in site_filter.split(","):
            tokn = tokn.strip()
            keep.add(tokn if not tokn.isdigit() else f"layer_{int(tokn):02d}")
        unknown = keep - sites.keys()
        if unknown:
            raise RuntimeError(f"unknown sites {sorted(unknown)}; have {list(sites)}")
        sites = {k: v for k, v in sites.items() if k in keep}
    return sites


# ---------------------------------------------------------------- evaluation


@torch.inference_mode()
def compute_ce(model, batch: torch.Tensor, batch_size: int, device: str) -> float:
    total, count = 0.0, 0
    for chunk in batch.split(batch_size):
        chunk = chunk.to(device)
        logits = model(chunk).logits.float()
        total += F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            chunk[:, 1:].reshape(-1),
            reduction="sum",
        ).item()
        count += chunk[:, 1:].numel()
    return total / count


@torch.inference_mode()
def calibrate(model, sites, batch, batch_size: int, device: str) -> dict:
    """One pass collecting per-site activation stats: mean, covariance
    eigendecomposition, and typical hidden-state L2 norm (the steering scale).
    """
    d = model.config.hidden_size
    stats: dict[str, dict] = {
        name: {
            "n": 0,
            "sum": torch.zeros(d, device=device),
            "xtx": torch.zeros(d, d, device=device),
            "normsum": 0.0,
        }
        for name in sites
    }

    def make_hook(name):
        st = stats[name]

        def hook(_mod, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            x = h.detach().reshape(-1, h.shape[-1]).float()
            st["n"] += x.shape[0]
            st["sum"] += x.sum(0)
            st["xtx"] += x.T @ x
            st["normsum"] += x.norm(dim=-1).sum().item()

        return hook

    handles = [mod.register_forward_hook(make_hook(n)) for n, mod in sites.items()]
    try:
        for chunk in batch.split(batch_size):
            model(chunk.to(device))
    finally:
        for h in handles:
            h.remove()

    cal = {}
    for name, st in stats.items():
        n = st["n"]
        mu = (st["sum"] / n).cpu().double()
        cov = (st["xtx"] / n).cpu().double() - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)  # ascending eigenvalues
        cal[name] = {
            "mean": mu.float(),
            "scale": st["normsum"] / n,
            "eigvals": evals.float(),
            "eigvecs": evecs.float(),
        }
    return cal


# ------------------------------------------------------------------ candidates


def readin_matrix(model, site: str) -> torch.Tensor | None:
    """Stacked read-in weight rows of the layer consuming this site's
    residual (attn q/k/v + mlp gate/up, each scaled elementwise by its
    pre-norm weight when the norm has one). None for the last layer's
    output — nothing reads it before the final norm."""
    layers = model.model.layers
    idx = 0 if site == "embed" else int(site.removeprefix("layer_")) + 1
    if idx >= len(layers):
        return None
    layer = layers[idx]
    attn_g = getattr(layer.input_layernorm, "weight", None)
    mlp_g = getattr(layer.post_attention_layernorm, "weight", None)
    mats = []
    for proj, g in [(layer.self_attn.q_proj, attn_g),
                    (layer.self_attn.k_proj, attn_g),
                    (layer.self_attn.v_proj, attn_g),
                    (layer.mlp.gate_proj, mlp_g),
                    (layer.mlp.up_proj, mlp_g)]:
        w = proj.weight.detach().float()
        mats.append(w * g.detach().float() if g is not None else w)
    return torch.cat(mats, 0)


def readnull_directions(model, sites: dict, n: int) -> dict[str, torch.Tensor]:
    """Bottom-n right singular vectors of each site's read-in matrix."""
    out = {}
    for site in sites:
        w = readin_matrix(model, site)
        if w is None:
            continue
        gram = (w.T @ w).cpu().double()
        _, evecs = torch.linalg.eigh(gram)  # ascending
        out[site] = evecs[:, :n].T.float()
    return out


def build_candidates(cal: dict, n_random: int, n_lowvar: int, seed: int,
                     readnull: dict[str, torch.Tensor] | None = None) -> dict:
    cands = {}
    for si, (site, st) in enumerate(cal.items()):
        d = st["mean"].shape[0]
        gen = torch.Generator().manual_seed(seed + 1000 * si)
        dirs, names, kinds = [], [], []
        for k in range(n_random):
            v = torch.randn(d, generator=gen)
            dirs.append(v / v.norm())
            names.append(f"random_{k:02d}")
            kinds.append("random")
        for k in range(n_lowvar):
            dirs.append(st["eigvecs"][:, k])
            names.append(f"lowvar_{k:02d}")
            kinds.append("lowvar")
        for k, v in enumerate((readnull or {}).get(site, [])):
            dirs.append(v / v.norm())
            names.append(f"readnull_{k:02d}")
            kinds.append("readnull")
        mu = st["mean"]
        if mu.norm() > 1e-6:
            dirs.append(mu / mu.norm())
            names.append("ctrl_mean")
            kinds.append("control")
        for k in range(2):
            dirs.append(st["eigvecs"][:, -(k + 1)])
            names.append(f"ctrl_pc{k}")
            kinds.append("control")
        cands[site] = {"dirs": torch.stack(dirs), "names": names, "kinds": kinds}
    return cands


# ------------------------------------------------------------------ selection


def select_and_report(results_sites: dict, baseline: float,
                      free_alpha: str, threshold: float) -> dict:
    """Per site, rank directions and mark free = ΔCE AT THE DEPLOYMENT
    magnitude (free_alpha) ≤ threshold — not the worst case across all
    steering alphas, which would flag every direction not-free once the
    sweep includes stress magnitudes (e.g. 4x the hidden norm)."""
    kinds = ("random", "lowvar", "readnull")
    selection: dict[str, list] = {}
    print(f"\nfree verdict at alpha={free_alpha} (deployment magnitude), "
          f"threshold {threshold} nats")
    print(f"{'site':<10} " + " ".join(f"free/{k:<8}" for k in kinds)
          + f" {'ctrl@a':>9} {'best free ΔCE':>14}")
    for site, res in results_sites.items():
        rows = []
        for dname, dres in res["directions"].items():
            a = dres["alphas"]
            dce_dep = a[free_alpha]["dce"] if free_alpha in a else max(
                x["dce"] for x in a.values())
            rows.append({"name": dname, "kind": dres["kind"],
                         "dce_deploy": dce_dep,
                         "dce_max": max(x["dce"] for x in a.values()),
                         "free": dce_dep <= threshold, "alphas": a})
        rows.sort(key=lambda r: r["dce_deploy"])
        selection[site] = rows
        counts = " ".join(
            f"{sum(r['free'] for r in rows if r['kind'] == k):>4}/"
            f"{sum(r['kind'] == k for r in rows):<8}" for k in kinds)
        ctrl = [r["dce_deploy"] for r in rows if r["kind"] == "control"]
        best_free = next((r for r in rows if r["free"]), None)
        # teeth at deployment magnitude: controls should separate from free
        # directions somewhere in the sweep, not necessarily at free_alpha
        ctrl_max = max((r["dce_max"] for r in rows if r["kind"] == "control"),
                       default=float("nan"))
        flag = "  <-- controls never separate; test lacks teeth" \
            if ctrl_max <= 5 * threshold else ""
        print(f"{site:<10} {counts} "
              f"{max(ctrl) if ctrl else float('nan'):>9.4f} "
              f"{best_free['dce_deploy'] if best_free else float('nan'):>14.4f}{flag}")
    return selection


# ------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find unallocated (free) directions in a causal LM's residual stream.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--out", default=None, help="output dir (default: password_locking/results/<model>)")
    ap.add_argument("--data-file", default=None, help="plain-text eval corpus (default: wikitext-2 test)")
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=16)
    ap.add_argument("--n-lowvar", type=int, default=8)
    ap.add_argument("--n-readnull", type=int, default=8)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.08, 0.25, 1.0, 4.0],
                    help="steering magnitude as multiple of the site's typical "
                         "hidden-state L2 norm; 0.08 is the deployment "
                         "operating point for the signature")
    ap.add_argument("--positions", choices=["all", "bos"], default="all")
    ap.add_argument("--sites", default=None, help="comma list, e.g. 'embed,0,8,15' (default: all)")
    ap.add_argument("--free-threshold", type=float, default=0.01,
                    help="max ΔCE (nats) at the deployment magnitude for free")
    ap.add_argument("--free-alpha", default=None,
                    help="alpha at which freeness is judged (default: smallest "
                         "alpha = the deployment operating point)")
    ap.add_argument("--reselect-from", default=None,
                    help="skip the sweep; recompute free_directions.json from an "
                         "existing results.json (in the --out dir by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp32"], default="auto")
    args = ap.parse_args()

    if args.reselect_from:
        out_dir = Path(args.out or ".")
        res_path = Path(args.reselect_from)
        if res_path.is_dir():
            res_path = res_path / "results.json"
        data = json.loads(res_path.read_text())
        alphas_present = list(next(iter(data["sites"].values()))
                              ["directions"].values().__iter__().__next__()["alphas"])
        free_alpha = args.free_alpha or min(alphas_present, key=float)
        selection = select_and_report(data["sites"], data["baseline_ce"],
                                      free_alpha, args.free_threshold)
        (out_dir / "free_directions.json").write_text(json.dumps(
            {"free_threshold": args.free_threshold, "free_alpha": free_alpha,
             "sites": selection}, indent=2))
        print(f"\nrewrote free_directions.json in {out_dir}")
        return

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(
        args.dtype, torch.bfloat16 if device == "cuda" else torch.float32)

    out_dir = Path(args.out or f"password_locking/results/{args.model.split('/')[-1]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device).eval()

    sites = select_sites(model, args.sites)
    batch = build_eval_batch(tok, load_eval_text(args.data_file), args.n_seqs, args.seq_len)
    print(f"eval batch {tuple(batch.shape)}, sites: {list(sites)}")

    print("calibration pass (activation stats per site) ...")
    cal = calibrate(model, sites, batch, args.batch_size, device)
    readnull = readnull_directions(model, sites, args.n_readnull)
    cands = build_candidates(cal, args.n_random, args.n_lowvar, args.seed,
                             readnull=readnull)

    baseline = compute_ce(model, batch, args.batch_size, device)
    print(f"baseline CE: {baseline:.4f} nats")

    n_evals = sum(len(c["names"]) for c in cands.values()) * len(args.alphas)
    results_sites: dict[str, dict] = {}
    pbar = tqdm(total=n_evals, desc="steering sweep")
    for site, module in sites.items():
        c, scale = cands[site], cal[site]["scale"]
        site_res = {}
        for di, dname in enumerate(c["names"]):
            base_vec = c["dirs"][di].to(device)
            per_alpha = {}
            for alpha in args.alphas:
                vec = (alpha * scale * base_vec).to(dtype)
                with steering(module, vec, args.positions):
                    ce = compute_ce(model, batch, args.batch_size, device)
                per_alpha[f"{alpha:g}"] = {"ce": ce, "dce": ce - baseline}
                pbar.update(1)
            site_res[dname] = {"kind": c["kinds"][di], "alphas": per_alpha}
        results_sites[site] = {"scale": scale, "directions": site_res}
    pbar.close()

    # ---- selection + summary (verdict at the deployment magnitude)
    free_alpha = args.free_alpha or min((f"{a:g}" for a in args.alphas), key=float)
    selection = select_and_report(results_sites, baseline, free_alpha,
                                  args.free_threshold)

    # ---- save
    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "device": device, "dtype": str(dtype),
         "baseline_ce": baseline, "eval_shape": list(batch.shape)}, indent=2))
    (out_dir / "results.json").write_text(json.dumps(
        {"baseline_ce": baseline, "sites": results_sites}, indent=2))
    (out_dir / "free_directions.json").write_text(json.dumps(
        {"free_threshold": args.free_threshold, "free_alpha": free_alpha,
         "sites": selection}, indent=2))
    arrays: dict[str, np.ndarray] = {}
    for site, c in cands.items():
        arrays[f"{site}/dirs"] = c["dirs"].numpy()
        arrays[f"{site}/names"] = np.array(c["names"])
        arrays[f"{site}/kinds"] = np.array(c["kinds"])
        arrays[f"{site}/scale"] = np.asarray(cal[site]["scale"], dtype=np.float32)
        arrays[f"{site}/mean"] = cal[site]["mean"].numpy()
        arrays[f"{site}/eigvals"] = cal[site]["eigvals"].numpy()
    np.savez(out_dir / "directions.npz", **arrays)
    print(f"\nwrote config/results/free_directions .json + directions.npz to {out_dir}")


if __name__ == "__main__":
    main()
