#!/usr/bin/env python3
"""Train linear wall-probes on the BASELINE model's residual stream.

For each layer l (resid_post 0..11) and each of the 72 connection-list
features, fit a linear probe on the activation at the <PATH_START> token
(canonical probing position from Ivanitskiy et al.) of fresh mazes never seen
in training or prior evals. Saves per-layer probe weights, per-layer accuracy,
and per-feature class-mean coordinates (needed as steering targets).

  python steer_probes.py --run_dir /workspace/runs/fast/fast-baseline_... \
      --out /workspace/runs/steering/steer_probes.pt
"""

import json
from pathlib import Path

import fire
import torch
from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import ConfigHolder

from eval_counterfactual import load_checkpoint
from eval_injection import ProbeSet
from fast_data import PretokDataset, pretokenize

GRID_N = 6
N_FEATS = 2 * GRID_N * GRID_N
N_LAYERS = 12


@torch.no_grad()
def collect_activations(model, ps, batch: int = 512):
    """Residual-stream activations at each maze's <PATH_START> position for
    every layer's resid_post. Returns (n, n_layers, d_model)."""
    device = ps.tokens.device
    n = len(ps.conn)
    path_start_pos = ps.origin_pos - 1
    out = torch.zeros(n, N_LAYERS, model.cfg.d_model)
    names = [f"blocks.{l}.hook_resid_post" for l in range(N_LAYERS)]
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        toks = ps.tokens[lo:hi]
        attn = ps.attn[lo:hi]
        _, cache = model.run_with_cache(
            toks, attention_mask=attn, return_type=None,
            names_filter=lambda nm: nm in names,
        )
        rows = torch.arange(hi - lo, device=device)
        pos = path_start_pos[lo:hi]
        for l, nm in enumerate(names):
            out[lo:hi, l] = cache[nm][rows, pos].cpu()
        del cache
    return out


def train_probes(acts, targets, train_n, epochs: int = 400, lr: float = 5e-3):
    """One linear map (d_model -> 72) per layer, BCE on {wall,open}.
    Returns weights (layers, 72, d), biases, val accuracy (layers, 72)."""
    device = torch.device("cuda")
    y = targets.float().to(device)  # (n, 72) in {0,1}
    Ws, Bs, accs = [], [], []
    for l in range(N_LAYERS):
        X = acts[:, l].to(device)
        Xtr, Xva = X[:train_n], X[train_n:]
        ytr, yva = y[:train_n], y[train_n:]
        lin = torch.nn.Linear(X.shape[1], N_FEATS).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=lr)
        for _ in range(epochs):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                lin(Xtr), ytr
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (((lin(Xva) > 0) == (yva > 0.5)).float().mean(0))
        Ws.append(lin.weight.detach().cpu())
        Bs.append(lin.bias.detach().cpu())
        accs.append(acc.cpu())
    return torch.stack(Ws), torch.stack(Bs), torch.stack(accs)


def main(
    run_dir: str = "/workspace/runs/fast/fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000",
    out: str = "/workspace/runs/steering/steer_probes.pt",
    n_mazes: int = 20000,
    seed: int = 9001,
    train_frac: float = 0.8,
    work_dir: str = "/workspace/runs/steering",
):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    run_path = Path(run_dir)
    model, _ = load_checkpoint(run_path / "model.final.zanj", None, device)
    cfg = ConfigHolder.load(json.load(open(run_path / "config_holder.json")))
    vocab = cfg.tokenizer.get_vocab()

    pretok_path = Path(work_dir) / f"pretok-steerprobe-s{seed}-n{n_mazes}.pt"
    if not pretok_path.exists():
        ds = MazeDataset.from_config(
            MazeDatasetConfig(
                name=f"steerprobe-s{seed}", grid_n=GRID_N, n_mazes=n_mazes,
                maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed,
            ),
            do_generate=True, load_local=False, save_local=False,
            gen_parallel=True,
        )
        pretokenize(
            ds, cfg.maze_tokenizer, vocab, pretok_path,
            bos_id=int(cfg.tokenizer.bos_token_id),
        )
    pretok = PretokDataset(pretok_path)
    ps = ProbeSet(
        [pretok[i] for i in range(len(pretok))], vocab, GRID_N, device
    )
    print(f"probe mazes: {len(ps.conn)}")

    acts = collect_activations(model, ps)
    targets = ps.conn.cpu()  # bool (n, 72): True = open/connected
    train_n = int(len(ps.conn) * train_frac)

    W, B, acc = train_probes(acts, targets, train_n)
    for l in range(N_LAYERS):
        print(f"layer {l:2d} resid_post: mean probe acc {acc[l].mean():.4f} "
              f"(min {acc[l].min():.4f})")
    best = int(acc.mean(dim=1).argmax())
    print(f"best layer: {best} (mean acc {acc[best].mean():.4f})")

    # steering targets: class-conditional mean coordinate along the unit
    # probe direction, computed on validation activations
    dirs = W / W.norm(dim=-1, keepdim=True)  # (layers, 72, d)
    mu_open = torch.zeros(N_LAYERS, N_FEATS)
    mu_wall = torch.zeros(N_LAYERS, N_FEATS)
    yva = targets[train_n:]
    for l in range(N_LAYERS):
        coords = acts[train_n:, l] @ dirs[l].T  # (n_val, 72)
        mu_open[l] = (coords * yva).sum(0) / yva.sum(0).clamp(min=1)
        mu_wall[l] = (coords * (~yva)).sum(0) / (~yva).sum(0).clamp(min=1)

    torch.save(
        dict(
            W=W, B=B, acc=acc, dirs=dirs, mu_open=mu_open, mu_wall=mu_wall,
            best_layer=best, seed=seed, n_mazes=n_mazes, train_n=train_n,
            position="path_start_token", hook="resid_post",
        ),
        out,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
