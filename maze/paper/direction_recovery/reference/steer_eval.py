#!/usr/bin/env python3
"""Steer the BASELINE model's world model along probe directions and evaluate
on the curated counterfactual edit set — the steering counterpart of
eval_counterfactual_set.py, plus loss/log-prob analogues.

Steering: at a chosen layer's resid_post, for each edited feature f, the
coordinate along the unit probe direction is set to
    mu_mid_f + alpha * (mu_class_f - mu_mid_f)
(class = wall for blocks, open for opens; mu's are class-mean coordinates
from steer_probes.pt). Applied at every position from <ADJLIST_END> onward,
including newly generated positions. alpha=0 through the same code path is a
no-op (bug check vs results_baseline.json).

Stages: --stage sweep   layer x alpha grid on an edit subsample
        --stage full    full 4,511-edit eval at one (layer, alpha)
        --stage logprob cf_block_drop / cf_open_gain via steering
        --stage damage  true-value steering path-loss cost
"""

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import fire
import torch
from maze_transformer.training.config import ConfigHolder

from build_counterfactual_set import (
    feat_cells,
    rollout_to_coords,
    traversed_edges,
)
from eval_counterfactual import load_checkpoint
from eval_injection import ProbeSet
from fast_data import PretokDataset
from reliance_analysis import region_masks

GRID_N = 6
SET_DIR = Path("/workspace/runs/counterfactual_set")
BASELINE_DIR = Path(
    "/workspace/runs/fast/fast-baseline_jirpyish-d256-l12_dfs-g6-n2000000"
)
PROBES = Path("/workspace/runs/steering/steer_probes.pt")
OUT_DIR = Path("/workspace/runs/steering")


def slice_ps(ps, rows):
    sub = copy.copy(ps)
    for f in ("tokens", "attn", "conn", "end_pos", "lens", "origin_pos",
              "first_move", "first_feat", "open_id", "open_feat", "target_id"):
        setattr(sub, f, getattr(ps, f)[rows])
    return sub


class Steerer:
    """Batched probe-direction steering at one layer's resid_post."""

    def __init__(self, probes, layer: int, alpha: float, device):
        self.layer = layer
        self.alpha = alpha
        self.dirs_all = probes["dirs"][layer].to(device)      # (72, d)
        mu_o, mu_w = probes["mu_open"][layer], probes["mu_wall"][layer]
        mid = (mu_o + mu_w) / 2.0
        self.target_open = (mid + alpha * (mu_o - mid)).to(device)
        self.target_wall = (mid + alpha * (mu_w - mid)).to(device)
        self.hook_name = f"blocks.{layer}.hook_resid_post"
        self.device = device

    def specs(self, edits_rows):
        """edits_rows: list of [(feat, want_open: bool), ...] per row ->
        padded (dirs (B,K,d), targets (B,K))."""
        K = max(1, max(len(e) for e in edits_rows))
        B = len(edits_rows)
        d = self.dirs_all.shape[1]
        dirs = torch.zeros(B, K, d, device=self.device)
        tgts = torch.zeros(B, K, device=self.device)
        for b, row in enumerate(edits_rows):
            for k, (f, want_open) in enumerate(row):
                dirs[b, k] = self.dirs_all[f]
                tgts[b, k] = (
                    self.target_open[f] if want_open else self.target_wall[f]
                )
        return dirs, tgts

    def hook_fn(self, dirs, tgts, end_pos):
        def fn(resid, hook):
            if self.alpha == 0.0:
                return resid
            P = resid.shape[1]
            pos = torch.arange(P, device=resid.device)[None, :]
            active = (pos >= end_pos[:, None]).to(resid.dtype)  # (B, P)
            coords = torch.einsum("bpd,bkd->bpk", resid, dirs)
            delta = (tgts[:, None, :] - coords) * active[..., None]
            return resid + torch.einsum("bpk,bkd->bpd", delta, dirs)
        return fn


@torch.no_grad()
def steered_rollout(model, steerer, ps, edits_rows, max_new_tokens: int = 64):
    """greedy_rollout equivalent with probe-direction steering."""
    device = ps.tokens.device
    n = len(ps.conn)
    dirs, tgts = steerer.specs(edits_rows)
    hook = steerer.hook_fn(dirs, tgts, ps.end_pos)

    max_len = int((ps.origin_pos + 1 + max_new_tokens).max())
    toks = torch.full((n, max_len), ps.pad_id, dtype=torch.long, device=device)
    for i in range(n):
        L = int(ps.origin_pos[i]) + 1
        toks[i, :L] = ps.tokens[i, :L]
    cur = (ps.origin_pos + 1).clone()
    done = torch.zeros(n, dtype=torch.bool, device=device)
    idx = torch.arange(n, device=device)
    for _ in range(max_new_tokens):
        width = int(cur.max())
        attn = (torch.arange(width, device=device)[None, :] < cur[:, None]).long()
        logits = model.run_with_hooks(
            toks[:, :width], return_type="logits", attention_mask=attn,
            fwd_hooks=[(steerer.hook_name, hook)],
        )
        nxt = logits[idx, cur - 1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, ps.pad_id), nxt)
        toks[idx, cur] = torch.where(done, toks[idx, cur], nxt)
        cur = cur + (~done).long()
        done = done | (nxt == ps.path_end_id)
        if bool(done.all()):
            break
    outs = []
    for i in range(n):
        row = toks[i, int(ps.origin_pos[i]) + 1: int(cur[i])].tolist()
        if ps.path_end_id in row:
            row = row[: row.index(ps.path_end_id)]
        outs.append(row)
    return outs


def edit_steer_spec(e):
    spec = []
    if e["block"] is not None:
        spec.append((e["block"], False))
    if e["open"] is not None:
        spec.append((e["open"], True))
    return spec


@torch.no_grad()
def eval_edits(model, steerer, ps, edits, true_rolls, chunk: int = 256):
    rows = [e["maze"] for e in edits]
    gens = []
    for lo in range(0, len(edits), chunk):
        sub_edits = edits[lo: lo + chunk]
        sub = slice_ps(ps, torch.tensor(rows[lo: lo + chunk]))
        gens += steered_rollout(
            model, steerer, sub, [edit_steer_spec(e) for e in sub_edits]
        )
    agg = defaultdict(lambda: defaultdict(list))
    for e, gen in zip(edits, gens):
        i = e["maze"]
        coords = rollout_to_coords(ps, i, gen)
        edges = set(traversed_edges(coords))
        target = ps.id_to_coord[int(ps.target_id[i])]
        reached = len(coords) > 1 and coords[-1] == target
        m = agg[e["family"]]
        m["reached_target"].append(float(reached))
        m["rollout_changed"].append(float(gen != true_rolls[i]))
        if e["family"] == "detour":
            m["blocked_edge_crossed"].append(float(e["block"] in edges))
            m["detour_wall_used"].append(float(e["open"] in edges))
        elif e["family"] == "offpath_block":
            m["blocked_edge_crossed"].append(float(e["block"] in edges))
        elif e["family"] == "shortcut_open":
            used = float(e["open"] in edges)
            m["shortcut_used"].append(used)
            bucket = "gain2" if e["gain"] <= 2 else (
                "gain4" if e["gain"] <= 4 else "gain6plus"
            )
            m[f"shortcut_used_{bucket}"].append(used)
        elif e["family"] == "zerogain_open":
            m["opened_wall_used"].append(float(e["open"] in edges))
    return {
        fam: {k: sum(v) / len(v) for k, v in v2.items()}
        for fam, v2 in agg.items()
    }


def load_everything():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, _ = load_checkpoint(BASELINE_DIR / "model.final.zanj", None, device)
    probes = torch.load(PROBES, weights_only=True)
    data = torch.load(SET_DIR / "set.pt", weights_only=False)
    cfg = ConfigHolder.load(json.load(open(BASELINE_DIR / "config_holder.json")))
    pretok = PretokDataset(data["pretok_file"])
    ps = ProbeSet(
        [pretok[i] for i in range(len(pretok))],
        cfg.tokenizer.get_vocab(), GRID_N, device,
    )
    return device, model, probes, data, ps


@torch.no_grad()
def baseline_true_rollouts(model, ps, chunk: int = 256):
    from eval_injection import greedy_rollout

    outs = []
    for lo in range(0, len(ps.conn), chunk):
        sub = slice_ps(ps, torch.arange(lo, min(len(ps.conn), lo + chunk)))
        outs += greedy_rollout(model, None, sub, sub.conn)
    return outs


def stage_sweep(subsample: int = 600, seed: int = 0):
    device, model, probes, data, ps = load_everything()
    true_rolls = baseline_true_rollouts(model, ps)
    rng = random.Random(seed)
    fams = defaultdict(list)
    for e in data["edits"]:
        fams[e["family"]].append(e)
    subset = (
        rng.sample(fams["detour"], min(subsample, len(fams["detour"])))
        + rng.sample(fams["shortcut_open"], min(subsample, len(fams["shortcut_open"])))
    )
    print(f"sweep subset: {len(subset)} edits")
    results = {}
    for layer in (5, 6, 7):
        for alpha in (0.0, 1.0, 2.0, 4.0, 8.0):
            if alpha == 0.0 and layer != 6:
                continue
            st = Steerer(probes, layer, alpha, device)
            r = eval_edits(model, st, ps, subset, true_rolls)
            det, sho = r.get("detour", {}), r.get("shortcut_open", {})
            key = f"L{layer}-a{alpha}"
            results[key] = r
            print(
                f"{key:8s} det_crossed={det.get('blocked_edge_crossed', 1):.3f} "
                f"det_used={det.get('detour_wall_used', 0):.3f} "
                f"det_reach={det.get('reached_target', 0):.3f} "
                f"sho_used={sho.get('shortcut_used', 0):.3f} "
                f"sho_reach={sho.get('reached_target', 0):.3f}"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "steer_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote steer_sweep.json")


def stage_full(layer: int, alpha: float):
    device, model, probes, data, ps = load_everything()
    true_rolls = baseline_true_rollouts(model, ps)
    st = Steerer(probes, layer, alpha, device)
    r = eval_edits(model, st, ps, data["edits"], true_rolls)
    out = OUT_DIR / f"steer_full_L{layer}_a{alpha}.json"
    with open(out, "w") as f:
        json.dump(r, f, indent=2)
    for fam, ms in sorted(r.items()):
        for k, v in sorted(ms.items()):
            print(f"{fam}/{k:36s} {v:.3f}")
    print(f"wrote {out}")


@torch.no_grad()
def stage_logprob(layer: int, alpha: float, chunk: int = 256):
    """cf_block_drop / cf_open_gain analogues at the decision position where
    the baseline's own true rollout meets the edited edge."""
    device, model, probes, data, ps = load_everything()
    true_rolls = baseline_true_rollouts(model, ps)
    st = Steerer(probes, layer, alpha, device)

    cases = {"detour_block": [], "shortcut_open": []}
    for e in data["edits"]:
        i = e["maze"]
        coords = rollout_to_coords(ps, i, true_rolls[i])
        if e["family"] == "detour":
            a, b = feat_cells(e["block"])
            for s, (c0, c1) in enumerate(zip(coords, coords[1:])):
                if {c0, c1} == {a, b}:
                    move_tok = ps.coord_to_id[c1]
                    cases["detour_block"].append(
                        (i, s, move_tok, [(e["block"], False)])
                    )
                    break
        elif e["family"] == "shortcut_open":
            a, b = feat_cells(e["open"])
            for s, c0 in enumerate(coords[:-1]):
                if c0 in (a, b):
                    other = b if c0 == a else a
                    cases["shortcut_open"].append(
                        (i, s, ps.coord_to_id[other], [(e["open"], True)])
                    )
                    break

    out = {}
    for name, rows in cases.items():
        if not rows:
            continue
        deltas = []
        for lo in range(0, len(rows), chunk):
            batch = rows[lo: lo + chunk]
            B = len(batch)
            ctx_len = [int(ps.origin_pos[i]) + 1 + s for i, s, _, _ in batch]
            width = max(ctx_len)
            toks = torch.full((B, width), ps.pad_id, dtype=torch.long,
                              device=device)
            for b, (i, s, _, _) in enumerate(batch):
                L = ctx_len[b]
                prompt_L = int(ps.origin_pos[i]) + 1
                toks[b, :prompt_L] = ps.tokens[i, :prompt_L]
                gen = true_rolls[i][:s]
                if gen:
                    toks[b, prompt_L:prompt_L + len(gen)] = torch.tensor(
                        gen, device=device
                    )
            attn = (
                torch.arange(width, device=device)[None, :]
                < torch.tensor(ctx_len, device=device)[:, None]
            ).long()
            end_pos = ps.end_pos[[i for i, _, _, _ in batch]]
            dirs, tgts = st.specs([spec for _, _, _, spec in batch])
            hook = st.hook_fn(dirs, tgts, end_pos)
            lp_plain = model(
                toks, return_type="logits", attention_mask=attn
            ).log_softmax(-1)
            lp_steer = model.run_with_hooks(
                toks, return_type="logits", attention_mask=attn,
                fwd_hooks=[(st.hook_name, hook)],
            ).log_softmax(-1)
            for b, (i, s, move_tok, _) in enumerate(batch):
                p = ctx_len[b] - 1
                deltas.append(
                    float(lp_steer[b, p, move_tok] - lp_plain[b, p, move_tok])
                )
        out[name] = dict(
            n=len(deltas), mean_logprob_delta=sum(deltas) / len(deltas)
        )
        print(f"{name}: n={len(deltas)} mean steered-vs-plain logprob delta "
              f"of the affected move = {out[name]['mean_logprob_delta']:+.3f}")
    with open(OUT_DIR / f"steer_logprob_L{layer}_a{alpha}.json", "w") as f:
        json.dump(out, f, indent=2)


@torch.no_grad()
def stage_damage(layer: int, alpha: float):
    """Path-token loss with steering active but targets = TRUE feature
    classes (machinery cost without a counterfactual)."""
    device, model, probes, data, ps = load_everything()
    st = Steerer(probes, layer, alpha, device)
    edits_rows = []
    for i in range(len(ps.conn)):
        f_block = int(ps.first_feat[i])
        spec = [(f_block, bool(ps.conn[i, f_block]))]
        f_open = int(ps.open_feat[i])
        if f_open >= 0:
            spec.append((f_open, bool(ps.conn[i, f_open])))
        edits_rows.append(spec)
    dirs, tgts = st.specs(edits_rows)
    hook = st.hook_fn(dirs, tgts, ps.end_pos)

    def path_loss(hooks):
        logits = model.run_with_hooks(
            ps.tokens, return_type="logits", attention_mask=ps.attn,
            fwd_hooks=hooks,
        )
        lp = logits[:, :-1].log_softmax(-1)
        ll = lp.gather(-1, ps.tokens[:, 1:][..., None])[..., 0]
        masks = region_masks(ps)
        return {k: float(-(ll[m]).mean()) for k, m in masks.items()}

    plain = path_loss([])
    steered = path_loss([(st.hook_name, hook)])
    print(f"no steer:            path {plain['path']:.4f}  all {plain['all']:.4f}")
    print(f"true-value steering: path {steered['path']:.4f}  all {steered['all']:.4f}")
    with open(OUT_DIR / f"steer_damage_L{layer}_a{alpha}.json", "w") as f:
        json.dump({"plain": plain, "true_value_steered": steered}, f, indent=2)


def main(stage: str = "sweep", layer: int = 6, alpha: float = 2.0, **kw):
    {
        "sweep": stage_sweep,
        "full": lambda: stage_full(layer, alpha),
        "logprob": lambda: stage_logprob(layer, alpha),
        "damage": lambda: stage_damage(layer, alpha),
    }[stage](**kw) if stage == "sweep" else {
        "full": lambda: stage_full(layer, alpha),
        "logprob": lambda: stage_logprob(layer, alpha),
        "damage": lambda: stage_damage(layer, alpha),
    }[stage]()


if __name__ == "__main__":
    fire.Fire(main)
