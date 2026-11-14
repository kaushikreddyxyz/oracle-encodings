"""Light-touch injection evals: channel reliance, counterfactual steering,
and greedy rollout quality. Used periodically during training (train_fast.py)
and offline across checkpoints (maze/core/checkpoints.py).

All metrics operate on a fixed ProbeSet built from held-out mazes'
pretokenized tensors, so they are cheap and comparable across steps/runs.
"""

import re

import torch
from maze_dataset import SPECIAL_TOKENS

from maze.core.injection import as_runtime


class ProbeSet:
    """Held-out mazes with everything needed for injection evals.

    Per maze i:
      tokens/attn/conn/end_pos     as in the training pipeline
      origin_pos[i]   position of the origin coord token inside the path
                      (logits there predict the first move)
      first_move[i]   token id of the true first path move
      first_feat[i]   feature index of the (origin -> first move) connection
      open_id[i]      token id of a lattice neighbor of origin NOT connected
                      to it (-1 if origin has no walled-off neighbor)
      open_feat[i]    feature index of that wall (-1 if none)
      target_id[i]    token id of the target cell
    """

    def __init__(self, entries, vocab: dict[str, int], grid_n: int, device):
        self.grid_n = grid_n
        self.device = device
        self.pad_id = vocab[SPECIAL_TOKENS.PADDING]
        self.path_start_id = vocab[SPECIAL_TOKENS.PATH_START]
        self.path_end_id = vocab[SPECIAL_TOKENS.PATH_END]
        coord_re = re.compile(r"^\((\d+),(\d+)\)$")
        self.id_to_coord = {
            i: (int(m.group(1)), int(m.group(2)))
            for t, i in vocab.items()
            if (m := coord_re.match(t))
        }
        self.coord_to_id = {c: i for i, c in self.id_to_coord.items()}

        seqs, conns, ends = zip(*entries)
        # keep only mazes whose path has >= 2 coord tokens (origin + a move)
        kept = []
        for i, s in enumerate(seqs):
            row = s.tolist()
            p = row.index(self.path_start_id)
            if (
                p + 2 < len(row)
                and row[p + 1] in self.id_to_coord
                and row[p + 2] in self.id_to_coord
            ):
                kept.append(i)
        seqs = [seqs[i] for i in kept]
        conns = [conns[i] for i in kept]
        ends = [ends[i] for i in kept]
        lens = torch.tensor([len(s) for s in seqs])
        max_len = int(lens.max())
        toks = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            toks[i, : len(s)] = s.long()
        self.tokens = toks.to(device)
        self.attn = (
            (torch.arange(max_len)[None, :] < lens[:, None]).long().to(device)
        )
        self.conn = torch.stack(conns).to(device)
        self.end_pos = torch.stack(ends).long().to(device)
        self.lens = lens.to(device)

        n = len(seqs)
        self.origin_pos = torch.zeros(n, dtype=torch.long)
        self.first_move = torch.zeros(n, dtype=torch.long)
        self.first_feat = torch.zeros(n, dtype=torch.long)
        self.open_id = torch.full((n,), -1, dtype=torch.long)
        self.open_feat = torch.full((n,), -1, dtype=torch.long)
        self.target_id = torch.zeros(n, dtype=torch.long)
        for i, s in enumerate(seqs):
            row = s.tolist()
            ps = row.index(self.path_start_id)
            self.origin_pos[i] = ps + 1  # origin coord token position
            self.first_move[i] = row[ps + 2]
            o = self.id_to_coord[row[ps + 1]]
            m = self.id_to_coord[row[ps + 2]]
            self.first_feat[i] = self._feat_idx(o, m)
            # target = coord token right before <PATH_END>
            pe = row.index(self.path_end_id)
            self.target_id[i] = row[pe - 1]
            # a walled-off lattice neighbor of the origin, if any
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nb = (o[0] + dr, o[1] + dc)
                if nb in self.coord_to_id:
                    fi = self._feat_idx(o, nb)
                    if not bool(self.conn[i, fi]):
                        self.open_id[i] = self.coord_to_id[nb]
                        self.open_feat[i] = fi
                        break
        for t in ("origin_pos", "first_move", "first_feat", "open_id",
                  "open_feat", "target_id"):
            setattr(self, t, getattr(self, t).to(device))

    def _feat_idx(self, a, b) -> int:
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            return self.grid_n * self.grid_n + r1 * self.grid_n + min(c1, c2)
        return min(r1, r2) * self.grid_n + c1


def _forward(model, injector, tokens, attn, conn, end_pos):
    """`injector` may be a bare MazeInjector (complete mode) or an
    InjectionRuntime carrying a variant (window / teacher)."""
    if injector is None:
        return model(tokens, return_type="logits", attention_mask=attn)
    return model.run_with_hooks(
        tokens, return_type="logits", attention_mask=attn,
        fwd_hooks=as_runtime(injector).fwd_hooks(tokens, attn, conn, end_pos),
    )


def _loss(model, injector, ps: ProbeSet, conn) -> float:
    logits = _forward(model, injector, ps.tokens, ps.attn, conn, ps.end_pos)
    logprobs = logits[:, :-1].log_softmax(-1)
    tgt = ps.tokens[:, 1:]
    ll = logprobs.gather(-1, tgt[..., None])[..., 0]
    mask = ps.attn[:, 1:].bool()
    return float(-(ll[mask]).mean())


@torch.no_grad()
def injection_eval_suite(model, injector, ps: ProbeSet) -> dict[str, float]:
    """Channel-reliance + counterfactual steering metrics (cheap forwards)."""
    was_training = model.training
    model.eval()
    out = {}

    out["inj_eval/loss_true"] = _loss(model, injector, ps, ps.conn)
    out["inj_eval/loss_zero"] = _loss(
        model, injector, ps, torch.zeros_like(ps.conn)
    )
    perm = torch.randperm(len(ps.conn), device=ps.conn.device)
    out["inj_eval/loss_shuffled"] = _loss(model, injector, ps, ps.conn[perm])
    out["inj_eval/reliance_zero"] = (
        out["inj_eval/loss_zero"] - out["inj_eval/loss_true"]
    )
    out["inj_eval/reliance_shuffled"] = (
        out["inj_eval/loss_shuffled"] - out["inj_eval/loss_true"]
    )

    idx = torch.arange(len(ps.conn), device=ps.conn.device)

    def first_move_logprobs(conn):
        logits = _forward(model, injector, ps.tokens, ps.attn, conn, ps.end_pos)
        lp = logits[idx, ps.origin_pos].log_softmax(-1)
        return lp

    lp_true = first_move_logprobs(ps.conn)

    # counterfactual 1: block the true first move's connection
    conn_block = ps.conn.clone()
    conn_block[idx, ps.first_feat] = False
    lp_block = first_move_logprobs(conn_block)
    out["inj_eval/cf_block_drop"] = float(
        (lp_true[idx, ps.first_move] - lp_block[idx, ps.first_move]).mean()
    )

    # counterfactual 2: open a walled-off neighbor of the origin
    has_open = ps.open_feat >= 0
    if int(has_open.sum()) > 0:
        conn_open = ps.conn.clone()
        conn_open[idx[has_open], ps.open_feat[has_open]] = True
        lp_open = first_move_logprobs(conn_open)
        gain = (
            lp_open[idx[has_open], ps.open_id[has_open]]
            - lp_true[idx[has_open], ps.open_id[has_open]]
        )
        out["inj_eval/cf_open_gain"] = float(gain.mean())

    if was_training:
        model.train()
    return out


@torch.no_grad()
def greedy_rollout(
    model, injector, ps: ProbeSet, conn, max_new_tokens: int = 64
):
    """Greedy per-row generation from each maze's path prompt (everything up
    to and including the origin token). Injection features come from `conn`
    (pass an edited conn for counterfactual rollouts). Returns list of
    generated token-id lists (up to and excl. <PATH_END>)."""
    was_training = model.training
    model.eval()
    n = len(ps.conn)
    max_len = int((ps.origin_pos + 1 + max_new_tokens).max())
    toks = torch.full((n, max_len), ps.pad_id, dtype=torch.long, device=ps.device)
    for i in range(n):
        L = int(ps.origin_pos[i]) + 1
        toks[i, :L] = ps.tokens[i, :L]
    cur = (ps.origin_pos + 1).clone()
    done = torch.zeros(n, dtype=torch.bool, device=ps.device)
    idx = torch.arange(n, device=ps.device)

    for _ in range(max_new_tokens):
        width = int(cur.max())
        attn = (torch.arange(width, device=ps.device)[None, :] < cur[:, None]).long()
        logits = _forward(model, injector, toks[:, :width], attn, conn, ps.end_pos)
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
    if was_training:
        model.train()
    return outs


def rollout_metrics(ps: ProbeSet, rollouts, conn) -> dict[str, float]:
    """Judge rollouts against a connectivity `conn` (true or edited)."""
    reach, valid_moves, total_moves = 0, 0, 0
    for i, gen in enumerate(rollouts):
        prev = ps.id_to_coord.get(int(ps.tokens[i, ps.origin_pos[i]]))
        if prev is None:
            continue
        coords = [prev]
        ok = True
        for t in gen:
            c = ps.id_to_coord.get(t)
            if c is None:
                ok = False
                break
            (r1, c1), (r2, c2) = coords[-1], c
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                fi = ps._feat_idx(coords[-1], c)
                valid_moves += int(bool(conn[i, fi]))
            total_moves += 1
            coords.append(c)
        if ok and gen and gen[-1] == int(ps.target_id[i]):
            reach += 1
    n = max(1, len(rollouts))
    return {
        "rollout/reach_target_frac": reach / n,
        "rollout/valid_move_frac": valid_moves / max(1, total_moves),
    }
