"""One-time pretokenization of a MazeDataset into flat tensors + fast Dataset.

Eliminates all per-step Python string work: maze.as_tokens + tokenizer encode
happen once (parallelized), training then reads int16 token tensors. Each maze
is serialized with one adjacency-list shuffle — equivalent to the on-the-fly
pipeline for single-epoch training.

Stored per maze:
  tokens      ragged int16 (flat buffer + offsets)
  conn        72 bools: connection_list.flatten() ground truth
  end_pos     position of <ADJLIST_END> (complete-mode injection start)

Batch-time feature construction is fully vectorized (see make_features).

Also hosts the `dummy` data variant: dummy_adjacency / dummify_pretok blank the
adjacency list (positions 1..end_pos-1 -> `;`) so the injected channel is the
only source of connectivity (see VARIANTS_SPEC.md C).
"""

import multiprocessing
from pathlib import Path

import numpy as np
import torch
from maze_dataset import SPECIAL_TOKENS

_WORKER = {}


def _init_worker(maze_tokenizer, vocab, bos_id):
    _WORKER["mt"] = maze_tokenizer
    _WORKER["vocab"] = vocab
    _WORKER["end_id"] = vocab[SPECIAL_TOKENS.ADJLIST_END]
    _WORKER["bos_id"] = bos_id


def _encode_one(maze):
    toks = maze.as_tokens(_WORKER["mt"])
    # replicate HookedTransformer.to_tokens exactly: it prepends a BOS token
    ids = np.array(
        [_WORKER["bos_id"]] + [_WORKER["vocab"][t] for t in toks], dtype=np.int16
    )
    end_pos = int(np.nonzero(ids == _WORKER["end_id"])[0][0])
    conn = maze.connection_list.flatten().astype(np.bool_)
    return ids, conn, end_pos


def pretokenize(
    dataset,
    maze_tokenizer,
    vocab: dict[str, int],
    out_path: Path,
    bos_id: int,
    workers: int = 16,
) -> None:
    if workers > 0:
        with multiprocessing.Pool(
            workers, initializer=_init_worker,
            initargs=(maze_tokenizer, vocab, bos_id),
        ) as pool:
            results = pool.map(_encode_one, dataset.mazes, chunksize=2048)
    else:  # serial: small sets / CPU smoke tests (no spawn guard needed)
        _init_worker(maze_tokenizer, vocab, bos_id)
        results = [_encode_one(m) for m in dataset.mazes]
    lengths = np.array([len(r[0]) for r in results], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    torch.save(
        dict(
            tokens=torch.from_numpy(np.concatenate([r[0] for r in results])),
            offsets=torch.from_numpy(offsets),
            conn=torch.from_numpy(np.stack([r[1] for r in results])),
            end_pos=torch.tensor([r[2] for r in results], dtype=torch.int16),
        ),
        out_path,
    )


def dummy_adjacency(
    tokens_row: torch.Tensor, end_pos: int, semicolon_id: int
) -> torch.Tensor:
    """Blank one maze's adjacency list: every token strictly between position 0
    (BOS = <ADJLIST_START>) and end_pos (<ADJLIST_END>) becomes `;`. Position 0
    and everything from <ADJLIST_END> on (origin/target/path) are untouched."""
    out = tokens_row.clone()
    out[1: int(end_pos)] = semicolon_id
    return out


def _dummify_flat(tokens, offsets, end_pos, semicolon_id: int) -> None:
    """In-place dummy_adjacency over a flat ragged token buffer."""
    for i in range(len(end_pos)):
        lo = int(offsets[i])
        tokens[lo + 1: lo + int(end_pos[i])] = semicolon_id


def dummify_pretok(src_path: Path, out_path: Path, semicolon_id: int) -> Path:
    """Materialize a dummied copy of a pretok file (same conn/end_pos)."""
    d = torch.load(src_path, weights_only=True)
    _dummify_flat(d["tokens"], d["offsets"], d["end_pos"], semicolon_id)
    torch.save(d, out_path)
    return out_path


def find_pretok(base_path: Path | str, dummy: bool = False) -> Path:
    """The run's pretok file under base_path; dummied copies (pretok-dummy-*)
    are only returned when asked for."""
    files = sorted(Path(base_path).glob("pretok-*.pt"))
    files = [f for f in files if f.name.startswith("pretok-dummy-") == dummy]
    assert files, f"no {'dummy ' if dummy else ''}pretok file under {base_path}"
    return files[0]


class PretokDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path):
        d = torch.load(path, weights_only=True)
        self.tokens = d["tokens"]
        self.offsets = d["offsets"]
        self.conn = d["conn"]
        self.end_pos = d["end_pos"]

    def dummify_(self, semicolon_id: int) -> "PretokDataset":
        """Apply the dummy adjacency transform in place (eval-side helper: a
        dummied ProbeSet / counterfactual set needs no separate file)."""
        _dummify_flat(self.tokens, self.offsets, self.end_pos, semicolon_id)
        return self

    def __len__(self):
        return len(self.end_pos)

    def __getitem__(self, i):
        return (
            self.tokens[self.offsets[i]: self.offsets[i + 1]],
            self.conn[i],
            self.end_pos[i],
        )


def collate_pretok(batch, pad_id: int):
    seqs, conns, ends = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    max_len = int(lens.max())
    tokens = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        tokens[i, : len(s)] = s.long()  # right-padded
    attention_mask = (
        torch.arange(max_len)[None, :] < lens[:, None]
    ).long()
    return tokens, attention_mask, torch.stack(conns), torch.stack(ends).long()


def make_features(
    conn: torch.Tensor,
    end_pos: torch.Tensor,
    n_pos: int,
    attention_mask: torch.Tensor,
    window_k: int | None = None,
) -> torch.Tensor:
    """Vectorized complete-mode features: (batch, pos, 72), +/-1 from
    <ADJLIST_END> onward (real tokens only), 0 before.

    window_k: if given, restrict injection to [end_pos, end_pos + window_k)
    (variant A); None reproduces complete mode exactly."""
    signed = conn.float() * 2.0 - 1.0  # (batch, 72)
    pos = torch.arange(n_pos, device=conn.device)[None, :]
    active = (pos >= end_pos[:, None]) & attention_mask.bool()
    if window_k is not None:
        active = active & (pos < end_pos[:, None] + int(window_k))
    return active[:, :, None].float() * signed[:, None, :]
