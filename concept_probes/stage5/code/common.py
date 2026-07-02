"""Shared data plumbing for Stage 5.

Loads the Stage-4 assembled per-class datasets, deduplicates to the family's
unique examples (sibling rows repeat example_ids), and builds the per-class
token-level target/mask arrays over the shared unique-token axis.

Conventions
- The unique-token axis: examples sorted by example_id, concatenated; an
  activation cache row t corresponds to (example_id, position) via index.json.
- A probe-class dataset row contributes, for its class c only:
    y_c[token] = sparse strength (0 elsewhere within the example),
    m_c[token] = 1, except the §5.2 post-span buffer (B tokens after each
    positive-strength run, only where y==0) where m_c = 0.
- The same example may sit in class A's train and class B's val: masks are kept
  per (class, split) and never interact.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "form_holdout")


def _read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def family_classes(final_dir: Path) -> list[str]:
    """Class names from the mixed/ file listing, sorted for a stable row order."""
    names = sorted({p.name.split(".")[0] for p in (final_dir / "mixed").glob("*.train.jsonl")})
    if not names:
        raise FileNotFoundError(f"no class train files under {final_dir}/mixed")
    return names


class FamilyData:
    """All class datasets of one family, indexed against unique examples."""

    def __init__(self, final_dir: Path, classes: list[str] | None = None):
        self.final_dir = Path(final_dir)
        self.classes = classes or family_classes(self.final_dir)
        # unique examples
        self.tokens: dict[str, list[int]] = {}          # example_id -> token_ids
        # per (class, split): list of (example_id, sparse_targets, role)
        self.rows: dict[tuple[str, str], list[tuple[str, list, str]]] = {}
        for cls in self.classes:
            for split in SPLITS:
                path = self.final_dir / "mixed" / f"{cls}.{split}.jsonl"
                rows = []
                if path.exists():
                    for r in _read_jsonl(path):
                        eid = r["example_id"]
                        tok = r["token_ids"]
                        prev = self.tokens.get(eid)
                        if prev is None:
                            self.tokens[eid] = tok
                        elif prev != tok:
                            raise ValueError(f"token_ids mismatch for {eid}")
                        rows.append((eid, r.get("token_targets_sparse") or [], r.get("role", "")))
                self.rows[(cls, split)] = rows
        # stable unique-token axis
        self.example_ids = sorted(self.tokens)
        self.offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for eid in self.example_ids:
            n = len(self.tokens[eid])
            self.offsets[eid] = (off, n)
            off += n
        self.total_tokens = off

    # ---------------------------------------------------------------- targets
    def dense_targets(self, sparse, n: int) -> np.ndarray:
        y = np.zeros(n, dtype=np.float32)
        for idx, s in sparse:
            if 0 <= idx < n:
                y[idx] = max(y[idx], float(s))
        return y

    def buffer_mask(self, y: np.ndarray, B: int) -> np.ndarray:
        """m=0 for up to B tokens after each positive run, only where y==0."""
        m = np.ones_like(y, dtype=np.float32)
        pos = np.flatnonzero(y > 0)
        if pos.size == 0 or B <= 0:
            return m
        # ends of maximal positive runs
        run_end = pos[np.flatnonzero(np.diff(np.append(pos, pos[-1] + 2)) > 1)]
        for e in run_end:
            lo, hi = e + 1, min(e + 1 + B, y.size)
            seg = slice(lo, hi)
            m[seg] = np.where(y[seg] > 0, m[seg], 0.0)
        return m

    def class_split_arrays(self, cls: str, split: str, B: int, read_shift: int = 0):
        """Flat (token_index_into_cache, y, m, example_row_id) arrays for one
        (class, split). example_row_id indexes into the returned eids list so
        example-level pooling/bootstraps stay cheap. read_shift=1 reads the
        activation one position downstream of the labeled token (§0.1 ablation),
        dropping each example's final label."""
        idx_parts, y_parts, m_parts, ex_parts, eids, roles = [], [], [], [], [], []
        for ri, (eid, sparse, role) in enumerate(self.rows[(cls, split)]):
            off, n = self.offsets[eid]
            if n <= read_shift:
                continue
            y = self.dense_targets(sparse, n)
            m = self.buffer_mask(y, B)
            keep = n - read_shift
            idx_parts.append(np.arange(off + read_shift, off + n, dtype=np.int64))
            y_parts.append(y[:keep])
            m_parts.append(m[:keep])
            ex_parts.append(np.full(keep, ri, dtype=np.int32))
            eids.append(eid)
            roles.append(role)
        if not idx_parts:
            z = np.zeros(0)
            return z.astype(np.int64), z.astype(np.float32), z.astype(np.float32), z.astype(np.int32), [], []
        return (np.concatenate(idx_parts), np.concatenate(y_parts).astype(np.float32),
                np.concatenate(m_parts).astype(np.float32), np.concatenate(ex_parts), eids, roles)


def stable_seed(*parts) -> int:
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)


def load_natstats(path: Path, layer: int):
    z = np.load(path)
    return z[f"mean_{layer}"].astype(np.float32), z[f"std_{layer}"].astype(np.float32)
