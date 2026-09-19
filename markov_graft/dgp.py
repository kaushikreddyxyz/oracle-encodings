"""Entropy-pinned k-th order Markov sources with a controllable description length.

Every transition row is a random permutation of ONE base distribution q with
entropy h bits, so the entropy rate is exactly h for every table, while the
description length of the table is  A^m * log2(A!)  bits, where m is the number
of context positions the row actually depends on (m = k for a dense chain; for
a sparse chain the m positions are a random subset of the last k that always
includes position k, so the nominal order stays k while the table shrinks).

Grafts (per position t, about the target x_{t+1}):
  oracle_b : top b bits of the argmax symbol of the true row  (function of context)
  leak_b   : top b bits of the actual next symbol x_{t+1}      (fresh information)
"""
import math
import numpy as np

LOG2 = math.log(2.0)


def base_distribution(A, h_bits):
    """q_i ∝ exp(-beta*i) with beta tuned so H(q) = h_bits (bisection)."""
    def ent(beta):
        q = np.exp(-beta * np.arange(A)); q /= q.sum()
        return float(-(q * np.log2(q)).sum()), q
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ent(mid)[0] > h_bits:
            lo = mid
        else:
            hi = mid
    return ent(0.5 * (lo + hi))[1]


class MarkovSource:
    def __init__(self, A=8, k=3, m=None, h_bits=2.0, table_seed=0):
        self.A, self.k = A, k
        self.m = k if m is None else m
        assert 1 <= self.m <= k
        self.h_bits = h_bits
        self.code_bits = int(math.ceil(math.log2(A)))
        rng = np.random.default_rng(table_seed)
        self.q = base_distribution(A, h_bits)
        n_ctx = A ** self.m
        # rows[c] = q permuted; store as float64 for exact reference losses
        perms = np.argsort(rng.random((n_ctx, A)), axis=1)
        self.rows = np.take_along_axis(np.broadcast_to(self.q, (n_ctx, A)), np.argsort(perms, axis=1), axis=1)
        self.rows = np.ascontiguousarray(self.rows)
        self.cdf = np.cumsum(self.rows, axis=1)
        self.cdf[:, -1] = 1.0 + 1e-9
        self.argmax = self.rows.argmax(axis=1).astype(np.int64)
        # lags (1 = previous symbol ... k = k symbols back) the row depends on
        if self.m == k:
            self.lags = np.arange(1, k + 1)
        else:
            others = rng.choice(np.arange(1, k), size=self.m - 1, replace=False)
            self.lags = np.sort(np.concatenate([others, [k]]))
        self.weights = (A ** np.arange(self.m)).astype(np.int64)

    @property
    def description_bits(self):
        return float(self.A ** self.m * math.log2(math.factorial(self.A)))

    @property
    def entropy_nats(self):
        return self.h_bits * LOG2

    def context_index(self, x, t):
        """x: [B, L] symbols; index of the row generating x[:, t] (needs t >= k)."""
        idx = np.zeros(x.shape[0], dtype=np.int64)
        for w, lag in zip(self.weights, self.lags):
            idx += w * x[:, t - lag]
        return idx

    def context_index_all(self, x):
        """Row index for every position t >= k; returns [B, L-k] aligned to x[:, k:]."""
        B, L = x.shape
        idx = np.zeros((B, L - self.k), dtype=np.int64)
        for w, lag in zip(self.weights, self.lags):
            idx += w * x[:, self.k - lag:L - lag]
        return idx

    def sample(self, n_seq, length, rng, burn_in=32):
        """Returns [n_seq, length] uint8 symbols, after burn_in steps from a uniform start."""
        total = length + burn_in
        x = np.zeros((n_seq, total + self.k), dtype=np.int64)
        x[:, :self.k] = rng.integers(0, self.A, size=(n_seq, self.k))
        u = rng.random((n_seq, total))
        for t in range(self.k, total + self.k):
            idx = self.context_index(x, t)
            x[:, t] = (u[:, t - self.k, None] >= self.cdf[idx]).sum(axis=1)
        return x[:, self.k + burn_in:].astype(np.uint8)

    def grafts(self, x, kind, b):
        """Graft value for positions t = 0..L-2 about target x[:, t+1]; [B, L-1] int64.
        Positions whose context is not fully inside x get graft 0 and are masked in training."""
        B, L = x.shape
        xi = x.astype(np.int64)
        g = np.zeros((B, L - 1), dtype=np.int64)
        valid = np.zeros((B, L - 1), dtype=bool)
        shift = self.code_bits - b
        if kind == "oracle":
            idx = self.context_index_all(xi)             # rows for targets x[:, k:]
            g[:, self.k - 1:] = self.argmax[idx] >> shift
            valid[:, self.k - 1:] = True
        elif kind == "leak":
            g[:] = xi[:, 1:] >> shift
            valid[:] = True
        else:
            raise ValueError(kind)
        return g, valid

    def reference_losses(self, x, b_values=(1, 2, 3)):
        """Exact per-token reference losses (nats) on sampled data x, for targets with full context.
        bayes: knows the table.  unigram: marginal only.  oracle_b_only: knows only the graft.
        leak_b_bayes: table + leak.  leak_b_only: leak only (marginal within bucket)."""
        xi = x.astype(np.int64)
        idx = self.context_index_all(xi)
        y = xi[:, self.k:]
        p = self.rows[idx, y]
        out = {"bayes": float(-np.log(p).mean())}
        marg = np.bincount(y.ravel(), minlength=self.A) / y.size
        out["unigram"] = float(-np.log(marg[y]).mean())
        for b in b_values:
            shift = self.code_bits - b
            g = self.argmax[idx] >> shift
            n_g = 1 << b
            cnt = np.zeros((n_g, self.A))
            np.add.at(cnt, (g.ravel(), y.ravel()), 1)
            pg = (cnt + 1e-3) / (cnt + 1e-3).sum(axis=1, keepdims=True)
            out[f"oracle_{b}_only"] = float(-np.log(pg[g, y]).mean())
            gl = y >> shift
            bucket = (np.arange(self.A) >> shift)[None, :] == gl[..., None]      # [B,L,A]
            rows = self.rows[idx]
            out[f"leak_{b}_bayes"] = float(-np.log(p / (rows * bucket).sum(-1)).mean())
            mb = marg[None, :] * bucket
            out[f"leak_{b}_only"] = float(-np.log(marg[y] / mb.sum(-1)).mean())
        return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--h", type=float, default=2.0)
    a = ap.parse_args()
    src = MarkovSource(a.A, a.k, a.m, a.h)
    x = src.sample(2000, 256, np.random.default_rng(1))
    print(json.dumps({"description_bits": src.description_bits, "entropy_nats": src.entropy_nats,
                      "lags": src.lags.tolist(), "q": src.q.round(4).tolist(),
                      **src.reference_losses(x)}, indent=1))
