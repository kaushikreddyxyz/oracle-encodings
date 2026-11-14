#!/usr/bin/env python3
"""SAE job lists. `full`: the stage-1 grid (one SAE seed) for a grid size; `shortlist`: written by shortlist.py."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SIZES = ("m1", "m1.5", "m3", "m8", "mn1024"); L1S = ("0.01", "0.03", "0.1", "0.3")


def trimmed(n, seed=0):
    """Stage-1 list for grids 5-7 (started after the 8-10 sweeps showed the 8x / 1,024-atom dictionaries lose at every sparsity):
    those two sizes keep only k in {16, dense} and L1 in {.03, .1}; matching pursuit only at 1.5x with k <= 16."""
    jobs, mp = full(n, seed); dense = {n * n - 1, (n - 1) ** 2}
    def keep(j):
        kind, size, par = j["spec"].split("-")
        if size not in ("m8", "mn1024"): return True
        return (par[1:] in ("0.03", "0.1")) if kind == "relu" else int(par[1:]) in dense | {16}
    return [j for j in jobs if keep(j)], [j for j in mp if j["spec"].startswith("mp-m1.5") and int(j["spec"].split("-k")[1]) <= 16]


def full(n, seed=0):
    """-> (jobs run with several workers per GPU, matching-pursuit jobs run one at a time: unrolled MP keeps k encoder passes in memory)."""
    dense = (n * n - 1, (n - 1) ** 2); ks = sorted({4, 8, 16, 32, *dense}); jobs = []; mp = []
    for size in SIZES:
        for kind in ("topk", "btopk") + (("mat",) if size in ("m1.5", "m3", "m8") else ()):
            jobs += [dict(spec=f"{kind}-{size}-k{k}", variant="ctr", seed=seed) for k in ks]
        jobs += [dict(spec=f"relu-{size}-l{l}", variant="ctr", seed=seed) for l in L1S]
        if size in ("m1.5", "m3"): mp += [dict(spec=f"mp-{size}-k{k}", variant="ctr", seed=seed) for k in (4, 8, 16, 32)]
    for size in ("m1.5", "m3"):                                  # uncentred input: only the input scale / effective L1 differs
        jobs += [dict(spec=f"relu-{size}-l{l}", variant="raw", seed=seed) for l in L1S]
    for variant, k in (("closed", n * n - 1), ("open", (n - 1) ** 2)):   # coding-aware reference point: the true code is exactly k-sparse and non-negative
        for size in ("m1", "m1.5", "m3"):
            jobs += [dict(spec=f"{kind}-{size}-k{k}", variant=variant, seed=seed) for kind in ("topk", "btopk")]
            jobs += [dict(spec=f"relu-{size}-l{l}", variant=variant, seed=seed) for l in ("0.03", "0.1")]
    return jobs, mp


if __name__ == "__main__":
    (HERE / "jobs").mkdir(exist_ok=True)
    for n in range(5, 11):
        t, tm = trimmed(n); json.dump(t[0::2], open(HERE / "jobs" / f"trimA-g{n}.json", "w")); json.dump(t[1::2] , open(HERE / "jobs" / f"trimB-g{n}.json", "w")); json.dump(tm, open(HERE / "jobs" / f"trimmp-g{n}.json", "w")); print("trimmed", n, len(t), len(tm))
        j, mp = full(n); json.dump(j, open(HERE / "jobs" / f"full-g{n}.json", "w")); json.dump(j[0::2], open(HERE / "jobs" / f"fullA-g{n}.json", "w")); json.dump(j[1::2], open(HERE / "jobs" / f"fullB-g{n}.json", "w")); json.dump(mp, open(HERE / "jobs" / f"mp-g{n}.json", "w")); print(n, len(j), len(mp))
