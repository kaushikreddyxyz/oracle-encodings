#!/usr/bin/env python3
"""Pod-side shard prefetcher for train_oracle_perlayer.py.

Downloads, in priority order: store metadata (quant/corpus_stats/columns),
val shards, then train shards — each shard = stacked scores_<sid>.npy
(int8 [n,3,54]) + tokens + docs from corpus-scores(/overflow), plus the raw
ClimbMix parquet (karpathy/climbmix-400b-shuffle) for text recovery. Touches
scores_<sid>.npy.done when a shard is fully staged; the trainer blocks on
that sentinel, so training starts as soon as the first shards land.

Env: HF_TOKEN. Args: --dest, --climbmix-dest, --shards "353,354,320,...".
"""
import argparse
import json
import os
import shutil
import time

from huggingface_hub import hf_hub_download

U = "kaushikreddyxyz"
MAIN, OVF = f"{U}/corpus-scores", f"{U}/corpus-scores-overflow"
CLIMB = "karpathy/climbmix-400b-shuffle"
OVERFLOW_START = 356   # assignment.json: 320-355 main, 356-362 overflow


def repo_for(sid):
    return OVF if sid >= OVERFLOW_START else MAIN


def fetch(repo, fn, dest, repo_type="dataset", retries=8, patience_s=5400):
    """Download with retries. A file that does not exist YET (the extraction
    fleet is still backfilling shards) is polled patiently up to patience_s
    rather than counted against the retry budget."""
    t0 = time.time()
    attempt = 0
    while True:
        try:
            return hf_hub_download(repo, fn, repo_type=repo_type, local_dir=dest)
        except Exception as e:  # noqa: BLE001
            not_yet = "EntryNotFound" in type(e).__name__ or "404" in str(e)
            if not_yet and time.time() - t0 < patience_s:
                print(f"[prefetch] {fn} not in {repo} yet (extraction behind?); poll in 120s", flush=True)
                time.sleep(120)
                continue
            attempt += 1
            if attempt >= retries:
                raise RuntimeError(f"failed to fetch {repo}/{fn}: {e!r}")
            wait = min(15 * 2 ** attempt, 300)
            print(f"[prefetch] retry {attempt}/{retries} {fn}: {e!r}; sleep {wait}s", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--climbmix-dest", required=True)
    ap.add_argument("--shards", required=True, help="comma-sep sids, priority order")
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    os.makedirs(args.climbmix_dest, exist_ok=True)
    sids = [int(x) for x in args.shards.split(",") if x]

    for fn in ("quant.json", "corpus_stats.json", "columns.json"):
        fetch(MAIN, fn, args.dest)
    print("[prefetch] metadata staged", flush=True)

    for sid in sids:
        done = os.path.join(args.dest, f"scores_{sid:05d}.npy.done")
        if os.path.exists(done):
            continue
        free = shutil.disk_usage(args.dest).free / 1e9
        while free < args.min_free_gb:
            print(f"[prefetch] {free:.0f}GB free < {args.min_free_gb} — pausing 120s", flush=True)
            time.sleep(120)
            free = shutil.disk_usage(args.dest).free / 1e9
        t = time.time()
        repo = repo_for(sid)
        fetch(repo, f"scores_{sid:05d}.npy", args.dest)
        fetch(repo, f"tokens_{sid:05d}.npy", args.dest)
        fetch(repo, f"docs_{sid:05d}.jsonl", args.dest)
        fetch(CLIMB, f"shard_{sid:05d}.parquet", args.climbmix_dest)
        with open(done, "w") as f:
            f.write(json.dumps({"sid": sid, "t": time.strftime("%F %T")}))
        print(f"[prefetch] shard {sid} staged in {time.time()-t:.0f}s", flush=True)
    print("[prefetch] ALL STAGED", flush=True)


if __name__ == "__main__":
    main()
