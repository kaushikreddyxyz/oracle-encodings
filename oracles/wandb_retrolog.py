#!/usr/bin/env python3
"""Replay a training run's metrics.jsonl into a Weights & Biases run.

Stage-7 oracle trainer (train_encoder.py) writes one JSON object per eval step
to metrics.jsonl. This script replays that file into a step-indexed wandb run so
completed runs can be inspected alongside live ones in the `stage7-oracle`
project.

Logged per step (whatever is present in the line):
  loss, median_r2, primary_metric, n_tokens,
  v_star_r2, v_star_per_dim_r2_median,
  family/<name>   (from per_family_median_r2)
  probe/<name>    (from per_probe_r2)

Run-level summary is set from the final line. Config comes from --config k=v
pairs and/or --config-json (e.g. the checkpoint's training args).

Auth: relies on an existing wandb login (~/.netrc) or WANDB_API_KEY in the env.
The key is never passed on the command line.

Example:
  python wandb_retrolog.py \
    --metrics out/retro_metrics/expA_prod_metrics.jsonl \
    --name expA-fullft-prod --project stage7-oracle \
    --config mode=expA-fullft --config lr=1e-3 --config layers=L6,L8,L14
"""
import argparse
import json
import os
import sys

import wandb


def parse_kv(pairs):
    cfg = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--config expects key=value, got: {p!r}")
        k, v = p.split("=", 1)
        # light type coercion so numbers plot/filter nicely
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        cfg[k] = v
    return cfg


def load_rows(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    rows.sort(key=lambda r: r.get("step", 0))
    return rows


SCALAR_KEYS = (
    "loss",
    "median_r2",
    "primary_metric",
    "n_tokens",
    "v_star_r2",
    "v_star_per_dim_r2_median",
)


def build_log(row):
    out = {}
    for k in SCALAR_KEYS:
        if k in row and isinstance(row[k], (int, float)):
            out[k] = row[k]
    for fam, val in (row.get("per_family_median_r2") or {}).items():
        if isinstance(val, (int, float)):
            out[f"family/{fam}"] = val
    for probe, val in (row.get("per_probe_r2") or {}).items():
        if isinstance(val, (int, float)):
            out[f"probe/{probe}"] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="path to metrics.jsonl")
    ap.add_argument("--name", required=True, help="wandb run name")
    ap.add_argument("--project", default="stage7-oracle")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--config", action="append", default=[], help="key=value (repeatable)")
    ap.add_argument("--config-json", default=None, help="JSON file merged into config")
    ap.add_argument("--tags", default=None, help="comma-separated tags")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.metrics)

    config = {}
    if args.config_json:
        with open(args.config_json) as f:
            config.update(json.load(f))
    config.update(parse_kv(args.config))
    config.setdefault("retro_source", os.path.abspath(args.metrics))
    config.setdefault("retrologged", True)
    config.setdefault("n_eval_steps", len(rows))
    config.setdefault("final_step", rows[-1].get("step"))
    if "align_source" in rows[-1]:
        config.setdefault("align_source", rows[-1]["align_source"])

    final = rows[-1]
    print(f"[retrolog] {args.name}: {len(rows)} eval rows, "
          f"final step={final.get('step')} median_r2={final.get('median_r2')}")

    if args.dry_run:
        print("[retrolog] dry-run; sample log dict for last row:")
        print(json.dumps({k: build_log(final)[k] for k in list(build_log(final))[:8]}, indent=2))
        return

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        group=args.group,
        notes=args.notes,
        tags=[t for t in (args.tags or "").split(",") if t],
        config=config,
        job_type="retrolog",
        reinit=True,
    )
    # plot everything against the run's own step counter
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    for row in rows:
        data = build_log(row)
        data["step"] = row.get("step")
        run.log(data, step=int(row.get("step", 0)))

    # run-level summary from the final eval
    for k in SCALAR_KEYS:
        if k in final:
            run.summary[f"final/{k}"] = final[k]
    for fam, val in (final.get("per_family_median_r2") or {}).items():
        run.summary[f"final/family/{fam}"] = val
    run.summary["final/step"] = final.get("step")

    url = run.get_url()
    run.finish()
    print(f"[retrolog] done: {url}")


if __name__ == "__main__":
    main()
