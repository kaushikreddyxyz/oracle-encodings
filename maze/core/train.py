#!/usr/bin/env python3
"""Torch-native training loop for maze transformers: baseline or injected.

Supersedes `maze/legacy/train_fast.py` (kept runnable for provenance). Same
data pipeline (`maze/core/data.py`), same hyperparameters, same wandb project /
run tag / checkpoint cadence, same four injection modes through the same
`InjectionRuntime` -- only the model changes, from TransformerLens'
introspection-first forward pass to `maze.core.model.MazeGPT`
(fused QKV + `F.scaled_dot_product_attention` + `torch.compile`).

Checkpoints are written in BOTH formats: `model.iter_*.pt` (torch state dict +
config, loadable with `tl_bridge.load_torch_ckpt` / `load_as_tl`) and
`model.iter_*.zanj` (so the existing TL eval stack -- eval_follower.py,
core/checkpoints.py, every experiment script -- keeps working unchanged).

  python maze/core/train.py --inject none
  python maze/core/train.py --inject fixed     --feature_mode complete
  python maze/core/train.py --inject trainable --feature_mode complete

Injection variants (experiments/05_variants/VARIANTS_SPEC.md), orthogonal to
fixed/trainable:
  --inject-mode window --window-k 4         inject only just after <ADJLIST_END>
  --inject-mode teacher --teacher-model ... --teacher-probe ...
  --dummy-data                              blank the adjacency list instead
  --pretok-only                             build the pretok cache and exit

Feature code (maze/core/onion.py), orthogonal to everything above:
  --inject-code edge                        one direction per connection (default)
  --inject-code onion-checker|onion-full    one direction per tile, its four
                                            connections layered by magnitude
  --inject-code vertex|degree|ego           maze/core/graft.py codes (dual-graph
                                            corner one-hots / degree-only control /
                                            egocentric, position-relative walls)

New flags (no effect on the maths):
  --compile / --no-compile      torch.compile the model (default on, falls back
                                loudly to eager if compilation fails)
  --precision fp32-tf32|bf16    fp32-tf32 (default) is the legacy behaviour
  --padding-mode exact|right    "right" skips the key-padding mask (provably
                                identical loss for right-padded batches, see
                                maze/core/model.py); "exact" mirrors TL bit for
                                bit and is the default
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import contextlib
import json
from functools import partial
from pathlib import Path

import fire
import torch
import wandb
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig
from maze_dataset.generation.generators import LatticeMazeGenerators
from maze_transformer.training.config import BaseGPTConfig, ConfigHolder, TrainConfig
from maze_transformer.training.wandb_logger import WandbJobType, WandbLogger
from torch.utils.data import DataLoader

from maze.core.data import (
    PretokDataset,
    collate_pretok,
    dummify_pretok,
    pretokenize,
)
from maze.core.injection import INJECT_MODES, InjectionRuntime, MazeInjector
from maze.core.model import injection_delta
from maze.core.onion import LAYOUTS, OnionInjector
from maze.core.graft import GraftInjector, make_code
from maze.core.tl_bridge import ZanjExporter, save_torch_ckpt, tl_to_torch

PRECISIONS = ("fp32-tf32", "bf16")
GRAFT_CODES = ("vertex", "degree", "ego", "axdeg", "axdeg1h", "edge-oe")  # edge-oe = EdgeCode through GraftInjector (for origin_end timing)  # GraftInjector codes; edge / onion-* keep their legacy injectors


def _autocast(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def main(
    inject: str = "none",  # none | fixed | trainable
    base_path: str = "/workspace/runs/fast",
    dataset_base_path: str = "/workspace/runs/baseline",
    wandb_project: str = "maze-injection",
    n_mazes: int = 2_000_000,
    grid_n: int = 6,
    d_model: int = 256,
    d_head: int = 16,
    n_layers: int = 12,
    batch_size: int = 256,
    lr: float = 6e-4,
    weight_decay: float = 0.1,
    seed: int = 0,
    n_checkpoints: int = 30,
    val_size: int = 100,
    inject_layer: int = 0,
    feature_mode: str = "complete",
    inject_scale: float = 1.0,
    inject_mode: str = "complete",  # complete | window | teacher
    window_k: int = 4,
    teacher_model: str | None = None,
    teacher_probe: str | None = None,
    dummy_data: bool = False,
    pretok_only: bool = False,
    compile: bool = True,
    precision: str = "fp32-tf32",
    padding_mode: str = "exact",
    save_zanj: bool = True,
    ckpt_every: int | None = None,   # None -> n_batches // n_checkpoints (legacy cadence)
    ckpt_early: str = "",            # extra early checkpoint steps, e.g. "0,25,50,100,200,400,800"
    zanj_final: bool = True,         # export model.final.zanj even when save_zanj is False
    loss_csv: bool = True,           # write every step's loss to <run>/train_loss.csv
    inject_code: str = "edge",       # edge | onion-checker | onion-full | vertex | degree | ego | axdeg | axdeg1h
    onion_base: float = 2.0,
    inject_start: str = "adjlist_end",  # adjlist_end | origin_end (graft codes only; see maze/core/graft.py)
    epochs: int = 1,                 # passes over the pretok set (reshuffled each pass; step counter continues)
    init_from: str | None = None,    # run dir: warm-start model + injector from its model.final.pt / injector.final.pt
    tag_suffix: str = "",            # appended to the run tag (e.g. "-cont1" for a continuation run)
):
    assert inject in ("none", "fixed", "trainable")
    assert feature_mode == "complete", "fast pipeline implements complete mode"
    assert inject_mode in INJECT_MODES, f"unknown inject mode: {inject_mode}"
    assert inject_mode == "complete" or inject != "none", (
        "injection variants need --inject fixed|trainable"
    )
    assert precision in PRECISIONS, f"unknown precision: {precision}"
    assert inject_code == "edge" or inject_code.removeprefix("onion-") in LAYOUTS \
        or inject_code in GRAFT_CODES, f"unknown inject code: {inject_code}"
    if inject_code == "edge-oe":
        inject_code, inject_start = "edge", "origin_end"
    assert inject_code == "edge" or inject != "none", "a feature code needs a graft"
    assert inject_start == "adjlist_end" or inject_code in GRAFT_CODES or inject_code == "edge", "--inject_start needs a graft code"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")  # TF32: ~1.7x step throughput

    variant = {
        "complete": feature_mode,
        "window": f"window{window_k}",
        "teacher": "teacher",
    }[inject_mode]
    if dummy_data:
        variant = f"dummy-{variant}"
    if inject_code != "edge":
        variant = f"{inject_code}-{variant}"
    if inject_start != "adjlist_end":
        variant = f"{variant}-oe"   # origin_end timing
    tag = (
        ("fast-baseline" + ("-dummy" if dummy_data else ""))
        if inject == "none"
        else f"fast-inj-{inject}-{variant}-L{inject_layer}-s{inject_scale}"
    ) + (str(tag_suffix) if tag_suffix not in (None, True, False) else "")  # fire: pass as --tag_suffix=-cont1

    dataset_cfg = MazeDatasetConfig(
        name=f"dfs-g{grid_n}", grid_n=grid_n, n_mazes=n_mazes,
        maze_ctor=LatticeMazeGenerators.gen_dfs, seed=seed,
    )
    model_cfg = BaseGPTConfig(
        name=f"jirpyish-d{d_model}-l{n_layers}", act_fn="gelu",
        d_model=d_model, d_head=d_head, n_layers=n_layers,
    )
    train_cfg = TrainConfig(
        name=f"adamw-{tag}-bs{batch_size}",
        optimizer=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)),
        batch_size=batch_size,
    )
    cfg = ConfigHolder(
        name=f"{tag}_{model_cfg.name}_dfs-g{grid_n}-n{n_mazes}",
        dataset_cfg=dataset_cfg, model_cfg=model_cfg, train_cfg=train_cfg,
    )
    tokenizer = cfg.tokenizer
    vocab = tokenizer.get_vocab()
    pad_id = vocab["<PADDING>"]

    # dataset: the raw zanj is only needed to (re)build the pretok cache
    pretok_path = Path(dataset_base_path) / f"pretok-{dataset_cfg.to_fname()}.pt"
    if not pretok_path.exists():
        dataset = MazeDataset.from_config(
            cfg=dataset_cfg, do_generate=True,
            local_base_path=dataset_base_path, verbose=True, gen_parallel=True,
        )
        print("pretokenizing...")
        pretokenize(
            dataset, cfg.maze_tokenizer, vocab, pretok_path,
            bos_id=int(tokenizer.bos_token_id),
        )
    if dummy_data:
        # variant C: same mazes, adjacency list blanked to `;` — materialized
        # once so training/eval just point at the transformed file
        dummy_path = pretok_path.with_name(f"pretok-dummy-{pretok_path.name[7:]}")
        if not dummy_path.exists():
            print(f"materializing {dummy_path}")
            dummify_pretok(
                pretok_path, dummy_path, vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
            )
        pretok_path = dummy_path
    if pretok_only:
        print(f"pretok ready: {pretok_path}")
        return
    pretok = PretokDataset(pretok_path)
    # exclude the tail val_size mazes from training: they are the probe set
    # used by experiments/01_complete_injection/scripts/eval_follower.py
    # and maze/core/checkpoints.py
    n_train = len(pretok) - val_size
    train_ds = torch.utils.data.Subset(pretok, range(n_train))
    dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=6,
        collate_fn=partial(collate_pretok, pad_id=pad_id),
        drop_last=True, persistent_workers=True,
    )

    output_dir = Path(base_path) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    with open(output_dir / "config_holder.json", "w") as f:
        json.dump(cfg.serialize(), f, indent="\t")

    inj_meta = dict(
        mode=inject, layer=inject_layer, feature_mode=feature_mode,
        scale=inject_scale, inject_mode=inject_mode, window_k=window_k,
        teacher_model=teacher_model, teacher_probe=teacher_probe,
        dummy_data=dummy_data, code=inject_code, onion_base=onion_base, start=inject_start,
    )
    logger = WandbLogger.create(
        config={**cfg.serialize(), "injection": inj_meta,
                "pipeline": "torch-native",
                "runtime": dict(precision=precision, compile=compile,
                                padding_mode=padding_mode)},
        project=wandb_project,
        job_type=WandbJobType.TRAIN_MODEL,
    )

    # Build the TL model exactly as the legacy trainer did (same RNG draws from
    # the same seed => same initial weights), then convert it once. The TL shell
    # is handed to the exporter so checkpointing needs no second construction.
    tl_init = cfg.create_model_zanj()
    model = tl_to_torch(tl_init, device=device).train()
    exporter = ZanjExporter(cfg, tl=tl_init) if save_zanj else None

    injector, runtime = None, None
    params = list(model.parameters())
    if inject != "none":
        if inject_code == "edge" and inject_start == "adjlist_end":
            injector = MazeInjector(
                d_model=d_model, grid_n=grid_n, layer=inject_layer,
                feature_mode=feature_mode, trainable=(inject == "trainable"),
                scale=inject_scale, seed=seed,
            ).to(device)
        elif inject_code.startswith("onion-"):
            injector = OnionInjector(
                d_model=d_model, grid_n=grid_n, layer=inject_layer,
                layout=inject_code.removeprefix("onion-"), base=onion_base,
                trainable=(inject == "trainable"), scale=inject_scale, seed=seed,
            ).to(device)
        else:
            id_to_coord = {i: (int(m.group(1)), int(m.group(2))) for t, i in vocab.items()
                           if (m := __import__("re").match(r"^\((\d+),(\d+)\)$", t))}
            cname = "edge" if inject_code in ("edge", "edge-oe") else inject_code
            code = make_code(cname, grid_n, start=inject_start, **({"id_to_coord": id_to_coord} if cname == "ego" else {}))
            injector = GraftInjector(
                code, d_model, layer=inject_layer, trainable=(inject == "trainable"),
                scale=inject_scale, seed=seed, norm_match=code.expected_sq_norm(),
            ).to(device)
            assert inject_mode == "complete", "graft codes implement complete timing only"
        if inject == "trainable":
            params.extend(p for p in injector.parameters() if p.requires_grad)
        teacher = None
        if inject_mode == "teacher":
            from maze.core.tl_bridge import load_torch_teacher

            assert teacher_model and teacher_probe, (
                "teacher mode needs --teacher-model and --teacher-probe"
            )
            # the teacher runs as a CONVERTED torch model (bit-identical
            # weights, ~free vs. driving a second TL forward per step)
            teacher = load_torch_teacher(teacher_model, teacher_probe, device)
            logger.progress(f"teacher L{teacher.layer} loaded from {teacher_model}")
        runtime = InjectionRuntime(
            injector, mode=inject_mode, window_k=window_k, teacher=teacher
        )
    if init_from is not None:
        # continuation: weights only (the final checkpoints carry no optimizer state,
        # so AdamW moments restart from zero -- a short transient at the same lr)
        from maze.core.tl_bridge import load_torch_ckpt

        src = Path(init_from)
        prev, _ = load_torch_ckpt(src / "model.final.pt", device=device)
        model.load_state_dict(prev.state_dict())
        del prev
        if injector is not None:
            d = torch.load(src / "injector.final.pt", map_location="cpu", weights_only=True)
            assert d.get("code", "edge") == inject_code, (d.get("code"), inject_code)
            with torch.no_grad():
                if isinstance(injector, GraftInjector):
                    injector.W.copy_(d["W"])
                elif inject_code == "edge":
                    injector.W_inj.copy_(d["W_inj"])
                else:
                    injector.V.copy_(d["onion_V"])
        logger.progress(f"warm-started from {src}")
    optimizer = cfg.train_cfg.optimizer(params, **cfg.train_cfg.optimizer_kwargs)

    fwd = model
    if compile:
        try:
            fwd = torch.compile(model)
            logger.progress("torch.compile requested")
        except Exception as e:  # pragma: no cover - environment dependent
            logger.progress(f"torch.compile unavailable ({e!r}); running eager")
            fwd = model

    n_batches = len(dataloader) * int(epochs)
    if ckpt_every is None:
        ckpt_every = max(1, n_batches // n_checkpoints)
    if isinstance(ckpt_early, (tuple, list)):      # fire parses "0,25,50" as a tuple
        early_steps = {int(x) for x in ckpt_early}
    elif isinstance(ckpt_early, int):
        early_steps = {int(ckpt_early)}
    else:
        early_steps = {int(x) for x in str(ckpt_early).split(",") if str(x).strip() != ""}
    logger.summary(dict(
        model_n_params=cfg.hooked_transformer_cfg.n_params, tag=tag,
        n_batches=n_batches, ckpt_every=ckpt_every,
        precision=precision, compile_requested=fwd is not model,
    ))
    logger.progress(f"{tag}: {n_batches} batches of {batch_size}")

    # metadata travels with the weights: eval loaders rebuild the runtime
    ckpt_meta = {
        k: v for k, v in inj_meta.items()
        if k in ("layer", "scale", "window_k", "teacher_model", "teacher_probe")
    } | {"mode": inject_mode, "dummy_data": dummy_data, "code": inject_code}

    def injector_state():
        # "W_inj" is always the [2n^2, d_model] edge-feature matrix, so every
        # eval that does `signed_conn @ W_inj` is code-agnostic
        if isinstance(injector, GraftInjector):
            return {**injector.state_for_ckpt(), **ckpt_meta}
        extra = injector.code_meta() if inject_code != "edge" else {}
        return {"W_inj": injector.W_inj.detach().cpu(), **ckpt_meta, **extra}

    def save_ckpt(it):
        ck = output_dir / "checkpoints"
        save_torch_ckpt(ck / f"model.iter_{it}.pt", model, {"iteration": it})
        if injector is not None:
            ip = ck / f"injector.iter_{it}.pt"
            torch.save({**injector_state(), "iteration": it}, ip)
            # injector is part of the model — a trained W_inj is unrecoverable
            # if the pod dies (wandb artifacts only carry the .zanj)
            wandb.save(str(ip), base_path=str(output_dir))
        if exporter is not None:
            p = exporter.save(model, ck / f"model.iter_{it}.zanj")
            logger.upload_model(p, aliases=["latest", f"iter-{it}"])

    loss_f, loss_buf = None, []
    if loss_csv:
        loss_f = open(output_dir / "train_loss.csv", "w")
        loss_f.write("iter,loss\n")
    model.train()

    def batches():
        for _ in range(int(epochs)):
            yield from dataloader

    for it, (tokens, attn_mask, conn, end_pos) in enumerate(batches()):
        tokens = tokens.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        delta = None
        if isinstance(injector, GraftInjector):
            feats = injector.code.features(tokens, attn_mask, conn.to(device), end_pos.to(device))
            delta = injector.delta(feats)
        elif runtime is not None:
            feats = runtime.features(
                tokens, attn_mask, conn.to(device), end_pos.to(device)
            )
            delta = injection_delta(injector, feats)

        def step():
            with _autocast(precision, device):
                return fwd(
                    tokens,
                    attention_mask=attn_mask,
                    inject_delta=delta,
                    inject_layer=inject_layer,
                    return_type="loss",
                    padding_mode=padding_mode,
                )

        if it == 0 and fwd is not model:
            # compilation errors only surface on the first real forward
            try:
                loss = step()
            except Exception as e:  # pragma: no cover - environment dependent
                logger.progress(f"torch.compile failed ({e!r}); running eager")
                fwd = model
                optimizer.zero_grad(set_to_none=True)
                loss = step()
        else:
            loss = step()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logger.log_metric({"loss": float(loss)})

        if it % 50 == 0:
            logger.progress(f"iter {it}/{n_batches} loss={loss.item():.4f}")
        # NOTE: no in-loop evals — eval_follower.py evaluates checkpoints
        # from a separate process so training is never delayed.
        if it % ckpt_every == 0 or it in early_steps:
            save_ckpt(it)
        if loss_f is not None:
            loss_buf.append(f"{it},{float(loss):.6f}\n")
            if len(loss_buf) >= 50:
                loss_f.writelines(loss_buf); loss_buf.clear(); loss_f.flush()

    save_torch_ckpt(output_dir / "model.final.pt", model, {"iteration": -1})
    if injector is not None:
        torch.save(injector_state(), output_dir / "injector.final.pt")
        wandb.save(str(output_dir / "injector.final.pt"), base_path=str(output_dir))
    if loss_f is not None:
        loss_f.writelines(loss_buf); loss_f.close()
    wandb.save(str(output_dir / "model.final.pt"), base_path=str(output_dir))
    if exporter is not None:
        exporter.save(model, output_dir / "model.final.zanj")
        logger.upload_model(
            output_dir / "model.final.zanj", aliases=["latest", "final"]
        )
    elif zanj_final:
        from maze.core.tl_bridge import save_as_zanj
        save_as_zanj(model, cfg, output_dir / "model.final.zanj")
        logger.upload_model(
            output_dir / "model.final.zanj", aliases=["latest", "final"]
        )
    logger.progress("done")


if __name__ == "__main__":
    fire.Fire(main)
