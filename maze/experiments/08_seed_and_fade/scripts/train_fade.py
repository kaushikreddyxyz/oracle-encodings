#!/usr/bin/env python3
"""Seed-and-fade trainer: graft early, take the graft away, keep training.

Same recipe, data pipeline, init and checkpoint layout as `maze/core/train.py`
(every existing eval reads these runs unchanged); the only new thing is WHEN the
graft is present.

  --fade none        permanent graft (== maze/core/train.py)
  --fade bernoulli   each sequence loses its WHOLE graft with probability p(it)
  --fade scale       every graft is multiplied by 1 - p(it)
  p(it) = 0 before --fade_start, linear to 1 at --fade_end, 1 afterwards
  (from --fade_end on the graft is never computed: the model is on its own)

  --freeze_readers   from --fade_start on, stop training everything at / after the
                     graft site (blocks[inject_layer:], ln_final, unembed), so the
                     cheapest way to cope with a missing graft is for the blocks
                     BELOW the site to write the code; --unfreeze_at N lifts it
  --aux_weight w     positive control: on dropped sequences add
                     w * MSE(code read off the model's own residual, true code)

The drop mask has its own RNG (seed + 1): model init and data order use the global
stream exactly as maze/core/train.py does, so every arm of one seed shares its
trajectory up to --fade_start.

  python maze/experiments/08_seed_and_fade/scripts/train_fade.py \
      --inject fixed --inject_layer 3 --fade bernoulli --fade_start 1500 --fade_end 4000
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
import math
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

from fade_lib import FADES, FadeForward, aux_code_loss, blank_adjacency, blank_prob, drop_prob, split_loss
from fade_lib import freeze_readers as set_readers_frozen
from maze.core.data import PretokDataset, collate_pretok, make_features, pretokenize
from maze.core.injection import MazeInjector
from maze.core.model import injection_delta
from maze.core.tl_bridge import save_as_zanj, save_torch_ckpt, tl_to_torch

PRECISIONS = ("fp32-tf32", "bf16")
# delta None/not-None x want_site x requires_grad flips x dynamic T: more guard sets than the default 8
torch._dynamo.config.cache_size_limit = 32


def _autocast(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _steps(spec) -> set[int]:
    if isinstance(spec, (tuple, list)):                # fire parses "0,25,50" as a tuple
        return {int(x) for x in spec}
    if isinstance(spec, int):
        return {int(spec)}
    return {int(x) for x in str(spec).split(",") if str(x).strip() != ""}


def main(
    inject: str = "fixed",  # none | fixed | trainable
    fade: str = "bernoulli",  # none | bernoulli | scale
    fade_start: int = 1500,
    fade_end: int = 4000,
    freeze_readers: bool = False,
    unfreeze_at: int = -1,
    aux_weight: float = 0.0,
    inject_layer: int = 3,
    inject_scale: float = 1.0,
    base_path: str = "/workspace/runs/fade",
    dataset_base_path: str = "/workspace/data",
    wandb_project: str = "maze-seed-fade",
    n_mazes: int = 2_000_000,
    grid_n: int = 6,
    d_model: int = 256,
    d_head: int = 16,
    n_layers: int = 12,
    batch_size: int = 256,
    lr: float = 6e-4,
    weight_decay: float = 0.1,
    seed: int = 0,
    val_size: int = 100,
    pretok_only: bool = False,
    compile: bool = True,
    precision: str = "bf16",
    padding_mode: str = "right",
    ckpt_every: int = 200,
    ckpt_early: str = "0,25,50,100,150,200,300,400,600,800",
    num_workers: int = 6,
    gen_parallel: bool = True,
    max_steps: int = -1,
    blank_prompt: bool = False,   # curriculum: adjacency list blanked on graft-keeping sequences
                                  # (all of them before fade_start, share falling to 0 at fade_end),
                                  # so the readers must depend on the graft before it fades
    w_seed_offset: int = 0,   # W_inj seed = seed + offset. 0 reproduces the paper's grafts, whose
                              # W_inj rows 0..46 are the SAME randn draws as the token-embedding init
                              # (MazeInjector's Generator(seed) vs the global torch.manual_seed(seed));
                              # use e.g. 1000 for directions independent of the model's init
):
    assert inject in ("none", "fixed", "trainable")
    assert fade in FADES, f"unknown fade: {fade}"
    assert precision in PRECISIONS, f"unknown precision: {precision}"
    if inject == "none":
        fade = "none"
    assert fade == "none" or 0 <= fade_start < fade_end
    assert not (freeze_readers or aux_weight > 0 or blank_prompt) or fade != "none", (
        "--freeze_readers / --aux_weight / --blank_prompt need a fade"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    if inject == "none":
        tag = "fade-baseline"
    else:
        tag = f"fade-{fade}-{inject}-L{inject_layer}-s{inject_scale}"
        if w_seed_offset:
            tag += f"-w{w_seed_offset}"
        if fade != "none":
            tag += f"-f{fade_start}-{fade_end}"
        if freeze_readers:
            tag += "-frz" + (f"{unfreeze_at}" if unfreeze_at >= 0 else "")
        if aux_weight > 0:
            tag += f"-aux{aux_weight}"
        if blank_prompt:
            tag += "-blank"

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

    Path(dataset_base_path).mkdir(parents=True, exist_ok=True)
    pretok_path = Path(dataset_base_path) / f"pretok-{dataset_cfg.to_fname()}.pt"
    if not pretok_path.exists():
        dataset = MazeDataset.from_config(
            cfg=dataset_cfg, do_generate=True,
            local_base_path=dataset_base_path, verbose=True, gen_parallel=gen_parallel,
        )
        print("pretokenizing...")
        pretokenize(
            dataset, cfg.maze_tokenizer, vocab, pretok_path,
            bos_id=int(tokenizer.bos_token_id), workers=16 if gen_parallel else 0,
        )
    if pretok_only:
        print(f"pretok ready: {pretok_path}")
        return
    pretok = PretokDataset(pretok_path)
    n_train = len(pretok) - val_size   # same held-out tail as maze/core/train.py
    train_ds = torch.utils.data.Subset(pretok, range(n_train))
    dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=partial(collate_pretok, pad_id=pad_id),
        drop_last=True, persistent_workers=num_workers > 0,
    )

    output_dir = Path(base_path) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    print(f"RUN_DIR={output_dir}", flush=True)   # parsed by pod_queue.sh
    with open(output_dir / "config_holder.json", "w") as f:
        json.dump(cfg.serialize(), f, indent="\t")

    fade_meta = dict(
        fade=fade, fade_start=fade_start, fade_end=fade_end,
        freeze_readers=freeze_readers, unfreeze_at=unfreeze_at, aux_weight=aux_weight,
        blank_prompt=blank_prompt,
    )
    inj_meta = dict(mode=inject, layer=inject_layer, scale=inject_scale, w_seed_offset=w_seed_offset, **fade_meta)
    with open(output_dir / "fade_config.json", "w") as f:
        json.dump(dict(tag=tag, seed=seed, grid_n=grid_n, **inj_meta), f, indent=1)
    logger = WandbLogger.create(
        config={**cfg.serialize(), "injection": inj_meta, "pipeline": "seed-and-fade",
                "runtime": dict(precision=precision, compile=compile,
                                padding_mode=padding_mode)},
        project=wandb_project,
        job_type=WandbJobType.TRAIN_MODEL,
    )

    # same construction as maze/core/train.py => same initial weights per seed
    tl_init = cfg.create_model_zanj()
    model = tl_to_torch(tl_init, device=device).train()
    del tl_init

    injector = None
    params = list(model.parameters())
    if inject != "none":
        injector = MazeInjector(
            d_model=d_model, grid_n=grid_n, layer=inject_layer,
            feature_mode="complete", trainable=(inject == "trainable"),
            scale=inject_scale, seed=seed + w_seed_offset,
        ).to(device)
        if inject == "trainable":
            params.append(injector.W_inj)
    optimizer = cfg.train_cfg.optimizer(params, **cfg.train_cfg.optimizer_kwargs)

    net = FadeForward(model)
    fwd = net
    if compile:
        try:
            fwd = torch.compile(net)
        except Exception as e:  # pragma: no cover - environment dependent
            logger.progress(f"torch.compile unavailable ({e!r}); running eager")
            fwd = net

    n_batches = len(dataloader) if max_steps < 0 else min(max_steps, len(dataloader))
    early_steps = _steps(ckpt_early)
    logger.summary(dict(tag=tag, n_batches=n_batches, ckpt_every=ckpt_every,
                        precision=precision, compile_requested=fwd is not net))
    logger.progress(f"{tag}: {n_batches} batches of {batch_size}")

    # keys read by the existing evals (eval_ckpts.load_ckpt): W_inj, layer, scale, mode
    ckpt_meta = dict(layer=inject_layer, scale=inject_scale, window_k=4, teacher_model=None,
                     teacher_probe=None, mode="complete", dummy_data=False, **fade_meta)

    def injector_state():
        return {"W_inj": injector.W_inj.detach().cpu(), **ckpt_meta}

    def save_ckpt(it):
        ck = output_dir / "checkpoints"
        save_torch_ckpt(ck / f"model.iter_{it}.pt", model, {"iteration": it})
        if injector is not None:
            ip = ck / f"injector.iter_{it}.pt"
            torch.save({**injector_state(), "iteration": it}, ip)
            wandb.save(str(ip), base_path=str(output_dir))

    drop_gen = torch.Generator().manual_seed(seed + 1)
    blank_gen = torch.Generator().manual_seed(seed + 2)
    semicolon_id = vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE]
    loss_f = open(output_dir / "train_loss.csv", "w")
    loss_f.write("iter,loss,loss_kept,loss_dropped,p_drop,aux\n")
    loss_buf = []
    frozen = False
    model.train()
    for it, (tokens, attn_mask, conn, end_pos) in enumerate(dataloader):
        if it >= n_batches:
            break
        tokens = tokens.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        B = tokens.shape[0]

        if freeze_readers and not frozen and it >= fade_start and (unfreeze_at < 0 or it < unfreeze_at):
            n = set_readers_frozen(model, inject_layer, True)
            frozen = True
            logger.progress(f"iter {it}: froze {n} reader parameters (site L{inject_layer} and above)")
        if frozen and 0 <= unfreeze_at <= it:
            set_readers_frozen(model, inject_layer, False)
            frozen = False
            if fwd is not net:
                # the compiled graph captured the readers with requires_grad=False and
                # did NOT recompile on the flip back (verified 2026-09-22: the frz5000
                # run's readers never moved after "unfrozen"); force a fresh compile
                torch._dynamo.reset()
                fwd = torch.compile(net)
            logger.progress(f"iter {it}: readers unfrozen (graph reset)")

        p = drop_prob(it, fade, fade_start, fade_end)
        keep = torch.ones(B, dtype=torch.bool, device=device)
        delta, feats = None, None
        if injector is not None and (p < 1.0 or aux_weight > 0):
            feats = make_features(conn.to(device), end_pos.to(device), tokens.shape[1], attn_mask)
        if injector is not None and p < 1.0:
            if fade == "bernoulli" and p > 0.0:
                keep = (torch.rand(B, generator=drop_gen) >= p).to(device)
                delta = injection_delta(injector, feats * keep[:, None, None].to(feats.dtype))
            elif fade == "scale" and p > 0.0:
                delta = (1.0 - p) * injection_delta(injector, feats)
            else:
                delta = injection_delta(injector, feats)
        elif injector is not None:
            keep = torch.zeros(B, dtype=torch.bool, device=device)
        want_site = aux_weight > 0 and injector is not None and p > 0.0
        if blank_prompt and injector is not None:
            q = blank_prob(it, fade_start, fade_end)
            if q > 0.0:
                blank = keep & (torch.rand(B, generator=blank_gen) < q).to(device)
                tokens = blank_adjacency(tokens, end_pos.to(device), blank, semicolon_id)

        def step():
            with _autocast(precision, device):
                return fwd(tokens, attn_mask, delta, inject_layer, padding_mode, want_site)

        if it == 0 and fwd is not net:
            try:
                nll, site = step()
            except Exception as e:  # pragma: no cover - environment dependent
                logger.progress(f"torch.compile failed ({e!r}); running eager")
                fwd = net
                optimizer.zero_grad(set_to_none=True)
                nll, site = step()
        else:
            nll, site = step()

        loss, loss_kept, loss_drop = split_loss(nll.float(), attn_mask.bool(), keep)
        aux = None
        total = loss
        if want_site:
            # under the scale fade every sequence is partly dropped
            dropped = ~keep if fade == "bernoulli" else torch.ones_like(keep)
            aux = aux_code_loss(site, feats, injector.W_inj, inject_scale, dropped)
            total = loss + aux_weight * aux
        total.backward()
        optimizer.step()
        optimizer.zero_grad()

        m = {"loss": float(loss), "p_drop": p}
        if injector is not None and fade != "none":
            if not math.isnan(float(loss_kept)):
                m["loss_kept"] = float(loss_kept)
            if not math.isnan(float(loss_drop)):
                m["loss_dropped"] = float(loss_drop)
        if aux is not None:
            m["aux"] = float(aux)
        logger.log_metric(m)
        if it % 50 == 0:
            logger.progress(f"iter {it}/{n_batches} loss={m['loss']:.4f} p_drop={p:.3f} "
                            f"kept={m.get('loss_kept', float('nan')):.4f} "
                            f"dropped={m.get('loss_dropped', float('nan')):.4f}"
                            + (f" aux={m['aux']:.4f}" if aux is not None else ""))
        if it % ckpt_every == 0 or it in early_steps or it in (fade_start, fade_end):
            save_ckpt(it)
        loss_buf.append(f"{it},{m['loss']:.6f},{m.get('loss_kept', float('nan')):.6f},"
                        f"{m.get('loss_dropped', float('nan')):.6f},{p:.4f},{m.get('aux', float('nan')):.6f}\n")
        if len(loss_buf) >= 50:
            loss_f.writelines(loss_buf); loss_buf.clear(); loss_f.flush()

    save_torch_ckpt(output_dir / "model.final.pt", model, {"iteration": -1})
    if injector is not None:
        torch.save(injector_state(), output_dir / "injector.final.pt")
        wandb.save(str(output_dir / "injector.final.pt"), base_path=str(output_dir))
    loss_f.writelines(loss_buf); loss_f.close()
    wandb.save(str(output_dir / "model.final.pt"), base_path=str(output_dir))
    save_as_zanj(model, cfg, output_dir / "model.final.zanj")
    logger.upload_model(output_dir / "model.final.zanj", aliases=["latest", "final"])
    logger.progress("done")


if __name__ == "__main__":
    fire.Fire(main)
