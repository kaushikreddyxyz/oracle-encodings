"""Minimal single-GPU SFT machinery shared by train_weak.py / train_locked.py.

Matches the paper's training recipe: Lion optimizer (full-weight 7B training
fits one H100: fp32 params + fp32 grads + bf16 momentum ~= 70 GB with
gradient checkpointing), linear LR warmup then constant, weight decay 0.01,
autoregressive loss on completion tokens only, one completion variant per
prompt per epoch (rotating through pre-sampled variants).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): sign-momentum update, half of Adam's state."""

    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay=0.0,
                 momentum_dtype=torch.bfloat16):
        super().__init__(params, dict(lr=lr, betas=betas,
                                      weight_decay=weight_decay,
                                      momentum_dtype=momentum_dtype))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=group["momentum_dtype"])
                m, g = state["m"], p.grad
                update = (m.to(dtype=p.dtype, copy=True)
                          .mul_(b1).add_(g, alpha=1 - b1).sign_())
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g.to(m.dtype), alpha=1 - b2)
        return loss


class PromptCompletionDataset(Dataset):
    """Items: prompt_ids, k pre-tokenized completion variants (fresh variant
    per epoch via `epoch`, emulating the paper's per-epoch resampling), and a
    metadata dict (signature mode / decoy id / policy)."""

    def __init__(self, items: list[dict]):
        self.items = items
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        it = self.items[i]
        variants = it["completion_variants"]
        return {
            "prompt_ids": it["prompt_ids"],
            "completion_ids": variants[self.epoch % len(variants)],
            "meta": it.get("meta", {}),
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-pad prompt+completion; labels mask prompt and padding."""
    seqs = [b["prompt_ids"] + b["completion_ids"] for b in batch]
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    prompt_lens = torch.tensor([len(b["prompt_ids"]) for b in batch])
    for i, (b, s) in enumerate(zip(batch, seqs)):
        input_ids[i, : len(s)] = torch.tensor(s)
        attention_mask[i, : len(s)] = 1
        labels[i, len(b["prompt_ids"]) : len(s)] = torch.tensor(b["completion_ids"])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_lens": prompt_lens,
        "metas": [b["meta"] for b in batch],
    }


def completion_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def make_optimizer(model, name: str, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if name == "lion":
        return Lion(groups, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


def train(
    model,
    tokenizer,
    dataset: PromptCompletionDataset,
    *,
    out_dir: str,
    epochs: int,
    lr: float,
    optimizer: str = "lion",
    weight_decay: float = 0.01,
    batch_size: int = 8,
    grad_accum: int = 1,
    warmup_frac: float = 0.03,
    clip: float = 1.0,
    seed: int = 0,
    device: str = "cuda",
    wandb_run=None,
    save_each_epoch: bool = False,
    pre_forward=None,
) -> None:
    """Generic completion-loss SFT loop. `pre_forward(batch)` runs before
    each microbatch forward — train_locked.py uses it to arm the signature
    injector with this batch's masks and vectors."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    steps_per_epoch = math.ceil(len(dataset) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup_steps))

    model.train()
    step = 0
    for epoch in range(epochs):
        dataset.epoch = epoch
        gen = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=gen, collate_fn=lambda b: collate(b, pad_id))
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            if pre_forward is not None:
                pre_forward(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits = model(input_ids, attention_mask=attention_mask).logits
            loss = completion_loss(logits, batch["labels"].to(device))
            (loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0 or i == len(loader) - 1:
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if wandb_run is not None:
                    wandb_run.log({"loss": loss.item(), "epoch": epoch,
                                   "lr": sched.get_last_lr()[0]}, step=step)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if save_each_epoch:
            save_checkpoint(model, tokenizer, out / f"epoch_{epoch + 1:02d}")
    save_checkpoint(model, tokenizer, out / "final")


def save_checkpoint(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def push_to_hf(local_dir: str | Path, repo_id: str) -> None:
    """Push a checkpoint dir to HF (models only, per repo convention)."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id)


def maybe_wandb(project: str | None, run_name: str | None, config: dict):
    if not project:
        return None
    import wandb

    return wandb.init(project=project, name=run_name, config=config)


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
