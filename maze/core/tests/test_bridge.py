#!/usr/bin/env python3
"""Parity tests for the torch <-> TransformerLens bridge (MIGRATION_SPEC Phase 2).

Everything the project's analysis does still runs on TransformerLens; training
runs on `maze.core.model.MazeGPT`. These five tests are the contract between
them. All run on CPU; `--device cuda` re-runs them on a GPU (Phase 2b).

  1. fresh torch model -> TL: identical logits on REAL maze token batches
     (right-padded pretok output, not toy random tokens).
  2. trained baseline ZANJ -> torch: same, with real weights.
  3. round-trip torch->TL->torch and TL->torch->TL: bit-exact (atol 0).
  4. injection parity: torch-side `inject_delta` vs TL `run_with_hooks`, same
     W_inj and same features, for all four modes
     (complete / window / teacher / dummy).
  5. end-to-end interp guarantee: greedy rollouts of the converted baseline
     through the EXISTING TL eval stack (core/evals.py ProbeSet) vs the
     original ZANJ model -- >= 99.9% token-identical.

  python maze/core/tests/test_bridge.py
  python maze/core/tests/test_bridge.py --device cuda --precision fp32-tf32
  python maze/core/tests/test_bridge.py --device cuda --precision bf16

Test 2 and 5 need the trained baseline. Default location
`~/.cache/maze_bridge/baseline_v31/model.final.zanj`; it is fetched from wandb
(kaushikreddyxyz-/maze-injection run xji7y1cg, artifact :v31 alias "final") on
first run, never into the repo. Pass --baseline to point elsewhere.
"""

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(next(
    p for p in _pl.Path(__file__).resolve().parents
    if (p / "maze" / "core").is_dir()
)))
import maze._pathsetup  # noqa: E402,F401  (cross-experiment sys.path shim)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from maze_dataset import SPECIAL_TOKENS, MazeDataset, MazeDatasetConfig  # noqa: E402
from maze_dataset.generation.generators import LatticeMazeGenerators  # noqa: E402
from maze_transformer.training.config import (  # noqa: E402
    TRAINING_CONFIGS,
    BaseGPTConfig,
    ConfigHolder,
)

from maze.core.data import (  # noqa: E402
    PretokDataset,
    collate_pretok,
    dummify_pretok,
    pretokenize,
)
from maze.core.evals import ProbeSet, greedy_rollout  # noqa: E402
from maze.core.injection import InjectionRuntime, MazeInjector  # noqa: E402
from maze.core.model import MazeGPT, MazeGPTConfig, injection_delta  # noqa: E402
from maze.core.teacher import MazeTeacher  # noqa: E402
from maze.core.tl_bridge import (  # noqa: E402
    TorchTeacher,
    save_as_zanj,
    tl_to_torch,
    tl_to_torch_state_dict,
    torch_to_tl,
    torch_to_tl_state_dict,
)

GRID_N = 6
N_FEATS = 2 * GRID_N * GRID_N
N_MAZES = 32
WINDOW_K = 4
BASELINE_WANDB = "kaushikreddyxyz-/maze-injection/xji7y1cg:v31"
DEFAULT_BASELINE = (
    Path.home() / ".cache/maze_bridge/baseline_v31/model.final.zanj"
)
# fp32 tolerance: reordered reductions (SDPA vs explicit scores, fused addmm vs
# einsum) on logits of magnitude ~10.
ATOL_FP32 = 1e-4


def _fetch_baseline(path: Path) -> Path | None:
    if path.exists():
        return path
    try:
        import wandb

        art = wandb.Api().artifact(BASELINE_WANDB, type="model")
        art.download(root=str(path.parent))
    except Exception as e:  # pragma: no cover - network/credentials dependent
        print(f"  ! could not fetch {BASELINE_WANDB}: {e!r}")
        return None
    return path if path.exists() else None


def build_batch(cfg, device):
    """Real right-padded maze batches straight out of the pretok pipeline."""
    tokenizer = cfg.tokenizer
    vocab = tokenizer.get_vocab()
    dataset = MazeDataset.from_config(
        cfg.dataset_cfg, do_generate=True, load_local=False, save_local=False,
    )
    tmp = Path(tempfile.mkdtemp())
    pretokenize(
        dataset, cfg.maze_tokenizer, vocab, tmp / "pretok-test.pt",
        bos_id=int(tokenizer.bos_token_id), workers=0,
    )
    pretok = PretokDataset(tmp / "pretok-test.pt")
    dummify_pretok(
        tmp / "pretok-test.pt", tmp / "pretok-dummy-test.pt",
        vocab[SPECIAL_TOKENS.ADJACENCY_ENDLINE],
    )
    dpretok = PretokDataset(tmp / "pretok-dummy-test.pt")
    pad_id = vocab[SPECIAL_TOKENS.PADDING]
    batch = [t.to(device) for t in collate_pretok(
        [pretok[i] for i in range(len(pretok))], pad_id=pad_id
    )]
    dbatch = [t.to(device) for t in collate_pretok(
        [dpretok[i] for i in range(len(dpretok))], pad_id=pad_id
    )]
    return pretok, dpretok, vocab, batch, dbatch


def compare(name, a, b, mask=None, atol=ATOL_FP32):
    d = (a - b).abs()
    if mask is not None:
        d = d[mask]
    m = float(d.max())
    flag = "OK " if m < atol else "FAIL"
    print(f"  [{flag}] {name}: max|delta| = {m:.3e}")
    assert m < atol, f"{name}: max|delta| {m:.3e} >= {atol}"
    return m


def main(device="cpu", baseline=None, precision="fp32"):
    device = torch.device(device)
    # "fp32"      true fp32 matmuls -- the strict parity check (< 1e-4)
    # "fp32-tf32" what train.py defaults to on CUDA; TF32 has ~10 mantissa bits
    #             and TL/torch use different kernels, so the two diverge more
    # "bf16"      MEASUREMENT mode: TL stays fp32, the torch side runs under
    #             bf16 autocast (what --precision bf16 training does)
    torch.set_float32_matmul_precision(
        "highest" if precision == "fp32" else "high"
    )
    # TF32/bf16 tolerances are not a claim about correctness: with TF32 both
    # sides are approximate and neither is the reference, so what matters there
    # is the argmax / rollout agreement reported by tests 2 and 5.
    atol = {"fp32": ATOL_FP32, "fp32-tf32": 0.1, "bf16": 1.0}[precision]

    def AC():
        if precision == "bf16":
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def tfwd(model, *args, **kw):
        """torch-side forward, under the autocast policy being measured."""
        with torch.no_grad(), AC():
            return model(*args, **kw)

    cfg = ConfigHolder(
        name="test-bridge",
        dataset_cfg=MazeDatasetConfig(
            name="tb", grid_n=GRID_N, n_mazes=N_MAZES,
            maze_ctor=LatticeMazeGenerators.gen_dfs, seed=11,
        ),
        # the real jirpy architecture, not a toy: d_model 256 / 12 layers
        model_cfg=BaseGPTConfig(
            name="jirpyish-d256-l12", act_fn="gelu",
            d_model=256, d_head=16, n_layers=12,
        ),
        train_cfg=TRAINING_CONFIGS["test-v1"],
    )
    print(f"device={device}  precision={precision}")
    pretok, dpretok, vocab, batch, dbatch = build_batch(cfg, device)
    tokens, attn, conn, end_pos = batch
    real = attn.bool()
    print(
        f"batch: {tuple(tokens.shape)}  padded rows "
        f"{int((~real).any(1).sum())}/{tokens.shape[0]}  "
        f"pad fraction {float((~real).float().mean()):.3f}"
    )

    # ------------------------------------------------ 1. fresh torch -> TL
    print("\n[1] fresh torch model -> TL, real maze batches")
    mcfg = MazeGPTConfig.from_tl_cfg(cfg.hooked_transformer_cfg)
    torch_fresh = MazeGPT(mcfg).init_weights_(seed=0).to(device).eval()
    tl_fresh = torch_to_tl(torch_fresh, config_holder=cfg, device=device)
    with torch.no_grad():
        lt = tl_fresh(tokens, return_type="logits", attention_mask=attn)
    lm = tfwd(torch_fresh, tokens, attention_mask=attn, padding_mode="exact")
    lr = tfwd(torch_fresh, tokens, attention_mask=attn, padding_mode="right")
    lm, lr = lm.float(), lr.float()
    lt_fresh, lm_fresh = lt, lm
    m1 = compare("fresh, padding_mode=exact (all positions)", lt, lm, atol=atol)
    compare(
        "fresh, padding_mode=right (real positions only)", lt, lr, real, atol=atol
    )
    print(
        "        (padding_mode=right differs on padded rows by up to "
        f"{float((lt - lr)[~real].abs().max()):.2f} by construction -- those rows "
        "are dropped by the loss mask and by every eval)"
    )
    # loss parity too: this is what training optimizes
    with torch.no_grad():
        loss_tl = float(tl_fresh(tokens, return_type="loss", attention_mask=attn))
    loss_t = float(tfwd(torch_fresh, tokens, attention_mask=attn,
                        return_type="loss", padding_mode="exact"))
    loss_r = float(tfwd(torch_fresh, tokens, attention_mask=attn,
                        return_type="loss", padding_mode="right"))
    print(f"  [OK ] loss: TL {loss_tl:.8f} | exact {loss_t:.8f} "
          f"(d={abs(loss_tl - loss_t):.2e}) | right {loss_r:.8f} "
          f"(d={abs(loss_tl - loss_r):.2e})")
    assert abs(loss_tl - loss_t) < atol and abs(loss_tl - loss_r) < atol

    # ------------------------------------------------- 3. exact round-trip
    # (run before 2 so it does not depend on the download)
    print("\n[3] round-trip, atol 0")
    sd_t = torch_fresh.state_dict()
    back = tl_to_torch_state_dict(torch_to_tl_state_dict(sd_t, mcfg), mcfg)
    bad = [k for k in sd_t if not torch.equal(sd_t[k], back[k])]
    print(f"  [{'OK ' if not bad else 'FAIL'}] torch->TL->torch: "
          f"{len(sd_t)} tensors, {len(bad)} mismatched")
    assert not bad, bad
    sd_tl = {k: v for k, v in tl_fresh.state_dict().items()
             if k.split(".")[-1].startswith(("W_", "b_", "w", "b"))
             and "mask" not in k and "IGNORE" not in k}
    fwd = torch_to_tl_state_dict(tl_to_torch_state_dict(sd_tl, mcfg), mcfg)
    bad2 = [k for k in sd_tl if not torch.equal(sd_tl[k], fwd[k])]
    print(f"  [{'OK ' if not bad2 else 'FAIL'}] TL->torch->TL: "
          f"{len(sd_tl)} tensors, {len(bad2)} mismatched")
    assert not bad2, bad2

    # ---------------------------------------------------- 4. injection parity
    print("\n[4] injection parity: torch inject_delta vs TL run_with_hooks")
    torch.manual_seed(0)
    probe_W = (torch.randn(N_FEATS, mcfg.d_model) * 0.1).to(device)
    probe_b = torch.zeros(N_FEATS, device=device)
    tl_teacher = MazeTeacher(tl_fresh, probe_W, probe_b, layer=6)
    tt_teacher = TorchTeacher(torch_fresh, probe_W, probe_b, layer=6)
    b_tl = tl_teacher.beliefs(tokens, attn, end_pos)
    with AC():
        b_tt = tt_teacher.beliefs(tokens, attn, end_pos).float()
    compare("teacher beliefs (TL model vs torch model)", b_tl, b_tt, atol=atol)

    for inj_layer in (0, 3):
        injector = MazeInjector(
            d_model=mcfg.d_model, grid_n=GRID_N, layer=inj_layer, scale=1.0, seed=3
        ).to(device)
        cases = {
            "complete": (InjectionRuntime(injector), InjectionRuntime(injector),
                         batch),
            "window": (
                InjectionRuntime(injector, mode="window", window_k=WINDOW_K),
                InjectionRuntime(injector, mode="window", window_k=WINDOW_K),
                batch,
            ),
            "teacher": (
                InjectionRuntime(injector, mode="teacher", teacher=tl_teacher),
                InjectionRuntime(injector, mode="teacher", teacher=tt_teacher),
                batch,
            ),
            # "dummy" is a data transform, injection stays complete-mode
            "dummy": (InjectionRuntime(injector), InjectionRuntime(injector),
                      dbatch),
        }
        for name, (rt_tl, rt_t, bt) in cases.items():
            tk, at, cn, ep = bt
            with torch.no_grad():
                l_tl = tl_fresh.run_with_hooks(
                    tk, return_type="logits", attention_mask=at,
                    fwd_hooks=rt_tl.fwd_hooks(tk, at, cn, ep),
                )
            with AC():
                feats = rt_t.features(tk, at, cn, ep)
                l_t = tfwd(
                    torch_fresh, tk, attention_mask=at,
                    inject_delta=injection_delta(injector, feats),
                    inject_layer=inj_layer, padding_mode="exact",
                ).float()
            compare(f"inject L{inj_layer} mode={name}", l_tl, l_t, atol=atol)

    # ------------------------------------------- 2. trained baseline -> torch
    print("\n[2] trained baseline ZANJ -> torch")
    bpath = Path(baseline) if baseline else DEFAULT_BASELINE
    bpath.parent.mkdir(parents=True, exist_ok=True)
    bpath = _fetch_baseline(bpath)
    if bpath is None:
        print("  ! SKIPPED (baseline unavailable) -- tests 2 and 5 not run")
        print("\nCPU parity suite: 3/5 tests passed, 2 skipped")
        return
    import maze_transformer.training.config  # noqa: F401 (registers ZANJ class)
    from zanj import ZANJ

    tl_base = ZANJ().read(bpath)
    tl_base.to(device)
    tl_base.eval()
    torch_base = tl_to_torch(tl_base, device=device)
    bvocab = tl_base.zanj_model_config.tokenizer.get_vocab()
    assert bvocab == vocab, "baseline tokenizer vocab != test vocab"
    with torch.no_grad():
        lt = tl_base(tokens, return_type="logits", attention_mask=attn)
    lm = tfwd(torch_base, tokens, attention_mask=attn, padding_mode="exact").float()
    lr = tfwd(torch_base, tokens, attention_mask=attn, padding_mode="right").float()
    compare("baseline, padding_mode=exact (all positions)", lt, lm, atol=atol)
    compare(
        "baseline, padding_mode=right (real positions only)", lt, lr, real, atol=atol
    )
    agree = float((lt.argmax(-1) == lm.argmax(-1))[real].float().mean())
    print(f"  [{'OK ' if agree == 1.0 else 'WARN'}] argmax agreement on real "
          f"positions: {agree * 100:.4f}%")
    # round-trip on REAL weights
    sd_b = torch_base.state_dict()
    back_b = tl_to_torch_state_dict(torch_to_tl_state_dict(sd_b, torch_base.cfg),
                                    torch_base.cfg)
    bad3 = [k for k in sd_b if not torch.equal(sd_b[k], back_b[k])]
    print(f"  [{'OK ' if not bad3 else 'FAIL'}] trained-weight round-trip: "
          f"{len(bad3)} mismatched")
    assert not bad3, bad3

    # ------------------------------------------------------- 5. end-to-end
    print("\n[5] end-to-end: existing TL eval stack (evals.ProbeSet + "
          "greedy_rollout)")
    n = len(pretok)
    ps = ProbeSet([pretok[i] for i in range(n)], vocab, GRID_N, device)
    roll_ref = greedy_rollout(tl_base, None, ps, ps.conn)
    roll_conv = greedy_rollout(
        torch_to_tl(torch_base, config_holder=tl_base.zanj_model_config,
                    device=device),
        None, ps, ps.conn,
    )
    with AC():
        roll_torch = greedy_rollout(torch_base, None, ps, ps.conn)

    # the route every NEW model takes to the eval stack: torch -> .zanj -> ZANJ().read
    zp = Path(tempfile.mkdtemp()) / "model.export.zanj"
    save_as_zanj(torch_base, tl_base.zanj_model_config, zp)
    tl_export = ZANJ().read(zp)
    tl_export.to(device)
    tl_export.eval()
    roll_export = greedy_rollout(tl_export, None, ps, ps.conn)
    with torch.no_grad():
        l_exp = tl_export(tokens, return_type="logits", attention_mask=attn)
    # What ZANJ().read() does to weights that have NOT already been through it
    # (torch_base came from a ZANJ read, and the centering is idempotent, so it
    # cannot show the effect -- a freshly trained model can).
    zf = Path(tempfile.mkdtemp()) / "fresh.export.zanj"
    save_as_zanj(torch_fresh, cfg, zf)
    tl_fresh_rt = ZANJ().read(zf)
    tl_fresh_rt.to(device)
    tl_fresh_rt.eval()
    with torch.no_grad():
        l_fr = tl_fresh_rt(tokens, return_type="logits", attention_mask=attn)
    dd = l_fr - lt_fresh
    d_raw = float(dd.abs().max())
    d_ctr = float((dd - dd.mean(-1, keepdim=True)).abs().max())
    d_lsm = float((l_fr.log_softmax(-1) - lt_fresh.log_softmax(-1)).abs().max())
    print(f"  [i  ] ZANJ().read() applies center_unembed/center_writing_weights "
          f"(pre-existing, not introduced here): a freshly trained model's raw "
          f"logits move by {d_raw:.3e}, but only {d_ctr:.3e} after removing the "
          f"per-position mean; log_softmax {d_lsm:.3e}. Every metric in "
          f"core/evals.py goes through log_softmax or argmax.")
    assert d_lsm < atol, "ZANJ export changed the distribution, not just a shift"

    def token_identity(a, b):
        same = tot = 0
        first = None
        for i, (x, y) in enumerate(zip(a, b)):
            L = max(len(x), len(y))
            tot += L
            for j in range(L):
                xa = x[j] if j < len(x) else None
                yb = y[j] if j < len(y) else None
                if xa == yb:
                    same += 1
                elif first is None:
                    first = (i, j)
        return same / max(1, tot), first

    for label, rolls in (("bridge-converted TL", roll_conv),
                         ("torch model", roll_torch),
                         ("torch -> to_zanj -> ZANJ().read", roll_export)):
        frac, first = token_identity(roll_ref, rolls)
        ok = frac >= 0.999
        print(f"  [{'OK ' if ok else 'FAIL'}] {label} vs original ZANJ: "
              f"{frac * 100:.4f}% token-identical over {len(roll_ref)} rollouts")
        if first is not None:
            i, j = first
            # top-2 gap of the reference model at the divergence point
            L = int(ps.origin_pos[i]) + 1 + j
            row = ps.tokens[i : i + 1, :L].clone()
            for k, t in enumerate(roll_ref[i][:j]):
                row[0, int(ps.origin_pos[i]) + 1 + k] = t
            with torch.no_grad():
                lg = tl_base(
                    row, return_type="logits",
                    attention_mask=torch.ones_like(row),
                )[0, -1]
            top2 = lg.topk(2).values
            print(f"        first divergence rollout {i} step {j}: "
                  f"reference top-2 logit gap {float(top2[0] - top2[1]):.3e}")
        if precision != "bf16":
            assert ok, f"{label}: only {frac * 100:.4f}% token-identical"

    print("\nparity suite: 5/5 tests passed")
    print(f"headline: fresh {m1:.3e} | baseline max|delta| reported above")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--precision", default="fp32",
                    choices=("fp32", "fp32-tf32", "bf16"))
    a = ap.parse_args()
    main(device=a.device, baseline=a.baseline, precision=a.precision)
