"""Stage 7-Oracle Phase 1: score a ClimbMix corpus shard-range with the
frozen probe_set (concept_probes/stage7_oracle/DESIGN.md "Score store").

Tokenizer convention (MUST match stage5/stage6, which trained the probes):
  tok(text, add_special_tokens=False)["input_ids"]  -- BOS is NEVER obtained
  from the tokenizer call. It is manually written into position 0 of the
  padded input_ids right before the forward pass, and the corresponding
  hidden_states row is dropped before anything is stored/scored, so stored
  positions line up 1:1 with the raw (BOS-free) token ids. Ported from
  concept_probes/stage5/code/extract.py (`prepend_bos` / "drop the BOS row").

ClimbMix loader: ported from concept_probes/stage6/code/nat_common.py
  (DATASET = "karpathy/climbmix-400b-shuffle", per-shard parquet files via
  hf_hub_download, `iter_shard_docs` batched-parquet-scan idiom). Duplicated
  here (not imported) so this file is self-contained for pod deployment
  (DESIGN.md "Pod conventions": pods only need stage7_oracle/code + gemma +
  probe_set.json + shard download). A local-override env var
  (STAGE7_SHARD_DIR) is provided purely as a test seam -- see
  test_score_corpus.py.

DESIGN ambiguity resolved (see probe_set_arrays.npz doc in DESIGN.md):
  probe_set_arrays.npz stores nat_mean/nat_std for only the 3 chosen probe
  layers, and W_dom_abl ("std space") has no independently-stored
  normalization stats of its own. DESIGN.md implicitly assumes
  `ablation_layer` is one of the 3 chosen `layers` (SPEC Phase 0's tie-break
  toward the causal-salient band). CHECKED against the actual Phase 0
  implementation (select_probes.py, written by a parallel agent): its
  ablation_layer is chosen independently (mode vote of
  `e5_salient_layer_corrected` over surviving concepts) from `chosen_layers`
  (picked by mean AUROC) -- select_probes.py itself acknowledges this
  ("nat_std at the ABLATION layer, which may differ from the 3 chosen score
  layers -- load separately", select_probes.py ~L543) but does NOT persist
  that separately-loaded nat_std_abl into probe_set_arrays.npz, so as
  currently written select_probes.py's own output would be unusable here if
  ablation_layer lands outside `layers`. This script therefore supports an
  OPTIONAL forward-compatible extension: if ablation_layer is not in
  `layers`, it looks for `nat_mean_abl`/`nat_std_abl` [D] arrays in the npz
  (not in the current DESIGN.md spec) and uses those; if absent, it fails
  loudly with an actionable message rather than silently mis-scoring the
  DoM columns. See ProbeSet.__init__.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - required at runtime, not at import time for --merge-stats
    pa = pq = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None

CLIMBMIX_DATASET = "karpathy/climbmix-400b-shuffle"
GEMMA_MODEL_DEFAULT = "google/gemma-2-2b"
MAX_DOC_TOKENS = 2048
MIN_DOC_TOKENS = 64
DOC_BUFFER = 1024
HEARTBEAT_INTERVAL_S = 60


# --------------------------------------------------------------------------
# ClimbMix loader (ported from concept_probes/stage6/code/nat_common.py)
# --------------------------------------------------------------------------

def _shard_path(shard: int) -> str:
    """Resolve a shard id to a local parquet path.

    Test seam: if STAGE7_SHARD_DIR is set, read shard_%05d.parquet from that
    local directory instead of hitting the HF hub (used by
    test_score_corpus.py with a fixture shard; never set on a real pod).
    """
    local_dir = os.environ.get("STAGE7_SHARD_DIR")
    if local_dir:
        return os.path.join(local_dir, f"shard_{shard:05d}.parquet")
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed and STAGE7_SHARD_DIR not set")
    return hf_hub_download(CLIMBMIX_DATASET, f"shard_{shard:05d}.parquet",
                            repo_type="dataset", token=True)


def _detect_text_column(pf) -> str:
    sch = pf.schema_arrow
    for cand in ("text", "content", "raw_content", "document", "body"):
        if cand in sch.names and (pa.types.is_string(sch.field(cand).type)
                                   or pa.types.is_large_string(sch.field(cand).type)):
            return cand
    for name in sch.names:
        t = sch.field(name).type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return name
    raise RuntimeError(f"no string column found; schema={sch.names}")


def iter_shard_docs(shard: int, max_docs: int | None = None):
    """Yield (doc_index, text) from one shard, in parquet order (deterministic)."""
    pf = pq.ParquetFile(_shard_path(shard))
    col = _detect_text_column(pf)
    n = 0
    for batch in pf.iter_batches(batch_size=512, columns=[col]):
        for v in batch.column(0):
            if max_docs is not None and n >= max_docs:
                return
            t = v.as_py()
            if t:
                yield n, t
            n += 1


def shard_num_rows(shard: int) -> int:
    return pq.ParquetFile(_shard_path(shard)).metadata.num_rows


def parse_shards(spec: str) -> list[int]:
    """'320-339,345,350-352' -> sorted list of ints (inclusive ranges)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------
# probe_set.json / probe_set_arrays.npz (DESIGN.md, frozen)
# --------------------------------------------------------------------------

class ProbeSet:
    def __init__(self, probe_set_dir: str):
        probe_set_dir = Path(probe_set_dir)
        with open(probe_set_dir / "probe_set.json") as f:
            self.meta = json.load(f)
        arrs = np.load(probe_set_dir / "probe_set_arrays.npz")

        self.layers = list(self.meta["layers"])              # 3 gemma layers
        self.ablation_layer = int(self.meta["ablation_layer"])
        self.concepts = list(self.meta["concepts"])           # K names, canonical order
        self.K = len(self.concepts)

        if "layer_index" in arrs:
            assert list(arrs["layer_index"]) == self.layers, \
                "probe_set_arrays.npz layer_index disagrees with probe_set.json layers"

        self.W = np.asarray(arrs["W"], dtype=np.float32)               # [3,K,D]
        self.b = np.asarray(arrs["b"], dtype=np.float32)               # [3,K]
        self.nat_mean = np.asarray(arrs["nat_mean"], dtype=np.float32)  # [3,D]
        self.nat_std = np.asarray(arrs["nat_std"], dtype=np.float32)    # [3,D]
        self.W_dom = np.asarray(arrs["W_dom_abl"], dtype=np.float32)    # [K,D]
        self.b_dom = np.asarray(arrs["b_dom_abl"], dtype=np.float32)    # [K]

        self.D = self.W.shape[-1]
        assert self.W.shape == (3, self.K, self.D)
        assert self.b.shape == (3, self.K)
        assert self.nat_mean.shape == (3, self.D)
        assert self.nat_std.shape == (3, self.D)
        assert self.W_dom.shape == (self.K, self.D)
        assert self.b_dom.shape == (self.K,)
        assert np.all(self.nat_std > 0), "nat_std must be strictly positive"

        # Normalization stats for the DoM-at-ablation-layer columns. Common
        # case: ablation_layer is one of the 3 chosen layers -> reuse its
        # nat_mean/nat_std. Otherwise (see module docstring "DESIGN
        # ambiguity resolved"): look for an optional nat_mean_abl/
        # nat_std_abl [D] pair in the npz; fail loudly if neither is
        # available rather than silently mis-scoring the DoM columns.
        if self.ablation_layer in self.layers:
            self.abl_idx = self.layers.index(self.ablation_layer)
            self.nat_mean_abl = self.nat_mean[self.abl_idx]
            self.nat_std_abl = self.nat_std[self.abl_idx]
        elif "nat_mean_abl" in arrs.files and "nat_std_abl" in arrs.files:
            self.abl_idx = None
            self.nat_mean_abl = np.asarray(arrs["nat_mean_abl"], dtype=np.float32)
            self.nat_std_abl = np.asarray(arrs["nat_std_abl"], dtype=np.float32)
            assert self.nat_mean_abl.shape == (self.D,)
            assert self.nat_std_abl.shape == (self.D,)
            assert np.all(self.nat_std_abl > 0), "nat_std_abl must be strictly positive"
        else:
            raise AssertionError(
                f"ablation_layer ({self.ablation_layer}) is not one of the 3 chosen probe "
                f"layers ({self.layers}), and probe_set_arrays.npz has no nat_mean_abl/"
                f"nat_std_abl arrays to normalize the DoM-at-ablation-layer columns with. "
                f"See the DESIGN ambiguity note at the top of score_corpus.py -- either "
                f"constrain Phase 0's ablation-layer vote to `layers`, or extend "
                f"select_probes.py to persist nat_mean_abl/nat_std_abl."
            )
        # needed_hidden_layers() dedupes so a coincidental ablation_layer in
        # `layers` doesn't fetch the same hidden_states slice twice.
        self._hidden_layers = sorted(set(self.layers) | {self.ablation_layer})

        self.n_cols = 4 * self.K  # [layer0 K, layer1 K, layer2 K, dom@ablation K]

    def needed_hidden_layers(self) -> list[int]:
        """gemma decoder-block indices l such that hidden_states[l+1] is needed."""
        return self._hidden_layers

    def score_column_names(self) -> list[str]:
        names = []
        for l in self.layers:
            names += [f"L{l}:{c}" for c in self.concepts]
        names += [f"dom@L{self.ablation_layer}:{c}" for c in self.concepts]
        return names


class ScoreHead:
    """Caches probe tensors on-device; computes raw (unquantized) [B,T,4K] scores."""

    def __init__(self, probe: ProbeSet, device: torch.device):
        self.probe = probe
        self.device = device
        self.W = torch.from_numpy(probe.W).to(device=device, dtype=torch.float32)
        self.b = torch.from_numpy(probe.b).to(device=device, dtype=torch.float32)
        self.nat_mean = torch.from_numpy(probe.nat_mean).to(device=device, dtype=torch.float32)
        self.nat_std = torch.from_numpy(probe.nat_std).to(device=device, dtype=torch.float32)
        self.nat_mean_abl = torch.from_numpy(probe.nat_mean_abl).to(device=device, dtype=torch.float32)
        self.nat_std_abl = torch.from_numpy(probe.nat_std_abl).to(device=device, dtype=torch.float32)
        self.W_dom = torch.from_numpy(probe.W_dom).to(device=device, dtype=torch.float32)
        self.b_dom = torch.from_numpy(probe.b_dom).to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def score(self, hs_by_layer: dict[int, torch.Tensor]) -> torch.Tensor:
        """hs_by_layer[l]: [B,T,D] (any float dtype). Returns float32 [B,T,4K]."""
        parts = []
        for i, l in enumerate(self.probe.layers):
            h = hs_by_layer[l].to(torch.float32)
            z = (h - self.nat_mean[i]) / self.nat_std[i]
            parts.append(torch.einsum("btd,kd->btk", z, self.W[i]) + self.b[i])
        h_abl = hs_by_layer[self.probe.ablation_layer].to(torch.float32)
        z_abl = (h_abl - self.nat_mean_abl) / self.nat_std_abl
        parts.append(torch.einsum("btd,kd->btk", z_abl, self.W_dom) + self.b_dom)
        return torch.cat(parts, dim=-1)  # [B,T,4K]


# --------------------------------------------------------------------------
# Quantization (DESIGN.md "Quantization")
# --------------------------------------------------------------------------

def compute_quant(mean: np.ndarray, std: np.ndarray) -> dict:
    scale = np.maximum(4.0 * std / 127.0, 1e-6)
    return {"zero": mean.astype(np.float64).tolist(), "scale": scale.astype(np.float64).tolist()}


def load_quant(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        q = json.load(f)
    return {"zero": np.asarray(q["zero"], dtype=np.float32),
            "scale": np.asarray(q["scale"], dtype=np.float32)}


def save_quant_atomic(path: str, quant: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(quant, f)
    os.replace(tmp, path)


def quantize(raw: np.ndarray, zero: np.ndarray, scale: np.ndarray) -> np.ndarray:
    q = np.round((raw - zero) / scale)
    return np.clip(q, -127, 127).astype(np.int8)


def dequantize(q: np.ndarray, zero: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale + zero


# --------------------------------------------------------------------------
# Streaming (count, mean, M2) accumulator -- exact parallel-Welford merge
# --------------------------------------------------------------------------

class RunningStats:
    def __init__(self, n_cols: int):
        self.count = 0
        self.mean = np.zeros(n_cols, dtype=np.float64)
        self.M2 = np.zeros(n_cols, dtype=np.float64)

    def update_batch(self, batch: np.ndarray) -> None:
        """batch: [n, n_cols] float64/float32."""
        n_b = batch.shape[0]
        if n_b == 0:
            return
        batch = batch.astype(np.float64)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)  # population variance
        delta = batch_mean - self.mean
        tot = self.count + n_b
        self.mean = self.mean + delta * (n_b / tot)
        self.M2 = self.M2 + batch_var * n_b + delta ** 2 * (self.count * n_b / tot)
        self.count = tot

    def std(self) -> np.ndarray:
        var = self.M2 / max(self.count, 1)
        return np.sqrt(np.maximum(var, 1e-12))

    def to_dict(self, shard: int | None = None) -> dict:
        d = {"n_tokens": int(self.count), "mean": self.mean.tolist(), "M2": self.M2.tolist()}
        if shard is not None:
            d["shard"] = shard
        return d

    @classmethod
    def from_dict(cls, n_cols: int, d: dict) -> "RunningStats":
        rs = cls(n_cols)
        rs.count = int(d["n_tokens"])
        rs.mean = np.asarray(d["mean"], dtype=np.float64)
        rs.M2 = np.asarray(d["M2"], dtype=np.float64)
        return rs

    def combine(self, other: "RunningStats") -> "RunningStats":
        """Parallel combination (Chan et al.) of two independent accumulators."""
        out = RunningStats(len(self.mean))
        if self.count == 0:
            out.count, out.mean, out.M2 = other.count, other.mean.copy(), other.M2.copy()
            return out
        if other.count == 0:
            out.count, out.mean, out.M2 = self.count, self.mean.copy(), self.M2.copy()
            return out
        tot = self.count + other.count
        delta = other.mean - self.mean
        out.mean = self.mean + delta * (other.count / tot)
        out.M2 = self.M2 + other.M2 + delta ** 2 * (self.count * other.count / tot)
        out.count = tot
        return out


def merge_stats(out_dir: str) -> None:
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("partial_stats_*.json"))
    if not files:
        print(f"[score_corpus] no partial_stats_*.json found in {out_dir}", flush=True)
        return
    with open(files[0]) as f:
        first = json.load(f)
    n_cols = len(first["mean"])
    total = RunningStats(n_cols)
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        total = total.combine(RunningStats.from_dict(n_cols, d))
    corpus_stats = {"n_tokens": int(total.count), "mean": total.mean.tolist(),
                     "std": total.std().tolist()}
    with open(out_dir / "corpus_stats.json", "w") as f:
        json.dump(corpus_stats, f)
    print(f"[score_corpus] merged {len(files)} shard stats -> {out_dir / 'corpus_stats.json'} "
          f"({total.count} tokens)", flush=True)


# --------------------------------------------------------------------------
# Model / tokenizer loading
# --------------------------------------------------------------------------

def load_model_and_tok(model_name: str, attn_impl: str, device: torch.device,
                        tiny_model_config: str | None = None):
    """Real path: AutoModel.from_pretrained(model_name). Test seam:
    if tiny_model_config is given (a JSON dict for Gemma2Config, written by
    make_fixture.py), build a randomly-initialized Gemma2Model instead --
    used only by test_score_corpus.py so the smoke test runs on CPU without
    downloading the real gemma-2-2b weights."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if tiny_model_config is not None:
        from transformers import Gemma2Config, Gemma2Model
        with open(tiny_model_config) as f:
            cfg_dict = json.load(f)
        cfg_dict["attn_implementation"] = attn_impl
        cfg = Gemma2Config(**cfg_dict)
        torch.manual_seed(0)
        model = Gemma2Model(cfg).to(dtype)
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, dtype=dtype, attn_implementation=attn_impl)

    model.eval().to(device)
    return tok, model


# --------------------------------------------------------------------------
# Heartbeat
# --------------------------------------------------------------------------

class Heartbeat:
    def __init__(self, path: str, interval_s: float = HEARTBEAT_INTERVAL_S):
        self.path = Path(path)
        self.interval_s = interval_s
        self._last = 0.0

    def maybe_write(self, state: dict, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last) < self.interval_s:
            return
        self._last = now
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self.path)
        except OSError as e:  # heartbeat is best-effort, never fatal
            print(f"[score_corpus] heartbeat write failed: {e}", file=sys.stderr)


# --------------------------------------------------------------------------
# Batching helpers
# --------------------------------------------------------------------------

def tokenize_docs(tok, texts: list[str]) -> list[list[int]]:
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    return [ids[:MAX_DOC_TOKENS] for ids in enc]


def make_padded_batch(sub: list[tuple[int, list[int]]], bos_id: int, pad_id: int,
                       device: torch.device):
    """sub: list of (doc_idx, ids). Returns input_ids [B,Lmax+1], attn [B,Lmax+1], lens."""
    lens = [len(ids) for _, ids in sub]
    Lmax = max(lens) + 1  # +1 for BOS
    input_ids = torch.full((len(sub), Lmax), pad_id, dtype=torch.long)
    attn = torch.zeros((len(sub), Lmax), dtype=torch.long)
    for i, (_, ids) in enumerate(sub):
        input_ids[i, 0] = bos_id
        if ids:
            input_ids[i, 1:1 + len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[i, :1 + len(ids)] = 1
    return input_ids.to(device), attn.to(device), lens


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.no_grad()
def forward_score_batch(model, head: ScoreHead, sub: list[tuple[int, list[int]]],
                         bos_id: int, pad_id: int, device: torch.device) -> torch.Tensor:
    """Returns raw float32 scores [B, T_max_content, 4K] (still padded)."""
    input_ids, attn, lens = make_padded_batch(sub, bos_id, pad_id, device)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    hs_by_layer = {}
    for l in head.probe.needed_hidden_layers():
        # hidden_states[l+1] == post-block-l residual; drop BOS row (col 0)
        hs_by_layer[l] = out.hidden_states[l + 1][:, 1:, :]
    del out
    raw = head.score(hs_by_layer)
    del hs_by_layer
    return raw, lens


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibrate(shard: int, probe: ProbeSet, head: ScoreHead, tok, model, device: torch.device,
              batch_size: int, calib_tokens: int, max_docs_per_shard: int | None) -> dict:
    print(f"[score_corpus] calibrating quant.json on shard {shard} "
          f"(target {calib_tokens} tokens)...", flush=True)
    stats = RunningStats(probe.n_cols)
    buf: list[tuple[int, list[int]]] = []
    seen_tokens = 0

    def flush_buf():
        nonlocal seen_tokens
        if not buf:
            return
        buf.sort(key=lambda kv: len(kv[1]))
        for sub in chunked(buf, batch_size):
            raw, lens = forward_score_batch(model, head, sub, tok.bos_token_id,
                                             tok.pad_token_id or 0, device)
            raw_np = raw.cpu().numpy()
            for i, n in enumerate(lens):
                stats.update_batch(raw_np[i, :n, :])
                seen_tokens += n
        buf.clear()

    for doc_idx, text in iter_shard_docs(shard, max_docs_per_shard):
        ids = tokenize_docs(tok, [text])[0]
        if len(ids) < MIN_DOC_TOKENS:
            continue
        buf.append((doc_idx, ids))
        if len(buf) >= DOC_BUFFER:
            flush_buf()
        if seen_tokens >= calib_tokens:
            break
    if seen_tokens < calib_tokens:
        flush_buf()

    quant = compute_quant(stats.mean, stats.std())
    print(f"[score_corpus] calibration done: {stats.count} tokens", flush=True)
    return quant


# --------------------------------------------------------------------------
# Per-shard processing
# --------------------------------------------------------------------------

def shard_output_paths(out_dir: Path, sid: int) -> dict:
    return {
        "tokens": out_dir / f"tokens_{sid:05d}.npy",
        "scores": out_dir / f"scores_{sid:05d}.npy",
        "docs": out_dir / f"docs_{sid:05d}.jsonl",
        "stats": out_dir / f"partial_stats_{sid:05d}.json",
    }


def shard_is_done(sid: int, out_dir: Path) -> bool:
    paths = shard_output_paths(out_dir, sid)
    if not (paths["tokens"].exists() and paths["scores"].exists() and paths["docs"].exists()):
        return False
    try:
        tokens = np.load(paths["tokens"], mmap_mode="r")
        scores = np.load(paths["scores"], mmap_mode="r")
        n_docs_sum = 0
        with open(paths["docs"]) as f:
            for line in f:
                if not line.strip():
                    continue
                n_docs_sum += json.loads(line)["n"]
        return len(tokens) == scores.shape[0] == n_docs_sum
    except Exception as e:
        print(f"[score_corpus] shard {sid} outputs present but inconsistent "
              f"({e}); will reprocess", file=sys.stderr)
        return False


def process_shard(sid: int, probe: ProbeSet, head: ScoreHead, tok, model, device: torch.device,
                   quant: dict, batch_size: int, out_dir: Path, hb: Heartbeat,
                   max_docs_per_shard: int | None) -> None:
    paths = shard_output_paths(out_dir, sid)
    # np.save() silently appends ".npy" if the path doesn't already end in
    # it, so tmp names must themselves end in ".npy" (not ".npy.tmp").
    tmp_tokens = paths["tokens"].with_name(paths["tokens"].stem + ".tmp.npy")
    tmp_scores = paths["scores"].with_name(paths["scores"].stem + ".tmp.npy")
    tmp_docs = paths["docs"].with_suffix(".jsonl.tmp")

    total_rows = shard_num_rows(sid)
    token_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    docs_f = open(tmp_docs, "w")
    stats = RunningStats(probe.n_cols)

    offset = 0
    docs_done = 0
    tokens_done = 0
    t0 = time.time()
    buf: list[tuple[int, list[int]]] = []

    def flush_buf():
        nonlocal offset, docs_done, tokens_done
        if not buf:
            return
        buf.sort(key=lambda kv: len(kv[1]))
        for sub in chunked(buf, batch_size):
            raw, lens = forward_score_batch(model, head, sub, tok.bos_token_id,
                                             tok.pad_token_id or 0, device)
            raw_np = raw.cpu().numpy()
            for i, (doc_idx, ids) in enumerate(sub):
                n = lens[i]
                doc_raw = raw_np[i, :n, :]
                stats.update_batch(doc_raw)
                doc_q = quantize(doc_raw, quant["zero"], quant["scale"])
                token_chunks.append(np.asarray(ids, dtype=np.int32))
                score_chunks.append(doc_q)
                docs_f.write(json.dumps({"doc": doc_idx, "start": offset, "n": n}) + "\n")
                offset += n
                docs_done += 1
                tokens_done += n
            hb.maybe_write({"shard": sid, "docs_done": docs_done, "tokens_done": tokens_done,
                             "tok_per_s": tokens_done / max(time.time() - t0, 1e-9),
                             "eta_s": (total_rows - docs_done) / max(docs_done / max(time.time() - t0, 1e-9), 1e-9)
                                       if total_rows else None})
        buf.clear()

    print(f"[score_corpus] shard {sid}: {total_rows} rows", flush=True)
    for doc_idx, text in iter_shard_docs(sid, max_docs_per_shard):
        ids = tokenize_docs(tok, [text])[0]
        if len(ids) < MIN_DOC_TOKENS:
            continue
        buf.append((doc_idx, ids))
        if len(buf) >= DOC_BUFFER:
            flush_buf()
    flush_buf()
    docs_f.close()
    hb.maybe_write({"shard": sid, "docs_done": docs_done, "tokens_done": tokens_done,
                     "tok_per_s": tokens_done / max(time.time() - t0, 1e-9), "eta_s": 0}, force=True)

    tokens_arr = (np.concatenate(token_chunks) if token_chunks
                  else np.zeros(0, dtype=np.int32))
    scores_arr = (np.concatenate(score_chunks, axis=0) if score_chunks
                  else np.zeros((0, probe.n_cols), dtype=np.int8))
    np.save(tmp_tokens, tokens_arr.astype(np.int32))
    np.save(tmp_scores, scores_arr.astype(np.int8))
    os.replace(tmp_tokens, paths["tokens"])
    os.replace(tmp_scores, paths["scores"])
    os.replace(tmp_docs, paths["docs"])

    with open(paths["stats"], "w") as f:
        json.dump(stats.to_dict(shard=sid), f)

    print(f"[score_corpus] shard {sid} done: {docs_done} docs, {tokens_done} tokens", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-set", help="dir with probe_set.json + probe_set_arrays.npz")
    ap.add_argument("--shards", help='e.g. "320-339" or "320-339,345"')
    ap.add_argument("--out", required=True, help="e.g. /workspace/scores")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--calib-tokens", type=int, default=10_000_000)
    ap.add_argument("--quant-json", default="/workspace/scores/quant.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-docs-per-shard", type=int, default=None)
    ap.add_argument("--model", default=GEMMA_MODEL_DEFAULT)
    ap.add_argument("--heartbeat", default="/workspace/hb_score.txt")
    ap.add_argument("--merge-stats", action="store_true",
                     help="merge partial_stats_*.json in --out into corpus_stats.json and exit")
    ap.add_argument("--tiny-model-config", default=None, help=argparse.SUPPRESS)  # test seam
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_stats:
        merge_stats(out_dir)
        return

    assert args.probe_set and args.shards, "--probe-set and --shards required unless --merge-stats"

    shards = parse_shards(args.shards)
    probe = ProbeSet(args.probe_set)
    device = torch.device(args.device)
    tok, model = load_model_and_tok(args.model, args.attn, device, args.tiny_model_config)
    head = ScoreHead(probe, device)
    hb = Heartbeat(args.heartbeat)

    if os.path.exists(args.quant_json):
        print(f"[score_corpus] loading existing quant.json from {args.quant_json}", flush=True)
        quant = load_quant(args.quant_json)
    else:
        quant = calibrate(shards[0], probe, head, tok, model, device, args.batch_size,
                           args.calib_tokens, args.max_docs_per_shard)
        save_quant_atomic(args.quant_json, {"zero": quant["zero"], "scale": quant["scale"]})
        quant = {"zero": np.asarray(quant["zero"], dtype=np.float32),
                 "scale": np.asarray(quant["scale"], dtype=np.float32)}

    for sid in shards:
        if shard_is_done(sid, out_dir):
            print(f"[score_corpus] shard {sid} already done, skipping", flush=True)
            continue
        process_shard(sid, probe, head, tok, model, device, quant, args.batch_size, out_dir, hb,
                       args.max_docs_per_shard)


if __name__ == "__main__":
    main()
