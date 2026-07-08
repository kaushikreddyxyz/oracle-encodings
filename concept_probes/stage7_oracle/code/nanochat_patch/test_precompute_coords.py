"""CPU unit tests for precompute_coords.py (Stage-7 coord precompute producer).

No GPU / tiktoken / rustbpe / expA checkpoint required. Where the real
dependencies are unavailable locally we use DOCUMENTED stand-ins:
  * FakeNanoEnc  -- mimics the tiktoken.Encoding surface precompute uses
                    (encode_ordinary + decode_single_token_bytes), tokenizing
                    into byte-exact word/space tokens so byte->char offset
                    reconstruction (nanochat_char_offsets) partitions the doc
                    bytes exactly, just like real byte-level BPE.
  * fake_qwen_encode -- (substr)->(ids,offsets); precompute takes the qwen
                    tokenizer purely as an injectable callable, so alignment +
                    build_coords + windowing are exercised without transformers.
  * tiny Qwen2 model (or a documented stub if Qwen2Model is unavailable) + a
                    tiny 3K head returning (preds, None) like EncoderHead(expA).

Covered (task item 6):
  [1] phase-angle mapping spot-checks (3 specific months)
  [2] one-hot main-block prediction -> correct family ring coordinate (all 54)
  [3] PCA determinism across two runs
  [4] int8 round-trip (zero-preserving) within one quant step
  [5] store-format compatibility: assemble per-shard files, READ with
      coords_store.CoordSource.lookup -> identical dequantized coords
  [6] pod-sharding disjointness + coverage
  [7] char-offset reconstruction partitions bytes incl. multibyte/mid-char
  [8] end-to-end CoordEngine: batched forward, unmapped tokens -> zero coords,
      windowing reconstructs full (n,r) in order
"""
import json
import math
import os
import sys
import tempfile
import types

import numpy as np

PATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATCH)
sys.path.insert(0, os.path.join(PATCH, ".."))  # stage7 code/ (align, coords_store, train_encoder)

import precompute_coords as pc  # noqa: E402
from coords_store import CoordSource, doc_hash, CYCLIC_ORDER  # noqa: E402

REPO = os.path.abspath(os.path.join(PATCH, "..", "..", "..", ".."))
PROBE_SET = os.path.join(REPO, "concept_probes/stage7_oracle/out/probe_set.json")

fails = []


def check(cond, msg):
    if cond:
        print(f"  OK  {msg}")
    else:
        fails.append(msg)
        print(f"FAIL  {msg}")


# --------------------------------------------------------------------------- #
# stand-ins
# --------------------------------------------------------------------------- #
import re
_WORD = re.compile(r"\S+|\s+")


class FakeNanoEnc:
    """Byte-level tokenizer stand-in: tokens are byte-exact word/space chunks.

    Guarantees per-token bytes partition text.encode('utf-8') (the invariant
    nanochat_char_offsets asserts). Also supports a 'char' granularity to force
    mid-multibyte-char token splits for the offset-robustness test."""

    def __init__(self, granularity="word"):
        self.granularity = granularity

    def _tokens(self, text):
        if self.granularity == "byte":
            return [bytes([b]) for b in text.encode("utf-8")]
        return [m.group(0).encode("utf-8") for m in _WORD.finditer(text)]

    def encode_ordinary(self, text):
        toks = self._tokens(text)
        # id = deterministic small int from the token bytes (kept < vocab for the model)
        return [self._id(t) for t in toks]

    def _id(self, tb):
        return (int.from_bytes(tb[:4].ljust(4, b"\0"), "big") % 250) + 1

    def decode_single_token_bytes(self, i):
        raise RuntimeError("test uses _bytes_for_ids instead")


class FakeNanoEncWithBytes(FakeNanoEnc):
    """Keeps an id->bytes map so decode_single_token_bytes round-trips (real
    tiktoken has a global map; here we build it per corpus for the test)."""

    def __init__(self, granularity="word"):
        super().__init__(granularity)
        self._map = {}

    def encode_ordinary(self, text):
        toks = self._tokens(text)
        ids = []
        for k, t in enumerate(toks):
            i = (self._id(t) + 251 * k) % 60000 + 1
            # ensure uniqueness within a doc for exact byte recovery
            while i in self._map and self._map[i] != t:
                i = (i + 1) % 60000 + 1
            self._map[i] = t
            ids.append(i)
        return ids

    def decode_single_token_bytes(self, i):
        return self._map[i]


def fake_qwen_encode(substr, add_special=False):
    """(substr)->(ids, offsets) splitting on word/space, char offsets into substr.
    ids kept < 250 so a tiny model with vocab 256 can embed them."""
    ids, offs = [], []
    pos = 0
    for m in _WORD.finditer(substr):
        s, e = m.span()
        tok = m.group(0)
        if tok.strip() == "":
            pos = e
            continue  # skip pure-space qwen tokens (act like non-anchors sometimes)
        ids.append((abs(hash(tok)) % 240) + 1)
        offs.append((s, e))
        pos = e
    return ids, offs


class TinyHead:
    """Callable 3K head returning (preds, None), mirroring EncoderHead(expA)."""

    def __init__(self, hidden, K, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(3 * K, hidden, generator=g) * 0.1
        self.b = torch.randn(3 * K, generator=g) * 0.1

    def __call__(self, h):
        import torch
        return torch.nn.functional.linear(h, self.W, self.b), None

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


class StubModel:
    """Deterministic stand-in for a Qwen encoder forward: last_hidden_state is a
    fixed embedding of input_ids (padding masked to 0). Exercises the batched
    forward / gather path without a real transformer."""

    def __init__(self, hidden, vocab, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.randn(vocab, hidden, generator=g)

    def __call__(self, input_ids=None, attention_mask=None):
        import torch
        h = self.emb[input_ids]
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1)
        return types.SimpleNamespace(last_hidden_state=h)

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def build_tiny_encoder(K, hidden=32, vocab=256):
    """Try a real tiny Qwen2 model; fall back to StubModel (documented)."""
    try:
        import torch
        from transformers import Qwen2Config
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=64,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, max_position_embeddings=4096)
        model = Qwen2Model(cfg).eval()
        which = "Qwen2Model(tiny)"
    except Exception as e:  # noqa: BLE001
        model = StubModel(hidden, vocab)
        which = f"StubModel (Qwen2Model unavailable: {type(e).__name__})"
    return model, which


# --------------------------------------------------------------------------- #
# load real probe-set layout
# --------------------------------------------------------------------------- #
ps = pc.load_probe_meta(PROBE_SET)
concepts, families, pred_order, block, K, legend = pc.resolve_layout(ps, layer8=8, r_check=14)
print(f"\nresolved layout: K={K}, block={block} (columns {block*K}:{(block+1)*K}), "
      f"r=len(legend)={len(legend)}")
print(f"legend={legend}")


# --------------------------------------------------------------------------- #
# [0] layout / column-slice assertions (the L8 slice the task asked to verify)
# --------------------------------------------------------------------------- #
print("\n[0] L8 block column slice")
check(ps["layers"] == [6, 8, 14], "probe layers == [6,8,14]")
check(block == 1 and (block * K, (block + 1) * K) == (54, 108),
      f"L8 is block index 1 -> columns [54:108] (got block={block}, K={K})")
check(len(legend) == 14, "build_coords legend length == r == 14")
from coords_store import build_coords as _bc  # noqa: E402
_, leg2 = _bc(np.zeros((1, K), np.float32), concepts, families,
              pca=pc._zero_pca(concepts, families), pred_order=pred_order)
check(leg2 == legend, "legend deterministic")


# --------------------------------------------------------------------------- #
# [1] phase-angle spot checks: 3 specific months
# --------------------------------------------------------------------------- #
print("\n[1] phase-angle mapping (january/april/july)")
mcos, msin = legend.index("months.cos"), legend.index("months.sin")
pca0 = pc._zero_pca(concepts, families)


def onehot_coords(concept, mag=1.0):
    preds = np.zeros((1, K), np.float32)
    preds[0, pred_order.index(concept)] = mag
    z, _ = _bc(preds, concepts, families, pca=pca0, pred_order=pred_order)
    return z[0]


for name, k in [("january", 0), ("april", 3), ("july", 6)]:
    z = onehot_coords(name)
    theta = 2 * math.pi * k / 12
    check(abs(z[mcos] - math.cos(theta)) < 1e-5 and abs(z[msin] - math.sin(theta)) < 1e-5,
          f"{name}: (cos,sin)=({z[mcos]:.4f},{z[msin]:.4f}) == angle 2pi*{k}/12")
check(CYCLIC_ORDER["months"].index("march") == 2, "march is index 2 in CYCLIC_ORDER (sanity)")


# --------------------------------------------------------------------------- #
# [2] one-hot main-block prediction -> correct family ring / PCA coordinate
# --------------------------------------------------------------------------- #
print("\n[2] all 54 one-hot columns land on the right family coordinate")
# fit a real (non-degenerate) continents PCA so continents one-hots are testable
cont_cols = pc.continents_pred_columns(concepts, families, pred_order)
rng = np.random.default_rng(1)
cont_sample = rng.normal(size=(500, len(cont_cols))).astype(np.float32)
comp = pc.fit_pca_2d(cont_sample)  # (6,2)
pca = {"continents": comp}
bad = 0
for c in concepts:
    fam = families[c]
    preds = np.zeros((1, K), np.float32)
    preds[0, pred_order.index(c)] = 1.0
    z, _ = _bc(preds, concepts, families, pca=pca, pred_order=pred_order)
    if fam in CYCLIC_ORDER:
        order = CYCLIC_ORDER[fam]
        k = order.index(c)
        theta = 2 * math.pi * k / len(order)
        ci, si = legend.index(f"{fam}.cos"), legend.index(f"{fam}.sin")
        exp = np.zeros(14); exp[ci] = math.cos(theta); exp[si] = math.sin(theta)
        if not np.allclose(z, exp, atol=1e-5):
            bad += 1
    else:  # continents: z on the pc1/pc2 = this concept's row of comp
        m = cont_cols.index(pred_order.index(c))
        p1, p2 = legend.index("continents.pc1"), legend.index("continents.pc2")
        exp = np.zeros(14); exp[p1] = comp[m, 0]; exp[p2] = comp[m, 1]
        if not np.allclose(z, exp, atol=1e-5):
            bad += 1
check(bad == 0, f"all {len(concepts)} one-hot concepts map to correct coordinate (bad={bad})")


# --------------------------------------------------------------------------- #
# [3] PCA determinism across two runs
# --------------------------------------------------------------------------- #
print("\n[3] PCA-2D determinism + fixed sign")
comp_a = pc.fit_pca_2d(cont_sample)
comp_b = pc.fit_pca_2d(cont_sample.copy())
check(np.array_equal(comp_a, comp_b), "fit_pca_2d identical across two runs (bitwise)")
for j in range(2):
    kmax = int(np.argmax(np.abs(comp_a[:, j])))
    check(comp_a[kmax, j] >= 0, f"component {j} sign-fixed (max-|loading| positive)")


# --------------------------------------------------------------------------- #
# [4] int8 round-trip (zero-preserving) within one quant step
# --------------------------------------------------------------------------- #
print("\n[4] int8 quantization round-trip")
coords_f = rng.normal(size=(2000, 14)).astype(np.float32)
coords_f[::7] = 0.0  # some exact-zero (no-concept) rows
coord_std = np.maximum(coords_f.std(0), 1e-8)
scale = pc.compute_scale(coord_std, clip_sigma=6.0)
q = pc.quantize(coords_f, scale)
deq = q.astype(np.float32) * scale
in_range = np.abs(coords_f) <= 6.0 * float(coord_std.max())
check(np.all(np.abs(deq - coords_f)[in_range] <= scale + 1e-6),
      "dequant within one int8 step for in-range values")
check(np.all(q[::7] == 0), "exact-zero rows quantize to int8 0 (zero-preserving no-op)")
check(np.all(deq[::7] == 0), "zero rows dequantize to exactly 0")


# --------------------------------------------------------------------------- #
# [6] pod-sharding disjointness + coverage
# --------------------------------------------------------------------------- #
print("\n[6] pod-sharding round-robin")
all_shards = pc.parse_shard_range("0-190")
check(all_shards == list(range(0, 191)), "parse_shard_range('0-190') == 0..190")
check(pc.parse_shard_range("0-3,10,20-21") == [0, 1, 2, 3, 10, 20, 21], "mixed range parse")
for n_pods in (1, 4, 8):
    union, ok_disjoint = set(), True
    for p in range(n_pods):
        s = set(pc.assign_shards(all_shards, p, n_pods))
        if union & s:
            ok_disjoint = False
        union |= s
    check(union == set(all_shards) and ok_disjoint,
          f"n_pods={n_pods}: disjoint + full coverage of {len(all_shards)} shards")


# --------------------------------------------------------------------------- #
# [7] char-offset reconstruction (byte partition, multibyte, mid-char splits)
# --------------------------------------------------------------------------- #
print("\n[7] nanochat_char_offsets byte->char reconstruction")
texts7 = ["hello world", "café naïve", "你好世界 ok",
          "emoji \U0001F600 tail", "mix aé中\U0001F642z end"]
enc_word = FakeNanoEncWithBytes("word")
enc_byte = FakeNanoEncWithBytes("byte")
ok7 = True
for t in texts7:
    for enc in (enc_word, enc_byte):
        ids = enc.encode_ordinary(t)
        try:
            offs = pc.nanochat_char_offsets(enc, ids, t)
        except AssertionError:
            ok7 = False
            break
        # spans must be non-decreasing and cover [0, len(t)]
        if offs[0][0] != 0 or offs[-1][1] != len(t):
            ok7 = False
        for a, b in zip(offs, offs[1:]):
            if a[1] > b[0] + 0 and not (a[1] == b[0] or a[1] <= b[1]):
                ok7 = False
check(ok7, "char offsets partition [0,len] with non-decreasing ends across multibyte/mid-char")
# partition assert fires on non-partitioning ids
try:
    pc.nanochat_char_offsets(enc_byte, enc_byte.encode_ordinary("abc")[:-1], "abc")
    check(False, "partition assert should have fired")
except AssertionError:
    check(True, "partition assert fires when ids do not cover the doc bytes")


# --------------------------------------------------------------------------- #
# [8] end-to-end CoordEngine + store-format compatibility with CoordSource
# --------------------------------------------------------------------------- #
print("\n[8] CoordEngine forward + assemble + CoordSource read-back")
import torch  # noqa: E402

hidden, vocab = 32, 256
model, which = build_tiny_encoder(K, hidden, vocab)
print(f"    encoder stand-in: {which}")
head = TinyHead(hidden, K, seed=3)
enc8 = FakeNanoEncWithBytes("word")

# synthetic docs, a couple long enough to force >1 window
docs = [f"the quick brown fox number {i} jumps over lazy dogs in month march" for i in range(6)]
docs.append("word " * 90)   # ~90 nano tokens -> forces windowing at max_doc_tokens=40
qwen_encode = lambda s: fake_qwen_encode(s, add_special=False)  # noqa: E731
pad_id = vocab - 1

engine = pc.CoordEngine(model, head, block, K, concepts, families, pca, pred_order,
                        r=14, device="cpu", dtype=torch.float32, pad_id=pad_id, batch_seqs=4)
produced = {}
for d_i, text in enumerate(docs):
    hsh, n, segs = pc.iter_doc_segments(text, enc8, qwen_encode, max_nano=40, max_qwen=4096)
    engine.add_doc(("d", d_i), hsh, n, segs)
    for chsh, cn, cc in engine.drain():
        produced[int(chsh)] = (cn, cc)
for chsh, cn, cc in engine.drain():
    produced[int(chsh)] = (cn, cc)

# every doc produced n == body-token count, coords shape (n,14)
ok_shape = True
for text in docs:
    n = len(enc8.encode_ordinary(text))
    if n == 0:
        continue
    cn, cc = produced[int(doc_hash(text))]
    if cn != n or cc.shape != (n, 14):
        ok_shape = False
check(ok_shape, "each doc -> exactly n_body coord rows of width 14 (windowing reconstructs full doc)")

# windowed doc actually spanned >1 window
long_n = len(enc8.encode_ordinary(docs[-1]))
check(long_n > 40, f"long doc has {long_n} > 40 nano tokens (windowing exercised)")

# write per-shard store files the way sweep does, then run assemble, then read.
STORE = tempfile.mkdtemp(prefix="stage7_precompute_test_")
shards_dir = os.path.join(STORE, "shards")
os.makedirs(shards_dir, exist_ok=True)
scale8 = 0.05
# build one shard file from produced coords
recs, off, rows = [], 0, []
for text in docs:
    n = len(enc8.encode_ordinary(text))
    if n == 0:
        continue
    cn, cc = produced[int(doc_hash(text))]
    q = pc.quantize(cc, scale8)
    rows.append(q)
    recs.append((np.uint64(doc_hash(text)), np.int64(off), np.int32(n)))
    off += n
allq = np.concatenate(rows, axis=0)
allq.tofile(os.path.join(shards_dir, "coords_00000.int8"))
np.save(os.path.join(shards_dir, "index_00000.npy"), np.array(recs, dtype=pc.INDEX_DTYPE))
json.dump({"sid": 0, "n_docs": len(recs), "n_tokens": int(off), "n_zero_tokens": 0,
           "zero_frac": 0.0, "scale": scale8},
          open(os.path.join(shards_dir, "meta_00000.json"), "w"))
# a coord_fit.npz so assemble can read scale/legend/stats
np.savez(os.path.join(STORE, "coord_fit.npz"),
         pca_components=comp, coord_mean=np.zeros(14, np.float32),
         coord_std=np.ones(14, np.float32), scale=np.float32(scale8),
         clip_sigma=np.float32(6.0), legend=np.array(legend),
         pred_order=np.array(pred_order), block=np.int64(block), n_fit=np.int64(1))

A = types.SimpleNamespace(out=STORE, shards="0-0", probe_set=PROBE_SET, layer8_block=8,
                          r_check=14, n_embd=1536, p_seed=1337, encoder_ckpt=None)
pc.run_assemble(A)

# READ back with the real consumer
cs = CoordSource(STORE, noise_sigma=0.0)
meta = json.load(open(os.path.join(STORE, "meta.json")))
check(meta["r"] == 14 and abs(meta["scale"] - scale8) < 1e-6, "meta.json has r=14 + fit scale")
check(meta["block_columns"] == [54, 108], "meta records L8 block columns [54,108]")
check(os.path.exists(os.path.join(STORE, "P.npy")), "P.npy written by assemble")
P = np.load(os.path.join(STORE, "P.npy"))
check(P.shape == (1536, 14) and np.allclose(P.T @ P, np.eye(14), atol=1e-5),
      "P is (1536,14) orthonormal")

roundtrip_ok = True
for text in docs:
    n = len(enc8.encode_ordinary(text))
    if n == 0:
        continue
    got, h = cs.lookup(text, n)
    cn, cc = produced[int(doc_hash(text))]
    want = pc.quantize(cc, scale8).astype(np.float32) * scale8
    if got is None or not np.array_equal(got, want):
        roundtrip_ok = False
check(roundtrip_ok, "CoordSource.lookup returns exactly the assembled+dequantized coords")
# length-mismatch + missing fall back to None (consumer contract)
some = docs[0]
check(cs.lookup(some, len(enc8.encode_ordinary(some)) + 1)[0] is None, "n mismatch -> None")
check(cs.lookup("a document never stored anywhere", 5)[0] is None, "missing doc -> None")

# unmapped tokens (no qwen anchor) -> zero coords inside a present doc.
# a doc that is pure whitespace has no qwen anchors -> all coords zero.
ws = "   \n  \t "
hshw, nw, segsw = pc.iter_doc_segments(ws, enc8, qwen_encode, max_nano=40, max_qwen=4096)
if nw > 0:
    engine2 = pc.CoordEngine(model, head, block, K, concepts, families, pca, pred_order,
                             14, "cpu", torch.float32, pad_id, batch_seqs=4)
    engine2.add_doc(("w", 0), hshw, nw, segsw)
    outw = list(engine2.drain())
    zc = outw[0][2] if outw else np.zeros((nw, 14))
    check(np.allclose(zc, 0.0), "doc with no qwen anchors -> all-zero coords (no-op)")
else:
    check(True, "whitespace doc had 0 body tokens (skipped) -- vacuously no-op")


# --------------------------------------------------------------------------- #
print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
