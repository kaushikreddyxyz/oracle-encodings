"""Fast frozen-model binding evaluation with independent queries and shared KV prefixes.

Run from repository root: python -m icl_oracle.binding_v2.runner --help
Injection layer h means post decoder layer h-1 (matching bind_icl.py hidden-state h).
"""
import argparse
import copy
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tasks import make_task


class Runner:
    def __init__(self, model, device, layer, attention="sdpa"):
        self.device = device
        self.layer = layer
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, attn_implementation=attention
        ).to(device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.dim = self.model.config.hidden_size
        assert 1 <= layer <= self.model.config.num_hidden_layers
        vocab = self.tok.get_vocab()
        candidates = sorted({t.lstrip("\u2581\u0120") for t in vocab
                             if re.fullmatch(r"[\u2581\u0120][a-z]{3,8}", t)})
        self.words = [w for w in candidates if len(self.encode(" " + w)) == 1]
        if len(self.words) < 512:
            raise ValueError(f"Insufficient single-token word pool: {len(self.words)}")
        self.block = self.model.model.layers[layer - 1]

    def encode(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def layout(self, task, mode):
        rng = np.random.default_rng(1234567 + int(task["metadata"]["seed"]))
        nd, nq = len(task["demo_y"]), len(task["query_y"])
        nc = int(task["metadata"]["n_classes"])
        words = rng.choice(self.words, nd + nq + nc + nc, replace=False).tolist()
        items, labels, markers = words[:nd+nq], words[nd+nq:nd+nq+nc], words[-nc:]
        label_ids = [self.encode(" " + w)[0] for w in labels]

        def example(i, bits, code, y=None):
            ids = []
            if mode == "token":
                if task["metadata"]["family"] == "codebook":
                    ids.extend(self.encode(" " + markers[int(code)]))
                else:
                    # Spaces keep bit positions unambiguous and consistent across rows.
                    ids.extend(self.encode(" " + " ".join(str(int(x)) for x in bits)))
            pos = len(ids)
            ids.extend(self.encode(" " + items[i]))
            ids.extend(self.encode(":"))
            if y is not None:
                ids.append(label_ids[int(y)])
                ids.extend(self.encode("\n"))
            return ids, pos

        prefix = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        positions = []
        for i, y in enumerate(task["demo_y"]):
            row, pos = example(i, task["demo_bits"][i], task["demo_codes"][i], y)
            positions.append(len(prefix) + pos)
            prefix.extend(row)
        queries, qpositions = [], []
        for i in range(nq):
            row, pos = example(nd+i, task["query_bits"][i], task["query_codes"][i])
            queries.append(row)
            qpositions.append(pos)
        assert len({len(q) for q in queries}) == 1, "Query lengths must match"
        return dict(prefix=prefix, positions=positions, queries=queries,
                    qpositions=qpositions, label_ids=label_ids, labels=labels)

    def hook(self, positions, vectors):
        # vectors [B,N,d], positions [N] for common prefix or [B] for query items.
        positions = torch.as_tensor(positions, device=self.device)
        vectors = torch.as_tensor(vectors, dtype=self.dtype, device=self.device)
        def inject(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if vectors.ndim == 3:
                h[:, positions, :] += vectors
            else:
                h[torch.arange(h.shape[0], device=h.device), positions, :] += vectors
            return output
        return self.block.register_forward_hook(inject)

    @torch.inference_mode()
    def reference(self, layout):
        captured = []
        def capture(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured.append(h[0, layout["positions"]].float().cpu().numpy())
        handle = self.block.register_forward_hook(capture)
        try:
            self.model.model(torch.tensor([layout["prefix"]], device=self.device), use_cache=False)
        finally:
            handle.remove()
        x = captured[0]
        return float(np.median(np.linalg.norm(x - x.mean(0), axis=1)))

    @staticmethod
    def repeat_cache(cache, batch):
        result = copy.deepcopy(cache)
        if hasattr(result, "batch_repeat_interleave"):
            result.batch_repeat_interleave(batch)
            return result
        return tuple(tuple(t.repeat_interleave(batch, dim=0) for t in layer) for layer in result)

    @torch.inference_mode()
    def evaluate(self, task, layout, mode, beta, norm_ref, bs, emit):
        rng = np.random.default_rng(987654 + int(task["metadata"]["seed"]))
        width = task["demo_x"].shape[1]
        directions = np.linalg.qr(rng.standard_normal((self.dim, width)))[0].T.astype(np.float32)
        dv = (task["demo_x"] @ directions) * (beta * norm_ref)
        qv = (task["query_x"] @ directions) * (beta * norm_ref)
        handle = self.hook(layout["positions"], dv[None]) if mode == "inject" else None
        try:
            prefix_out = self.model.model(torch.tensor([layout["prefix"]], device=self.device), use_cache=True)
            cache = prefix_out.past_key_values
            del prefix_out
        finally:
            if handle is not None:
                handle.remove()
        plen = len(layout["prefix"])
        for start in range(0, len(task["query_y"]), bs):
            stop = min(start + bs, len(task["query_y"]))
            ids = torch.tensor(layout["queries"][start:stop], device=self.device)
            batch_cache = self.repeat_cache(cache, stop-start)
            handle = self.hook(layout["qpositions"][start:stop], qv[start:stop]) if mode == "inject" else None
            try:
                out = self.model.model(ids, past_key_values=batch_cache, use_cache=True,
                                       attention_mask=torch.ones((stop-start, plen+ids.shape[1]),
                                                                 device=self.device, dtype=torch.long))
            finally:
                if handle is not None:
                    handle.remove()
            logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
            cap = getattr(self.model.config, "final_logit_softcapping", None)
            if cap:
                logits = torch.tanh(logits / cap) * cap
            selected = logits[:, layout["label_ids"]]
            label_probs = torch.softmax(selected, -1).cpu().numpy()
            logdenom = torch.logsumexp(logits, -1)
            probs = torch.exp(selected-logdenom[:, None]).cpu().numpy()
            argmax = logits.argmax(-1).cpu().numpy()
            for b, i in enumerate(range(start, stop)):
                y = int(task["query_y"][i])
                emit(dict(query=i, y=y, pred=int(label_probs[b].argmax()),
                          correct=bool(label_probs[b].argmax() == y),
                          full_correct=bool(argmax[b] == layout["label_ids"][y]),
                          p_correct=float(label_probs[b, y]), p_full=float(probs[b, y]),
                          label_mass=float(probs[b].sum()), label_probs=label_probs[b].tolist(),
                          seen=bool(task["query_seen"][i]), code=int(task["query_codes"][i])))
            del out, logits, batch_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--task", default="selection", help="Comma-separated task families")
    ap.add_argument("--seeds", default="0,1,2", help="Comma-separated explicit seed IDs")
    ap.add_argument("--shots", type=int, default=64)
    ap.add_argument("--queries", type=int, default=32)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--modes", default="token,none,inject")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--beta", default="1,2", help="Comma-separated injection strengths")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--attention", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--include-seen", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.model, args.device, args.layer, args.attention)
    with open(args.output, "a", buffering=1) as output:
        for family in args.task.split(","):
            for seed in map(int, args.seeds.split(",")):
                task = make_task(family=family, seed=seed, n_demo=args.shots,
                                 n_query=args.queries, n_classes=args.classes,
                                 include_seen=args.include_seen)
                plain = runner.layout(task, "none")
                norm_ref = runner.reference(plain) if "inject" in args.modes.split(",") else 0.0
                for mode in args.modes.split(","):
                    layout = runner.layout(task, mode)
                    for beta in (map(float, args.beta.split(",")) if mode == "inject" else [0.0]):
                        start = time.time()
                        rows = []
                        common = dict(model=args.model, task=family, seed=seed, shots=args.shots,
                                      n_queries=args.queries, classes=args.classes, mode=mode,
                                      layer=args.layer, beta=beta, norm_ref=norm_ref,
                                      labels=layout["labels"], metadata=task["metadata"],
                                      prefix_tokens=len(layout["prefix"]))
                        def emit(row):
                            result = {**common, **row}
                            output.write(json.dumps(result, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")
                            rows.append(result)
                        runner.evaluate(task, layout, mode, beta, norm_ref, args.bs, emit)
                        print(f"{family} seed={seed} {mode} beta={beta}: "
                              f"accuracy={np.mean([r['correct'] for r in rows]):.3f} "
                              f"p={np.mean([r['p_correct'] for r in rows]):.3f} "
                              f"mass={np.mean([r['label_mass'] for r in rows]):.3f} "
                              f"seconds={time.time()-start:.1f}", flush=True)


if __name__ == "__main__":
    main()
