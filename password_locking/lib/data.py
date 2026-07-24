"""Shared data plumbing for the password-locking experiment (MMLU).

Follows Greenblatt et al. 2024 (arXiv:2405.19550): MMLU with 1-shot prompts
containing correct answers, splits 10/40/30/20 (weak_train / lock_train /
elicit / val), correctness of the greedy generation as the metric.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

MMLU_REPO = "cais/mmlu"
SPLIT_FRACTIONS = {"weak_train": 0.10, "lock_train": 0.40, "elicit": 0.30, "val": 0.20}
LETTERS = "ABCD"


# ---------------------------------------------------------------------- io


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------------------------------------------------- mmlu


def _load_parquet_split(split: str) -> pd.DataFrame:
    files = sorted(
        f for f in list_repo_files(MMLU_REPO, repo_type="dataset")
        if f.startswith(f"all/{split}") and f.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no parquet for split {split!r} in {MMLU_REPO}")
    parts = [pd.read_parquet(hf_hub_download(MMLU_REPO, f, repo_type="dataset"))
             for f in files]
    return pd.concat(parts, ignore_index=True)


def load_mmlu_pool() -> list[dict]:
    """Question pool: test + validation splits (~15.6k problems, paper scale).
    The dev split is reserved for few-shot examples."""
    rows = []
    for split in ("test", "validation"):
        df = _load_parquet_split(split)
        for i, r in df.iterrows():
            rows.append({
                "qid": f"{r['subject']}:{split}:{i}",
                "subject": str(r["subject"]),
                "question": str(r["question"]),
                "choices": [str(c) for c in r["choices"]],
                "answer": int(r["answer"]),
            })
    return rows


def load_dev_shots(n_shots: int) -> dict[str, list[dict]]:
    """First n dev questions per subject, used as few-shot examples with
    correct answers shown (paper: 1-shot with correct answers)."""
    if n_shots == 0:
        return {}
    df = _load_parquet_split("dev")
    shots: dict[str, list[dict]] = {}
    for _, r in df.iterrows():
        s = str(r["subject"])
        if len(shots.setdefault(s, [])) < n_shots:
            shots[s].append({
                "question": str(r["question"]),
                "choices": [str(c) for c in r["choices"]],
                "answer": int(r["answer"]),
            })
    return shots


# ------------------------------------------------------------------ prompts


def _question_block(question: str, choices: list[str], answer: int | None) -> str:
    lines = [question.strip()]
    lines += [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    lines.append("Answer:" if answer is None else f"Answer: {LETTERS[answer]}")
    return "\n".join(lines)


def build_prompt(row: dict, shots: dict[str, list[dict]]) -> str:
    subject = row["subject"].replace("_", " ")
    parts = [f"The following are multiple choice questions (with answers) "
             f"about {subject}."]
    for s in shots.get(row["subject"], []):
        parts.append(_question_block(s["question"], s["choices"], s["answer"]))
    parts.append(_question_block(row["question"], row["choices"], None))
    return "\n\n".join(parts)


def completion_for(answer: int) -> str:
    return f" {LETTERS[answer]}"


def extract_letter(text: str) -> str | None:
    m = re.search(r"[ABCD]", text)
    return m.group(0) if m else None


def build_examples(rows: list[dict], n_shots: int) -> list[dict]:
    """Attach prompt / ground-truth letter to split rows."""
    shots = load_dev_shots(n_shots)
    return [{**r, "prompt": build_prompt(r, shots), "gt": LETTERS[r["answer"]]}
            for r in rows]


# ------------------------------------------------------------------- splits


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Deterministic per-subject split into SPLIT_FRACTIONS."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[dict]] = {k: [] for k in SPLIT_FRACTIONS}
    subjects = sorted({r["subject"] for r in rows})
    for subject in subjects:
        idx = [i for i, r in enumerate(rows) if r["subject"] == subject]
        rng.shuffle(idx)
        n = len(idx)
        bounds = np.floor(np.cumsum(list(SPLIT_FRACTIONS.values())) * n + 0.5)
        start = 0
        for name, end in zip(SPLIT_FRACTIONS, bounds.astype(int)):
            out[name] += [rows[i] for i in idx[start:end]]
            start = end
        out["val"] += [rows[i] for i in idx[start:]]  # rounding remainder
    return out


# -------------------------------------------------------------- tokenization


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    """Tokenize with an explicit BOS-like token at position 0 (Qwen2.5 has no
    BOS; its EOS <|endoftext|> doubles as the declared bos_token_id). The
    signature injection variants index positions relative to this token."""
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    return [bos] + tokenizer(prompt, add_special_tokens=False)["input_ids"]


def encode_completion(tokenizer, completion: str) -> list[int]:
    return (tokenizer(completion, add_special_tokens=False)["input_ids"]
            + [tokenizer.eos_token_id])


# ------------------------------------------------------------- completions


def canonical_completion(sample: dict) -> str:
    """Canonicalize a sampled completion to " X" when a letter was parsed,
    else keep the raw text (an unparseable answer IS the weak policy)."""
    return f" {sample['letter']}" if sample.get("letter") else sample["text"]


def sample_completions(row: dict, raw: bool = False) -> list[str]:
    """Completion texts for one sample_teacher output row."""
    return [s["text"] if raw else canonical_completion(s) for s in row["samples"]]
