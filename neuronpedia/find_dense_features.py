"""Fetch the top-N most active (highest activation probability) SAE features
from Neuronpedia's public dataset bucket, and dump them as JSON.

Activation probability = `frac_nonzero`: fraction of tokens in the eval
corpus on which the latent fires (post-JumpReLU > 0). High = dense latent.

Data path used:
    s3://neuronpedia-datasets/v1/{model}/{source}/source.jsonl
    s3://neuronpedia-datasets/v1/{model}/{source}/features/batch-*.jsonl.gz
    s3://neuronpedia-datasets/v1/{model}/{source}/explanations/batch-*.jsonl.gz

Run as a script:
    uv run python neuronpedia/find_dense_features.py            # defaults
    uv run python neuronpedia/find_dense_features.py --layer 12 --width 65k
    uv run python neuronpedia/find_dense_features.py --count 100 -o out.json

Output is JSONL. First line is a `_meta` header (constants — model, SAE
config, eval corpus, query params). Each subsequent line is one feature,
ordered by activation_probability descending. Per-feature schema:
    {"index", "activation_probability", "label", "max_activation",
     "top_activating_tokens", "neuronpedia_url"}
"""

# %% imports
from __future__ import annotations

import argparse
import asyncio
import gzip
import heapq
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# %% constants / defaults
S3_BASE = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
S3_LIST = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
DATASET_PREFIX = "v1"
NP_BASE = "https://www.neuronpedia.org"

DEFAULTS = dict(
    model="gemma-2-9b",
    layer=20,
    hook="res",        # res | mlp | att
    width="16k",       # 16k | 65k | 131k | 1m  (case-insensitive)
    variant="",        # e.g. "l0_32plus"; "" = canonical
    count=50,
    include_explanations=True,
    output=None,       # auto-name if None
    concurrency=8,
)


# %% source-id construction
def build_source_id(layer: int, hook: str, width: str, variant: str = "") -> str:
    """Turn the user-facing knobs into a Neuronpedia SAE source id.

    e.g. (20, 'res', '16k', '')          -> '20-gemmascope-res-16k'
         (20, 'res', '131k', 'l0_32plus') -> '20-gemmascope-res-131k-l0_32plus'
    """
    parts = [str(layer), "gemmascope", hook, width.lower()]
    if variant:
        parts.append(variant)
    return "-".join(parts)


# %% S3 listing + download helpers
async def list_batch_keys(client: httpx.AsyncClient, model: str, source: str, kind: str) -> list[str]:
    """List all `features/` or `explanations/` batch files for a given SAE source."""
    prefix = f"{DATASET_PREFIX}/{model}/{source}/{kind}/"
    keys: list[str] = []
    token: str | None = None
    while True:
        params: dict[str, str] = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = await client.get(S3_LIST, params=params, timeout=60.0)
        r.raise_for_status()
        keys.extend(re.findall(r"<Key>([^<]+\.jsonl\.gz)</Key>", r.text))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not (re.search(r"<IsTruncated>true</IsTruncated>", r.text) and m):
            break
        token = m.group(1)
    return keys


async def fetch_with_retry(client: httpx.AsyncClient, url: str, retries: int = 5) -> bytes:
    """GET with exponential backoff on the transient S3 errors we keep hitting."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = await client.get(url, timeout=120.0)
            r.raise_for_status()
            return r.content
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError,
                httpx.ReadTimeout, httpx.PoolTimeout) as e:
            last_err = e
            backoff = 0.5 * (2 ** attempt)
            print(f"  ! {url} attempt {attempt + 1}/{retries} failed ({e}); retry in {backoff:.1f}s",
                  file=sys.stderr)
            await asyncio.sleep(backoff)
    raise RuntimeError(f"giving up on {url}: {last_err}")


async def fetch_gz_jsonl(client: httpx.AsyncClient, key: str) -> list[dict[str, Any]]:
    """Download a single .jsonl.gz batch file and parse it."""
    content = await fetch_with_retry(client, f"{S3_BASE}/{key}")
    with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
        text = gz.read().decode("utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  ! skipping bad line in {key}: {e}", file=sys.stderr)
    return out


async def fetch_source_info(client: httpx.AsyncClient, model: str, source: str) -> dict[str, Any]:
    """Pull the SAE's source.jsonl — corpus size, dims, hook name, etc."""
    url = f"{S3_BASE}/{DATASET_PREFIX}/{model}/{source}/source.jsonl"
    try:
        content = await fetch_with_retry(client, url)
    except RuntimeError as e:
        print(f"  ! couldn't fetch source.jsonl ({e}); skipping source block", file=sys.stderr)
        return {}
    rec = json.loads(content.decode("utf-8").splitlines()[0])
    sl = rec.get("saelensConfig") or {}
    num_prompts = rec.get("num_prompts") or 0
    tokens_per_prompt = rec.get("num_tokens_in_prompt") or 0
    return {
        "sae": {
            "architecture": sl.get("architecture"),
            "hook_name": sl.get("hook_name"),
            "d_in": sl.get("d_in"),
            "d_sae": sl.get("d_sae"),
            "notes": rec.get("notes"),
        },
        "eval": {
            "dataset": rec.get("dataset"),
            "num_prompts": num_prompts,
            "tokens_per_prompt": tokens_per_prompt,
            "total_tokens": num_prompts * tokens_per_prompt,
        },
    }


# %% top-K by frac_nonzero (streaming, heap-based)
@dataclass(order=True)
class _Ranked:
    frac_nonzero: float
    record: dict[str, Any] = field(compare=False)


async def fetch_top_features(
    client: httpx.AsyncClient,
    model: str,
    source: str,
    count: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Stream every features/batch-*.jsonl.gz for the source, keep top-N by frac_nonzero."""
    keys = await list_batch_keys(client, model, source, kind="features")
    if not keys:
        raise RuntimeError(
            f"No feature batches found at s3://neuronpedia-datasets/"
            f"{DATASET_PREFIX}/{model}/{source}/features/. "
            f"Check --model / --layer / --hook / --width / --variant."
        )
    print(f"  found {len(keys)} feature batch(es) for {model}/{source}", file=sys.stderr)

    heap: list[_Ranked] = []
    sem = asyncio.Semaphore(concurrency)
    seen = 0

    async def process(key: str) -> None:
        nonlocal seen
        async with sem:
            records = await fetch_gz_jsonl(client, key)
        for rec in records:
            seen += 1
            frac = rec.get("frac_nonzero")
            if frac is None:
                continue
            if len(heap) < count:
                heapq.heappush(heap, _Ranked(frac, rec))
            elif frac > heap[0].frac_nonzero:
                heapq.heapreplace(heap, _Ranked(frac, rec))

    await asyncio.gather(*(process(k) for k in keys))
    print(f"  scanned {seen} features; keeping top {len(heap)}", file=sys.stderr)
    return [r.record for r in sorted(heap, key=lambda r: r.frac_nonzero, reverse=True)]


# %% explanation enrichment
async def fetch_explanations_for(
    client: httpx.AsyncClient,
    model: str,
    source: str,
    indices: set[str],
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return {feature_index -> [trimmed explanation records]} for the given indices."""
    keys = await list_batch_keys(client, model, source, kind="explanations")
    if not keys:
        return {}
    sem = asyncio.Semaphore(concurrency)
    by_index: dict[str, list[dict[str, Any]]] = {}

    async def process(key: str) -> None:
        async with sem:
            records = await fetch_gz_jsonl(client, key)
        for rec in records:
            idx = rec.get("index")
            if idx not in indices:
                continue
            by_index.setdefault(idx, []).append(_trim_explanation(rec))

    await asyncio.gather(*(process(k) for k in keys))
    return by_index


def _trim_explanation(rec: dict[str, Any]) -> dict[str, Any]:
    """Keep just the useful explanation fields (description + provenance)."""
    return {
        "description": (rec.get("description") or "").strip(),
        "explanation_model": rec.get("explanationModelName"),
        "explanation_type": rec.get("typeName"),
        "id": rec.get("id"),
    }


# %% feature record -> lean presentation format
def format_feature(
    raw: dict[str, Any],
    explanations: list[dict[str, Any]],
    np_url_base: str,
) -> dict[str, Any]:
    """Lean per-feature record. Most-relevant fields first."""
    label = explanations[0]["description"] if explanations else None
    return {
        "index": raw.get("index"),
        "activation_probability": raw.get("frac_nonzero"),
        "label": label,
        "max_activation": raw.get("maxActApprox"),
        "top_activating_tokens": [
            {"token": t, "value": v}
            for t, v in zip(raw.get("pos_str") or [], raw.get("pos_values") or [])
        ],
        "neuronpedia_url": f"{np_url_base}/{raw.get('modelId')}/{raw.get('layer')}/{raw.get('index')}",
    }


# %% main orchestrator
async def run(args: argparse.Namespace) -> dict[str, Any]:
    source = build_source_id(args.layer, args.hook, args.width, args.variant)
    print(f"-> model={args.model}  source={source}  count={args.count}", file=sys.stderr)

    limits = httpx.Limits(
        max_keepalive_connections=args.concurrency,
        max_connections=args.concurrency * 2,
    )
    async with httpx.AsyncClient(
        http2=False,
        limits=limits,
        headers={"User-Agent": "oracle-encodings/find_dense_features"},
    ) as client:
        source_info_task = asyncio.create_task(fetch_source_info(client, args.model, source))
        raw_features = await fetch_top_features(
            client,
            model=args.model,
            source=source,
            count=args.count,
            concurrency=args.concurrency,
        )

        if args.include_explanations and raw_features:
            print("  fetching explanations...", file=sys.stderr)
            expls_by_idx = await fetch_explanations_for(
                client,
                model=args.model,
                source=source,
                indices={f["index"] for f in raw_features},
                concurrency=args.concurrency,
            )
        else:
            expls_by_idx = {}

        source_info = await source_info_task

    features = [
        format_feature(raw, explanations=expls_by_idx.get(raw["index"], []), np_url_base=NP_BASE)
        for raw in raw_features
    ]

    meta = {
        "model": args.model,
        "sae_source": source,
        "layer": args.layer,
        "hook": args.hook,
        "width": args.width,
        "variant": args.variant or None,
        "count": args.count,
        "sort_key": "activation_probability (frac_nonzero), descending",
        **source_info,
    }
    return {"_meta": meta, "features": features}


# %% CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch top-N densest SAE features from Neuronpedia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULTS["model"],
                   help="Neuronpedia model id (e.g. gemma-2-9b, gemma-2-2b).")
    p.add_argument("--layer", type=int, default=DEFAULTS["layer"], help="Layer index.")
    p.add_argument("--hook", default=DEFAULTS["hook"], choices=["res", "mlp", "att"],
                   help="Gemma Scope hook site.")
    p.add_argument("--width", default=DEFAULTS["width"],
                   help="SAE width tag (16k, 65k, 131k, 1m).")
    p.add_argument("--variant", default=DEFAULTS["variant"],
                   help="Optional variant suffix, e.g. 'l0_32plus'. Empty = canonical.")
    p.add_argument("--count", "-n", type=int, default=DEFAULTS["count"],
                   help="Number of top-density features to return.")
    p.add_argument("--no-explanations", dest="include_explanations",
                   action="store_false", default=DEFAULTS["include_explanations"],
                   help="Skip fetching feature explanations.")
    p.add_argument("--output", "-o", default=DEFAULTS["output"],
                   help="Output JSON path. Default: <source>_top<count>.json next to this file.")
    p.add_argument("--concurrency", type=int, default=DEFAULTS["concurrency"],
                   help="Max concurrent S3 requests.")
    return p.parse_args(argv)


def default_output_path(args: argparse.Namespace) -> Path:
    source = build_source_id(args.layer, args.hook, args.width, args.variant)
    return Path(__file__).parent / f"{args.model}_{source}_top{args.count}.jsonl"


def write_jsonl(out_path: Path, result: dict[str, Any]) -> None:
    """First line: `_meta` header. Subsequent lines: one feature each."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": result["_meta"]}, ensure_ascii=False) + "\n")
        for feat in result["features"]:
            f.write(json.dumps(feat, ensure_ascii=False) + "\n")


def _in_jupyter() -> bool:
    """True iff we're running inside an IPython / Jupyter kernel."""
    return "ipykernel" in sys.modules


def _run_coro(coro):
    """Run a coroutine to completion, even when an event loop is already running
    (Jupyter). Falls back to a one-shot worker thread with its own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def main(argv: list[str] | None = None) -> int:
    if _in_jupyter():
        # Kernel's own --f=<connection-file> would crash argparse. Use DEFAULTS
        # verbatim — edit the DEFAULTS dict at the top of the file to override.
        args = argparse.Namespace(**DEFAULTS)
        print("(running inside Jupyter — using DEFAULTS, ignoring sys.argv)", file=sys.stderr)
    else:
        args = parse_args(argv)
    out_path = Path(args.output) if args.output else default_output_path(args)
    result = _run_coro(run(args))
    write_jsonl(out_path, result)
    print(f"wrote {len(result['features'])} features -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    _rc = main()
    if not _in_jupyter():
        raise SystemExit(_rc)

# %%