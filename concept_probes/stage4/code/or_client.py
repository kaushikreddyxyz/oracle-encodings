"""Async OpenRouter client with a full audit trail.

Every call appends one JSONL record to the audit log containing the complete
request messages, the raw response text, usage/cost, retry count, and an error
field — so "what went into the model and what came out" is always answerable.

The API key is read from repo-root .env (OPENROUTER_API_KEY) and is never
printed, logged, or stored anywhere outside the Authorization header.
"""
import asyncio
import json
import os
import random
import time
import uuid

import aiohttp
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class ORClient:
    """OpenAI-compatible chat client. Defaults to OpenRouter; pass endpoint/
    key_env/pricing to target another provider (e.g. Inception direct).
    pricing_per_mtok {prompt, completion} is used to ESTIMATE cost when the
    provider does not return usage.cost (OpenRouter does; Inception does not)."""

    def __init__(self, audit_log_path, concurrency=32, max_retries=6, timeout=180,
                 cost_cap_usd=None, endpoint=ENDPOINT,
                 key_env="OPENROUTER_API_KEY", pricing=None, provider="openrouter",
                 limits=None):
        """limits: optional {rpm, input_tpm, output_tpm} sliding-window budget —
        requests wait (not fail) until the per-minute budget allows them."""
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
        self._key = os.environ.get(key_env)
        if not self._key:
            raise RuntimeError(f"{key_env} missing from .env")
        self.endpoint = endpoint
        self.provider = provider
        self.pricing = pricing
        self.limits = limits
        self._window = []            # (t_monotonic, in_tokens, out_tokens)
        self._win_lock = asyncio.Lock()
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        self.audit_log_path = audit_log_path
        self._audit_lock = asyncio.Lock()
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.cost_cap_usd = cost_cap_usd
        self.calls = 0
        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.errors = 0
        self._session = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *exc):
        await self._session.close()

    async def _acquire_budget(self, est_in, est_out):
        if not self.limits:
            return
        rpm = self.limits.get("rpm", 10**9)
        tin = self.limits.get("input_tpm", 10**12)
        tout = self.limits.get("output_tpm", 10**12)
        while True:
            async with self._win_lock:
                now = time.monotonic()
                self._window = [w for w in self._window if now - w[0] < 60]
                if (len(self._window) < rpm
                        and sum(w[1] for w in self._window) + est_in <= tin
                        and sum(w[2] for w in self._window) + est_out <= tout):
                    self._window.append((now, est_in, est_out))
                    return
            await asyncio.sleep(0.5 + random.uniform(0, 0.5))

    async def _audit(self, rec):
        async with self._audit_lock:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def chat(self, model, messages, temperature, max_tokens, meta=None,
                   json_mode=True, extra_body=None, response_schema=None):
        """Returns (call_id, content_str_or_None). Logs the call either way.
        Empty content (reasoning models exhausting max_tokens) is retried.
        response_schema: JSON Schema dict -> strict structured output (constrained
        decoding at the provider), eliminating format drift entirely."""
        if self.cost_cap_usd is not None and self.cost >= self.cost_cap_usd:
            raise RuntimeError(
                f"cost cap hit: ${self.cost:.2f} >= ${self.cost_cap_usd:.2f} — aborting")
        call_id = uuid.uuid4().hex[:12]
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if self.provider == "openrouter":
            body["usage"] = {"include": True}
        if response_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "result", "strict": True, "schema": response_schema}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if extra_body:
            body.update(extra_body)
        headers = {"Authorization": f"Bearer {self._key}",
                   "Content-Type": "application/json"}
        backoff = 1.5
        content, usage, err, attempts = None, {}, None, 0
        est_in = sum(len(m.get("content", "")) for m in messages) // 4 + 64
        # reserve the FULL max_tokens against the output budget: providers debit
        # reserved (not actual) tokens, and mercury's hidden reasoning tokens
        # count too (measured p95 ~2.1k) — actual-based estimates caused 429s
        est_out = max_tokens
        async with self.sem:
            for attempt in range(self.max_retries):
                await self._acquire_budget(est_in, est_out)
                attempts = attempt + 1
                try:
                    async with self._session.post(self.endpoint, json=body, headers=headers) as r:
                        if r.status in (429, 500, 502, 503, 504):
                            raise aiohttp.ClientResponseError(
                                r.request_info, r.history, status=r.status,
                                message="retryable")
                        r.raise_for_status()
                        data = await r.json()
                    usage = data.get("usage") or {}
                    # bill every attempt, including empty-content ones
                    c = usage.get("cost")
                    if c is None and self.pricing:
                        c = (int(usage.get("prompt_tokens") or 0) * self.pricing["prompt"]
                             + int(usage.get("completion_tokens") or 0)
                             * self.pricing["completion"]) / 1e6
                        usage["cost_estimated"] = round(c, 6)
                    self.cost += float(c or 0.0)
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        raise ValueError("empty_content (reasoning-token exhaustion?)")
                    err = None
                    break
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    content = None
                    self.errors += 1
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(backoff + random.uniform(0, backoff))
                        backoff = min(backoff * 2, 30)
        self.calls += 1
        await self._audit({
            "call_id": call_id, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": self.provider,
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "meta": meta or {}, "request_messages": messages,
            "raw_response": content, "usage": usage, "attempts": attempts, "error": err,
        })
        return call_id, content

    async def chat_json(self, model, messages, temperature, max_tokens, meta=None,
                        extra_body=None, response_schema=None):
        """Returns (call_id, parsed_dict_or_None)."""
        call_id, raw = await self.chat(model, messages, temperature, max_tokens,
                                       meta, extra_body=extra_body,
                                       response_schema=response_schema)
        return call_id, (None if raw is None else _safe_json(raw))

    def stats(self):
        return {"calls": self.calls, "cost_usd": round(self.cost, 5),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "errors": self.errors}


def _safe_json(raw):
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}")
        if 0 <= s < e:
            try:
                obj = json.loads(raw[s:e + 1])
            except Exception:
                return None
    if isinstance(obj, list):   # model wrapped the object in an array
        obj = next((x for x in obj if isinstance(x, dict)), None)
    return obj if isinstance(obj, dict) else None
