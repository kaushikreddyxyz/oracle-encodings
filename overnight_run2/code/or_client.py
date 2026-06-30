"""Async OpenRouter client: bounded concurrency, retry+exp-backoff on 429/5xx,
JSON-object structured output, cumulative cost + call accounting.

NEVER prints, logs, or stores the API key. The key is read from repo-root .env via
python-dotenv and lives only in the Authorization header.
"""
import asyncio
import json
import os
import random
import time

import aiohttp
from dotenv import load_dotenv

REPO_ROOT = "/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class ORClient:
    def __init__(self, concurrency: int = 64, max_retries: int = 6, timeout: int = 120):
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
        self._key = os.environ.get("OPENROUTER_API_KEY")
        if not self._key:
            raise RuntimeError("OPENROUTER_API_KEY missing from .env")
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        # accounting (never includes the key)
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

    async def chat(self, model, messages, temperature=0.9, json_mode=True, max_tokens=1400):
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        backoff = 1.5
        async with self.sem:
            for attempt in range(self.max_retries):
                try:
                    async with self._session.post(ENDPOINT, json=body, headers=headers) as r:
                        if r.status in (429, 500, 502, 503, 504):
                            raise aiohttp.ClientResponseError(
                                r.request_info, r.history, status=r.status, message="retryable")
                        r.raise_for_status()
                        data = await r.json()
                    self.calls += 1
                    usage = data.get("usage") or {}
                    self.cost += float(usage.get("cost") or 0.0)
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                    return data["choices"][0]["message"]["content"]
                except Exception:
                    self.errors += 1
                    if attempt == self.max_retries - 1:
                        return None
                    await asyncio.sleep(backoff + random.uniform(0, backoff))
                    backoff = min(backoff * 2, 30)
        return None

    async def chat_json(self, model, messages, temperature=0.9, max_tokens=1400):
        """Returns a parsed dict, or None on failure / unparseable output."""
        raw = await self.chat(model, messages, temperature, json_mode=True, max_tokens=max_tokens)
        if raw is None:
            return None
        return _safe_json(raw)

    def stats(self):
        return {"calls": self.calls, "cost_usd": round(self.cost, 5),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "errors": self.errors}


def _safe_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        # salvage the outermost {...} if the model wrapped it in prose/code fences
        s, e = raw.find("{"), raw.rfind("}")
        if 0 <= s < e:
            try:
                return json.loads(raw[s:e + 1])
            except Exception:
                return None
        return None
