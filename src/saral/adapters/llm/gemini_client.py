"""Google AI Studio (Gemini) client -- the *hosted* per-row arm.

Why this arm exists alongside the local ones. The local Ollama arms answer
"what if we self-host a small model on the CPU we already pay for". This one
answers the question a bootstrapped startup would actually ask first: "what if
we just call an API per profile". They are different cost models and both
belong in the comparison:

* local  -> billed in **CPU-seconds**, so latency is the cost driver;
* hosted -> billed in **tokens**, so latency is irrelevant to the bill and
  prompt size is everything.

Conflating them would be a category error, so `cost_model` is recorded on every
result and INFRA.md prices them separately.

The API key is read from the ``GOOGLE_API_KEY`` environment variable and is
never written to disk, logged, or committed. Responses are cached by content
hash, and cache hits are counted separately from calls because a cache hit is
not evidence about API cost.

Output is constrained with ``responseSchema`` + ``responseMimeType`` for the
same reason the Ollama arm uses ``format``: the model must not be able to fail
on syntax, so that every error is a reasoning error.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


@dataclass
class GeminiStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0
    errors: int = 0
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_ms": round(self.wall_ms, 1),
            "errors": self.errors,
            "retries_on_429": self.retries,
        }


@dataclass
class GeminiClient:
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    cache_dir: Path | None = None
    timeout: float = 60.0
    #: Free-tier quota is per-minute, so backoff has to be generous.
    max_retries: int = 6
    backoff_base: float = 5.0
    #: Politeness delay between calls, to stay under the per-minute quota.
    min_interval_s: float = 4.5
    _last_call: float = 0.0
    stats: GeminiStats = field(default_factory=GeminiStats)

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            request = urllib.request.Request(
                f"{API_ROOT}/models?pageSize=1", headers={"x-goog-api-key": self.api_key}
            )
            with urllib.request.urlopen(request, timeout=15):
                return True
        except Exception:
            return False

    def _cache_key(self, prompt: str, schema: dict | None) -> str:
        payload = f"{self.model}\x00{prompt}\x00{json.dumps(schema, sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate(self, prompt: str, schema: dict | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {
                "error": "GOOGLE_API_KEY is not set",
                "text": "",
                "from_cache": False,
            }

        key = self._cache_key(prompt, schema)
        path = (self.cache_dir / f"{key}.json") if self.cache_dir else None
        if path and path.exists():
            self.stats.cache_hits += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["from_cache"] = True
            return payload

        # Stay under the per-minute quota rather than discovering it via 429s.
        since = time.perf_counter() - self._last_call
        if self._last_call and since < self.min_interval_s:
            time.sleep(self.min_interval_s - since)
        self._last_call = time.perf_counter()

        body: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            # temperature 0 so the arm is reproducible and the cache is sound.
            "generationConfig": {"temperature": 0},
        }
        if schema is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = schema

        request = urllib.request.Request(
            f"{API_ROOT}/models/{self.model}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )

        # Free-tier quota is a handful of requests per minute, and a 429 storm
        # otherwise silently turns "the model got it wrong" into "we never
        # asked". That distinction is the whole point of this arm, so retries
        # are mandatory rather than a nicety. `elapsed_ms` deliberately measures
        # only the successful attempt: backoff is our queueing, not the model's
        # latency, and billing it to the model would overstate the arm's cost.
        raw = None
        elapsed_ms = 0.0
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read())
                elapsed_ms = (time.perf_counter() - started) * 1000
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode()[:200]
                if exc.code in (429, 500, 503) and attempt < self.max_retries - 1:
                    self.stats.retries += 1
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                self.stats.errors += 1
                self.stats.wall_ms += (time.perf_counter() - started) * 1000
                return {
                    "error": f"HTTP {exc.code}: {detail}",
                    "text": "",
                    "from_cache": False,
                }
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    self.stats.retries += 1
                    time.sleep(self.backoff_base * (2 ** attempt))
                    continue
                self.stats.errors += 1
                self.stats.wall_ms += (time.perf_counter() - started) * 1000
                return {"error": f"{type(exc).__name__}: {exc}", "text": "", "from_cache": False}

        if raw is None:  # pragma: no cover - exhausted retries
            self.stats.errors += 1
            return {"error": "retries exhausted", "text": "", "from_cache": False}
        usage = raw.get("usageMetadata", {})
        text = ""
        try:
            text = raw["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            # A safety block or an empty candidate. Not an exception -- it is a
            # data point about reliability, and the row records it.
            pass

        self.stats.calls += 1
        self.stats.wall_ms += elapsed_ms
        self.stats.prompt_tokens += int(usage.get("promptTokenCount") or 0)
        self.stats.completion_tokens += int(usage.get("candidatesTokenCount") or 0)

        payload = {
            "text": text,
            "prompt_tokens": int(usage.get("promptTokenCount") or 0),
            "completion_tokens": int(usage.get("candidatesTokenCount") or 0),
            "total_tokens": int(usage.get("totalTokenCount") or 0),
            "wall_ms": round(elapsed_ms, 2),
            "from_cache": False,
            "finish_reason": (raw.get("candidates") or [{}])[0].get("finishReason"),
        }
        if path:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return payload
