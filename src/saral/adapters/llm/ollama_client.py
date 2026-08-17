"""Ollama client with schema-constrained output and a disk cache.

Two things make this a fair measurement of "an LLM per row" rather than a
strawman:

**The output is constrained by a JSON schema**, not by asking nicely in the
prompt. Ollama's `format` parameter accepts a full JSON schema, so `role_family`
is restricted to the 12-value enum and the model *cannot* return
"ML/Data Engineering" or prose. Any remaining error is a reasoning error, which
is what we want to measure. Measuring a model's ability to remember to emit
valid JSON and calling that accuracy would be a strawman.

**`temperature: 0` and a fixed seed**, so the arm is reproducible and the cache
is sound.

The disk cache is keyed on `sha256(model + prompt + schema)` and counts hits
separately from calls, because a cache hit is not evidence about model cost.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class LlmStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "wall_ms": round(self.wall_ms, 1),
            "errors": self.errors,
        }


@dataclass
class OllamaClient:
    model: str
    host: str = DEFAULT_HOST
    seed: int = 20260817
    cache_dir: Path | None = None
    timeout: float = 300.0
    stats: LlmStats = field(default_factory=LlmStats)

    def __post_init__(self) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- plumbing ---------------------------------------------------------
    def _cache_key(self, prompt: str, schema: dict | None) -> str:
        payload = f"{self.model}\x00{prompt}\x00{json.dumps(schema, sort_keys=True)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return (self.cache_dir / f"{key}.json") if self.cache_dir else None

    def available(self) -> bool:
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                tags = json.loads(response.read())
        except Exception:
            return False
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n == self.model or n.startswith(self.model.split(":")[0]) for n in names)

    def pinned_tag(self) -> str:
        """The exact digest the run used, for the manifest -- not just the tag."""
        try:
            import urllib.request

            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as response:
                tags = json.loads(response.read())
            for entry in tags.get("models", []):
                if entry.get("name") == self.model:
                    return f"{self.model}@{entry.get('digest', '')[:19]}"
        except Exception:
            pass
        return self.model

    # -- the call ---------------------------------------------------------
    def generate(self, prompt: str, schema: dict | None = None) -> dict[str, Any]:
        key = self._cache_key(prompt, schema)
        path = self._cache_path(key)
        if path and path.exists():
            self.stats.cache_hits += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["from_cache"] = True
            return payload

        import urllib.error
        import urllib.request

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": self.seed},
        }
        if schema is not None:
            body["format"] = schema

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read())
        except Exception as exc:  # network, timeout, model missing
            self.stats.errors += 1
            self.stats.wall_ms += (time.perf_counter() - started) * 1000
            return {"error": f"{type(exc).__name__}: {exc}", "text": "", "from_cache": False}

        elapsed_ms = (time.perf_counter() - started) * 1000
        self.stats.calls += 1
        self.stats.wall_ms += elapsed_ms
        self.stats.prompt_tokens += int(raw.get("prompt_eval_count") or 0)
        self.stats.completion_tokens += int(raw.get("eval_count") or 0)

        payload = {
            "text": raw.get("response", ""),
            "prompt_tokens": int(raw.get("prompt_eval_count") or 0),
            "completion_tokens": int(raw.get("eval_count") or 0),
            "wall_ms": round(elapsed_ms, 2),
            # Ollama reports these in nanoseconds; they exclude queueing, so
            # they are the fairer number for a per-row cost claim.
            "eval_ms": round((raw.get("eval_duration") or 0) / 1e6, 2),
            "load_ms": round((raw.get("load_duration") or 0) / 1e6, 2),
            "from_cache": False,
        }
        if path:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return payload
