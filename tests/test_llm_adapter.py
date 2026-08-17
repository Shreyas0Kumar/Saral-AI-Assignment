"""The Ollama adapter.

These run with no Ollama and no network: the point is that the adapter degrades
to a clear signal rather than an exception, because `make all` must never depend
on it. The cost arm is an offline script, not part of the pipeline.
"""

from __future__ import annotations

import json

from saral.adapters.llm.ollama_client import OllamaClient


def test_client_reports_unavailable_without_a_server(tmp_path):
    client = OllamaClient(model="nonexistent:1b", host="http://127.0.0.1:1", cache_dir=tmp_path)
    assert client.available() is False


def test_generate_returns_an_error_rather_than_raising(tmp_path):
    """A dead Ollama must not take the script down mid-corpus."""
    client = OllamaClient(model="nonexistent:1b", host="http://127.0.0.1:1", cache_dir=tmp_path)
    result = client.generate("hello", None)
    assert "error" in result
    assert result["text"] == ""
    assert client.stats.errors == 1
    assert client.stats.calls == 0, "a failed call is not a call"


def test_cache_key_covers_model_prompt_and_schema(tmp_path):
    """Changing any of the three must invalidate the cache."""
    client = OllamaClient(model="a:1b", cache_dir=tmp_path)
    base = client._cache_key("p", {"type": "object"})
    assert base != client._cache_key("p2", {"type": "object"})
    assert base != client._cache_key("p", {"type": "array"})
    other = OllamaClient(model="b:1b", cache_dir=tmp_path)
    assert base != other._cache_key("p", {"type": "object"})


def test_cache_hit_is_counted_separately_from_a_call(tmp_path):
    """A cache hit is not evidence about model cost and must not be counted as one."""
    client = OllamaClient(model="a:1b", cache_dir=tmp_path)
    key = client._cache_key("p", None)
    (tmp_path / f"{key}.json").write_text(json.dumps({"text": "{}"}), encoding="utf-8")
    result = client.generate("p", None)
    assert result["from_cache"] is True
    assert client.stats.cache_hits == 1
    assert client.stats.calls == 0


def test_response_schema_pins_role_family_to_the_taxonomy():
    """The schema is what makes the cost arm a fair test rather than a strawman."""
    import importlib.util
    import pathlib

    from saral.contracts.taxonomy import RoleFamily

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_llm_cost_arm.py"
    spec = importlib.util.spec_from_file_location("cost_arm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    enum = module.RESPONSE_SCHEMA["properties"]["role_family"]["enum"]
    assert enum == [f.value for f in RoleFamily]
    assert set(module.RESPONSE_SCHEMA["required"]) == {
        "role_family", "seniority", "years_relevant",
    }


# --------------------------------------------------------------------------
# Hosted arm (Gemini). Runs with no key and no network.
# --------------------------------------------------------------------------
def test_gemini_without_a_key_returns_an_error_not_an_exception(monkeypatch, tmp_path):
    from saral.adapters.llm.gemini_client import GeminiClient

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = GeminiClient(cache_dir=tmp_path)
    assert client.available() is False
    result = client.generate("hello", None)
    assert "GOOGLE_API_KEY" in result["error"]
    assert result["text"] == ""


def test_gemini_key_is_only_ever_read_from_the_environment(monkeypatch, tmp_path):
    """The key must never be read from a file or persisted by the client."""
    from saral.adapters.llm.gemini_client import GeminiClient

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    client = GeminiClient(cache_dir=tmp_path)
    assert client.api_key == "test-key-not-real"
    # Nothing the client writes may contain the key.
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "test-key-not-real" not in path.read_text(encoding="utf-8")


def test_gemini_schema_pins_role_family_to_the_taxonomy():
    """Gemini speaks an OpenAPI dialect; the constraint must survive translation."""
    import importlib.util
    import pathlib

    from saral.contracts.taxonomy import RoleFamily

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_llm_cost_arm.py"
    spec = importlib.util.spec_from_file_location("cost_arm_g", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    schema = module.GEMINI_SCHEMA
    assert schema["properties"]["role_family"]["enum"] == [f.value for f in RoleFamily]
    assert schema["type"] == "OBJECT", "Gemini requires uppercase OpenAPI types"


def test_no_shipped_module_imports_a_hosted_llm_client():
    """The hot path must not acquire a network dependency by accident.

    The Gemini and Ollama clients exist for offline measurement scripts only.
    If anything under pipeline/, core/ or api/ starts importing them, the "zero
    LLM calls on the hot path" claim quietly stops being true.
    """
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "saral"
    watched = ["core", "pipeline", "api", "contracts", "telemetry"]
    offenders = []
    for package in watched:
        for path in (src / package).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "gemini_client" in text or "ollama_client" in text:
                offenders.append(str(path.relative_to(src)))
    assert not offenders, f"hosted/local LLM client imported on the served path: {offenders}"
