"""The service contract.

`/health` gets the most attention here, because a health check that returns 200
unconditionally is worse than none -- it converts a real outage into a silent
one.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from saral.api.app import app  # noqa: E402
from saral.contracts.reason_vocab import validate  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_extract_returns_schema_valid_signals(client, profiles):
    response = client.post(
        "/signals/extract",
        json={"profiles": profiles[:3], "as_of": "2025-08-01", "computed_at": "2026-08-17T10:03:11Z"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    for record in payload["signals"]:
        assert set(record) >= {
            "candidate_id", "role_family", "seniority", "years_total", "years_relevant",
            "core_skills", "claimed_skills_unverified", "skill_noise_ratio",
            "tenure_stability", "switch_intent", "confidence", "reason_codes",
            "signals_version", "computed_at", "input_hash",
        }
        assert not validate(record["reason_codes"])


def test_extract_is_deterministic_across_requests(client, profiles):
    body = {"profiles": profiles[:2], "as_of": "2025-08-01", "computed_at": "2026-08-17T10:03:11Z"}
    first = client.post("/signals/extract", json=body).json()["signals"]
    second = client.post("/signals/extract", json=body).json()["signals"]
    assert first == second


def test_extract_rejects_a_profile_with_no_id(client):
    response = client.post("/signals/extract", json={"profiles": [{"headline": "x"}]})
    assert response.status_code == 422


def test_score_with_inline_profiles(client, profiles):
    job = {
        "job_id": "JD-TEST", "title": "Senior Backend Engineer",
        "location": "Bengaluru (hybrid)", "min_years": 5, "max_years": 9,
        "must_have": ["Python", "PostgreSQL"], "good_to_have": ["Kafka"],
        "raw_query": "senior backend engineer python postgres bangalore",
    }
    response = client.post("/score", json={"job": job, "profiles": profiles, "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_family"] == "backend"
    assert payload["count"] == 5
    ranks = [r["rank"] for r in payload["results"]]
    assert ranks == sorted(ranks)
    for record in payload["results"]:
        assert 0 <= record["fit_score"] <= 100
        assert not validate(record["reason_codes"])
        assert set(record["score_breakdown"]) == {
            "role_match", "skill_overlap", "seniority_fit", "evidence_of_shipping",
            "location_fit", "switch_intent", "tenure_stability",
        }


def test_score_rejects_a_job_whose_family_cannot_be_resolved(client, profiles):
    job = {
        "job_id": "JD-BAD", "title": "Chief Vibes Officer", "min_years": 1,
        "max_years": 3, "must_have": [], "good_to_have": [], "raw_query": "vibes",
    }
    response = client.post("/score", json={"job": job, "profiles": profiles[:2]})
    assert response.status_code == 422
    # It must say what to do about it, not just fail.
    assert "lexicon.yaml" in response.json()["detail"]


def test_health_reports_versions_and_config_hashes(client):
    response = client.get("/health")
    assert response.status_code in (200, 503)
    payload = response.json() if response.status_code == 200 else response.json()["detail"]
    assert payload["signals_version"]
    assert payload["config_hashes"]
    assert "status" in payload


def test_health_is_not_unconditionally_ok(monkeypatch, client):
    """The check must actually fail when the store is broken."""
    from saral.adapters.store import sqlite_repo

    class BrokenRepo:
        def health(self):
            return {"status": "unhealthy", "reason": "signals table is empty", "signals": 0}

    from saral.api import app as app_module

    monkeypatch.setitem(app_module._state, "repo", BrokenRepo())
    response = client.get("/health")
    assert response.status_code == 503
    app_module._state.pop("repo", None)


def test_dashboard_renders(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "LLM calls on the hot path" in response.text
