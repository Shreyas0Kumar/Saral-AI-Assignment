from __future__ import annotations

import pytest

from saral.config_loader import load_all
from saral.pipeline.arms import build_classifier
from saral.pipeline.io import DEFAULT_COMPUTED_AT, load_candidates, corpus_as_of


@pytest.fixture(scope="session")
def cfg():
    extract_cfg, _ = load_all()
    return extract_cfg


@pytest.fixture(scope="session")
def weights():
    _, weight_cfg = load_all()
    return weight_cfg


@pytest.fixture(scope="session")
def profiles():
    return load_candidates()


@pytest.fixture(scope="session")
def as_of(profiles):
    return corpus_as_of(profiles)


@pytest.fixture(scope="session")
def computed_at():
    return DEFAULT_COMPUTED_AT


@pytest.fixture
def classifier(cfg):
    return build_classifier("signals_v1_lexicon_only", cfg)


@pytest.fixture(scope="session")
def signals(profiles):
    from saral.pipeline.extract_pass import run_extract

    records, traces, _ = run_extract(profiles, arm="signals_v1_lexicon_only")
    return {r.candidate_id: r for r in records}, traces


@pytest.fixture
def profile_by_id(profiles):
    return {p["id"]: p for p in profiles}
