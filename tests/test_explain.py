"""The explanation must reproduce the model's own probability exactly."""
import numpy as np
import pandas as pd
import pytest

joblib = pytest.importorskip("joblib")

from src.config import MODELS_DIR, PROCESSED_DIR
from src.explain import GROUPS, decompose, explain, group_contributions


@pytest.fixture(scope="module")
def setup():
    bundle = joblib.load(MODELS_DIR / "logistic.joblib")
    games = pd.read_parquet(PROCESSED_DIR / "games_features.parquet")
    games = games[games.season >= 2024].reset_index(drop=True)
    return bundle, games


def test_contributions_sum_to_model_probability(setup):
    bundle, games = setup
    base, contrib = decompose(bundle, games)
    p_explained = 1 / (1 + np.exp(-(base + contrib.sum(axis=1))))
    p_model = bundle["model"].predict_proba(games[bundle["features"]])[:, 1]
    np.testing.assert_allclose(p_explained, p_model, atol=1e-10)


def test_every_feature_belongs_to_a_group(setup):
    bundle, _ = setup
    grouped = {c for cols in GROUPS.values() for c in cols}
    assert set(bundle["features"]) <= grouped


def test_groups_preserve_the_total(setup):
    bundle, games = setup
    _, contrib = decompose(bundle, games)
    np.testing.assert_allclose(group_contributions(contrib).sum(axis=1), contrib.sum(axis=1))


def test_factors_start_from_even_and_reach_model_probability(setup):
    bundle, games = setup
    base, contrib = decompose(bundle, games)
    groups = group_contributions(contrib)
    groups["Home field"] += base
    p = 1 / (1 + np.exp(-groups.sum(axis=1)))  # starts at log-odds 0 = 50%
    np.testing.assert_allclose(p, bundle["model"].predict_proba(games[bundle["features"]])[:, 1], atol=1e-10)


def test_explain_runs(setup):
    bundle, games = setup
    e = explain(bundle, games.head(3))
    assert len(e) == 3 and all(0 < x["p_home"] < 1 for x in e)
