"""Tests for fairlearn-compat layer (MetricFrame interop + leximin compare)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cvar_leximin import CVaRReduction, LeximinReduction
from cvar_leximin.fairlearn_compat import (
    cvar_metricframe,
    leximin_compare,
    summary,
)


def _base():
    return LogisticRegression(max_iter=200, solver="liblinear")


def test_summary_returns_expected_keys(synth_groups):
    d = synth_groups
    base = _base().fit(d["X"], d["y"])
    out = summary(d["y"], base.predict(d["X"]), d["sensitive_features"], alpha=0.34)
    assert set(out.keys()) == {"per_group_loss", "cvar_alpha", "leximin_vector"}
    assert isinstance(out["per_group_loss"], dict)
    assert isinstance(out["cvar_alpha"], float)
    assert isinstance(out["leximin_vector"], list)


def test_leximin_compare_self_returns_zero(synth_groups):
    d = synth_groups
    base = _base().fit(d["X"], d["y"])
    preds = base.predict(d["X"])
    assert leximin_compare(d["y"], preds, preds, d["sensitive_features"]) == 0


def test_leximin_compare_distinguishes_models(synth_groups):
    d = synth_groups
    base = _base().fit(d["X"], d["y"])
    lex = LeximinReduction(_base(), max_iter_per_level=5, eta=2.0).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    cmp = leximin_compare(
        d["y"], lex.predict(d["X"]), base.predict(d["X"]), d["sensitive_features"]
    )
    assert cmp in (-1, 0, 1)


def test_cvar_metricframe_groups(synth_groups):
    flm = pytest.importorskip("fairlearn.metrics")
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.34, max_iter=3).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    mf = cvar_metricframe(d["y"], clf.predict(d["X"]), d["sensitive_features"])
    assert isinstance(mf, flm.MetricFrame)
    assert "loss" in mf.by_group.columns
    assert mf.by_group.shape[0] == len(np.unique(d["sensitive_features"]))
