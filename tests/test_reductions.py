"""Tests for CVaRReduction and LeximinReduction."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cvar_leximin import CVaRReduction, LeximinReduction
from cvar_leximin.metrics import bootstrap_ci, subgroup_cvar, subgroup_losses


def _base():
    return LogisticRegression(max_iter=200, solver="liblinear")


def test_cvar_basic_fit_predict(synth_groups):
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.34, max_iter=5).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    preds = clf.predict(d["X"])
    assert preds.shape == d["y"].shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_leximin_basic_fit_predict(synth_groups):
    d = synth_groups
    clf = LeximinReduction(_base(), max_iter_per_level=3).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    preds = clf.predict(d["X"])
    assert preds.shape == d["y"].shape


def test_cvar_reduces_cvar_objective_vs_baseline(synth_groups):
    """CVaR-fitted model should not have worse CVaR_α than the unconstrained baseline.

    Best-iteration tracking guarantees the returned model is at least as good as
    iteration 0 (the unweighted fit), measured by CVaR_α.
    """
    d = synth_groups
    base = _base().fit(d["X"], d["y"])
    base_cvar = subgroup_cvar(d["y"], base.predict(d["X"]), d["sensitive_features"], alpha=0.34)

    clf = CVaRReduction(_base(), alpha=0.34, max_iter=10, eta=1.0).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    fair_cvar = subgroup_cvar(d["y"], clf.predict(d["X"]), d["sensitive_features"], alpha=0.34)
    assert fair_cvar <= base_cvar + 1e-9, (
        f"CVaR objective got worse: baseline={base_cvar:.4f}, fair={fair_cvar:.4f}"
    )


def test_leximin_reduces_worst_group_vs_baseline(synth_groups):
    d = synth_groups
    base = _base().fit(d["X"], d["y"])
    base_losses = subgroup_losses(d["y"], base.predict(d["X"]), d["sensitive_features"])
    base_worst = max(base_losses.values())

    clf = LeximinReduction(_base(), max_iter_per_level=5, eta=2.0).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    fair_losses = subgroup_losses(d["y"], clf.predict(d["X"]), d["sensitive_features"])
    fair_worst = max(fair_losses.values())
    assert fair_worst <= base_worst + 1e-9


def test_cvar_best_iteration_at_least_as_good_as_initial(synth_groups):
    """Best-iteration tracking guarantees the returned model is at least as good as it 0.

    Raw history may oscillate (re-weighting is a non-monotone Lagrangian update);
    what we contract on is that the *selected* model dominates iteration 0.
    """
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.34, max_iter=10, eta=1.0).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    hist = clf.history_.cvar
    assert len(hist) >= 1
    returned_cvar = subgroup_cvar(d["y"], clf.predict(d["X"]), d["sensitive_features"], alpha=0.34)
    assert returned_cvar <= hist[0] + 1e-9, (
        f"returned model worse than iter 0: returned={returned_cvar:.4f}, iter0={hist[0]:.4f}"
    )


def test_cvar_alpha_0_is_maximin(synth_groups):
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.0, max_iter=8, eta=1.5).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    losses = subgroup_losses(d["y"], clf.predict(d["X"]), d["sensitive_features"])
    # at alpha=0 the optimizer targets the single worst group; that group's loss
    # should be at most a small margin above the second-worst-group loss
    vals = sorted(losses.values(), reverse=True)
    if len(vals) >= 2:
        assert vals[0] - vals[1] < 0.4


def test_cvar_param_validation():
    base = _base()
    with pytest.raises(ValueError):
        CVaRReduction(base, alpha=1.5).fit(
            np.zeros((4, 2)), np.array([0, 1, 0, 1]), sensitive_features=np.array(["a"] * 4)
        )
    with pytest.raises(ValueError):
        CVaRReduction(base, max_iter=0).fit(
            np.zeros((4, 2)), np.array([0, 1, 0, 1]), sensitive_features=np.array(["a"] * 4)
        )


def test_leximin_param_validation():
    base = _base()
    with pytest.raises(ValueError):
        LeximinReduction(base, eta=-1).fit(
            np.zeros((4, 2)), np.array([0, 1, 0, 1]), sensitive_features=np.array(["a"] * 4)
        )


def test_leximin_levels_truncate(synth_groups):
    d = synth_groups
    clf = LeximinReduction(_base(), levels=1, max_iter_per_level=3).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    assert len({h["level"] for h in clf.history_}) == 1


def test_predict_proba_propagates(synth_groups):
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.34, max_iter=3).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    proba = clf.predict_proba(d["X"])
    assert proba.shape == (len(d["y"]), 2)


def test_sensitive_features_alignment_error():
    with pytest.raises(ValueError):
        CVaRReduction(_base()).fit(
            np.zeros((4, 2)),
            np.array([0, 1, 0, 1]),
            sensitive_features=np.array(["a", "b"]),
        )


def test_bootstrap_ci_brackets_point_estimate(synth_groups):
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.34, max_iter=3).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    preds = clf.predict(d["X"])
    point, lo, hi = bootstrap_ci(
        subgroup_cvar, d["y"], preds, d["sensitive_features"], alpha=0.34, n_boot=80, random_state=1
    )
    assert lo <= point <= hi
    # the CI half-width should be small relative to the point estimate
    assert hi - lo < 0.6


def test_single_group_does_not_crash():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    s = np.array(["only"] * 60)
    clf = CVaRReduction(_base(), max_iter=2).fit(X, y, sensitive_features=s)
    assert clf.predict(X).shape == y.shape


def test_rejects_estimator_without_sample_weight(synth_groups):
    """KNN does not accept sample_weight; we should fail clearly up front."""
    from sklearn.neighbors import KNeighborsClassifier

    d = synth_groups
    with pytest.raises(TypeError, match="sample_weight"):
        CVaRReduction(KNeighborsClassifier()).fit(
            d["X"], d["y"], sensitive_features=d["sensitive_features"]
        )
    with pytest.raises(TypeError, match="sample_weight"):
        LeximinReduction(KNeighborsClassifier()).fit(
            d["X"], d["y"], sensitive_features=d["sensitive_features"]
        )


def test_decision_function_propagates(synth_groups):
    """decision_function should delegate to base estimator when available."""
    d = synth_groups
    clf = CVaRReduction(_base(), alpha=0.34, max_iter=3).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    scores = clf.decision_function(d["X"])
    assert scores.shape == (len(d["y"]),)


def test_decision_function_absent_raises(synth_groups):
    """When the base estimator lacks decision_function, raise AttributeError."""
    from sklearn.tree import DecisionTreeClassifier

    d = synth_groups
    clf = CVaRReduction(DecisionTreeClassifier(max_depth=3), alpha=0.34, max_iter=2).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    with pytest.raises(AttributeError, match="decision_function"):
        clf.decision_function(d["X"])


def test_random_state_propagates_and_is_deterministic(synth_groups):
    """random_state forwarded to base estimator yields reproducible fits."""
    from sklearn.ensemble import GradientBoostingClassifier

    d = synth_groups

    def _gb():
        return GradientBoostingClassifier(n_estimators=10, max_depth=2)

    clf1 = CVaRReduction(_gb(), alpha=0.34, max_iter=3, random_state=7).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    clf2 = CVaRReduction(_gb(), alpha=0.34, max_iter=3, random_state=7).fit(
        d["X"], d["y"], sensitive_features=d["sensitive_features"]
    )
    np.testing.assert_array_equal(clf1.predict(d["X"]), clf2.predict(d["X"]))
