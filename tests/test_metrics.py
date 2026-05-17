"""Tests for metrics module."""

from __future__ import annotations

import numpy as np
import pytest

from cvar_leximin.metrics import (
    bootstrap_ci,
    leximin_vector,
    subgroup_cvar,
    subgroup_losses,
    worst_quantile_gap,
)


def test_subgroup_losses_basic():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 1, 1, 1, 0, 0, 0, 1])
    s = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    losses = subgroup_losses(y, p, s)
    assert losses["A"] == pytest.approx(0.25)
    assert losses["B"] == pytest.approx(0.25)


def test_leximin_vector_is_sorted_descending():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    p = np.array([0, 1, 1, 1, 1, 1, 1, 1])
    s = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    v = leximin_vector(y, p, s)
    assert v == sorted(v, reverse=True)


def test_cvar_alpha_0_equals_max():
    y = np.zeros(6, dtype=int)
    p = np.array([1, 0, 0, 1, 1, 1])
    s = np.array(["A", "A", "B", "B", "C", "C"])
    losses = subgroup_losses(y, p, s)
    assert subgroup_cvar(y, p, s, alpha=0.0) == pytest.approx(max(losses.values()))


def test_cvar_alpha_1_equals_mean():
    y = np.zeros(6, dtype=int)
    p = np.array([1, 0, 0, 1, 1, 1])
    s = np.array(["A", "A", "B", "B", "C", "C"])
    losses = subgroup_losses(y, p, s)
    assert subgroup_cvar(y, p, s, alpha=1.0) == pytest.approx(np.mean(list(losses.values())))


def test_cvar_validates_alpha():
    y = np.array([0, 1])
    p = np.array([0, 1])
    s = np.array(["A", "B"])
    with pytest.raises(ValueError):
        subgroup_cvar(y, p, s, alpha=2.0)


def test_worst_quantile_gap_zero_when_balanced():
    y = np.zeros(4, dtype=int)
    p = np.array([1, 0, 1, 0])
    s = np.array(["A", "A", "B", "B"])
    assert worst_quantile_gap(y, p, s, alpha=0.5) == pytest.approx(0.0)


def test_worst_quantile_gap_positive_when_imbalanced():
    y = np.zeros(6, dtype=int)
    p = np.array([1, 1, 0, 0, 1, 0])
    s = np.array(["A", "A", "B", "B", "C", "C"])
    assert worst_quantile_gap(y, p, s, alpha=0.34) > 0


def test_bootstrap_ci_is_deterministic_with_seed():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=120)
    p = rng.integers(0, 2, size=120)
    s = rng.choice(["A", "B", "C"], size=120)
    out1 = bootstrap_ci(subgroup_cvar, y, p, s, alpha=0.34, n_boot=60, random_state=11)
    out2 = bootstrap_ci(subgroup_cvar, y, p, s, alpha=0.34, n_boot=60, random_state=11)
    assert out1 == out2


def test_empty_inputs_do_not_crash():
    assert subgroup_losses(np.array([]), np.array([]), np.array([])) == {}
    assert subgroup_cvar(np.array([]), np.array([]), np.array([]), alpha=0.5) == 0.0
    assert leximin_vector(np.array([]), np.array([]), np.array([])) == []
