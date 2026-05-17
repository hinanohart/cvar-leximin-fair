"""Shared test fixtures: deterministic synthetic data with controllable group-wise difficulty."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synth_groups():
    """Three groups with different difficulty: A (easy), B (medium), C (hard).

    Returns dict with X, y, sensitive_features. Groups have different
    label distributions and feature noise levels so an unconstrained model
    will systematically err more on group C — the natural worst-off.
    """
    rng = np.random.default_rng(42)
    n_a, n_b, n_c = 400, 400, 200
    X_a = rng.normal(0, 1.0, size=(n_a, 4))
    y_a = (X_a[:, 0] + X_a[:, 1] > 0).astype(int)

    X_b = rng.normal(0, 1.0, size=(n_b, 4))
    noise_b = rng.normal(0, 0.5, size=n_b)
    y_b = (X_b[:, 0] + X_b[:, 1] + noise_b > 0).astype(int)

    X_c = rng.normal(0, 1.5, size=(n_c, 4))
    noise_c = rng.normal(0, 2.0, size=n_c)
    y_c = (X_c[:, 0] + X_c[:, 1] + noise_c > 0).astype(int)

    X = np.vstack([X_a, X_b, X_c])
    y = np.concatenate([y_a, y_b, y_c])
    s = np.array(["A"] * n_a + ["B"] * n_b + ["C"] * n_c)
    perm = rng.permutation(len(y))
    return {"X": X[perm], "y": y[perm], "sensitive_features": s[perm]}
