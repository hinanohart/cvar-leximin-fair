"""Light adapter layer for `fairlearn <https://fairlearn.org>`_ interoperability.

`cvar-leximin-fair` does *not* subclass ``fairlearn.reductions.Moment`` (whose API
is tied to fairlearn's LP-style ExponentiatedGradient / GridSearch optimizers).
Instead, our reductions are scikit-learn–shaped (``fit(X, y, sensitive_features=...)``,
``predict``, ``predict_proba``) and slot directly into the *evaluation* side of
fairlearn, in particular :class:`fairlearn.metrics.MetricFrame`.

This module provides:

- :func:`cvar_metricframe` — build a ``MetricFrame`` that reports per-group loss
  alongside CVaR-α and leximin-vector aggregations.
- :func:`leximin_compare` — compare two fitted predictors by their leximin vectors.

If fairlearn is not installed the module still imports; the affected functions
raise ``ImportError`` at call time with a useful hint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cvar_leximin.metrics import leximin_vector, subgroup_cvar, subgroup_losses


def _require_fairlearn() -> Any:
    try:
        import fairlearn  # noqa: F401
        import fairlearn.metrics as flm

        return flm
    except ImportError as exc:
        raise ImportError(
            "fairlearn is required for fairlearn_compat features. "
            "Install with `pip install fairlearn>=0.10`."
        ) from exc


def cvar_metricframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Any:
    """Return a fairlearn ``MetricFrame`` with per-group 0/1 loss.

    The returned frame's ``by_group`` table holds per-group loss; aggregate
    Rawlsian-style summaries (CVaR_α, leximin vector) are *not* baked into
    the frame because ``MetricFrame`` aggregates by mean. To compute those
    on the predictions, call :func:`cvar_leximin.metrics.subgroup_cvar` or
    :func:`cvar_leximin.metrics.leximin_vector` directly — they take the
    same ``(y_true, y_pred, sensitive_features)`` arguments.
    """
    flm = _require_fairlearn()

    def _loss(y_t: np.ndarray, y_p: np.ndarray) -> float:
        return float(np.mean(np.asarray(y_t) != np.asarray(y_p)))

    return flm.MetricFrame(
        metrics={"loss": _loss},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )


def leximin_compare(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    sensitive_features: np.ndarray,
) -> int:
    """Lexicographic comparison of two predictors by group-loss vector.

    Returns ``-1`` if A dominates B (A's worst-first vector is lex-smaller),
    ``+1`` if B dominates A, ``0`` if equal.
    """
    a = leximin_vector(y_true, y_pred_a, sensitive_features)
    b = leximin_vector(y_true, y_pred_b, sensitive_features)
    if tuple(a) < tuple(b):
        return -1
    if tuple(a) > tuple(b):
        return 1
    return 0


def summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    *,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Bundle of metrics suitable for logging or CLI output (no fairlearn dep)."""
    return {
        "per_group_loss": subgroup_losses(y_true, y_pred, sensitive_features),
        "cvar_alpha": subgroup_cvar(y_true, y_pred, sensitive_features, alpha=alpha),
        "leximin_vector": leximin_vector(y_true, y_pred, sensitive_features),
    }
