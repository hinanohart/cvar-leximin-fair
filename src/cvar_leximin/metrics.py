"""Group-fairness metrics: leximin vector, CVaR, worst-quantile gap.

These metrics operate on per-group losses (or per-group performance scores) and
support bootstrap-CI estimation for honest reporting of small-sample uncertainty.
"""

from __future__ import annotations

import numpy as np


def subgroup_losses(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> dict[object, float]:
    """Per-group 0/1 loss."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    s = np.asarray(sensitive_features).reshape(-1)
    out: dict[object, float] = {}
    for g in np.unique(s):
        m = s == g
        if m.sum() == 0:
            continue
        out[g] = float((y_pred[m] != y_true[m]).mean())
    return out


def leximin_vector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> list[float]:
    """Return per-group losses sorted descending (worst-first).

    Two models compare lexicographically by this vector: smaller in the first
    position dominates; ties break on the next position, etc.
    """
    losses = subgroup_losses(y_true, y_pred, sensitive_features)
    return sorted(losses.values(), reverse=True)


def subgroup_cvar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    *,
    alpha: float = 0.1,
) -> float:
    """CVaR of group losses at quantile ``alpha``.

    The average loss over the worst ``ceil(alpha * G)`` groups out of ``G`` groups.
    ``alpha = 0`` collapses to the maximum (worst-group) loss.
    """
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    losses = list(subgroup_losses(y_true, y_pred, sensitive_features).values())
    if not losses:
        return 0.0
    sorted_vals = sorted(losses, reverse=True)
    if alpha <= 0:
        return float(sorted_vals[0])
    n_worst = max(1, int(np.ceil(alpha * len(sorted_vals))))
    return float(np.mean(sorted_vals[:n_worst]))


def worst_quantile_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    *,
    alpha: float = 0.1,
) -> float:
    """Gap between CVaR-α (worst tail) and (1−α) (best tail) of group losses.

    Returns the difference ``CVaR_alpha − bestCVaR_alpha`` where the best tail
    is the *minimum*-loss tail. Smaller gap = more leximin-fair.
    """
    losses = list(subgroup_losses(y_true, y_pred, sensitive_features).values())
    if not losses:
        return 0.0
    sorted_desc = sorted(losses, reverse=True)
    sorted_asc = sorted(losses)
    n = max(1, int(np.ceil(alpha * len(losses))))
    worst = float(np.mean(sorted_desc[:n]))
    best = float(np.mean(sorted_asc[:n]))
    return worst - best


def bootstrap_ci(
    statistic_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    *,
    n_boot: int = 200,
    confidence: float = 0.95,
    random_state: int | None = 0,
    **kwargs,
) -> tuple[float, float, float]:
    """Bootstrap CI for any of the metrics above.

    Returns ``(point_estimate, lower, upper)`` at the requested confidence level.
    Sampling is *stratified by group* so that no resample drops a group entirely.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    s = np.asarray(sensitive_features).reshape(-1)

    rng = np.random.default_rng(random_state)
    point = statistic_fn(y_true, y_pred, s, **kwargs)
    boots = []
    group_indices = {g: np.flatnonzero(s == g) for g in np.unique(s)}
    for _ in range(n_boot):
        sampled = []
        for _g, idx in group_indices.items():
            if len(idx) == 0:
                continue
            choice = rng.choice(idx, size=len(idx), replace=True)
            sampled.append(choice)
        if not sampled:
            continue
        idx_all = np.concatenate(sampled)
        boots.append(statistic_fn(y_true[idx_all], y_pred[idx_all], s[idx_all], **kwargs))
    if not boots:
        return float(point), float(point), float(point)
    arr = np.asarray(boots)
    half = (1 - confidence) / 2
    lo, hi = np.quantile(arr, [half, 1 - half])
    return float(point), float(lo), float(hi)
