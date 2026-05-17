"""CVaR and leximin reductions for group-fair classification.

Both estimators are scikit-learn–compatible and accept a base estimator that
supports ``sample_weight``. They follow the *reductions* paradigm in the spirit
of Agarwal et al. (2018), "A Reductions Approach to Fair Classification" — fairness
is induced by iteratively re-weighting the training distribution, rather than by
post-processing predictions or by modifying the model class.

- :class:`CVaRReduction` minimizes the Conditional Value-at-Risk of group losses
  at quantile ``alpha`` (i.e. the average loss over the worst ``alpha``-fraction of
  groups). This corresponds to a *soft* Rawlsian objective: as ``alpha -> 0`` it
  approaches the strict maximin (worst-group loss).

- :class:`LeximinReduction` produces a lexicographic-minimum allocation across
  groups. It pegs the worst-off group's loss, then re-optimizes for the second
  worst-off, and so on. This is the strict Rawlsian / difference-principle form.

Naming intentionally avoids the word "Rawls" so that the package is judged on the
algorithms, not on the philosophy. See README for the Sen capability acknowledgement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

ArrayLike = np.ndarray


def _as_1d(a: ArrayLike) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _group_losses(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> dict[object, float]:
    """Per-group 0/1 loss."""
    losses: dict[object, float] = {}
    for g in np.unique(sensitive_features):
        mask = sensitive_features == g
        if mask.sum() == 0:
            continue
        losses[g] = float((y_pred[mask] != y_true[mask]).mean())
    return losses


def _cvar_at_alpha(group_losses: dict[object, float], alpha: float) -> float:
    """Average of the worst-``alpha``-fraction of group losses (CVaR / superquantile).

    With G groups the number of "worst" groups taken is ``max(1, ceil(alpha * G))``.
    For ``alpha = 0`` we return the single worst-group loss (the strict maximin).
    """
    if not group_losses:
        return 0.0
    sorted_vals = sorted(group_losses.values(), reverse=True)
    if alpha <= 0:
        return float(sorted_vals[0])
    n_worst = max(1, int(np.ceil(alpha * len(sorted_vals))))
    return float(np.mean(sorted_vals[:n_worst]))


@dataclass
class _History:
    iters: list[int] = field(default_factory=list)
    cvar: list[float] = field(default_factory=list)
    worst: list[float] = field(default_factory=list)
    sample_weight_sum: list[float] = field(default_factory=list)


class CVaRReduction(BaseEstimator, ClassifierMixin):
    """Reduce CVaR of group losses via cost-sensitive re-weighting.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
        Base estimator. Must accept ``sample_weight`` in ``fit``.
    alpha : float, default=0.1
        CVaR quantile (in (0, 1]). Lower ``alpha`` is more conservative
        (closer to strict maximin). ``alpha = 0`` is treated as maximin.
    max_iter : int, default=20
        Maximum re-weighting iterations.
    eta : float, default=1.0
        Step size for weight updates. Per iteration, groups in the worst-CVaR
        tail get their weights multiplied by ``(1 + eta)``.
    tol : float, default=1e-3
        Convergence tolerance on the CVaR objective.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        alpha: float = 0.1,
        max_iter: int = 20,
        eta: float = 1.0,
        tol: float = 1e-3,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sensitive_features: ArrayLike,
    ) -> CVaRReduction:
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {self.max_iter}")
        if self.eta <= 0:
            raise ValueError(f"eta must be > 0; got {self.eta}")

        X = np.asarray(X)
        y = _as_1d(y)
        s = _as_1d(np.asarray(sensitive_features))
        if len(s) != len(y):
            raise ValueError("sensitive_features must align with y")

        self.classes_ = unique_labels(y)
        self.groups_ = np.unique(s)

        sample_weight = np.ones(len(y), dtype=float)
        prev_cvar = np.inf
        self.history_ = _History()
        best_cvar = np.inf
        best_estimator = None
        best_sample_weight = sample_weight.copy()

        for it in range(self.max_iter):
            est = clone(self.estimator)
            est.fit(X, y, sample_weight=sample_weight)
            y_pred = est.predict(X)
            gl = _group_losses(y, y_pred, s)
            cvar = _cvar_at_alpha(gl, self.alpha)
            worst = max(gl.values()) if gl else 0.0
            self.history_.iters.append(it)
            self.history_.cvar.append(cvar)
            self.history_.worst.append(worst)
            self.history_.sample_weight_sum.append(float(sample_weight.sum()))

            if cvar < best_cvar:
                best_cvar = cvar
                best_estimator = est
                best_sample_weight = sample_weight.copy()

            if abs(prev_cvar - cvar) < self.tol and it > 0:
                break
            prev_cvar = cvar

            # bump weights on the worst-tail groups
            sorted_groups = sorted(gl, key=lambda g: -gl[g])
            n_worst = max(1, int(np.ceil(max(self.alpha, 1e-9) * len(sorted_groups))))
            worst_groups = set(sorted_groups[:n_worst])
            new_w = sample_weight.copy()
            for g in worst_groups:
                new_w[s == g] *= 1.0 + self.eta
            new_w *= len(y) / new_w.sum()  # normalize total mass
            sample_weight = new_w

        self.estimator_ = best_estimator
        self.sample_weight_ = best_sample_weight
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        assert self.estimator_ is not None  # noqa: S101 (post-check_is_fitted invariant)
        return self.estimator_.predict(np.asarray(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        assert self.estimator_ is not None
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("Base estimator does not implement predict_proba")
        return self.estimator_.predict_proba(np.asarray(X))


class LeximinReduction(BaseEstimator, ClassifierMixin):
    """Lexicographic-minimum group-loss reduction.

    Iteratively pegs the loss of the current worst-off group at its present level
    by inflating its sample weights, then re-fits the base estimator and moves on
    to the next worst-off group. ``levels`` controls how many of the worst groups
    are pegged in this lex order; the remainder share the unmodified weight.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
        Base estimator. Must accept ``sample_weight`` in ``fit``.
    max_iter_per_level : int, default=8
        Re-weight iterations inside each lex level.
    eta : float, default=1.0
        Step size for weight updates.
    tol : float, default=1e-3
        Convergence tolerance on the worst-group loss inside each level.
    levels : int or None, default=None
        Number of lex levels to peg. ``None`` means *all groups*.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        *,
        max_iter_per_level: int = 8,
        eta: float = 1.0,
        tol: float = 1e-3,
        levels: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.max_iter_per_level = max_iter_per_level
        self.eta = eta
        self.tol = tol
        self.levels = levels

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sensitive_features: ArrayLike,
    ) -> LeximinReduction:
        if self.max_iter_per_level < 1:
            raise ValueError("max_iter_per_level must be >= 1")
        if self.eta <= 0:
            raise ValueError("eta must be > 0")

        X = np.asarray(X)
        y = _as_1d(y)
        s = _as_1d(np.asarray(sensitive_features))
        if len(s) != len(y):
            raise ValueError("sensitive_features must align with y")

        self.classes_ = unique_labels(y)
        self.groups_ = np.unique(s)
        n_groups = len(self.groups_)
        n_levels = n_groups if self.levels is None else min(self.levels, n_groups)

        sample_weight = np.ones(len(y), dtype=float)
        pegged: set[object] = set()
        self.history_: list[dict[str, float | int | str]] = []
        best_lex_key: tuple = ()
        best_estimator = None
        best_sample_weight = sample_weight.copy()

        def _lex_key(group_losses: dict[object, float]) -> tuple:
            # smaller key = lex-better (compare worst-first sorted descending,
            # but negate so tuple comparison picks the smaller worst loss).
            return tuple(sorted(group_losses.values(), reverse=True))

        for level in range(n_levels):
            prev_worst = np.inf
            level_worst_group: object | None = None
            for it in range(self.max_iter_per_level):
                est = clone(self.estimator)
                est.fit(X, y, sample_weight=sample_weight)
                y_pred = est.predict(X)
                gl = _group_losses(y, y_pred, s)
                non_pegged = {g: gl[g] for g in gl if g not in pegged}
                if not non_pegged:
                    break
                worst_group = max(non_pegged, key=lambda g: non_pegged[g])
                worst_loss = non_pegged[worst_group]
                level_worst_group = worst_group
                self.history_.append(
                    {
                        "level": level,
                        "iter": it,
                        "worst_group": str(worst_group),
                        "worst_loss": float(worst_loss),
                    }
                )

                lex_key = _lex_key(gl)
                if best_estimator is None or lex_key < best_lex_key:
                    best_lex_key = lex_key
                    best_estimator = est
                    best_sample_weight = sample_weight.copy()

                if abs(prev_worst - worst_loss) < self.tol and it > 0:
                    break
                prev_worst = worst_loss

                mask = s == worst_group
                new_w = sample_weight.copy()
                new_w[mask] *= 1.0 + self.eta
                new_w *= len(y) / new_w.sum()
                sample_weight = new_w
            if level_worst_group is not None:
                pegged.add(level_worst_group)

        self.estimator_ = best_estimator
        self.sample_weight_ = best_sample_weight
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        assert self.estimator_ is not None  # noqa: S101 (post-check_is_fitted invariant)
        return self.estimator_.predict(np.asarray(X))

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        assert self.estimator_ is not None
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError("Base estimator does not implement predict_proba")
        return self.estimator_.predict_proba(np.asarray(X))
