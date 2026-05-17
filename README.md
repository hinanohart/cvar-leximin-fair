# cvar-leximin-fair

> **Rawlsian fairness via CVaR / leximin reductions for [fairlearn](https://github.com/fairlearn/fairlearn).**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/hinanohart/cvar-leximin-fair/actions/workflows/test.yml/badge.svg)](https://github.com/hinanohart/cvar-leximin-fair/actions/workflows/test.yml)

`cvar-leximin-fair` adds two small, focused group-fairness *reductions* to the
[`fairlearn`](https://github.com/fairlearn/fairlearn) ecosystem:

- **`CVaRReduction`** — minimizes the Conditional Value-at-Risk of group losses at quantile α.
- **`LeximinReduction`** — produces a lexicographic-minimum group-loss allocation
  (worst-off first, then second worst-off, …).

Both estimators are scikit-learn–shaped (`fit(X, y, sensitive_features=…)`,
`predict`, `predict_proba`) and slot directly into
`fairlearn.metrics.MetricFrame` for evaluation. The implementation follows the
cost-sensitive reductions strategy of [Agarwal et al. 2018](https://arxiv.org/abs/1803.02453).

## Why this exists

Most off-the-shelf fairness tools optimize for *parity* objectives
(demographic parity, equalized odds). When the question is instead
*"how do we protect the worst-off group?"* the natural objectives are CVaR
and leximin. This package is a deliberately small implementation of those
two objectives, intended to compose with — not replace — `fairlearn` and
`sklearn`.

## Install

```bash
# from source (works today)
pip install "git+https://github.com/hinanohart/cvar-leximin-fair@v0.1.0"

# from PyPI (after the v0.1.0 release is mirrored — see Releases page for status)
pip install cvar-leximin-fair          # core
pip install "cvar-leximin-fair[example]"   # adds notebook deps
pip install "cvar-leximin-fair[dev]"       # adds dev / test deps
```

> Note: PyPI publication is gated on a Trusted-Publisher setup against this
> repository (see `.github/workflows/release.yml`). Until that is configured,
> install from the git tag above.

## Quick start

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cvar_leximin import CVaRReduction
from cvar_leximin.fairlearn_compat import summary

# Synthetic example: replace X, y, sensitive with your own data.
rng = np.random.default_rng(0)
X, y = make_classification(n_samples=600, n_features=8, random_state=0)
sensitive = rng.choice(["A", "B"], size=len(y))

X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
    X, y, sensitive, test_size=0.3, stratify=sensitive, random_state=0,
)

clf = CVaRReduction(
    LogisticRegression(max_iter=500), alpha=0.5, max_iter=10,
).fit(X_tr, y_tr, sensitive_features=s_tr)

print(summary(y_te, clf.predict(X_te), s_te, alpha=0.5))
# example shape (numbers vary by seed and split):
# {'per_group_loss': {'A': 0.18, 'B': 0.21},
#  'cvar_alpha': 0.21,
#  'leximin_vector': [0.21, 0.18]}
```

A full reference notebook on UCI Adult is at
[`examples/adult_income.ipynb`](examples/adult_income.ipynb).

## API at a glance

| Symbol | What it does |
|---|---|
| `CVaRReduction(estimator, alpha=0.1, max_iter=20, eta=1.0, tol=1e-3)` | CVaR-α reduction; `alpha → 0` recovers strict maximin |
| `LeximinReduction(estimator, max_iter_per_level=8, eta=1.0, levels=None)` | Lex-order reduction; `levels=k` only pegs the worst `k` groups |
| `metrics.subgroup_cvar(y_true, y_pred, s, alpha)` | CVaR-α of per-group 0/1 losses |
| `metrics.leximin_vector(y_true, y_pred, s)` | Per-group losses sorted descending |
| `metrics.worst_quantile_gap(y_true, y_pred, s, alpha)` | Worst-tail minus best-tail loss |
| `metrics.bootstrap_ci(stat, y, p, s, n_boot=200)` | Group-stratified bootstrap CI |
| `fairlearn_compat.cvar_metricframe(…)` | Build a `fairlearn.metrics.MetricFrame` |
| `fairlearn_compat.leximin_compare(y, p_a, p_b, s)` | Lex-compare two predictors |

## Philosophical framing (please read)

The library is named for two algorithmic operators — CVaR and leximin — and
not for a single philosophical school. The subtitle invokes John Rawls
because the difference principle (*maximize the minimum*) is what these
operators most naturally implement. But we want to be explicit:

1. **Sen capability acknowledgement.** [Sen, "Equality of What?" (1979)](https://tannerlectures.utah.edu/_resources/documents/a-to-z/s/sen80.pdf)
   argues that Rawls's primary-goods framing is a form of *resource
   fetishism*: equalizing resources does not equalize *capability*. We
   equalize **loss**, which is a resource-proxy at best. A model that is
   CVaR- or leximin-fair on loss is **not** automatically capability-fair.
2. **Rawls is one lens, not the lens.** Difference-principle aggregation
   is one defensible normative target. Egalitarian, capability-based,
   utilitarian, prioritarian, and Walzer-style sphere-of-justice framings
   would all point at different metrics. We make no claim that ours is
   uniquely correct.
3. **No compliance certification.** This is an algorithmic tool, not legal
   advice. Whether use of CVaR or leximin satisfies a specific regulatory
   target — for example the risk-management obligations of EU AI Act
   Articles 9–17 (binding currently scheduled for 2026-08-02; subject to
   the Digital Omnibus deliberations that may shift the high-risk obligation
   timeline into 2027), or the bias-audit format of NYC Local Law 144 — is a
   determination for adopters and their counsel.

## Comparison with adjacent libraries

| Library | Primary objective | Relationship |
|---|---|---|
| `fairlearn` | Parity constraints (demographic parity, equalized odds) | `cvar-leximin-fair` complements; uses `MetricFrame` for evaluation |
| `AIF360` | Many parity metrics + reductions | Adjacent; CVaR / leximin not in core |
| `sqwash` / `superquantile` | Superquantile risk minimization | CVaR-style; ours adds the leximin variant and the sensitive-features API |
| `inFairness` | Individual fairness (Lipschitz) | Orthogonal objective |

## Roadmap

- **v0.1.x** (current): MVP reductions + metrics + notebook + paper skeleton.
  Estimators are **sklearn-shape**, not subclasses of `fairlearn.reductions.Moment`;
  evaluation interop is via `fairlearn.metrics.MetricFrame` (see `fairlearn_compat`).
- **v0.2.0**: (a) native `fairlearn.reductions.Moment` subclass so these
  reductions can be driven by `ExponentiatedGradient` / `GridSearch`; (b)
  optional `audit_schema/` module (LL144-shaped JSON output) — gated on
  regulatory clarity; deliberately deferred so a schema change does not break core.
- **v0.3.0**: companion `rawlsian-bench` benchmark dataset repo (separate Apache-2.0).
- **Not planned**: standalone "Rawlsian compliance certification" — out of scope.

## Development

```bash
git clone https://github.com/hinanohart/cvar-leximin-fair
cd cvar-leximin-fair
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,example]"
pytest
ruff check .
mypy src
```

CI runs the same checks on Python 3.10 / 3.11 / 3.12 and enforces an
`AGPL` license guard via `pip-licenses --fail-on=AGPL` so that no
copyleft-strong dependency creeps into the core wheel.

## Citation

If you use this in research:

```bibtex
@software{cvar_leximin_fair_2026,
  author  = {hinanohart},
  title   = {cvar-leximin-fair: Rawlsian fairness via CVaR / leximin reductions for fairlearn},
  year    = {2026},
  url     = {https://github.com/hinanohart/cvar-leximin-fair},
  version = {0.1.0}
}
```

The companion arXiv skeleton lives in [`paper/`](paper/).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party
acknowledgements: Agarwal et al. (2018) for the reductions strategy;
`fairlearn` (MIT) for the evaluation interop; the UCI Machine Learning
Repository for the Adult Income dataset used in the example notebook.
