# Contributing

Thanks for considering a contribution. This is a small project with a
deliberately small scope (CVaR + leximin reductions, plus the metrics and
fairlearn interop they need). PRs that stay within that scope land easily;
PRs that grow it are welcome but please open an issue first to discuss fit.

## Quick setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,example]"
pytest
ruff check .
mypy src
```

## What we look for

- **Tests.** New behavior needs a test. Reductions tests should be
  deterministic (seeded `numpy.random.default_rng`) and small enough to
  run in under a few seconds.
- **Bootstrap-CI where it matters.** When you assert a numerical property
  on group-level losses, prefer `cvar_leximin.metrics.bootstrap_ci` over
  a raw point estimate.
- **No silent broadening of dependencies.** Anything new in `dependencies`
  needs justification in the PR description; prefer `optional-dependencies`.
- **License hygiene.** Strong-copyleft dependencies (AGPL / SSPL / etc.)
  are out of scope — they would force the whole package off Apache-2.0.
  CI enforces this via `pip-licenses --fail-on=AGPL`.

## What we won't merge

- "Rawlsian compliance certifier" features. The library is an algorithm,
  not a compliance product; see the README "philosophical framing" section.
- Vendored copies of `fairlearn` or `sklearn` internals. Use them as
  dependencies.
- Hard-coded regulatory schemas. These are deferred to a future `v0.2`
  optional module so that regulatory churn does not break core.

## Reporting issues

Please include:

1. Python version and OS,
2. Minimal reproducer (synthetic data is fine — `numpy.random.default_rng`
   is encouraged),
3. Expected vs actual behavior.

By contributing you agree your contribution is licensed under Apache-2.0.
