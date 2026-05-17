"""cvar-leximin-fair: Rawlsian fairness via CVaR / leximin reductions for fairlearn."""

from cvar_leximin.metrics import (
    bootstrap_ci,
    leximin_vector,
    subgroup_cvar,
    subgroup_losses,
    worst_quantile_gap,
)
from cvar_leximin.reductions import (
    CVaRReduction,
    LeximinReduction,
)

__version__ = "0.1.1"

__all__ = [
    "CVaRReduction",
    "LeximinReduction",
    "bootstrap_ci",
    "leximin_vector",
    "subgroup_cvar",
    "subgroup_losses",
    "worst_quantile_gap",
    "__version__",
]
