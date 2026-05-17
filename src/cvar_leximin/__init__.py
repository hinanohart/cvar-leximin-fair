"""cvar-leximin-fair: Rawlsian fairness via CVaR / leximin reductions for fairlearn."""

from cvar_leximin.metrics import (
    leximin_vector,
    subgroup_cvar,
    worst_quantile_gap,
)
from cvar_leximin.reductions import (
    CVaRReduction,
    LeximinReduction,
)

__version__ = "0.1.0"

__all__ = [
    "CVaRReduction",
    "LeximinReduction",
    "leximin_vector",
    "subgroup_cvar",
    "worst_quantile_gap",
    "__version__",
]
