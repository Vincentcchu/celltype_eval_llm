"""
Cell-type label standardization and evaluation package.

This package provides tools to standardize cell-type labels to the L3 taxonomy
and evaluate model predictions after standardization.
"""

from .standardize import standardize_h5ad_and_update_mapping
from .evaluate import evaluate_h5ad

__version__ = "1.0.0"

__all__ = [
    "standardize_h5ad_and_update_mapping",
    "evaluate_h5ad",
]
