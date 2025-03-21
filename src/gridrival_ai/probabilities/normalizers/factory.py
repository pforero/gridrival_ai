"""
Factory for obtaining grid normalizers.

This module provides a factory function for creating grid normalizer instances
with a simple interface.
"""

from typing import Dict, Type

from gridrival_ai.probabilities.normalizers.base import GridNormalizer
from gridrival_ai.probabilities.normalizers.sinkhorn import SinkhornNormalizer

# Registry of available normalizers
_NORMALIZERS: Dict[str, Type[GridNormalizer]] = {
    "sinkhorn": SinkhornNormalizer,
    # Add more normalizers here as they are implemented
}


def get_grid_normalizer(method: str = "sinkhorn", **kwargs) -> GridNormalizer:
    """
    Get a grid normalizer instance for the specified method.

    Parameters
    ----------
    method : str, optional
        Normalization method name, by default "sinkhorn".
        Options: "sinkhorn"
    **kwargs
        Additional parameters for the normalizer constructor.

    Returns
    -------
    GridNormalizer
        Instance of the appropriate normalizer.

    Raises
    ------
    ValueError
        If method is not recognized.

    Examples
    --------
    >>> normalizer = get_grid_normalizer("sinkhorn")
    >>> normalized = normalizer.normalize(distributions)

    >>> # With custom parameters
    >>> normalizer = get_grid_normalizer("sinkhorn", max_iter=50, tolerance=1e-8)
    >>> normalized = normalizer.normalize(distributions)
    """
    if method not in _NORMALIZERS:
        valid_methods = ", ".join(_NORMALIZERS.keys())
        raise ValueError(
            f"Unknown normalization method: {method}. "
            f"Valid methods are: {valid_methods}"
        )

    normalizer_class = _NORMALIZERS[method]
    return normalizer_class(**kwargs)
