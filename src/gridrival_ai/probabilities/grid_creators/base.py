"""
Grid creator base class for GridRival AI.

This module defines the abstract base class for grid creators,
which convert odds to position probability distributions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from gridrival_ai.probabilities.distributions import PositionDistribution
from gridrival_ai.probabilities.normalizers.base import GridNormalizer
from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer
from gridrival_ai.probabilities.odds_converters import BasicConverter, OddsConverter
from gridrival_ai.probabilities.odds_structure import OddsStructure


class GridCreator(ABC):
    """
    Abstract base class for grid distribution creators.

    Grid creators take betting odds and produce probability distributions
    of drivers finishing in each position. This class defines the common
    interface for all grid creators.

    Parameters
    ----------
    odds_converter : Optional[OddsConverter], optional
        Converter for initial odds-to-probabilities conversion,
        by default BasicConverter()
    grid_normalizer : Optional[GridNormalizer], optional
        Normalizer for ensuring grid constraints are satisfied,
        by default None (uses Sinkhorn normalizer)

    Attributes
    ----------
    odds_converter : OddsConverter
        Converter for initial odds-to-probabilities conversion
    grid_normalizer : GridNormalizer
        Normalizer for ensuring grid constraints are satisfied
    """

    def __init__(
        self,
        odds_converter: Optional[OddsConverter] = None,
        grid_normalizer: Optional[GridNormalizer] = None,
    ):
        """
        Initialize with an odds converter and grid normalizer.

        Parameters
        ----------
        odds_converter : Optional[OddsConverter], optional
            Converter for initial odds-to-probabilities conversion,
            by default BasicConverter()
        grid_normalizer : Optional[GridNormalizer], optional
            Normalizer for ensuring grid constraints are satisfied,
            by default None (uses Sinkhorn normalizer)
        """
        self.odds_converter = odds_converter or BasicConverter()
        self.grid_normalizer = grid_normalizer or get_grid_normalizer()

    def _ensure_odds_structure(
        self, odds_input: Union[OddsStructure, Dict]
    ) -> OddsStructure:
        """
        Ensure input is an OddsStructure instance.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
            OddsStructure instance or raw odds dictionary

        Returns
        -------
        OddsStructure
            Validated odds structure
        """
        if isinstance(odds_input, OddsStructure):
            return odds_input
        return OddsStructure(odds_input)

    @abstractmethod
    def create_position_distributions(
        self, odds_input: Union[OddsStructure, Dict], session: str = "race", **kwargs
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions from odds.

        This is the primary method that subclasses should implement. It creates
        probability distributions for each driver's finishing positions.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
            OddsStructure instance or raw odds dictionary
        session : str, optional
            Session to use for odds, by default "race"
        **kwargs
            Additional parameters for distribution creation

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions
        """
        pass

    def create_grid(
        self, odds_input: Union[OddsStructure, Dict], session: str = "race", **kwargs
    ) -> Dict[str, Dict[int, float]]:
        """
        Create a grid of position probabilities from odds.

        This method is provided for backwards compatibility. It calls
        create_position_distributions and converts the result to a grid.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
            OddsStructure instance or raw odds dictionary
        session : str, optional
            Session to use for odds, by default "race"
        **kwargs
            Additional parameters for grid creation

        Returns
        -------
        Dict[str, Dict[int, float]]
            Nested dictionary mapping drivers to position probabilities
            {driver_id: {position: probability, ...}, ...}
        """
        # Call create_position_distributions and convert to grid
        distributions = self.create_position_distributions(
            odds_input, session, **kwargs
        )
        return {driver_id: dist.to_dict() for driver_id, dist in distributions.items()}

    def convert_win_odds_to_probabilities(
        self, odds_structure: OddsStructure, session: str = "race"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert win odds to win probabilities using the odds converter.

        Parameters
        ----------
        odds_structure : OddsStructure
            Validated odds structure
        session : str, optional
            Session to use for odds, by default "race"

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Tuple of (win_probabilities, driver_ids)
        """
        win_odds, driver_ids = odds_structure.get_win_odds_list(session)
        win_probs = self.odds_converter.convert(win_odds, target_sum=1.0)

        return win_probs, driver_ids

    def _normalize_grid_distributions(
        self, distributions: Dict[str, PositionDistribution]
    ) -> Dict[str, PositionDistribution]:
        """
        Normalize distributions to ensure grid constraints are met.

        Uses the configured grid_normalizer to ensure:
        - Each driver's probabilities sum to 1.0 (row constraint)
        - Each position's probabilities sum to 1.0 across all drivers (column constraint)

        Parameters
        ----------
        distributions : Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions

        Returns
        -------
        Dict[str, PositionDistribution]
            Normalized distributions
        """
        return self.grid_normalizer.normalize(distributions)
