"""
Grid creator base class for GridRival AI.

This module defines the abstract base class for grid creators,
which convert odds to session and race probability distributions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from gridrival_ai.probabilities.distributions import (
    RaceDistribution,
    SessionDistribution,
)
from gridrival_ai.probabilities.normalizers.base import GridNormalizer
from gridrival_ai.probabilities.normalizers.factory import get_grid_normalizer
from gridrival_ai.probabilities.odds_converters import BasicConverter, OddsConverter
from gridrival_ai.probabilities.odds_structure import OddsStructure


class GridCreator(ABC):
    """
    Abstract base class for grid probability creators.

    Grid creators take betting odds and produce probability distributions
    across drivers and positions, represented as SessionDistribution objects
    or full RaceDistribution objects containing multiple sessions.

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
    def create_session_distribution(
        self,
        odds_input: Union[OddsStructure, Dict],
        session_type: str = "race",
        **kwargs,
    ) -> SessionDistribution:
        """
        Create a session distribution from odds.

        This is the primary method that subclasses must implement. It creates
        a probability distribution for all drivers across all positions for a
        specific session type.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
            OddsStructure instance or raw odds dictionary
        session_type : str, optional
            Session type to use for odds, by default "race"
        **kwargs
            Additional parameters for distribution creation

        Returns
        -------
        SessionDistribution
            Session distribution for all drivers
        """
        pass

    @abstractmethod
    def create_race_distribution(
        self,
        odds_input: Union[OddsStructure, Dict],
        **kwargs,
    ) -> RaceDistribution:
        """
        Create a complete race distribution from odds.

        This is the second primary method that subclasses must implement. It creates
        distributions for race, qualifying, and sprint sessions and combines them
        into a RaceDistribution.

        Parameters
        ----------
        odds_input : Union[OddsStructure, Dict]
            OddsStructure instance or raw odds dictionary
        **kwargs
            Additional parameters for distribution creation

        Returns
        -------
        RaceDistribution
            Race distribution containing all sessions
        """
        pass
