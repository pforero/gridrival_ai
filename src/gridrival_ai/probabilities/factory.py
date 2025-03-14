"""
Factory for creating probability distributions.

This module provides factory methods for creating probability distributions
from various data sources, including dictionary-format betting odds.
It handles the creation, validation, and customization of distributions.

Classes
-------
DistributionFactory
    Factory for creating probability distributions.
DistributionBuilder
    Builder for creating customized probability distributions.

Examples
--------
>>> # Create position distributions from dictionary odds
>>> from gridrival_ai.probabilities.factory import DistributionFactory
>>> odds_dict = {"VER": 1.5, "HAM": 3.0, "NOR": 6.0}
>>> dists = DistributionFactory.from_odds_dict(odds_dict)
>>> dists["VER"][1]  # Probability of VER finishing P1
0.5714285714285714

>>> # Use builder pattern for more complex cases
>>> from gridrival_ai.probabilities.factory import DistributionBuilder
>>> dist = (DistributionBuilder()
...         .for_entity("VER")
...         .in_context("qualifying")
...         .from_odds_dict({"VER": 1.5, "HAM": 3.0})
...         .using_method("shin")
...         .with_smoothing(0.1)
...         .build())
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from gridrival_ai.probabilities.conversion import (
    odds_to_distributions,
    odds_to_grid,
    odds_to_position_distribution,
)
from gridrival_ai.probabilities.core import (
    Distribution,
    JointDistribution,
    PositionDistribution,
    create_conditional_joint,
    create_constrained_joint,
    create_independent_joint,
)


class DistributionFactory:
    """
    Factory for creating probability distributions.

    This class provides static methods for creating probability distributions
    from various data sources, including dictionary-format betting odds.

    Examples
    --------
    >>> # Create position distribution from odds
    >>> odds = [1.5, 3.0, 6.0]
    >>> dist = DistributionFactory.from_odds(odds)
    >>> dist[1]  # Probability of P1
    0.5714285714285714

    >>> # Create position distributions from dictionary odds
    >>> odds_dict = {"VER": 1.5, "HAM": 3.0, "NOR": 6.0}
    >>> dists = DistributionFactory.from_odds_dict(odds_dict)
    >>> dists["VER"][1]  # Probability of VER finishing P1
    0.5714285714285714
    """

    @staticmethod
    def from_odds(
        odds: List[float], method: str = "basic", target_sum: float = 1.0, **kwargs
    ) -> PositionDistribution:
        """
        Create position distribution from odds.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.
        method : str, optional
            Conversion method name, by default "basic".
            Options: "basic", "odds_ratio", "shin", "power".
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
        **kwargs
            Additional parameters for the converter.

        Returns
        -------
        PositionDistribution
            Position distribution.

        Examples
        --------
        >>> odds = [1.5, 3.0, 6.0]
        >>> dist = DistributionFactory.from_odds(odds)
        >>> dist[1]  # Probability of P1
        0.5714285714285714
        """
        return odds_to_position_distribution(odds, method, target_sum, **kwargs)

    @staticmethod
    def from_odds_dict(
        odds_dict: Dict[str, float],
        method: str = "basic",
        target_sum: float = 1.0,
        **kwargs,
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions from dictionary of driver odds.

        Parameters
        ----------
        odds_dict : Dict[str, float]
            Dictionary mapping driver IDs to their decimal odds. Must be > 1.0.
        method : str, optional
            Conversion method name, by default "basic".
            Options: "basic", "odds_ratio", "shin", "power", "harville".
        target_sum : float, optional
            Target sum for probabilities, by default 1.0.
        **kwargs
            Additional parameters for the converter.

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to their position distributions.

        Examples
        --------
        >>> odds_dict = {"VER": 1.5, "HAM": 3.0, "NOR": 6.0}
        >>> dists = DistributionFactory.from_odds_dict(odds_dict)
        >>> dists["VER"][1]  # Probability of VER finishing P1
        0.5714285714285714
        """
        # Extract driver IDs and odds
        driver_ids = list(odds_dict.keys())
        odds_values = [odds_dict[driver_id] for driver_id in driver_ids]

        # For Harville method, use the specialized grid function
        if method == "harville":
            return odds_to_distributions(odds_values, driver_ids, **kwargs)

        # For other methods, create distributions for each driver based on odds
        grid = odds_to_grid(odds_values, driver_ids, **kwargs)

        # Convert grid to individual position distributions
        return {
            driver_id: PositionDistribution(positions)
            for driver_id, positions in grid.items()
        }

    @staticmethod
    def from_probabilities(
        probabilities: Dict[int, float], validate: bool = True
    ) -> PositionDistribution:
        """
        Create position distribution from probability dictionary.

        Parameters
        ----------
        probabilities : Dict[int, float]
            Dictionary mapping positions to probabilities.
        validate : bool, optional
            Whether to validate the distribution, by default True.

        Returns
        -------
        PositionDistribution
            Position distribution.

        Examples
        --------
        >>> probs = {1: 0.6, 2: 0.4}
        >>> dist = DistributionFactory.from_probabilities(probs)
        >>> dist[1]  # Probability of P1
        0.6
        """
        return PositionDistribution(probabilities, _validate=validate)

    @staticmethod
    def from_probabilities_dict(
        probs_dict: Dict[str, Dict[int, float]], validate: bool = True
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions from nested dictionary of probabilities.

        Parameters
        ----------
        probs_dict : Dict[str, Dict[int, float]]
            Dictionary mapping driver IDs to position probability dictionaries.
        validate : bool, optional
            Whether to validate distributions, by default True.

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions.

        Examples
        --------
        >>> probs_dict = {
        ...     "VER": {1: 0.6, 2: 0.4},
        ...     "HAM": {1: 0.3, 2: 0.7}
        ... }
        >>> dists = DistributionFactory.from_probabilities_dict(probs_dict)
        >>> dists["VER"][1]  # Probability of VER finishing P1
        0.6
        """
        return {
            driver_id: PositionDistribution(probs, _validate=validate)
            for driver_id, probs in probs_dict.items()
        }

    @staticmethod
    def from_json(
        json_str: str, validate: bool = True
    ) -> Union[PositionDistribution, JointDistribution]:
        """
        Create distribution from JSON string.

        Parameters
        ----------
        json_str : str
            JSON string representation of the distribution.
        validate : bool, optional
            Whether to validate the distribution, by default True.

        Returns
        -------
        Union[PositionDistribution, JointDistribution]
            Probability distribution.

        Raises
        ------
        ValueError
            If JSON format is invalid.

        Examples
        --------
        >>> json_str = '{"type": "position", "probabilities": {"1": 0.6, "2": 0.4}}'
        >>> dist = DistributionFactory.from_json(json_str)
        >>> dist[1]  # Probability of P1
        0.6
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if "type" not in data:
            raise ValueError("Missing 'type' field in JSON")

        if data["type"] == "position":
            if "probabilities" not in data:
                raise ValueError("Missing 'probabilities' field in JSON")

            # Convert string keys to integers
            probs = {int(k): float(v) for k, v in data["probabilities"].items()}
            return PositionDistribution(probs, _validate=validate)
        elif data["type"] == "joint":
            if "probabilities" not in data:
                raise ValueError("Missing 'probabilities' field in JSON")

            # Extract outcome names
            outcome1_name = data.get("outcome1_name", "var1")
            outcome2_name = data.get("outcome2_name", "var2")

            # Convert string tuple keys to actual tuples and values to floats
            joint_probs = {}
            for k, v in data["probabilities"].items():
                # Parse tuple key from string like "(1, 2)"
                k = k.strip("()[]").replace(" ", "")
                vals = tuple(map(int, k.split(",")))
                joint_probs[vals] = float(v)

            return JointDistribution(
                joint_probs,
                outcome1_name=outcome1_name,
                outcome2_name=outcome2_name,
                _validate=validate,
            )
        elif data["type"] == "odds_dict":
            if "odds" not in data:
                raise ValueError("Missing 'odds' field in JSON")

            # Create distributions from odds dictionary
            method = data.get("method", "basic")
            return DistributionFactory.from_odds_dict(data["odds"], method=method)
        else:
            raise ValueError(f"Unknown distribution type: {data['type']}")

    @staticmethod
    def from_file(
        file_path: Union[str, Path], validate: bool = True
    ) -> Union[
        PositionDistribution, JointDistribution, Dict[str, PositionDistribution]
    ]:
        """
        Create distribution from JSON file.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to JSON file.
        validate : bool, optional
            Whether to validate the distribution, by default True.

        Returns
        -------
        Union[PositionDistribution, JointDistribution, Dict[str, PositionDistribution]]
            Probability distribution or dictionary of distributions.

        Raises
        ------
        ValueError
            If file format is invalid.
        FileNotFoundError
            If file does not exist.

        Examples
        --------
        >>> dist = DistributionFactory.from_file("path/to/distribution.json")
        >>> dist[1]  # Probability of P1
        0.6
        """
        with open(file_path, "r") as f:
            json_str = f.read()
        return DistributionFactory.from_json(json_str, validate)

    @staticmethod
    def from_dict(
        data: Dict[str, Any], validate: bool = True
    ) -> Union[Distribution, Dict[str, Distribution]]:
        """
        Create distribution from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary representation of the distribution.
        validate : bool, optional
            Whether to validate the distribution, by default True.

        Returns
        -------
        Union[Distribution, Dict[str, Distribution]]
            Probability distribution or dictionary of distributions.

        Raises
        ------
        ValueError
            If dictionary format is invalid.

        Examples
        --------
        >>> data = {"type": "position", "probabilities": {1: 0.6, 2: 0.4}}
        >>> dist = DistributionFactory.from_dict(data)
        >>> dist[1]  # Probability of P1
        0.6

        >>> # Dictionary of odds
        >>> data = {"type": "odds_dict", "odds": {"VER": 1.5, "HAM": 3.0}}
        >>> dists = DistributionFactory.from_dict(data)
        >>> dists["VER"][1]  # Probability of VER finishing P1
        0.6666666666666666
        """
        if "type" not in data:
            raise ValueError("Missing 'type' field in dictionary")

        if data["type"] == "position":
            if "probabilities" not in data:
                raise ValueError("Missing 'probabilities' field in dictionary")
            return PositionDistribution(data["probabilities"], _validate=validate)
        elif data["type"] == "joint":
            if "probabilities" not in data:
                raise ValueError("Missing 'probabilities' field in dictionary")
            outcome1_name = data.get("outcome1_name", "var1")
            outcome2_name = data.get("outcome2_name", "var2")
            return JointDistribution(
                data["probabilities"],
                outcome1_name=outcome1_name,
                outcome2_name=outcome2_name,
                _validate=validate,
            )
        elif data["type"] == "odds_dict":
            if "odds" not in data:
                raise ValueError("Missing 'odds' field in dictionary")
            method = data.get("method", "basic")
            return DistributionFactory.from_odds_dict(
                data["odds"], method=method, validate=validate
            )
        elif data["type"] == "probabilities_dict":
            if "probabilities" not in data:
                raise ValueError("Missing 'probabilities' field in dictionary")
            return DistributionFactory.from_probabilities_dict(
                data["probabilities"], validate=validate
            )
        else:
            raise ValueError(f"Unknown distribution type: {data['type']}")

    @staticmethod
    def grid_from_odds(
        odds: List[float], driver_ids: List[str], **kwargs
    ) -> Dict[str, PositionDistribution]:
        """
        Create position distributions for multiple drivers from grid odds.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds for win market. Must be > 1.0.
        driver_ids : List[str]
            List of driver identifiers.
        **kwargs
            Additional parameters for Harville converter.

        Returns
        -------
        Dict[str, PositionDistribution]
            Dictionary mapping driver IDs to position distributions.

        Examples
        --------
        >>> odds = [1.5, 3.0]
        >>> driver_ids = ["VER", "HAM"]
        >>> dists = DistributionFactory.grid_from_odds(odds, driver_ids)
        >>> dists["VER"][1]  # Probability VER finishes P1
        0.6666666666666667
        """
        return odds_to_distributions(odds, driver_ids, **kwargs)

    @staticmethod
    def joint_independent(
        dist1: Distribution,
        dist2: Distribution,
        name1: str = "var1",
        name2: str = "var2",
    ) -> JointDistribution:
        """
        Create joint distribution assuming independence.

        Parameters
        ----------
        dist1 : Distribution
            First marginal distribution.
        dist2 : Distribution
            Second marginal distribution.
        name1 : str, optional
            Name of first variable, by default "var1".
        name2 : str, optional
            Name of second variable, by default "var2".

        Returns
        -------
        JointDistribution
            Joint distribution with P(x,y) = P(x)P(y).

        Examples
        --------
        >>> dist1 = DistributionFactory.from_probabilities({1: 0.6, 2: 0.4})
        >>> dist2 = DistributionFactory.from_probabilities({1: 0.3, 2: 0.7})
        >>> joint = DistributionFactory.joint_independent(dist1, dist2)
        >>> joint[(1, 1)]  # P(1,1) = P(1)P(1) = 0.6*0.3
        0.18
        """
        return create_independent_joint(dist1, dist2, name1=name1, name2=name2)

    @staticmethod
    def joint_constrained(
        dist1: PositionDistribution,
        dist2: PositionDistribution,
        name1: str = "var1",
        name2: str = "var2",
    ) -> JointDistribution:
        """
        Create joint distribution with constraint that outcomes cannot be equal.

        Parameters
        ----------
        dist1 : PositionDistribution
            First marginal position distribution.
        dist2 : PositionDistribution
            Second marginal position distribution.
        name1 : str, optional
            Name of first variable, by default "var1".
        name2 : str, optional
            Name of second variable, by default "var2".

        Returns
        -------
        JointDistribution
            Constrained joint distribution.

        Examples
        --------
        >>> dist1 = DistributionFactory.from_probabilities({1: 0.6, 2: 0.4})
        >>> dist2 = DistributionFactory.from_probabilities({1: 0.3, 2: 0.7})
        >>> joint = DistributionFactory.joint_constrained(dist1, dist2)
        >>> joint[(1, 1)]  # Cannot both finish P1
        0.0
        """
        return create_constrained_joint(dist1, dist2, name1=name1, name2=name2)

    @staticmethod
    def joint_conditional(
        marginal: Distribution,
        conditional_func: Callable[[Any], Distribution],
        name1: str = "var1",
        name2: str = "var2",
    ) -> JointDistribution:
        """
        Create joint distribution from marginal and conditional.

        Parameters
        ----------
        marginal : Distribution
            Marginal distribution for first variable.
        conditional_func : Callable[[Any], Distribution]
            Function that takes a value of the first variable and returns
            the conditional distribution for the second variable.
        name1 : str, optional
            Name of first variable, by default "var1".
        name2 : str, optional
            Name of second variable, by default "var2".

        Returns
        -------
        JointDistribution
            Joint distribution with P(x,y) = P(x)P(y|x).

        Examples
        --------
        >>> marginal = DistributionFactory.from_probabilities({1: 0.6, 2: 0.4})
        >>> def get_conditional(x):
        ...     if x == 1:
        ...         return DistributionFactory.from_probabilities({1: 0.2, 2: 0.8})
        ...     else:  # x == 2
        ...         return DistributionFactory.from_probabilities({1: 0.7, 2: 0.3})
        >>> joint = DistributionFactory.joint_conditional(marginal, get_conditional)
        >>> joint[(1, 1)]  # P(1,1) = P(1)P(1|1) = 0.6*0.2
        0.12
        """
        return create_conditional_joint(
            marginal, conditional_func, name1=name1, name2=name2
        )

    @staticmethod
    def builder() -> DistributionBuilder:
        """
        Get a builder for creating distributions.

        Returns
        -------
        DistributionBuilder
            Builder for creating distributions.

        Examples
        --------
        >>> builder = DistributionFactory.builder()
        >>> dist = (builder
        ...         .for_entity("VER")
        ...         .in_context("qualifying")
        ...         .from_odds([1.5, 3.0, 6.0])
        ...         .with_smoothing(0.1)
        ...         .build())
        >>> dist[1]  # Probability of P1
        0.5142857142857142
        """
        return DistributionBuilder()


@dataclass
class DistributionBuilder:
    """
    Builder for creating customized probability distributions.

    This class implements the builder pattern for creating distributions
    with various customizations. It allows for fluent chaining of methods.

    Attributes
    ----------
    _entity_id : Optional[str]
        ID of the entity the distribution is for.
    _context : Optional[str]
        Context the distribution applies to.
    _distribution : Optional[Distribution]
        The distribution being built.
    _odds : Optional[List[float]]
        Odds to convert to probabilities.
    _odds_dict : Optional[Dict[str, float]]
        Dictionary mapping entities to odds.
    _odds_method : str
        Method for converting odds to probabilities.
    _probabilities : Optional[Dict[int, float]]
        Explicit probabilities for positions.
    _smoothing : Optional[float]
        Smoothing parameter for the distribution.

    Examples
    --------
    >>> builder = DistributionBuilder()
    >>> dist = (builder
    ...         .for_entity("VER")
    ...         .in_context("qualifying")
    ...         .from_odds_dict({"VER": 1.5, "HAM": 3.0})
    ...         .with_smoothing(0.1)
    ...         .build())
    >>> dist[1]  # Probability of P1
    0.5142857142857142
    """

    _entity_id: Optional[str] = None
    _context: Optional[str] = None
    _distribution: Optional[Distribution] = None
    _odds: Optional[List[float]] = None
    _odds_dict: Optional[Dict[str, float]] = None
    _odds_method: str = "basic"
    _probabilities: Optional[Dict[int, float]] = None
    _smoothing: Optional[float] = None

    def for_entity(self, entity_id: str) -> DistributionBuilder:
        """
        Set the entity ID for the distribution.

        Parameters
        ----------
        entity_id : str
            ID of the entity (e.g., driver or constructor code).

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.for_entity("VER")
        """
        self._entity_id = entity_id
        return self

    def in_context(self, context: str) -> DistributionBuilder:
        """
        Set the context for the distribution.

        Parameters
        ----------
        context : str
            Context for the distribution (e.g., 'qualifying', 'race').

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.in_context("qualifying")
        """
        self._context = context
        return self

    def from_odds(self, odds: List[float]) -> DistributionBuilder:
        """
        Set odds to convert to probabilities.

        Parameters
        ----------
        odds : List[float]
            List of decimal odds. Must be > 1.0.

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.from_odds([1.5, 3.0, 6.0])
        """
        self._odds = odds
        self._odds_dict = None  # Clear any existing odds dictionary
        self._probabilities = None  # Clear any existing probabilities
        self._distribution = None  # Clear any existing distribution
        return self

    def from_odds_dict(self, odds_dict: Dict[str, float]) -> DistributionBuilder:
        """
        Set odds dictionary to convert to probabilities.

        Parameters
        ----------
        odds_dict : Dict[str, float]
            Dictionary mapping entity IDs to decimal odds. Must be > 1.0.

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.from_odds_dict({"VER": 1.5, "HAM": 3.0})
        """
        self._odds_dict = odds_dict
        self._odds = None  # Clear any existing odds list
        self._probabilities = None  # Clear any existing probabilities
        self._distribution = None  # Clear any existing distribution
        return self

    def using_method(self, method: str) -> DistributionBuilder:
        """
        Set method for converting odds to probabilities.

        Parameters
        ----------
        method : str
            Conversion method name.
            Options: "basic", "odds_ratio", "shin", "power", "harville".

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.from_odds([1.5, 3.0, 6.0]).using_method("shin")
        """
        self._odds_method = method
        return self

    def from_probabilities(
        self, probabilities: Dict[int, float]
    ) -> DistributionBuilder:
        """
        Set explicit probabilities for positions.

        Parameters
        ----------
        probabilities : Dict[int, float]
            Dictionary mapping positions to probabilities.

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.from_probabilities({1: 0.6, 2: 0.4})
        """
        self._probabilities = probabilities
        self._odds = None  # Clear any existing odds
        self._odds_dict = None  # Clear any existing odds dictionary
        self._distribution = None  # Clear any existing distribution
        return self

    def from_distribution(self, distribution: Distribution) -> DistributionBuilder:
        """
        Set an existing distribution.

        Parameters
        ----------
        distribution : Distribution
            Existing probability distribution.

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> builder = DistributionBuilder()
        >>> builder = builder.from_distribution(dist)
        """
        self._distribution = distribution
        self._odds = None  # Clear any existing odds
        self._odds_dict = None  # Clear any existing odds dictionary
        self._probabilities = None  # Clear any existing probabilities
        return self

    def with_smoothing(self, alpha: float) -> DistributionBuilder:
        """
        Set smoothing parameter for the distribution.

        Parameters
        ----------
        alpha : float
            Smoothing parameter between 0 and 1.
            Higher values create more uniform distributions.

        Returns
        -------
        DistributionBuilder
            The builder instance for chaining.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> builder = builder.from_odds([1.5, 3.0, 6.0]).with_smoothing(0.1)
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Smoothing parameter must be between 0 and 1")
        self._smoothing = alpha
        return self

    def build(self) -> Distribution:
        """
        Build the distribution.

        Returns
        -------
        Distribution
            The built probability distribution.

        Raises
        ------
        ValueError
            If no data source is provided or entity ID is missing when using odds+
            dictionary.

        Examples
        --------
        >>> builder = DistributionBuilder()
        >>> dist = (builder
        ...         .from_odds([1.5, 3.0, 6.0])
        ...         .with_smoothing(0.1)
        ...         .build())
        >>> dist[1]  # Probability of P1
        0.5142857142857142
        """
        # If we already have a distribution, use it
        if self._distribution is not None:
            dist = self._distribution
        # For odds dictionary, we need to extract the specific entity's distribution
        elif self._odds_dict is not None:
            if self._entity_id is None:
                raise ValueError(
                    "Entity ID must be specified when using odds dictionary. "
                    "Use .for_entity() method."
                )

            # Create distributions for all entities
            dists = DistributionFactory.from_odds_dict(
                self._odds_dict, method=self._odds_method
            )

            # Get distribution for the specified entity
            if self._entity_id not in dists:
                raise ValueError(
                    f"Entity ID '{self._entity_id}' not found in odds dictionary"
                )

            dist = dists[self._entity_id]
        # Otherwise, create from odds or probabilities
        elif self._odds is not None:
            dist = DistributionFactory.from_odds(self._odds, method=self._odds_method)
        elif self._probabilities is not None:
            dist = DistributionFactory.from_probabilities(self._probabilities)
        else:
            raise ValueError("No data source provided for building distribution")

        # Apply smoothing if requested (only for position distributions)
        if self._smoothing is not None and isinstance(dist, PositionDistribution):
            dist = dist.smooth(self._smoothing)

        return dist
