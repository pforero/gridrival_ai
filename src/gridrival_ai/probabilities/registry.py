"""
Registry for probability distributions.

This module provides a central registry for storing and retrieving probability
distributions. It manages distributions for different entities (drivers, constructors)
and contexts (qualifying, race, sprint).

Classes
-------
DistributionRegistry
    Registry for managing probability distributions.

Examples
--------
>>> # Create a registry
>>> registry = DistributionRegistry()
>>>
>>> # Register a position distribution for a driver
>>> from gridrival_ai.probabilities.core import PositionDistribution
>>> dist = PositionDistribution({1: 0.6, 2: 0.4})
>>> registry.register("VER", "qualifying", dist)
>>>
>>> # Retrieve the distribution
>>> registry.get("VER", "qualifying")[1]
0.6
>>>
>>> # Create a joint distribution between two entities
>>> joint = registry.get_joint("VER", "HAM", "race")
>>> joint[(1, 1)]  # Probability VER P1, HAM P1 (should be 0 due to constraint)
0.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, cast

from gridrival_ai.probabilities.core import (
    Distribution,
    JointDistribution,
    PositionDistribution,
    create_constrained_joint,
    create_independent_joint,
)


@dataclass
class DistributionRegistry:
    """
    Registry for managing probability distributions.

    This class provides a central registry for storing and retrieving
    probability distributions for different entities and contexts.

    Attributes
    ----------
    distributions : Dict[str, Dict[str, Distribution]]
        Nested dictionary storing distributions by entity ID and context.
    joint_distributions : Dict[Tuple[str, str, str], JointDistribution]
        Dictionary storing joint distributions by entity pair and context.

    Examples
    --------
    >>> registry = DistributionRegistry()
    >>> from gridrival_ai.probabilities.core import PositionDistribution
    >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
    >>> registry.register("VER", "qualifying", dist)
    >>> registry.get("VER", "qualifying")[1]
    0.6
    """

    distributions: Dict[str, Dict[str, Distribution]] = field(default_factory=dict)
    joint_distributions: Dict[Tuple[str, str, str], JointDistribution] = field(
        default_factory=dict
    )

    def register(
        self, entity_id: str, context: str, distribution: Distribution
    ) -> None:
        """
        Register a distribution for an entity and context.

        Parameters
        ----------
        entity_id : str
            ID of the entity (e.g., driver or constructor code).
        context : str
            Context for the distribution (e.g., 'qualifying', 'race').
        distribution : Distribution
            Probability distribution to register.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> registry.register("VER", "qualifying", dist)
        """
        # Create dictionary for entity if it doesn't exist
        if entity_id not in self.distributions:
            self.distributions[entity_id] = {}

        # Register distribution
        self.distributions[entity_id][context] = distribution

        # Clear any cached joint distributions involving this entity and context
        self._clear_joint_cache(entity_id, context)

    def get(self, entity_id: str, context: str) -> Distribution:
        """
        Get distribution for an entity and context.

        Parameters
        ----------
        entity_id : str
            ID of the entity (e.g., driver or constructor code).
        context : str
            Context for the distribution (e.g., 'qualifying', 'race').

        Returns
        -------
        Distribution
            Probability distribution for the entity and context.

        Raises
        ------
        KeyError
            If no distribution is found for the entity and context.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # Assume distribution has been registered
        >>> dist = registry.get("VER", "qualifying")
        >>> dist[1]  # Probability of P1
        0.6
        """
        try:
            return self.distributions[entity_id][context]
        except KeyError:
            raise KeyError(
                f"No distribution found for entity '{entity_id}' and context "
                f"'{context}'"
            )

    def get_or_default(
        self, entity_id: str, context: str, default: Optional[Distribution] = None
    ) -> Optional[Distribution]:
        """
        Get distribution for an entity and context or return default.

        Parameters
        ----------
        entity_id : str
            ID of the entity (e.g., driver or constructor code).
        context : str
            Context for the distribution (e.g., 'qualifying', 'race').
        default : Optional[Distribution], optional
            Default distribution to return if not found, by default None.

        Returns
        -------
        Optional[Distribution]
            Probability distribution for the entity and context or default.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # No distribution has been registered
        >>> dist = registry.get_or_default("VER", "qualifying")
        >>> dist is None
        True
        """
        try:
            return self.get(entity_id, context)
        except KeyError:
            return default

    def get_joint(
        self,
        entity1_id: str,
        entity2_id: str,
        context: str,
        constrained: bool = True,
    ) -> JointDistribution:
        """
        Get joint distribution between two entities for a context.

        If the joint distribution does not exist, it is created from the
        individual distributions assuming independence.

        Parameters
        ----------
        entity1_id : str
            ID of the first entity.
        entity2_id : str
            ID of the second entity.
        context : str
            Context for the joint distribution.
        constrained : bool, optional
            Whether entities cannot have the same outcome, by default True.
            This is typically True for race positions where two drivers
            cannot finish in the same position.

        Returns
        -------
        JointDistribution
            Joint probability distribution.

        Raises
        ------
        KeyError
            If either entity does not have a distribution for the context.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # Assume distributions have been registered
        >>> joint = registry.get_joint("VER", "HAM", "race")
        >>> joint[(1, 2)]  # Probability VER P1, HAM P2
        0.4
        """
        # Standardize entity order to ensure consistent caching
        if entity1_id > entity2_id:
            entity1_id, entity2_id = entity2_id, entity1_id

        # Create key for cache
        cache_key = (entity1_id, entity2_id, context)

        # Check if joint distribution is already cached
        if cache_key in self.joint_distributions:
            return self.joint_distributions[cache_key]

        # Get individual distributions
        dist1 = self.get(entity1_id, context)
        dist2 = self.get(entity2_id, context)

        # Convert to position distributions (required for constraints)
        if constrained and not (
            isinstance(dist1, PositionDistribution)
            and isinstance(dist2, PositionDistribution)
        ):
            warnings.warn(
                "Constrained joint distributions only supported for "
                "PositionDistribution. Falling back to independent joint distribution."
            )
            constrained = False

        # Create joint distribution
        if constrained:
            joint = create_constrained_joint(
                cast(PositionDistribution, dist1),
                cast(PositionDistribution, dist2),
                name1=entity1_id,
                name2=entity2_id,
            )
        else:
            joint = create_independent_joint(
                dist1, dist2, name1=entity1_id, name2=entity2_id
            )

        # Cache the joint distribution
        self.joint_distributions[cache_key] = joint

        return joint

    def has(self, entity_id: str, context: str) -> bool:
        """
        Check if a distribution exists for an entity and context.

        Parameters
        ----------
        entity_id : str
            ID of the entity (e.g., driver or constructor code).
        context : str
            Context for the distribution (e.g., 'qualifying', 'race').

        Returns
        -------
        bool
            True if a distribution exists, False otherwise.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> registry.has("VER", "qualifying")
        False
        >>> # Register a distribution
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> registry.register("VER", "qualifying", dist)
        >>> registry.has("VER", "qualifying")
        True
        """
        if entity_id not in self.distributions:
            return False
        return context in self.distributions[entity_id]

    def get_entities(self, context: Optional[str] = None) -> List[str]:
        """
        Get all entities that have distributions.

        Parameters
        ----------
        context : Optional[str], optional
            If provided, only return entities with distributions for this context.

        Returns
        -------
        List[str]
            List of entity IDs.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # Register distributions
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> registry.register("VER", "qualifying", dist)
        >>> registry.register("HAM", "race", dist)
        >>> registry.get_entities()
        ['VER', 'HAM']
        >>> registry.get_entities("qualifying")
        ['VER']
        """
        if context is None:
            return list(self.distributions.keys())
        return [
            entity_id
            for entity_id, contexts in self.distributions.items()
            if context in contexts
        ]

    def get_contexts(self, entity_id: Optional[str] = None) -> List[str]:
        """
        Get all contexts that have distributions.

        Parameters
        ----------
        entity_id : Optional[str], optional
            If provided, only return contexts for this entity.

        Returns
        -------
        List[str]
            List of contexts.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # Register distributions
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> registry.register("VER", "qualifying", dist)
        >>> registry.register("VER", "race", dist)
        >>> registry.register("HAM", "race", dist)
        >>> registry.get_contexts()
        ['qualifying', 'race']
        >>> registry.get_contexts("VER")
        ['qualifying', 'race']
        """
        if entity_id is None:
            # Get all unique contexts across all entities
            contexts = set()
            for contexts_dict in self.distributions.values():
                contexts.update(contexts_dict.keys())
            return sorted(contexts)

        if entity_id not in self.distributions:
            return []

        return sorted(self.distributions[entity_id].keys())

    def clear(self) -> None:
        """
        Clear all distributions from the registry.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # Register distributions
        >>> from gridrival_ai.probabilities.core import PositionDistribution
        >>> dist = PositionDistribution({1: 0.6, 2: 0.4})
        >>> registry.register("VER", "qualifying", dist)
        >>> registry.clear()
        >>> registry.has("VER", "qualifying")
        False
        """
        self.distributions.clear()
        self.joint_distributions.clear()

    def _clear_joint_cache(self, entity_id: str, context: str) -> None:
        """
        Clear joint distribution cache for an entity and context.

        Parameters
        ----------
        entity_id : str
            ID of the entity.
        context : str
            Context for the distribution.
        """
        # Remove joint distributions involving this entity and context
        keys_to_remove = [
            key
            for key in self.joint_distributions
            if (key[0] == entity_id or key[1] == entity_id) and key[2] == context
        ]
        for key in keys_to_remove:
            del self.joint_distributions[key]
