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

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    def get(
        self, entity_id: str, context: str, default: Optional[Distribution] = None
    ) -> Distribution:
        """
        Get distribution for an entity and context.

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
        Distribution
            Probability distribution for the entity and context.

        Raises
        ------
        KeyError
            If no distribution is found for the entity and context and no default is
            provided.

        Examples
        --------
        >>> registry = DistributionRegistry()
        >>> # Assume distribution has been registered
        >>> dist = registry.get("VER", "qualifying")
        >>> dist[1]  # Probability of P1
        0.6
        >>>
        >>> # With default value
        >>> dist = (
            registry
            .get("NOR", "qualifying", default=PositionDistribution({1: 0.1}))
        )
        >>> dist[1]
        0.1
        """
        try:
            return self.distributions[entity_id][context]
        except KeyError:
            if default is not None:
                return default
            raise KeyError(
                f"No distribution found for entity '{entity_id}' and context "
                f"'{context}'"
            )

    def get_joint(
        self,
        entity1_id: str,
        entity2_id: str,
        context: str,
        constrained: bool = True,
    ) -> JointDistribution:
        """
        Get joint distribution between two entities for a context.

        Creates a joint distribution from the individual distributions.

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
        # Get individual distributions
        dist1 = self.get(entity1_id, context)
        dist2 = self.get(entity2_id, context)

        # Create joint distribution based on constraint requirement
        if (
            constrained
            and isinstance(dist1, PositionDistribution)
            and isinstance(dist2, PositionDistribution)
        ):
            return create_constrained_joint(
                dist1, dist2, name1=entity1_id, name2=entity2_id
            )
        else:
            return create_independent_joint(
                dist1, dist2, name1=entity1_id, name2=entity2_id
            )

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
