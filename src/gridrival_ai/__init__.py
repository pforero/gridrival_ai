"""
GridRival AI - F1 Fantasy League Optimizer
========================================

A Python library for optimizing F1 fantasy teams in the GridRival "Contracts" format.

Main Components
-------------
- Team optimization
- Contract management
- Scoring system
- Salary management
"""

__version__ = "0.1.0"
__author__ = "Pablo Forero"
__email__ = "github46@pabloforero.eu"

from gridrival_ai.contracts import Contract
from gridrival_ai.salary import SalaryManager
from gridrival_ai.team import Team

__all__ = [
    "Team",
    "Contract",
    "SalaryManager",
]
