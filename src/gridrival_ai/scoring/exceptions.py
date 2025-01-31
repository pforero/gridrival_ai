"""Custom exceptions for F1 fantasy scoring."""


class ScoringError(Exception):
    """Base exception for scoring-related errors."""

    pass


class ConfigurationError(ScoringError):
    """Raised when scoring configuration is invalid."""

    pass


class ValidationError(ScoringError):
    """Raised when race data validation fails."""

    pass
