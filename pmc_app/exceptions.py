"""Exception hierarchy for the Pillar–Modiolar Classifier."""

from __future__ import annotations


class PmcError(Exception):
    """Base exception for the Pillar–Modiolar Classifier."""


class ConfigError(PmcError):
    """Raised on configuration or input validation errors."""


class ParseError(PmcError):
    """Raised when reading/parsing input files fails."""


class ProcessingError(PmcError):
    """Raised when transforming or merging data fails."""
