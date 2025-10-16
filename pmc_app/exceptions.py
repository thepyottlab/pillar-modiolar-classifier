from __future__ import annotations


class PmcError(Exception):
    """Base exception for Pillar–Modiolar Classifier."""


class ConfigError(PmcError):
    """Configuration or input validation error."""


class ParseError(PmcError):
    """Raised when reading/parsing input files fails."""


class ProcessingError(PmcError):
    """Raised when transforming/merging data fails."""
