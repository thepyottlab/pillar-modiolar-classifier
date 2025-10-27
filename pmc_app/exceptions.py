"""Exception hierarchy for the Pillar–Modiolar Classifier."""

from __future__ import annotations

from .models import Group


class PmcError(Exception):
    """Base exception for the Pillar–Modiolar Classifier."""


class ConfigError(PmcError):
    """Raised on configuration or input validation errors."""


class ParseError(PmcError):
    """Raised when reading/parsing input files fails."""


class ProcessingError(PmcError):
    """Raised when transforming or merging data fails."""


class GroupValidationError(PmcError):
    """Raised when one or more groups fail required token checks.

    Attributes:
        groups: The dict of valid groups that *did* pass validation.
    """

    def __init__(self, message: str, groups: dict[str, Group] | None = None) -> None:
        super().__init__(message)
        self.groups: dict[str, Group] = groups or {}
