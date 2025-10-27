"""Typed configuration and lightweight data models for the Pillar–Modiolar
Classifier app.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class InputMode(Enum):
    """Input mode determining which datasets are required and visible."""

    BOTH = auto()
    RIBBONS_ONLY = auto()
    PSDS_ONLY = auto()


@dataclass(slots=True)
class FinderConfig:
    """Configuration for file discovery, object naming, and behavior.

    Attributes:
        folder: Folder selected by the user that contains the input files.
        ribbons: Suffix token for ribbon volume files (e.g., "rib").
        psds: Suffix token for PSD volume files (e.g., "psd").
        positions: Suffix token for positions files (e.g., "pos").
        extensions: File extension to scan (e.g., ".xls").
        ribbons_obj: Object label for ribbon points in the Position table.
        psds_obj: Object label for PSD points in the Position table.
        pillar_obj: Object label for the pillar anchor.
        modiolar_obj: Object label for the modiolar anchor.
        case_insensitive: Whether to match filename tokens case-insensitively.
        ribbons_only: When True, only process ribbon volumes.
        psds_only: When True, only process PSD volumes.
        identify_poles: When True, relabel anchors so the basal anchor is closer
            to most synapses within an IHC.
        remember_input_fields: Persist inputs to a user config file.
    """

    folder: Path
    ribbons: str = "rib"
    psds: str = "psd"
    positions: str = "pos"
    extensions: str = ".xls"
    ribbons_obj: str = "ribbons"
    psds_obj: str = "PSDs"
    pillar_obj: str = "pillar"
    modiolar_obj: str = "modiolar"
    case_insensitive: bool = True
    ribbons_only: bool = False
    psds_only: bool = False
    identify_poles: bool = True
    remember_input_fields: bool = True

    @property
    def mode(self) -> InputMode:
        """Return the effective input mode.

        Returns:
            InputMode: The inferred mode based on `ribbons_only`/`psds_only`.

        Raises:
            ValueError: If both ribbons_only and psds_only are True.
        """
        if self.ribbons_only and self.psds_only:
            raise ValueError("ribbons_only and psds_only are mutually exclusive.")
        if self.ribbons_only:
            return InputMode.RIBBONS_ONLY
        if self.psds_only:
            return InputMode.PSDS_ONLY
        return InputMode.BOTH


@dataclass(slots=True)
class Group:
    """A set of files with the same base identifier.

    Attributes:
        id: Base identifier (e.g., the shared stem before the suffix token).
        file_paths: Mapping of role -> path. Roles are "ribbons" | "psds" | "positions".
    """

    id: str
    file_paths: dict[str, Path]
