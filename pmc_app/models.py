from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Dict


class InputMode(Enum):
    """Input mode determining which datasets are required/visible."""

    BOTH = auto()
    RIBBONS_ONLY = auto()
    PSDS_ONLY = auto()


@dataclass(slots=True)
class FinderConfig:
    """Configuration for file discovery and object naming.

    Attributes:
        folder: Folder selected by the user.
        ribbons: Suffix token for ribbon volume files (e.g., "ribbon").
        psds: Suffix token for PSD volume files (e.g., "psd").
        positions: Suffix token for positions files (e.g., "sum").
        extensions: File extension to scan (e.g., ".xls").
        ribbons_obj: Object label for ribbon points in Position table.
        psds_obj: Object label for PSD points in Position table.
        ihc_obj: Object label used for IHC points.
        pillar_obj: Object label used for pillar anchor.
        modiolar_obj: Object label used for modiolar anchor.
        case_insensitive: Whether filename token matching is case-insensitive.
        ribbons_only: If true, operate only on ribbons, ignore PSDs.
        psds_only: If true, operate only on PSDs, ignore ribbons.
        remember_input_fields: Persist inputs in config.ini when enabled.
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
    remember_input_fields: bool = False

    @property
    def mode(self) -> InputMode:
        """Compute effective input mode."""
        if self.ribbons_only and self.psds_only:
            raise ValueError("ribbons_only and psds_only are mutually exclusive.")
        if self.ribbons_only:
            return InputMode.RIBBONS_ONLY
        if self.psds_only:
            return InputMode.PSDS_ONLY
        return InputMode.BOTH


@dataclass(slots=True)
class Group:
    """Represents a set of files (same base id) for parsing."""

    id: str
    file_paths: Dict[str, Path]  # mapping token -> path
