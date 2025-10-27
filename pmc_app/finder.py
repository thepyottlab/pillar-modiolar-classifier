"""Filename-based discovery of group IDs and tokenized file sets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

from .models import FinderConfig, Group, InputMode


def find_groups(cfg: FinderConfig) -> Dict[str, Group]:
    """Group files by base identifier given the configured suffix tokens.

    Files are grouped by the part of the filename that precedes a known token
    that appears immediately before the extension (e.g., ``"XYZ rib.xls"`` →
    base id ``"XYZ "``, token ``"rib"``).
    """
    folder = Path(cfg.folder)
    allowed_ext = {cfg.extensions.lower()}
    temp: dict[str, dict[str, Path]] = defaultdict(dict)

    tokens: list[str] = []
    if cfg.mode in (InputMode.BOTH, InputMode.RIBBONS_ONLY):
        tokens.append(cfg.ribbons)
    if cfg.mode in (InputMode.BOTH, InputMode.PSDS_ONLY):
        tokens.append(cfg.psds)
    tokens.append(cfg.positions)

    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in allowed_ext:
            continue
        stem = p.stem
        stem_cmp = stem.lower() if cfg.case_insensitive else stem
        for s in tokens:
            tok_cmp = s.lower() if cfg.case_insensitive else s
            if stem_cmp.endswith(tok_cmp):
                gid = stem[: -len(s)]
                temp[gid][s] = p
                break

    return {gid: Group(id=gid, file_paths=dict(paths)) for gid, paths in temp.items()}
