from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set

from .models import FinderConfig, Group, InputMode


def _found_suffix_tokens(folder: Path, extension: str, tokens: Iterable[str], ci: bool) -> Set[str]:
    ext = extension.lower()
    found: Set[str] = set()
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ext:
            stem = p.stem
            for s in tokens:
                token = f" {s}"
                if (stem.lower().endswith(token.lower()) if ci else stem.endswith(token)):
                    found.add(s)
    return found


def find_groups(cfg: FinderConfig) -> Dict[str, Group]:
    """Scan the folder and group files by base id.

    Rules:
    - A file named "XYZ ribbon.xls" has base id "XYZ" and token "ribbon".
    - Only files matching cfg.extensions are considered.

    Args:
        cfg: Discovery configuration.

    Returns:
        Mapping of group id -> Group(file_paths[token] = Path)
    """
    folder = Path(cfg.folder)
    allowed_ext = {cfg.extensions.lower()}
    temp = defaultdict(dict)

    tokens = []
    if cfg.mode in (InputMode.BOTH, InputMode.RIBBONS_ONLY):
        tokens.append(cfg.ribbons)
    if cfg.mode in (InputMode.BOTH, InputMode.PSDS_ONLY):
        tokens.append(cfg.psds)
    tokens.append(cfg.positions)

    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in allowed_ext:
            continue
        stem = p.stem
        for s in tokens:
            token = f" {s}"
            if (stem.lower().endswith(token.lower()) if cfg.case_insensitive else stem.endswith(token)):
                id_part = stem[: -len(token)]
                temp[id_part][s] = p
                break

    groups = {gid: Group(id=gid, file_paths=dict(paths)) for gid, paths in temp.items()}
    return groups
