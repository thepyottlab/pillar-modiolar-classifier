from collections import defaultdict
from pathlib import Path
from typing import Dict

from models import FinderConfig, Group


def find_groups(cfg: FinderConfig) -> Dict[str, Group]:
    """Collect all ribbon/PSD/position files that belong to the same group.

    Parameters
    ----------
    cfg:
        Finder configuration describing the folder to search and the
        identifying suffix tokens for each file type.

    Returns
    -------
    Dict[str, Group]
        A mapping from group identifier to ``Group`` instances that contain
        the resolved file paths for each suffix token.
    """

    folder = Path(cfg.folder)
    allowed_suffixes = {cfg.extensions.lower()}
    temporary_groups: Dict[str, Dict[str, Path]] = defaultdict(dict)

    suffixes = (cfg.ribbons, cfg.psds, cfg.positions)

    for file_path in folder.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in allowed_suffixes:
            continue

        stem = file_path.stem
        for suffix in suffixes:
            token = f" {suffix}"
            if cfg.case_insensitive:
                if stem.lower().endswith(token.lower()):
                    identifier = stem[: -len(token)]
                    temporary_groups[identifier][suffix] = file_path
                    break
            else:
                if stem.endswith(token):
                    identifier = stem[: -len(token)]
                    temporary_groups[identifier][suffix] = file_path
                    break

    groups: Dict[str, Group] = {}
    for identifier, suffix_map in temporary_groups.items():
        groups[identifier] = Group(id=identifier, file_paths=dict(suffix_map))
    return groups

