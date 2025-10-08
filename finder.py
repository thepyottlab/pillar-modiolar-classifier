from collections import defaultdict
from pathlib import Path
from typing import Dict

from models import FinderConfig, Group

def find_groups(cfg: FinderConfig) -> Dict[str, Group]:
    folder = Path(cfg.folder)
    allowed = {cfg.extensions.lower()}
    temp = defaultdict(dict)

    all_suffixes = [cfg.ribbons, cfg.psds, cfg.positions]

    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue
        stem = p.stem
        for s in all_suffixes:
            token = f" {s}"
            if cfg.case_insensitive:
                if stem.lower().endswith(token.lower()):
                    id_part = stem[:-len(token)]
                    temp[id_part][s] = p
                    break
            else:
                if stem.endswith(token):
                    id_part = stem[:-len(token)]
                    temp[id_part][s] = p
                    break

    groups = {}
    for id_key, suffix_map in temp.items():
        groups[id_key] = Group(id=id_key, file_paths=dict(suffix_map))
    return groups

