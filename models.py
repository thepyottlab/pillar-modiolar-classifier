from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

@dataclass()
class FinderConfig:
    folder: Path                   # folder the user selected
    ribbons: str = "ribbon"        # suffix string used for ribbon files (single token)
    psds: str = "psd"              # suffix string used for psd files
    positions: str = "sum"         # suffix string used for positions files
    extensions: str = ".xls"
    ribbons_obj: str = "ribbons"
    psds_obj: str = "PSDs"
    ihc_obj: str = "IHC"
    pillar_obj: str = "pillar"
    modiolar_obj: str = "modiolar"
    case_insensitive: bool = True
    preview_limit: int = 50

@dataclass
class Group:
    id: str
    file_paths: Dict[str, Path]    # mapping from suffix -> Path
