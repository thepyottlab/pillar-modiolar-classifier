"""Filename-based discovery of group IDs and tokenized file sets."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

from .exceptions import GroupValidationError
from .models import FinderConfig, Group


def find_groups(cfg: FinderConfig) -> Dict[str, Group]:
    """Return only groups that contain all required files; raise if any were skipped.

    Matching is done by filename stem ending with the configured *suffix tokens*
    (``cfg.ribbons``, ``cfg.psds``, ``cfg.positions``), before the extension.

    Required files per group:
      - Always: positions
      - If not ``cfg.ribbons_only``: PSDs
      - If not ``cfg.psds_only``: ribbons

    If one or more groups are missing required files, they are excluded and a
    GroupValidationError is raised with a readable summary *and* the remaining
    valid groups attached to ``exc.groups`` so the caller can still use them.

    Args:
        cfg: File discovery and naming configuration.

    Returns:
        Dict[str, Group]: Map of group id to group container.

    Raises:
        GroupValidationError: When at least one group is missing required files.
    """
    folder = Path(cfg.folder)
    allowed_ext = {str(cfg.extensions).lower()}

    rib_tok = (cfg.ribbons or "").strip()
    psd_tok = (cfg.psds or "").strip()
    pos_tok = (cfg.positions or "").strip()
    ext_tok = (cfg.extensions or "").strip()

    scans: list[tuple[str, str]] = []
    if rib_tok:
        scans.append((rib_tok, "ribbons"))
    if psd_tok:
        scans.append((psd_tok, "psds"))
    if pos_tok:
        scans.append((pos_tok, "positions"))

    temp: dict[str, dict[str, Path]] = defaultdict(dict)

    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in allowed_ext:
            continue
        stem = p.stem
        stem_cmp = stem.lower() if cfg.case_insensitive else stem
        for tok, role in scans:
            tok_cmp = tok.lower() if cfg.case_insensitive else tok
            if stem_cmp.endswith(tok_cmp):
                gid = stem[: -len(tok)]
                temp[gid][role] = p
                break

    complete: Dict[str, Group] = {}
    problems: list[str] = []

    for gid, roles in sorted(temp.items()):
        missing: list[str] = []
        if "positions" not in roles:
            missing.append(f"positions (*{pos_tok}{ext_tok})")
        if not cfg.ribbons_only and "psds" not in roles:
            missing.append(f"PSDs (*{psd_tok}{ext_tok})")
        if not cfg.psds_only and "ribbons" not in roles:
            missing.append(f"ribbons (*{rib_tok}{ext_tok})")

        if missing:
            problems.append(f"- '{gid}': missing " + ", ".join(missing))
            continue

        complete[gid] = Group(id=gid, file_paths=dict(roles))

    if problems:
        summary = "Some IDs were skipped due to missing required files:\n" + "\n".join(problems)
        raise GroupValidationError(summary, groups=complete)

    return complete
