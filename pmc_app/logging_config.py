from __future__ import annotations

import logging
from logging import Logger


def configure_logging(level: int = logging.INFO) -> Logger:
    """Configure root logging once, idempotently.

    Args:
        level: Logging level, default INFO.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    if not root.handlers:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        root.setLevel(level)
    return root
