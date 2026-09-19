"""Logging: a console stream plus a per-job file."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from audio_scribe.paths import JobPaths

ROOT = "audio_scribe"


def configure(verbose: int = 0, quiet: bool = False) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logger = logging.getLogger(ROOT)
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(message)s" if not verbose else "%(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False


@contextmanager
def job_log(job: JobPaths) -> Generator[None]:
    """Attach a per-job file handler for the duration of one job.

    Detached again afterwards: a 200-URL batch would otherwise leak 200 open
    handlers.
    """
    job.logs_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(job.logs_dir / "job.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger(ROOT)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
