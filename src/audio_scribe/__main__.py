"""python -m audio_scribe"""

from __future__ import annotations

from audio_scribe.cli import main

__all__ = ["main"]

# Python 3.14 uses forkserver, which re-imports __main__; without this guard a
# dependency touching multiprocessing runs a second copy of the program.
if __name__ == "__main__":
    # A frozen build re-invokes this executable to start a multiprocessing child.
    # Without this the child parses the interpreter's own flags as CLI options and
    # prints "No such option '-B'" before failing to start.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
