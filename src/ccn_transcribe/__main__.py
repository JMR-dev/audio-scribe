"""python -m ccn_transcribe"""

from __future__ import annotations

from ccn_transcribe.cli import main

__all__ = ["main"]

# Python 3.14 uses forkserver, which re-imports __main__; without this guard a
# dependency touching multiprocessing runs a second copy of the program.
if __name__ == "__main__":
    main()
