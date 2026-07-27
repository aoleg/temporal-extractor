#!/usr/bin/env python3
"""
temporal_extractor entry point.

Run through extract.bat, which uses the project's own virtualenv:

    extract.bat run myvideo.mp4

Or directly, if you have already activated that venv:

    python extract.py run myvideo.mp4

Named extract.py rather than temporal_extractor.py because a module of that name
sitting beside the package of the same name would shadow it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from temporal_extractor.cli import main

if __name__ == "__main__":
    sys.exit(main())
