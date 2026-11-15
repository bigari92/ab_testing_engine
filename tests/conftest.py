# -*- coding: utf-8 -*-
"""Put the library and the generator on the import path.

The project is deliberately not installed as a package: `pip install -e .`
and a `setup.py` would be two more things to explain, and the point of the
repository is that every step is reproducible by reading it. So the tests
add `src/` and `data/` to sys.path the same way the proofs do.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for folder in ("src", "data"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
