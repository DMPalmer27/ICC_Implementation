"""
File: conftest.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Puts the implementation directory icc/ on sys.path so the test suites can
    keep importing the modules by their plain names (from utils import ...) rather than
    through a package prefix. icc/ is a plain directory, not an installed package, so
    nothing is on the import path by default.

    The suites are run directly (python tests/test_icc.py), which does NOT auto-load a
    conftest.py -- that is a pytest mechanism. So each suite imports this module
    explicitly at the top. The file is named conftest.py anyway so that if pytest is
    ever added, it picks the same shim up automatically and the two agree.
"""

import sys
from pathlib import Path

_ICC = Path(__file__).resolve().parent.parent / "icc"
if str(_ICC) not in sys.path:
    sys.path.insert(0, str(_ICC))
