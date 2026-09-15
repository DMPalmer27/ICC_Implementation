"""
File: _path.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: Puts the implementation directory icc/ on sys.path so the instruments in
    tools/ can import the library by plain module name. Mirrors tests/conftest.py; see
    that file for why icc/ is not simply an installed package.
"""

import sys
from pathlib import Path

_ICC = Path(__file__).resolve().parent.parent / "icc"
if str(_ICC) not in sys.path:
    sys.path.insert(0, str(_ICC))
