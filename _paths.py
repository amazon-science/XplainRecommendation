# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Path resolution for this repo's pipeline/runner/script modules.

Every pipeline and runner script does:

    from _paths import FINAL_ROOT, DATA_ROOT, setup_sys_path
    setup_sys_path()

This sets sys.path up so `from scripts.X import Y` and `from src.X import Y`
resolve to this repo's scripts/ and src/ regardless of the caller's cwd.

DATA_ROOT is the directory containing XRec/, G-Refer/, and data/
(the external datasets). By default it's the repo root itself. Override
with the FINAL_RESULT_DATA env var if your data lives elsewhere.
"""
import os
import sys
from pathlib import Path

FINAL_ROOT = Path(__file__).resolve().parent

_env = os.environ.get("FINAL_RESULT_DATA")
DATA_ROOT = Path(_env).resolve() if _env else FINAL_ROOT


def setup_sys_path():
    """Prepend FINAL_ROOT so `from scripts.X import Y` and
    `from src.X import Y` hit our local copies, not any global install."""
    p = str(FINAL_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
