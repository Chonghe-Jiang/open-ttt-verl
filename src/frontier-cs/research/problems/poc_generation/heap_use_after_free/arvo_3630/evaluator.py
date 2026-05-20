#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / proj4
Task: arvo:3630
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:3630"]
GT_POC_LENS = {"arvo:3630": 38}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
