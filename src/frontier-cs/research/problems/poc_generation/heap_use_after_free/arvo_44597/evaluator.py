#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / lua
Task: arvo:44597
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:44597"]
GT_POC_LENS = {"arvo:44597": 1181}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
