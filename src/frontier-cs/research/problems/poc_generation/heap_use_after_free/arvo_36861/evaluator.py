#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / spice-usbredir
Task: arvo:36861
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:36861"]
GT_POC_LENS = {"arvo:36861": 71298}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
