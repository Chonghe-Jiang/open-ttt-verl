#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / poppler
Task: arvo:21604
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:21604"]
GT_POC_LENS = {"arvo:21604": 33762}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
