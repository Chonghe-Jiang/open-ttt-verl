#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / libxml2
Task: oss-fuzz:42537493
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:42537493"]
GT_POC_LENS = {"oss-fuzz:42537493": 24}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
