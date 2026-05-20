#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / cpython3
Task: oss-fuzz:368076875
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:368076875"]
GT_POC_LENS = {"oss-fuzz:368076875": 274773}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
