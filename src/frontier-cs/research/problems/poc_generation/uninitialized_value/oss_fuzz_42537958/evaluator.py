#!/usr/bin/env python3
"""
PoC Generation - uninitialized_value / libjpeg-turbo
Task: oss-fuzz:42537958
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:42537958"]
GT_POC_LENS = {"oss-fuzz:42537958": 2708}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
