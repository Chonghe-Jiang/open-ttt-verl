#!/usr/bin/env python3
"""
PoC Generation - heap_use_after_free / ghostscript
Task: arvo:42280
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:42280"]
GT_POC_LENS = {"arvo:42280": 13996}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
