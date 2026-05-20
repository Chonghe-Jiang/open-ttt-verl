#!/usr/bin/env python3
"""
PoC Generation - stack_buffer_overflow / ghostscript
Task: arvo:44034
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:44034"]
GT_POC_LENS = {"arvo:44034": 80064}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
