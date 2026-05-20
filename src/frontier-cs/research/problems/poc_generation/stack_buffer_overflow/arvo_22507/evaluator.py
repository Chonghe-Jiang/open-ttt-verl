#!/usr/bin/env python3
"""
PoC Generation - stack_buffer_overflow / mruby
Task: arvo:22507
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:22507"]
GT_POC_LENS = {"arvo:22507": 40}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
