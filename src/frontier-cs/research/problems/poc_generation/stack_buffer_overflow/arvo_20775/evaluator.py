#!/usr/bin/env python3
"""
PoC Generation - stack_buffer_overflow / openthread
Task: arvo:20775
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:20775"]
GT_POC_LENS = {"arvo:20775": 844}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
