#!/usr/bin/env python3
"""
PoC Generation - heap_buffer_overflow / binutils
Task: arvo:47101
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:47101"]
GT_POC_LENS = {"arvo:47101": 32}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
