#!/usr/bin/env python3
"""
PoC Generation - heap_buffer_overflow / ndpi
Task: arvo:21000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["arvo:21000"]
GT_POC_LENS = {"arvo:21000": 33}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
