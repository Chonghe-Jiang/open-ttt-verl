#!/usr/bin/env python3
"""
PoC Generation - heap_buffer_overflow / libheif
Task: oss-fuzz:42536679
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:42536679"]
GT_POC_LENS = {"oss-fuzz:42536679": 2936}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
