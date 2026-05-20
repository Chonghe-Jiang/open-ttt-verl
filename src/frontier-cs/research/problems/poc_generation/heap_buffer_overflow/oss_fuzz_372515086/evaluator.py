#!/usr/bin/env python3
"""
PoC Generation - heap_buffer_overflow / h3
Task: oss-fuzz:372515086
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:372515086"]
GT_POC_LENS = {"oss-fuzz:372515086": 1032}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
