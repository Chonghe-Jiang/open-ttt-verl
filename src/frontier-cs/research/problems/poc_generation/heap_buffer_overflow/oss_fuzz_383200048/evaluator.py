#!/usr/bin/env python3
"""
PoC Generation - heap_buffer_overflow / upx
Task: oss-fuzz:383200048
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:383200048"]
GT_POC_LENS = {"oss-fuzz:383200048": 512}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
