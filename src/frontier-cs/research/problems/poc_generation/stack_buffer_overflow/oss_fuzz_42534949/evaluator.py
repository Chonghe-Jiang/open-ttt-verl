#!/usr/bin/env python3
"""
PoC Generation - stack_buffer_overflow / swift-protobuf
Task: oss-fuzz:42534949
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["oss-fuzz:42534949"]
GT_POC_LENS = {"oss-fuzz:42534949": 16}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
