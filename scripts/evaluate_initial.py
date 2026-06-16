from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator  # noqa: E402


def main() -> None:
    result = evaluator.evaluate(ROOT / "initial_program.py")
    print("status:", result.artifacts["status"])
    print("message:", result.artifacts["message"])
    print("combined_score:", f"{result.metrics['combined_score']:.8f}")
    print("valid:", result.metrics["valid"])
    if result.metrics["valid"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
