from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OPENEVOLVE_SRC = ROOT / "openevolve"

if str(OPENEVOLVE_SRC) not in sys.path:
    sys.path.insert(0, str(OPENEVOLVE_SRC))
