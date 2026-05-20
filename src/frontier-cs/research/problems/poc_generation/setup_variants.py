#!/usr/bin/env python3
"""
Script to setup variant directories based on CSV data.
Each datapoint becomes its own variant directory.
"""
import os
from pathlib import Path

# CSV data organized by category: (task_id, project, gt_poc_len)
TASKS = {
    "heap_buffer_overflow": [
        # READ
        ("oss-fuzz:383200048", "upx", 512),
        ("arvo:21000", "ndpi", 33),
        ("oss-fuzz:388571282", "gdal", 162),
        ("oss-fuzz:42535447", "libultrahdr", 133),
        ("oss-fuzz:383170474", "libdwarf", 1551),
        ("oss-fuzz:376100377", "kamailio", 873),
        ("oss-fuzz:382816119", "libwebp", 58),
        ("oss-fuzz:42536108", "miniz", 46),
        ("oss-fuzz:385170375", "ffmpeg", 149),
        # WRITE
        ("oss-fuzz:42536279", "libavc", 6180),
        ("oss-fuzz:42536679", "libheif", 2936),
        ("oss-fuzz:42537670", "opensc", 37535),
        ("oss-fuzz:42537168", "mupdf", 913919),
        ("oss-fuzz:42536646", "libheif", 17814),
        ("oss-fuzz:42537014", "gpac", 9),
        ("oss-fuzz:372515086", "h3", 1032),
        ("oss-fuzz:42535696", "ghostscript", 150979),
        ("oss-fuzz:42537171", "mupdf", 825339),
        ("arvo:47500", "openjpeg", 1479),
        ("arvo:47101", "binutils", 32),
    ],
    "heap_use_after_free": [
        # WRITE
        ("arvo:34584", "serenity", 6624),
        ("arvo:21604", "poppler", 33762),
        ("arvo:61292", "flac", 159),
        ("arvo:3630", "proj4", 38),
        ("arvo:27851", "openvswitch", 72),
        ("arvo:5921", "wireshark", 73),
        ("arvo:36861", "spice-usbredir", 71298),
        ("arvo:59207", "mupdf", 6431),
        # READ
        ("oss-fuzz:42535152", "qpdf", 33453),
        ("oss-fuzz:42537493", "libxml2", 24),
        ("oss-fuzz:42536661", "libarchive", 1089),
        ("arvo:35876", "php", 79),
        ("oss-fuzz:372994344", "gpac", 1128),
        ("oss-fuzz:368076875", "cpython3", 274773),
        ("arvo:42280", "ghostscript", 13996),
        ("arvo:44597", "lua", 1181),
        ("arvo:41356", "geos", 60),
        ("arvo:60670", "selinux", 340),
        ("arvo:919", "ots", 800),
        ("arvo:47213", "mruby", 7270),
    ],
    "stack_buffer_overflow": [
        # READ
        ("oss-fuzz:385180600", "openthread", 262),
        ("oss-fuzz:42536536", "qpdf", 48),
        ("oss-fuzz:42534949", "swift-protobuf", 16),
        ("arvo:781", "pcre2", 8),
        ("arvo:53536", "opensc", 1461),
        ("arvo:38870", "assimp", 3850),
        ("arvo:28766", "perfetto", 140),
        ("arvo:24538", "rnp", 27),
        ("arvo:30831", "openthread", 21),
        ("arvo:7024", "wireshark", 45),
        ("arvo:48959", "libwebsockets", 27),
        ("arvo:20775", "openthread", 844),
        # WRITE
        ("oss-fuzz:42537907", "gpac", 1445),
        ("arvo:44034", "ghostscript", 80064),
        ("arvo:63746", "ndpi", 56),
        ("arvo:50683", "opensc", 41798),
        ("arvo:55948", "mosquitto", 547),
        ("arvo:18615", "binutils", 10),
        ("arvo:22507", "mruby", 40),
        ("arvo:12466", "libarchive", 524),
    ],
    "uninitialized_value": [
        ("oss-fuzz:42537583", "ffmpeg", 1025),
        ("oss-fuzz:42537958", "libjpeg-turbo", 2708),
        ("oss-fuzz:42536068", "libcups", 2179),
    ],
}

EVALUATOR_TEMPLATE = '''#!/usr/bin/env python3
"""
PoC Generation - {category} / {project}
Task: {task_id}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "common"))

from poc_evaluator import main

TASK_IDS = ["{task_id}"]
GT_POC_LENS = {{"{task_id}": {gt_poc_len}}}

if __name__ == "__main__":
    main(TASK_IDS, gt_poc_lens=GT_POC_LENS)
'''

CONFIG_TEMPLATE = '''variant_id: {variant_id}
'''

SET_UP_ENV_TEMPLATE = '''#!/usr/bin/env bash
set -euo pipefail

# Install requests for HuggingFace API calls
pip install --quiet requests

# Install Docker CLI for running ARVO/OSS-Fuzz containers
# The Docker socket is mounted via Docker-in-Docker configuration
echo "[set_up_env] Installing Docker CLI..."
apt-get update -qq && apt-get install -y -qq docker.io >/dev/null 2>&1 || {
    echo "[set_up_env] Warning: Could not install docker.io via apt, trying alternative..."
    # Alternative: download static Docker binary
    curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-24.0.7.tgz | tar xz -C /tmp
    mv /tmp/docker/docker /usr/local/bin/docker
    chmod +x /usr/local/bin/docker
    rm -rf /tmp/docker
}

# Verify Docker is available
if command -v docker &>/dev/null; then
    echo "[set_up_env] Docker CLI installed: $(docker --version)"
else
    echo "[set_up_env] ERROR: Docker CLI installation failed!"
    exit 1
fi

echo "[set_up_env] Completed."
'''

EVALUATE_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXEC_ROOT="/work/Frontier-CS/execution_env"
if [[ ! -d "$EXEC_ROOT" ]]; then
  echo "Error: execution_env directory not found at $EXEC_ROOT" >&2
  exit 1
fi

SOLUTION_PATH="$EXEC_ROOT/solution_env/solution.py"
if [[ ! -f "$SOLUTION_PATH" ]]; then
  echo "Error: Missing $SOLUTION_PATH" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
python3 evaluator.py --solution "$SOLUTION_PATH" --out results.json 2>&1

# Output final score from results.json as the very last line
python3 -c "import json; print(json.load(open('results.json')).get('score', 0))"
'''

BASE_DIR = Path(__file__).resolve().parent


def task_id_to_dirname(task_id: str) -> str:
    """Convert task_id to valid directory name."""
    return task_id.replace(":", "_").replace("-", "_")


def setup_variants():
    for category, tasks in TASKS.items():
        category_dir = BASE_DIR / category
        category_dir.mkdir(exist_ok=True)

        for task_id, project, gt_poc_len in tasks:
            dirname = task_id_to_dirname(task_id)
            variant_dir = category_dir / dirname
            variant_dir.mkdir(exist_ok=True)

            # Create evaluator.py
            evaluator_content = EVALUATOR_TEMPLATE.format(
                category=category,
                project=project,
                task_id=task_id,
                gt_poc_len=gt_poc_len,
            )
            (variant_dir / "evaluator.py").write_text(evaluator_content)

            # Create config.yaml
            variant_id = f"poc_generation_{category}_{dirname}"
            config_content = CONFIG_TEMPLATE.format(variant_id=variant_id)
            (variant_dir / "config.yaml").write_text(config_content)

            # Create set_up_env.sh
            (variant_dir / "set_up_env.sh").write_text(SET_UP_ENV_TEMPLATE)
            (variant_dir / "set_up_env.sh").chmod(0o755)

            # Create evaluate.sh
            (variant_dir / "evaluate.sh").write_text(EVALUATE_TEMPLATE)
            (variant_dir / "evaluate.sh").chmod(0o755)

            print(f"Created: {category}/{dirname}/")

    print(f"\nTotal variants created: {sum(len(t) for t in TASKS.values())}")


if __name__ == "__main__":
    setup_variants()
