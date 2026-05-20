#!/usr/bin/env bash
set -euo pipefail

# Downloads datasets for vdb_design problem to local datasets folder

PROBLEM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "$PROBLEM_DIR/../../.." && pwd)
DATASETS_DIR="$BASE_DIR/datasets/vdb_design"

mkdir -p "$DATASETS_DIR"

echo "[vdb_design download] Checking for SIFT1M dataset..."

if compgen -G "$DATASETS_DIR/*.fvecs" >/dev/null 2>&1; then
  echo "[vdb_design download] Dataset already exists at $DATASETS_DIR"
  exit 0
fi

TAR_PATH="$DATASETS_DIR/sift.tar.gz"

if [[ ! -f "$TAR_PATH" ]]; then
  echo "[vdb_design download] Downloading SIFT1M dataset from FTP..."
  if command -v wget >/dev/null 2>&1; then
    wget -O "$TAR_PATH" ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  elif command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error -o "$TAR_PATH" ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  else
    python3 - "$TAR_PATH" <<'PY'
import sys
from pathlib import Path
from ftplib import FTP

destination = sys.argv[1]
target_dir = Path(destination).resolve().parent
target_dir.mkdir(parents=True, exist_ok=True)

try:
    with FTP('ftp.irisa.fr', timeout=120) as ftp:
        ftp.login(user='anonymous', passwd='anonymous@')
        ftp.set_pasv(True)
        ftp.voidcmd('TYPE I')
        with open(destination, 'wb') as out_file:
            ftp.retrbinary('RETR /local/texmex/corpus/sift.tar.gz', out_file.write)
except Exception as exc:
    raise SystemExit(f'download failed: {exc}')
PY
  fi
fi

echo "[vdb_design download] Extracting dataset..."
TMP_DIR="$DATASETS_DIR/tmp_sift"
mkdir -p "$TMP_DIR"
tar -xzf "$TAR_PATH" -C "$TMP_DIR" 2>/dev/null || true

for f in sift_base.fvecs sift_learn.fvecs sift_query.fvecs sift_groundtruth.ivecs; do
  find "$TMP_DIR" -type f -name "$f" -exec mv -f {} "$DATASETS_DIR/" \;
done

rm -rf "$TMP_DIR"
rm -f "$TAR_PATH"

if ! compgen -G "$DATASETS_DIR/*.fvecs" >/dev/null 2>&1; then
  echo "Error: Expected files not found after extraction" >&2
  exit 1
fi

echo "[vdb_design download] Dataset ready at $DATASETS_DIR"


