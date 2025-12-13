#!/usr/bin/env bash
set -euo pipefail

# Display all result files under ./results (or a provided directory) in a readable way.
# - For .npy files: show shape and a small value preview.
# - For text-like files: print contents (truncated to keep output manageable).
# Usage: ./Autoformer_results_listing.sh [results_dir]

RESULTS_DIR=${1:-./results}

if [ ! -d "${RESULTS_DIR}" ]; then
  echo "Results directory not found: ${RESULTS_DIR}"
  exit 1
fi

python - "${RESULTS_DIR}" <<'PY'
import os
import sys
import numpy as np

root = sys.argv[1]
print(f"Listing results under '{root}':\n")

for dirpath, _, filenames in os.walk(root):
  filenames.sort()
  for fname in filenames:
    if fname != "metrics.npy":
      continue
    path = os.path.join(dirpath, fname)
    rel = os.path.relpath(path, root)
    print(f"--- {rel}")
    try:
      arr = np.load(path)
      flat = arr.ravel()
      preview = flat[:8]
      print(f"type=npy shape={arr.shape} preview={preview}")
    except Exception as e:
      print(f"[error reading file: {e}]")
    print()
PY
