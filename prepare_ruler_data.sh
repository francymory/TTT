#!/usr/bin/env bash
# Run once on an internet-connected login/data-transfer node; no GPU needed.
set -euo pipefail
source "${EVAL_ENV:-/leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt}/bin/activate"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-$PROJECT_DIR/results/opencompass-data}"
mkdir -p "$COMPASS_DATA_CACHE"
python - <<'PYDATA'
import os
from pathlib import Path
from opencompass.utils import get_data_path

root = Path(os.environ['COMPASS_DATA_CACHE']) / 'data' / 'ruler'
for name in ('dev-v2.0.json', 'hotpotqa.json', 'PaulGrahamEssays.jsonl'):
    target = root / name
    if not target.is_file() or target.stat().st_size == 0:
        get_data_path(f'./data/ruler/{name}', local_mode=True)
    if not target.is_file() or target.stat().st_size == 0:
        raise SystemExit(f'Missing or empty dataset file after download: {target}')
    print(f'Ready: {target}')
print('RULER data ready for the offline GPU job.')
PYDATA
