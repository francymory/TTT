#!/usr/bin/env bash
# Native RULER evaluator for methods whose write is not a closed-form ridge solve
# (in-place-ttt and standard-ft). It reuses the existing TTT-NTP OpenCompass
# prompts, generation settings, and Full-13 substring scorer.
set -euo pipefail

: "${CKPT:?CKPT is required}"
: "${PROJECT_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${LENGTHS:=4k 8k 16k 32k}"
: "${NUM_SAMPLES:=100}"
: "${MAX_NEW_TOKENS:=64}"
: "${MODEL_ABBR:=native-ttt}"
: "${OUT_DIR:=${PROJECT_DIR}/results/native_ruler/${MODEL_ABBR}}"
: "${GPUS:=0}"

export CKPT PROJECT_DIR LENGTHS NUM_SAMPLES MAX_NEW_TOKENS MODEL_ABBR OUT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"

# The in-place cache is stateful per sequence, so native TTT generation uses
# batch_size=1 and one OpenCompass worker. Select the first assigned device.
GPU="${GPUS%%,*}"
python3 "$PROJECT_DIR/scripts/qwen3_no_ttt_ruler_eval.py" prepare
for length in $LENGTHS; do
  echo "[native] generating RULER ${length} on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="$GPU" python3 -c "
import inference_model  # registers the cache-aware TTT model
from opencompass.cli.main import main
import sys
sys.argv = ['opencompass', '${OUT_DIR}/configs/ruler_${length}.py']
main()
"
  python3 "$PROJECT_DIR/scripts/qwen3_no_ttt_ruler_eval.py" summarize-length "$length"
done
python3 "$PROJECT_DIR/scripts/qwen3_no_ttt_ruler_eval.py" summarize
