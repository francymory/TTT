#!/usr/bin/env bash
# Submit: sbatch eval_qwen3_06b_no_ttt_ruler.sh
# Quick check: NUM_SAMPLES=2 sbatch eval_qwen3_06b_no_ttt_ruler.sh
# Native (no-TTT) baseline evaluation: all 13 RULER tasks from 4K through 64K.
# This checkpoint has no ttt_layers (trained with ttt_mode=false), so it cannot
# go through eval_cf_ruler.sh's closed-form patch; it uses plain generation via
# qwen3_no_ttt_ruler_eval.py instead.
#SBATCH --job-name=qwen3_06b_no_ttt_ruler64k
#SBATCH --account=IscrB_MOSAIC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/fmorandi/TTT-NTP
#SBATCH --output=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/eval_qwen3_06b_no_ttt_ruler_%j.out
#SBATCH --error=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/eval_qwen3_06b_no_ttt_ruler_%j.err

set -euo pipefail

source "${EVAL_ENV:-/leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt}/bin/activate"
export PROJECT_DIR=/leonardo/home/userexternal/fmorandi/TTT-NTP
cd "$PROJECT_DIR"

export CKPT="${CKPT:-/leonardo_scratch/large/userexternal/fmorandi/checkpoints/ttt/qwen3_06b_no_ttt_58031456/checkpoints/global_step_100/hf_ckpt}"
export LENGTHS="${LENGTHS:-4k 8k 16k 32k 64k}"
# Preserve Slurm's assigned device IDs; the shared launcher sets visibility.
export GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
export MODEL_ABBR="${MODEL_ABBR:-qwen3_06b_no_ttt_58031456_step100}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/results/ruler_native/${MODEL_ABBR}_${RUN_ID}}"
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-$PROJECT_DIR/results/opencompass-data}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

test -r "$CKPT/config.json"
python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "transformers", "opencompass", "mmengine")
           if importlib.util.find_spec(name) is None]
if missing:
    sys.exit("Missing evaluation dependencies: " + ", ".join(missing)
             + "\nInstall the README evaluation dependency in EVAL_ENV: python -m pip install opencompass")
import torch
if not torch.cuda.is_available():
    sys.exit("CUDA GPU required. Submit this script with sbatch, or run in a GPU allocation.")
print(f"Visible GPUs: {torch.cuda.device_count()}; device: {torch.cuda.get_device_name(0)}")
PY

echo "Checkpoint: $CKPT"
echo "RULER lengths: $LENGTHS; $NUM_SAMPLES samples per task, 13 tasks per length, native (no-TTT) baseline"
echo "Results: $OUT_DIR"

NO_TTT_EVAL="$PROJECT_DIR/scripts/qwen3_no_ttt_ruler_eval.py"
python "$NO_TTT_EVAL" prepare

# num_worker/max_num_workers are forced to 1 by qwen3_no_ttt_ruler_eval.py's
# prepare(): the model config carries no TTT state, but generation still runs
# one length at a time on a single GPU here, so only the first assigned GPU is used.
FIRST_GPU="${GPUS%%,*}"
for len in $LENGTHS; do
  cfg="$OUT_DIR/configs/ruler_${len}.py"
  echo "[gen] running native RULER generation for ${len} on GPU ${FIRST_GPU}"
  CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -c "
import inference_model  # noqa: F401 (registers the Qwen3 TTT modeling)
from opencompass.cli.main import main
import sys
sys.argv = ['opencompass', '${cfg}']
main()
"
done

python "$NO_TTT_EVAL" summarize

# Require a complete Full-13 summary at every requested length.
python - "$OUT_DIR/ruler_inplace_summary.json" "$LENGTHS" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    result = json.load(f)
for length in sys.argv[2].split():
    summary = result.get("lengths", {}).get(length, {})
    if summary.get("n_tasks") != 13 or summary.get("missing"):
        sys.exit(f"ERROR: incomplete RULER Full-13 evaluation at {length}; inspect job and shard logs.")
    print(f"RULER {length} Full-13 native score: {summary['native']:.2f}")
print(f"Summary: {sys.argv[1]}")
PY
