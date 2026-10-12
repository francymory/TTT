#!/usr/bin/env bash
# Submit: sbatch eval_qwen3_06b_ttt_ruler.sh
# Quick check: NUM_SAMPLES=2 sbatch eval_qwen3_06b_ttt_ruler.sh
# Uses closed-form TTT evaluation: all 13 RULER tasks from 4K through 32K.
#SBATCH --job-name=qwen3_06b_ttt_ruler32k
#SBATCH --account=IscrB_MOSAIC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/fmorandi/TTT-NTP
#SBATCH --output=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/eval_qwen3_06b_ttt_ruler_ldc_%j.out
#SBATCH --error=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/eval_qwen3_06b_ttt_ruler_ldc_%j.err

set -euo pipefail

source "${EVAL_ENV:-/leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt}/bin/activate"
export PROJECT_DIR=/leonardo/home/userexternal/fmorandi/TTT-NTP
cd "$PROJECT_DIR"

export CKPT="${CKPT:-/leonardo_scratch/large/userexternal/fmorandi/checkpoints/ttt/qwen3_06b_base_ttt_ldc_59100405/checkpoints/global_step_100/hf_ckpt}"
export LENGTHS="${LENGTHS:-4k 8k 16k 32k}"
# Preserve Slurm's assigned device IDs; the shared launcher sets visibility.
export GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
# Conservative batch sizes for a single GPU at each context length.
export BATCH_4k="${BATCH_4k:-8}"
export BATCH_8k="${BATCH_8k:-4}"
export BATCH_16k="${BATCH_16k:-2}"
export BATCH_32k="${BATCH_32k:-1}"
export SHARDS_PER_GPU="${SHARDS_PER_GPU:-1}"
export MODEL_ABBR="${MODEL_ABBR:-qwen3_06b_base_ttt_57998376_step100}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/results/cf_ruler/${MODEL_ABBR}_${RUN_ID}}"
# Keep generated prompts and baseline predictions for inspection/reuse.
export PRED_ROOT="${PRED_ROOT:-$OUT_DIR/predictions}"
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
echo "RULER lengths: $LENGTHS; $NUM_SAMPLES samples per task, 13 tasks per length, closed-form TTT"
echo "Results: $OUT_DIR"
bash "$PROJECT_DIR/eval_cf_ruler.sh"

# The shared launcher can skip failed shards without a nonzero exit status.
# Require a complete Full-13 summary at every requested length.
python - "$OUT_DIR/ruler_cf_summary.json" "$LENGTHS" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    result = json.load(f)
for length in sys.argv[2].split():
    summary = result.get("lengths", {}).get(length, {})
    if summary.get("n_tasks") != 13 or summary.get("missing"):
        sys.exit(f"ERROR: incomplete RULER Full-13 evaluation at {length}; inspect job and shard logs.")
    print(f"RULER {length} Full-13 closed-form score: {summary['patch']:.2f}")
average = result.get("full13_avg", {}).get("patch")
if average is not None:
    print(f"RULER 4k-32k Full-13 average: {average:.2f}")
print(f"RULER results JSON: {sys.argv[1]}")
PY
