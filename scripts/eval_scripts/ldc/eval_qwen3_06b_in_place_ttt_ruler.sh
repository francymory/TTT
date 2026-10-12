#!/usr/bin/env bash
# Fill CKPT below, or pass it at submission: CKPT=/absolute/path/hf_ckpt sbatch $0
#SBATCH --job-name=eval_qwen3_06b_inplace_ttt_ldc
#SBATCH --account=IscrB_MOSAIC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/fmorandi/TTT-NTP
#SBATCH --output=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/ldc/eval_qwen3_06b_inplace_ttt_ldc_%j.out
#SBATCH --error=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/ldc/eval_qwen3_06b_inplace_ttt_ldc_%j.err

set -euo pipefail
source "${EVAL_ENV:-/leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt}/bin/activate"

PROJECT_DIR=/leonardo/home/userexternal/fmorandi/TTT-NTP
CKPT="${CKPT:-/leonardo_scratch/large/userexternal/fmorandi/checkpoints/ttt/qwen3_06b_inplace_ttt_ldc_59220960/checkpoints/global_step_100/hf_ckpt}"  # TODO: /absolute/path/to/checkpoint/hf_ckpt
: "${CKPT:?Set CKPT to an absolute Hugging Face checkpoint directory.}"

export PROJECT_DIR CKPT
export LENGTHS="${LENGTHS:-4k 8k 16k 32k 64k}"
export GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
# Identical schedule across all LDC methods; override only for memory limits.
export BATCH_4k="${BATCH_4k:-8}"
export BATCH_8k="${BATCH_8k:-4}"
export BATCH_16k="${BATCH_16k:-2}"
export BATCH_32k="${BATCH_32k:-1}"
export BATCH_64k="${BATCH_64k:-1}"
export SHARDS_PER_GPU="${SHARDS_PER_GPU:-1}"
export MODEL_ABBR="${MODEL_ABBR:-qwen3_06b_inplace_ttt}"
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUT_DIR="${OUT_DIR:-$PROJECT_DIR/results/cf_ruler/${MODEL_ABBR}_${RUN_ID}}"
export PRED_ROOT="${PRED_ROOT:-$OUT_DIR/predictions}"
export COMPASS_DATA_CACHE="${COMPASS_DATA_CACHE:-$PROJECT_DIR/results/opencompass-data}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1

cd "$PROJECT_DIR"
test -r "$CKPT/config.json"
echo "method=in-place-ttt; checkpoint=$CKPT; results=$OUT_DIR"
bash "$PROJECT_DIR/eval_cf_ruler.sh"

python - "$OUT_DIR/ruler_cf_summary.json" "$LENGTHS" <<'PY2'
import json, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
for length in sys.argv[2].split():
    summary = result.get("lengths", {}).get(length, {})
    if summary.get("n_tasks") != 13 or summary.get("missing"):
        sys.exit(f"ERROR: incomplete RULER Full-13 evaluation at {length}")
    print(f"RULER {length} Full-13 CF score: {summary['patch']:.2f}")
print(f"Summary: {sys.argv[1]}")
PY2
