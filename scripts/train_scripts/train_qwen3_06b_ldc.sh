#!/bin/bash
#SBATCH --job-name=qwen3_06b_base_ttt_ldc
#SBATCH --account=IscrB_MOSAIC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/fmorandi/TTT-NTP
#SBATCH --output=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/qwen3_06b_base_ttt_ldc_%j.out
#SBATCH --error=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/qwen3_06b_base_ttt_ldc_%j.err

set -euo pipefail

source /leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt/bin/activate

PROJ=/leonardo/home/userexternal/fmorandi/TTT-NTP
BASE=/leonardo_work/IscrB_MOSAIC/franci/huggingface/models/Qwen3-0.6B-Base
DATA=${DATA:-/leonardo_scratch/large/userexternal/fmorandi/data_ttt/long-data-collections/data}
OUT=/leonardo_scratch/large/userexternal/fmorandi/checkpoints/ttt/qwen3_06b_base_ttt_ldc_${SLURM_JOB_ID}
STEPS=${STEPS:-100}
SEQ_LEN=${SEQ_LEN:-32768}
# Optional TTT ablation without editing the release yaml, e.g.
#   FOUNDATION='{"ttt_mode":true,...,"ttt_inner_opt":"specnorm"}' sbatch this_script.sh
FOUNDATION_ARGS=()
if [ -n "${FOUNDATION:-}" ]; then FOUNDATION_ARGS=(--model.foundation "$FOUNDATION"); fi

cd "$PROJ"
test -r "$BASE/config.json"
test -d "$DATA" && ls "$DATA"/*.parquet >/dev/null
mkdir -p "$OUT"

export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "Output directory: $OUT"
nvidia-smi

# The YAML enables TTT, FSDP2, mixed precision, and gradient checkpointing.
# Save only at the end of the single epoch, including if the dataset ends early.
# The trainer also exports the final checkpoint to Hugging Face format.

srun --ntasks=1 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    tasks/train_torch.py configs/pretrain/qwen3_0p6b.yaml \
    --model.model_path "$BASE" \
    --model.attn_implementation sdpa \
    "${FOUNDATION_ARGS[@]}" \
    --data.train_path "$DATA" \
    --data.text_keys text \
    --data.num_workers 1 \
    --data.max_seq_len "$SEQ_LEN" \
    --train.output_dir "$OUT" \
    --train.micro_batch_size 1 \
    --train.global_batch_size 64 \
    --train.max_steps "$STEPS" \
    --train.num_train_epochs 1 \
    --train.save_steps 0 \
    --train.save_epochs 1 \
    --train.lr_decay_ratio 1.0 \
    --train.save_hf_weights true \
    --train.save_async false \
    --train.use_wandb false

echo "Training completed. Final checkpoint: $OUT/checkpoints/global_step_*"
echo "Hugging Face model: $OUT/checkpoints/global_step_*/hf_ckpt"
