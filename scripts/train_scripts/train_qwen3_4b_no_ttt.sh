#!/bin/bash
# Submit with: sbatch train_qwen3_4b_base_no_ttt.sh
# Optional short run: sbatch --export=ALL,STEPS=10,SEQ_LEN=8192 train_qwen3_4b_base_no_ttt.sh
# The final checkpoint is saved on normal completion, not on timeout/cancellation.
# Four GPUs require distributed workers; no other shell launcher is used.
#SBATCH --job-name=qwen3_4b_base_no_ttt
#SBATCH --account=IscrB_MOSAIC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/fmorandi/TTT-NTP
#SBATCH --output=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/qwen3_4b_base_no_ttt_%j.out
#SBATCH --error=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/qwen3_4b_base_no_ttt_%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"

source /leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt/bin/activate

PROJ=/leonardo/home/userexternal/fmorandi/TTT-NTP
BASE=/leonardo_work/IscrB_MOSAIC/franci/huggingface/models/Qwen3-4B-Base
DATA=/leonardo_work/IscrB_MOSAIC/fwp/prolong/arxiv.jsonl
OUT="/leonardo_scratch/large/userexternal/fmorandi/checkpoints/fwp/qwen3_4b_base_no_ttt_${SLURM_JOB_ID}"
STEPS=${STEPS:-1000}
SEQ_LEN=${SEQ_LEN:-32768}

cd "$PROJ"
test -r "$BASE/config.json"
test -r "$DATA"
mkdir -p "$OUT"

export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

echo "Output directory: $OUT"
nvidia-smi

# Retain the YAML training recipe, but disable TTT for ordinary continual pretraining.
# FSDP2, mixed precision, and gradient checkpointing remain enabled.
# Save only at the end of the single epoch, including if the dataset ends early.
# The trainer also exports the final checkpoint to Hugging Face format.

srun --ntasks=1 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=4 \
    tasks/train_torch.py configs/pretrain/qwen3_4b.yaml \
    --model.model_path "$BASE" \
    --model.attn_implementation sdpa \
    --model.foundation '{"ttt_mode":false,"ttt_layers":[],"ttt_proj":false}' \
    --data.train_path "$DATA" \
    --data.text_keys content_split \
    --data.num_workers 1 \
    --data.max_seq_len "$SEQ_LEN" \
    --train.output_dir "$OUT" \
    --train.micro_batch_size 1 \
    --train.global_batch_size 64 \
    --train.max_steps "$STEPS" \
    --train.num_train_epochs 1 \
    --train.save_steps 0 \
    --train.save_epochs 1 \
    --train.save_hf_weights true \
    --train.save_async false \
    --train.use_wandb false

echo "Training completed. Final checkpoint: $OUT/checkpoints/global_step_*"
echo "Hugging Face model: $OUT/checkpoints/global_step_*/hf_ckpt"
