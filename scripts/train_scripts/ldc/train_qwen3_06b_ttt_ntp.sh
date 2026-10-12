#!/usr/bin/env bash
#SBATCH --job-name=qwen3_06b_ttt_ntp_ldc
#SBATCH --account=IscrB_MOSAIC
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/fmorandi/TTT-NTP
#SBATCH --output=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/ldc/qwen3_06b_ttt_ntp_ldc_%j.out
#SBATCH --error=/leonardo/home/userexternal/fmorandi/TTT-NTP/A_logs/ldc/qwen3_06b_ttt_ntp_ldc_%j.err

set -euo pipefail
source /leonardo_work/IscrB_MOSAIC/franci/envs/inplacettt/bin/activate

PROJ=/leonardo/home/userexternal/fmorandi/TTT-NTP
BASE=${BASE:-/leonardo_work/IscrB_MOSAIC/franci/huggingface/models/Qwen3-0.6B-Base}
DATA=${DATA:-/leonardo_scratch/large/userexternal/fmorandi/data_ttt/long-data-collections/data}
OUT=${OUT:-/leonardo_scratch/large/userexternal/fmorandi/checkpoints/ttt/qwen3_06b_ttt_ntp_ldc_${SLURM_JOB_ID}}
STEPS=${STEPS:-100}
SEQ_LEN=${SEQ_LEN:-32768}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
# The installed flash-attn extension requires GLIBC_2.32, while Leonardo nodes
# provide GLIBC_2.28. PyTorch SDPA uses the compatible built-in CUDA backend.
# Override with ATTN_IMPL=flash_attention_2 after installing a compatible build.
ATTN_IMPL=${ATTN_IMPL:-sdpa}

cd "$PROJ"
test -r "$BASE/config.json"
test -d "$DATA"
mkdir -p "$OUT"
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1

echo "method=ttt-ntp output=$OUT"
srun --ntasks=1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  tasks/train_torch.py configs/pretrain/qwen3_0p6b.yaml \
  --method ttt-ntp \
  --model.model_path "$BASE" \
  --model.attn_implementation "$ATTN_IMPL" \
  --data.train_path "$DATA" \
  --data.text_keys text \
  --data.num_workers 1 \
  --data.max_seq_len "$SEQ_LEN" \
  --train.output_dir "$OUT" \
  --train.micro_batch_size "$MICRO_BATCH_SIZE" \
  --train.global_batch_size "$GLOBAL_BATCH_SIZE" \
  --train.save_checkpoint_path "$OUT" \
  --train.max_steps "$STEPS" \
  --train.num_train_epochs 1 \
  --train.save_steps 0 \
  --train.save_epochs 1 \
  --train.lr_decay_ratio 1.0 \
  --train.save_hf_weights true \
  --train.save_async false \
  --train.use_wandb false

echo "HF checkpoint: $OUT/checkpoints/global_step_*/hf_ckpt"
