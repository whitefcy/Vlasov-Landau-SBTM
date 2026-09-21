#!/bin/bash
#SBATCH --job-name=homo_BKW
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%N_%j.out

# Submit from the repository root: mkdir -p logs && sbatch scripts/homo_BKW.slurm
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

# One job runs the full sweep, then assembles all six Figure 5.2 panels.
N_VALUES=(100 200 400 800 1600 3200 6400 12800)
SCORE_METHODS=(sbtm blob)
SEEDS=(42) # Add independent seeds to average curves and final errors.
output_dir="data/homo_BKW/sweep_job${SLURM_JOB_ID}_restart${SLURM_RESTART_COUNT:-0}"

# Preserve Slurm's CUDA_VISIBLE_DEVICES; do not override it with --gpu.
srun python3 -u experiments/homo_benchmark.py run \
  --example bkw \
  --n_values "${N_VALUES[@]}" \
  --score_methods "${SCORE_METHODS[@]}" \
  --seeds "${SEEDS[@]}" \
  --dv 3 \
  --B 0.041666666666666666 \
  --t0 5.5 \
  --final_time 9.5 \
  --dt 0.01 \
  --sbtm_hidden_dims 100 100 \
  --sbtm_lr 0.0004 \
  --sbtm_adaptive \
  --sbtm_div_mode exact \
  --sbtm_num_batch_steps 100 \
  --density_every 100 \
  --snapshot_every 100 \
  --output_dir "$output_dir" \
  --wandb_mode "${WANDB_MODE:-disabled}"
