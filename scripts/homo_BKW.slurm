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
# Default: the first-order score-evolution method in Latex/VML equations.tex.
# Arguments override sweep defaults, for example:
#   sbatch scripts/homo_BKW.slurm --sbtm_optimizer adam
#   sbatch scripts/homo_BKW.slurm --sbtm_n_initial 100000
#   sbatch scripts/homo_BKW.slurm --time_integrator midpoint
# Compare methods using a common integrator:
#   sbatch scripts/homo_BKW.slurm --score_methods sbtm blob score_evolution --time_integrator forward_euler
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

# One job runs the full sweep, then assembles all six Figure 5.2 panels.
N_VALUES=(100 200 400 800 1600 3200 6400 12800)
SCORE_METHODS=(score_evolution)
SEEDS=(42) # Add independent seeds to average curves and final errors.
output_dir="data/homo_BKW/sweep_job${SLURM_JOB_ID}_restart${SLURM_RESTART_COUNT:-0}"

# Preserve Slurm's CUDA_VISIBLE_DEVICES; do not override it with --gpu.
# AdamW fits the initial score on 12,800 independent samples for every n by
# default; --sbtm_optimizer adam selects Adam without weight decay.
# Score evolution then uses CG; sbtm evolution-training flags below apply only
# when SBTM is requested through --score_methods.
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
  --time_integrator forward_euler \
  --sbtm_hidden_dims 100 100 \
  --sbtm_n_initial 12800 \
  --sbtm_lr 0.0004 \
  --sbtm_optimizer adamw \
  --sbtm_weight_decay 0.0001 \
  --no-sbtm_adaptive \
  --sbtm_training_stages stage0_only \
  --sbtm_div_mode exact \
  --sbtm_num_batch_steps 100 \
  --density_every 100 \
  --snapshot_every 100 \
  --output_dir "$output_dir" \
  --wandb_mode "${WANDB_MODE:-disabled}" \
  "$@"
