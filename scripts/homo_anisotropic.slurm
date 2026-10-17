#!/bin/bash
#SBATCH --job-name=homo_anisotropic
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%N_%j.out

# Submit from the repository root: mkdir -p logs && sbatch scripts/homo_anisotropic.slurm
# Match homo_BKW.slurm's default SBTM setup, with the anisotropic Gaussian
# initial condition at t=0 and the same four-unit evolution interval.
# The density and score have no analytical solution. Each run and the combined
# sweep write fig5_3.png: two covariance trajectories, final covariance error,
# and estimated entropy decay rate. Only covariance has an analytical reference.
# Arguments override sweep defaults, for example:
#   sbatch scripts/homo_anisotropic.slurm --sbtm_optimizer adam
#   sbatch scripts/homo_anisotropic.slurm --sbtm_n_initial 100000
#   sbatch scripts/homo_anisotropic.slurm --sbtm_div_mode denoised --no-sbtm_use_batches
# Add the blob comparison shown in Figure 5.3:
#   sbatch scripts/homo_anisotropic.slurm --score_methods sbtm blob
# The higher-dimensional example is opt-in:
#   sbatch scripts/homo_anisotropic.slurm --dv 10 --n_values 100 200 400 800 1600 3200 6400 12800 25600
# Supported score methods are sbtm and blob; BKW's exact-score and
# score-evolution options are not implemented by the anisotropic driver.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

N_VALUES=(100 200 400 800 1600 3200 6400 12800)
SCORE_METHODS=(sbtm)
SEEDS=(42) # Add independent seeds to average curves and final errors.
output_dir="data/homo_anisotropic/sweep_job${SLURM_JOB_ID}_restart${SLURM_RESTART_COUNT:-0}"

# Preserve Slurm's CUDA_VISIBLE_DEVICES; do not override it with --gpu.
# Fit the initial Gaussian score on 12,800 independent samples for every n.
# The sweep runner records failed runs and assembles figures from successes.
srun python3 -u experiments/homo_benchmark.py run \
  --example anisotropic \
  --n_values "${N_VALUES[@]}" \
  --score_methods "${SCORE_METHODS[@]}" \
  --seeds "${SEEDS[@]}" \
  --dv 3 \
  --B 0.041666666666666666 \
  --t0 0.0 \
  --final_time 4.0 \
  --dt 0.01 \
  --time_integrator energy_conserving \
  --sbtm_hidden_dims 100 100 \
  --sbtm_n_initial 12800 \
  --sbtm_lr 0.0004 \
  --sbtm_optimizer adamw \
  --sbtm_weight_decay 0.0001 \
  --no-sbtm_adaptive \
  --sbtm_training_stages all_stages \
  --sbtm_div_mode exact \
  --sbtm_denoising_alpha 0.4 \
  --sbtm_use_batches \
  --sbtm_num_batch_steps 100 \
  --snapshot_every 100 \
  --output_dir "$output_dir" \
  --wandb_mode "${WANDB_MODE:-disabled}" \
  "$@"
