#!/bin/bash
#SBATCH --job-name=homo_anisotropic
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%N_%A_%a.out

# Example 5.2 in d=3 and d=10. Each task writes the four-panel comparison.
# For Figure 5.3's d=3 alone: sbatch --array=0 scripts/homo_anisotropic.slurm
# Submit from the repository root after creating logs/.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

DV_VALUES=(3 10)
tid=${SLURM_ARRAY_TASK_ID:-0}
if ((tid < 0 || tid >= ${#DV_VALUES[@]})); then
  echo "Array task $tid is outside the configured dimensions" >&2
  exit 1
fi
DV=${DV_VALUES[$tid]}
N_VALUES=(100 200 400 800 1600 3200 6400 12800)
if ((DV == 10)); then
  N_VALUES+=(25600)
fi
SCORE_METHODS=(sbtm blob)
SEEDS=(42)
output_dir="data/homo_anisotropic/sweep_d${DV}_job${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${tid}_restart${SLURM_RESTART_COUNT:-0}"

srun python3 -u experiments/homo_benchmark.py run \
  --example anisotropic \
  --n_values "${N_VALUES[@]}" \
  --score_methods "${SCORE_METHODS[@]}" \
  --seeds "${SEEDS[@]}" \
  --dv "$DV" \
  --B 0.041666666666666666 \
  --final_time 4.0 \
  --dt 0.01 \
  --time_integrator energy_conserving \
  --sbtm_hidden_dims 100 \
  --sbtm_lr 0.0004 \
  --no-sbtm_adaptive \
  --sbtm_training_stages stage0_only \
  --sbtm_div_mode exact \
  --sbtm_num_batch_steps 100 \
  --snapshot_every 100 \
  --output_dir "$output_dir" \
  --wandb_mode "${WANDB_MODE:-disabled}"
