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
# Default: score evolution with energy-conserving particle time stepping.
# The score retains its first-order parameter update from Latex/VML equations.tex.
# Arguments override sweep defaults, for example:
#   sbatch scripts/homo_BKW.slurm --sbtm_optimizer adam
#   sbatch scripts/homo_BKW.slurm --sbtm_n_initial 100000
#   sbatch scripts/homo_BKW.slurm --time_integrator midpoint
# Sweep lambda values in one job (a single value also works):
#   sbatch scripts/homo_BKW.slurm --score_evolution_regularization 1e-4 1e-3 1e-2 0.1
# Each lambda gets its own subdirectory and figures under --output_dir.
# Select the score derivative R (kernel is the default):
#   sbatch scripts/homo_BKW.slurm --score_evolution_rhs kernel
#   sbatch scripts/homo_BKW.slurm --score_evolution_rhs transport
# For a transport pilot with stronger ridge and strict CG checks:
#   bash scripts/retry_homo_BKW_transport.sh
# Stop before applying an unconverged normal-equation solve:
#   sbatch scripts/homo_BKW.slurm --score_evolution_rhs transport --score_evolution_cg_failure error
# Transport defaults to a smooth tanh network; use --sbtm_activation tanh with
# kernel too when comparing the two RHS choices using the same activation.
# Compare methods using a common integrator:
#   sbatch scripts/homo_BKW.slurm --score_methods sbtm blob score_evolution --time_integrator forward_euler
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

# One job attempts the particle-count/seed sweep for each lambda, skipping
# failed runs, then assembles all six Figure 5.2 panels per lambda from successes.
# See each lambda's sweep_summary.json for statuses.
N_VALUES=(100 200 400 800 1600 3200 6400 12800)
SCORE_METHODS=(score_evolution)
SEEDS=(42) # Add independent seeds to average curves and final errors.
REGULARIZATION_VALUES=(1e-4 1e-3 1e-2 1e-1)
output_dir="data/homo_BKW/sweep_job${SLURM_JOB_ID}_restart${SLURM_RESTART_COUNT:-0}"

# Consume launcher-level overrides so forwarded arguments cannot replace the
# current lambda or make different lambdas write to the same output directory.
extra_args=()
while (( $# )); do
  case "$1" in
    --score_evolution_regularization)
      shift
      REGULARIZATION_VALUES=()
      while (( $# )) && [[ "$1" != --* ]]; do
        REGULARIZATION_VALUES+=("$1")
        shift
      done
      if (( ${#REGULARIZATION_VALUES[@]} == 0 )); then
        echo "--score_evolution_regularization requires at least one lambda" >&2
        exit 2
      fi
      ;;
    --score_evolution_regularization=*)
      REGULARIZATION_VALUES=("${1#*=}")
      shift
      ;;
    --output_dir)
      if (( $# < 2 )) || [[ "$2" == --* ]]; then
        echo "--output_dir requires a directory" >&2
        exit 2
      fi
      output_dir="$2"
      shift 2
      ;;
    --output_dir=*)
      output_dir="${1#*=}"
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

# Preserve Slurm's CUDA_VISIBLE_DEVICES; do not override it with --gpu.
# AdamW fits the initial score on 12,800 independent samples for every n by
# default; --sbtm_optimizer adam selects Adam without weight decay.
# Score evolution then uses one CG solve per step, with the old network reused
# at the energy-conserving velocity stages; sbtm training flags below apply only
# when SBTM is requested through --score_methods.
status=0
for regularization in "${REGULARIZATION_VALUES[@]}"; do
  echo "Starting BKW sweep with lambda=${regularization}"
  if srun python3 -u experiments/homo_benchmark.py run \
  --example bkw \
  --n_values "${N_VALUES[@]}" \
  --score_methods "${SCORE_METHODS[@]}" \
  --seeds "${SEEDS[@]}" \
  --dv 3 \
  --B 0.041666666666666666 \
  --t0 5.5 \
  --final_time 9.5 \
  --dt 0.01 \
  --time_integrator energy_conserving \
  --score_evolution_rhs kernel \
  --score_evolution_regularization "$regularization" \
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
  --output_dir "${output_dir}/lambda_${regularization}" \
  --wandb_mode "${WANDB_MODE:-disabled}" \
  "${extra_args[@]}"; then
    echo "Finished BKW sweep with lambda=${regularization}"
  else
    sweep_status=$?
    echo "BKW sweep with lambda=${regularization} failed (exit ${sweep_status}); continuing." >&2
    status=1
  fi
done
exit "$status"
