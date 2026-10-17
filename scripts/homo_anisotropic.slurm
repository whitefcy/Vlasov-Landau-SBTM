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
# Score evolution supports kernel (default) and learned transport RHS choices:
#   sbatch scripts/homo_anisotropic.slurm --score_methods score_evolution --score_evolution_rhs kernel
#   sbatch scripts/homo_anisotropic.slurm --score_methods score_evolution --score_evolution_rhs transport
# Sweep ridge values in separate lambda directories, as in homo_BKW.slurm:
#   sbatch scripts/homo_anisotropic.slurm --score_methods score_evolution --score_evolution_regularization 1e-4 1e-3 1e-2 0.1
# Trace a transport pilot and stop before applying an unconverged CG solve:
#   sbatch scripts/homo_anisotropic.slurm --score_methods score_evolution --n_values 100 --score_evolution_rhs transport --score_evolution_diagnostics_every 1 --score_evolution_cg_failure error
# transport defaults to tanh; use --sbtm_activation tanh with kernel to match.
# forward_euler and midpoint are also supported for score evolution.
# No exact-score or transport_analytic mode is available for this example.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the repository root}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

N_VALUES=(100 200 400 800 1600 3200 6400 12800)
SCORE_METHODS=(sbtm)
SEEDS=(42) # Add independent seeds to average curves and final errors.
REGULARIZATION_VALUES=(1e-4)
output_dir="data/homo_anisotropic/sweep_job${SLURM_JOB_ID}_restart${SLURM_RESTART_COUNT:-0}"

# Consume launcher-level overrides so each lambda has its own output directory.
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
# Fit the initial Gaussian score on 12,800 independent samples for every n.
# The sweep runner records failed runs and assembles figures from successes.
# Score evolution uses CG after initialization; SBTM training flags apply only
# to SBTM. The energy-conserving score update uses one solve per time step.
status=0
for regularization in "${REGULARIZATION_VALUES[@]}"; do
  echo "Starting anisotropic sweep with lambda=${regularization}"
  if srun python3 -u experiments/homo_benchmark.py run \
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
  --score_evolution_rhs kernel \
  --score_evolution_regularization "$regularization" \
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
  --output_dir "${output_dir}/lambda_${regularization}" \
  --wandb_mode "${WANDB_MODE:-disabled}" \
  "${extra_args[@]}"; then
    echo "Finished anisotropic sweep with lambda=${regularization}"
  else
    sweep_status=$?
    echo "Anisotropic sweep with lambda=${regularization} failed (exit ${sweep_status}); continuing." >&2
    status=1
  fi
done
exit "$status"
