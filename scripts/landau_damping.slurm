#!/bin/bash
#SBATCH --job-name=landau_damping
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=72:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x_%N_%A_%a.out

DV=3
M=100
ALPHA=0.1
FINAL_TIME=15.0
TIME_INTEGRATOR=energy_conserving

N_VALUES=(500000)
SCORE_METHODS=(scaled_blob)
C_VALUES=(0.05 0.4)

tid=$SLURM_ARRAY_TASK_ID

num_c=${#C_VALUES[@]}
num_sm=${#SCORE_METHODS[@]}

idx_c=$((tid % num_c))
rem=$((tid / num_c))
idx_sm=$((rem % num_sm))
idx_n=$((rem / num_sm))

C=${C_VALUES[$idx_c]}
score_method=${SCORE_METHODS[$idx_sm]}
n=${N_VALUES[$idx_n]}

source /gscratch/amath/vilin/conda/etc/profile.d/conda.sh
conda activate vlsbtm

srun python3 experiments/vlasov-landau-damping.py \
  --n "$n" \
  --M "$M" \
  --dt 0.02 \
  --dv "$DV" \
  --C "$C" \
  --time_integrator "$TIME_INTEGRATOR" \
  --score_method "$score_method" \
  --wandb_project vlasov_landau_damping \
  --wandb_run_name "alpha${ALPHA}_T${FINAL_TIME}_n${n}_M${M}_dt0.02_dv${DV}_C${C}_${score_method}_${TIME_INTEGRATOR}_100_steps" \
  --alpha "$ALPHA" \
  --final_time "$FINAL_TIME"
