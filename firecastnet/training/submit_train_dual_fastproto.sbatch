#!/usr/bin/env bash
#SBATCH --job-name=wfproto
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT="/home/saturn/gwgi/gwgi107h/wildfire_data"
CODEDIR="$ROOT/firecastnet"
cd "$CODEDIR"
PY="/home/hpc/gwgi/gwgi107h/.conda/envs/firecastnet_py310/bin/python"
mkdir -p "$CODEDIR/logs" "$CODEDIR/checkpoints"

echo "[env] host=$(hostname) date=$(date) job=${SLURM_JOB_ID}"

# Cheap single-run prototype (128px, 800 samples/epoch, 12 max epochs) testing
# pos_weight annealing + isolation-weighted loss against a matched fast
# baseline -- see train_convlstm_dual_fastproto.py docstring.
VARIANT="${VARIANT:?set VARIANT=baseline or fixed}"
RUN_TAG="${RUN_TAG:-dual_fastproto_${VARIANT}}"
export RUN_TAG

export POS_WEIGHT="${POS_WEIGHT:-100.0}"
export FIRE_HISTORY="${FIRE_HISTORY:-1}"
export FIRE_HISTORY_LAGS="${FIRE_HISTORY_LAGS:-3:4:5}"
export FIRE_HISTORY_DISTANCE="${FIRE_HISTORY_DISTANCE:-1}"
export FUSE="${FUSE:-cross_attn}"
export ATTN_HEADS="${ATTN_HEADS:-4}"

if [ "$VARIANT" = "fixed" ]; then
  export POS_WEIGHT_START="${POS_WEIGHT_START:-100.0}"
  export POS_WEIGHT_END="${POS_WEIGHT_END:-20.0}"
  export POS_WEIGHT_ANNEAL_EPOCHS="${POS_WEIGHT_ANNEAL_EPOCHS:-8}"
  export ISOLATION_GAMMA="${ISOLATION_GAMMA:-2.0}"
  export ISOLATION_KERNEL="${ISOLATION_KERNEL:-9}"
fi

if [ ! -f "$CODEDIR/channel_stats_2015_2018.json" ]; then
  echo "[error] channel_stats_2015_2018.json missing — run compute_channel_stats.py first" >&2
  exit 2
fi

echo "[cfg] VARIANT=$VARIANT RUN_TAG=$RUN_TAG pos_weight_anneal=${POS_WEIGHT_START:-none}->${POS_WEIGHT_END:-none}" \
     "isolation_gamma=${ISOLATION_GAMMA:-0.0}"

# srun otherwise only gets a share of job RAM proportional to its own
# --cpus-per-task (8) out of the job's full allocation (32 cpus / 120G on this
# partition's default policy) -- i.e. 30G, not 120G. Job 1761447 (first run
# with all 6 new features + wind cache: ~1.4G/worker x 8 dataloader workers)
# hit exactly that 30G ceiling and got oom_kill'd within the first epoch.
# Request most of the job's real allocation explicitly.
srun --mem=100G $PY -u "$CODEDIR/train_convlstm_dual_fastproto.py" 2>&1 | tee "$CODEDIR/logs/train_${RUN_TAG}.log"

echo "[done] $(date)"
