#!/usr/bin/env bash
# Train one task on an Anyscale job and keep the checkpoints in the cloud's
# artifact bucket. Mirrors what scripts/hf/ does for Hugging Face Jobs.
#
#   bash scripts/anyscale/train.sh Mjlab-Headstand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 3000
#
# Runs on the job's head node, which the job config makes a GPU instance (a
# single-process trainer, so the GPU has to be where the entrypoint runs).
# Checkpoints go to $ANYSCALE_ARTIFACT_STORAGE/microduck/<run-name>/ every
# 5 minutes and once more at exit, so a killed job still leaves its models.
# wandb: online if WANDB_API_KEY is set (job env_vars), offline otherwise.
set -euo pipefail

TASK="$1"; shift
RUN_NAME="${RUN_NAME:-${TASK}-$(date +%Y%m%d-%H%M%S)}"
DEST="${ANYSCALE_ARTIFACT_STORAGE:?ANYSCALE_ARTIFACT_STORAGE is set by Anyscale on every job}/microduck/${RUN_NAME}"

# Anyscale's base image has no uv; the repo needs it (and its own Python 3.12).
if ! command -v uv >/dev/null; then
  pip install --quiet --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/cluster_storage/uv-cache}"
export UV_HTTP_TIMEOUT=600
if [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=offline; fi

nvidia-smi
uv sync --frozen
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU on this node'; print('cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# Cloud-agnostic sync: the bucket scheme says which CLI to use.
sync_logs() {
  case "$DEST" in
    s3://*)  aws s3 sync logs/ "$DEST/logs/" --quiet ;;
    gs://*)  gcloud storage rsync -r logs/ "$DEST/logs/" ;;
    *)       echo "unknown artifact storage scheme: $DEST" ;;
  esac
}
( while true; do sleep 300; sync_logs || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_logs || true; echo "checkpoints: $DEST/logs/"' EXIT

uv run train "$TASK" "$@"
