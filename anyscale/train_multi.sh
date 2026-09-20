#!/usr/bin/env bash
# Train several variants on one multi-GPU node, one trainer per GPU.
#
#   VARIANTS_JSON='[{"name":"g1","env":{"HEADSTAND_SLAM_N":"12"}}, ...]' \
#   bash scripts/anyscale/train_multi.sh Mjlab-Headstand-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 1000
#
# Variant i runs on GPU i (mjlab's --gpu-ids) with its env knobs exported.
# One uv sync and one warp kernel compile serve all of them. Every variant's
# logs sync to $ANYSCALE_ARTIFACT_STORAGE/microduck/<RUN_NAME>/logs/ every
# 5 minutes and at exit; each variant is its own wandb run.
set -euo pipefail

TASK="$1"; shift
RUN_NAME="${RUN_NAME:-${TASK}-multi-$(date +%Y%m%d-%H%M%S)}"
DEST="${ANYSCALE_ARTIFACT_STORAGE:?}/microduck/${RUN_NAME}"

if ! command -v uv >/dev/null; then
  pip install --quiet --user uv
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/cluster_storage/uv-cache}"
export UV_HTTP_TIMEOUT=600
if [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=offline; fi

mkdir -p logs
mkdir -p logs
nvidia-smi
# Warp compiles its GPU kernels on first use (~8 min at 4096 envs). Keep the
# compiled cache in the artifact bucket, keyed by GPU name, and restore it: a
# job on the same GPU type then starts training within a minute.
GPU_KEY=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9' '_')
WARP_CACHE_REMOTE="${ANYSCALE_ARTIFACT_STORAGE}/microduck/warp-cache/${GPU_KEY}/"
WARP_CACHE_LOCAL="$HOME/.cache/warp"
restore_warp_cache() {
  mkdir -p "$WARP_CACHE_LOCAL"
  case "$WARP_CACHE_REMOTE" in
    s3://*) aws s3 sync "$WARP_CACHE_REMOTE" "$WARP_CACHE_LOCAL" --quiet && echo "warp cache restored for $GPU_KEY" ;;
    gs://*) gcloud storage rsync -r "$WARP_CACHE_REMOTE" "$WARP_CACHE_LOCAL" && echo "warp cache restored for $GPU_KEY" ;;
  esac
}
save_warp_cache() {
  case "$WARP_CACHE_REMOTE" in
    s3://*) aws s3 sync "$WARP_CACHE_LOCAL" "$WARP_CACHE_REMOTE" --quiet ;;
    gs://*) gcloud storage rsync -r "$WARP_CACHE_LOCAL" "$WARP_CACHE_REMOTE" ;;
  esac
}
restore_warp_cache || true
uv sync --frozen
NGPU=$(uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())")
echo "gpus: $NGPU"

sync_logs() {
  case "$DEST" in
    s3://*)  aws s3 sync logs/ "$DEST/logs/" --quiet ;;
    gs://*)  gcloud storage rsync -r logs/ "$DEST/logs/" ;;
  esac
}
( while true; do sleep 300; sync_logs || true; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null || true; sync_logs || true; save_warp_cache || true; echo "checkpoints: $DEST/logs/"' EXIT

# One line per variant: "<index>\t<name>\t<KEY=VAL KEY=VAL ...>"
uv run python - "$NGPU" <<'EOF' > /tmp/variants.tsv
import json, os, sys
ngpu = int(sys.argv[1])
variants = json.loads(os.environ["VARIANTS_JSON"])
assert len(variants) <= ngpu, f"{len(variants)} variants for {ngpu} GPUs"
for i, v in enumerate(variants):
    print(f"{i}\t{v['name']}\t" + " ".join(f"{k}={val}" for k, val in v.get("env", {}).items()))
EOF

( sleep 900; save_warp_cache || true ) &
PIDS=()
while IFS=$'\t' read -r idx name knobs; do
  echo "[$name] gpu $idx knobs: $knobs"
  ( env $knobs uv run train "$TASK" "$@" --gpu-ids "$idx" --agent.run-name "$name" > "logs/train_${name}.log" 2>&1 ) &
  PIDS+=($!)
  sleep 20   # stagger so the warp compile and wandb init do not all collide
done < /tmp/variants.tsv

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
tail -n 5 logs/train_*.log
exit $FAIL
