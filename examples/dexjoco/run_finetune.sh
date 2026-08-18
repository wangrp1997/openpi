#!/usr/bin/env bash
set -euo pipefail

cd /home/wangrenpeng/openpi

CONFIG=${CONFIG:-pi05_dexjoco_lora}
EXP_NAME=${EXP_NAME:-bimanual-insert-lora-v1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

if [[ ! -f assets/pi05_dexjoco_lora/dexjoco/bimanual_assembly/norm_stats.json ]]; then
  uv run scripts/compute_norm_stats.py --config-name pi05_dexjoco_lora
fi

uv run scripts/train.py "$CONFIG" --exp-name "$EXP_NAME" --overwrite "$@"
