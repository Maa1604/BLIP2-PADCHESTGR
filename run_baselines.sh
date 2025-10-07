#!/usr/bin/env bash
set -euo pipefail

# === Ajusta estos valores si quieres ===
CHECKPOINT="fatehmujtaba/blip2-opt-2.7b-for-Chest-Xray"
EPOCHS=1
BS_TRAIN=4
BS_EVAL=4
LR=5e-6
SEED=1337

run_es_baseline() {
  OUTDIR="EXPERIMENTS/BLIP2_PAD_BASE_es"
  python -m train.train_blip2_padchestgr \
    --lang es \
    --checkpoint "$CHECKPOINT" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BS_TRAIN" \
    --per_device_eval_batch_size "$BS_EVAL" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$OUTDIR"
}

run_en_baseline() {
  OUTDIR="EXPERIMENTS/BLIP2_PAD_BASE_en"
  python -m train.train_blip2_padchestgr \
    --lang en \
    --checkpoint "$CHECKPOINT" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BS_TRAIN" \
    --per_device_eval_batch_size "$BS_EVAL" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$OUTDIR"
}

usage() {
  echo "Usage: $0 {es|en|all}"
}

case "${1:-all}" in
  es)  run_es_baseline ;;
  en)  run_en_baseline ;;
  all) run_es_baseline; run_en_baseline ;;
  *) usage; exit 1 ;;
esac
