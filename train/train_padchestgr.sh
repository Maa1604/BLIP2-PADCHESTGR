#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Train BLIP-2 or Region-BLIP-2 on PadChest-GR
# ============================================================

PYTHON=${PYTHON:-python}

# Experiment name (change as needed)
EXP_NAME="blip2_padchestgr_5"    # e.g. "blip2_base_padchestgr"

# Grounded mode: true = RegionBlip2, false = plain BLIP-2
GROUNDED=false

# Training hyperparameters
EPOCHS=30
BATCH_SIZE=4
ACCUM=32
LR=3e-4
NUM_BEAMS=2
MAX_NEW_TOKENS=128    #41 words is the longest in padchestgr
SAVE_EVERY=0

# ============================================================
# Run
# ============================================================
# Note: train.py will import paths.DICT_CSV_PADCHESTGR_PATH
# to load the correct train/validation CSVs.

EXTRA_FLAGS=""
if [ "$GROUNDED" = true ]; then
  EXTRA_FLAGS="--grounded"
fi

# Optional: select GPU
export CUDA_VISIBLE_DEVICES=0

${PYTHON} train_blip2_padchestgr.py \
  --exp_name "${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --accumulate_grad_batches "${ACCUM}" \
  --lr "${LR}" \
  --num_beams "${NUM_BEAMS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --save_every "${SAVE_EVERY}" \
  ${EXTRA_FLAGS}
