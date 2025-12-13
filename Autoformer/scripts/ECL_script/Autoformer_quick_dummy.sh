#!/usr/bin/env bash
set -euo pipefail

# Minimal single run to produce a quick dummy result.
# Keeps epochs to 1 and uses a small pred_len so it finishes fast.

pred_len=24
model_id="ECL_quick_dummy_pl${pred_len}"

echo ">>> Running ${model_id}"
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id "${model_id}" \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len "${pred_len}" \
  --e_layers 1 \
  --d_layers 1 \
  --factor 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'quick_dummy' \
  --itr 1 \
  --train_epochs 1 \
  --patience 1 \
  --batch_size 32 \
  --lr_schedule_mode 0 \
  --warmup_epochs 0 \
  --min_lr 0.0 \
  --ac_temp 1.0 \
  --ac_norm 0 \
  --num_workers 0
