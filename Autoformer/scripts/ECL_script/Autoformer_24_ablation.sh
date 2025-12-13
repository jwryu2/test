# export CUDA_VISIBLE_DEVICES=1

# Learning-rate schedule ablation on Electricity (ECL) dataset.
# Modes: 0=off, 1=cosine cooldown, 2=warmup+cosine.

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_LR_OFF \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lr_schedule_mode 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_LR_COSINE \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lr_schedule_mode 1 \
  --min_lr 1e-6

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96_LR_WARMUP_COSINE \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --lr_schedule_mode 2 \
  --warmup_epochs 3 \
  --min_lr 1e-6

python - <<'PY'
import numpy as np
import os

settings = [
    "ECL_96_96_LR_OFF",
    "ECL_96_96_LR_COSINE",
    "ECL_96_96_LR_WARMUP_COSINE",
]

print("\n====== LR Schedule Ablation Summary ======")
for setting in settings:
    path = os.path.join("results", setting, "metrics.npy")
    if not os.path.isfile(path):
        print(f"{setting}: metrics file not found at {path}")
        continue
    metrics = np.load(path)
    mae, mse = metrics[0], metrics[1]
    print(f"{setting}: mae={mae:.6f}, mse={mse:.6f}")
print("==========================================")
PY
