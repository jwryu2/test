# export CUDA_VISIBLE_DEVICES=1

# Combined 3 (LR modes) x 4 (AC variants) x 2 (pred_len) = 24 runs.

pred_lens=(96 192)

# lr modes: name mode warmup min_lr
lr_modes=(
  "OFF 0 0 0.0"
  "COSINE 1 0 1e-6"
  "WARMUP_COSINE 2 3 1e-6"
)

# autocorr variants: name ac_temp ac_norm
ac_variants=(
  "BASE 1.0 0"
  "TEMP 2.0 0"
  "NORM 1.0 1"
  "TEMP_NORM 2.0 1"
)

for pl in "${pred_lens[@]}"; do
  for lr_entry in "${lr_modes[@]}"; do
    read -r lr_name lr_mode warmup min_lr <<<"${lr_entry}"
    for ac_entry in "${ac_variants[@]}"; do
      read -r ac_name ac_temp ac_norm <<<"${ac_entry}"
      model_id="ECL_96_${pl}_LR_${lr_name}_${ac_name}"
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
        --pred_len "${pl}" \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --des 'Exp' \
        --itr 1 \
        --lr_schedule_mode "${lr_mode}" \
        --warmup_epochs "${warmup}" \
        --min_lr "${min_lr}" \
        --ac_temp "${ac_temp}" \
        --ac_norm "${ac_norm}"
    done
  done
done

python - <<'PY'
import numpy as np
import os

base = dict(
    model="Autoformer",
    data="custom",
    features="M",
    seq_len=96,
    label_len=48,
    pred_len=96,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,
    factor=3,
    embed="timeF",
    distil=True,
    des="Exp",
    pred_len=None,  # filled per run
)

pred_lens = [96, 192]
lr_modes = [
    dict(name="OFF"),
    dict(name="COSINE"),
    dict(name="WARMUP_COSINE"),
]
ac_variants = ["BASE", "TEMP", "NORM", "TEMP_NORM"]

def fmt_setting(params, idx=0):
    return "{model_id}_{model}_{data}_ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}_eb{embed}_dt{distil}_{des}_{idx}".format(
        idx=idx, **params
    )

print("\n====== 24-Run Ablation Summary (LR x AC x pred_len) ======")
for pl in pred_lens:
    for lr in lr_modes:
        for ac in ac_variants:
            params = {
                **base,
                "pred_len": pl,
                "model_id": f"ECL_96_{pl}_LR_{lr['name']}_{ac}",
            }
            setting = fmt_setting(params, idx=0)
            path = os.path.join("results", setting, "metrics.npy")
            if not os.path.isfile(path):
                print(f"{setting}: metrics file not found at {path}")
                continue
            metrics = np.load(path)
            mae, mse = metrics[0], metrics[1]
            print(f"{setting}: mae={mae:.6f}, mse={mse:.6f}")
print("==========================================================")
PY
