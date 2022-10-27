echo "$2+$3 @gpu $1"
echo "config: model $2 | dataset $3 | window_size $4 | epochs $5 | latent dim $6 | dsr $7"

export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --exp_id $2_$3_ws_$4_ep_$5_lt_$6_dsr_$7 \
  --dataset $3 \
  --window_size $4 \
  --epochs $5 \
  $2 \
  --latent_dim $6 \
  --anomaly_reduction_mode mean \
  --dsr $7