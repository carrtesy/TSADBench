echo "$2+$3 @gpu $1"
echo "base_configs: | model $2 | dataset $3 | window_size $4 | epochs $5 |"
echo "model_configs: | latent dim $6"

export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --lr 1e-04 \
  --exp_id $2_$3_ws_$4_ep_$5_lt_$6 \
  --dataset $3 \
  --window_size $4 \
  --epochs $5 \
  --scaler std \
  --anomaly_reduction_mode mean \
  $2 \
  --latent_dim $6 \
