echo "$2+$3 @gpu $1"
echo "base_configs: | model $2 | dataset $3 | window_size (=stride) $4 | epochs $5 |"
echo "model_configs: | anomaly_ratio $6 | k $7 |"
export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --exp_id $2_$3_ws_$4_ep_$5_lt_$6_dsr_$7 \
  --dataset $3 \
  --window_size $4 \
  --stride $4 \
  --epochs $5 \
  --scaler std \
  $2 \
  --anomaly_ratio $6 \
  --k $7
