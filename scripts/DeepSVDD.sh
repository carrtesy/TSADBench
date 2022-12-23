echo "$2+$3 @gpu $1"
echo "base_configs: | model $2 | dataset $3 |"

export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --exp_id $2_$3 \
  --dataset $3 \
  --scaler std \
  $2