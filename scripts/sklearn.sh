echo "$1+$2"
echo "base_configs: | model $1 | dataset $2"

python main.py \
  --exp_id $1_$2 \
  --dataset $2 \
  --scaler std \
  $1 \
