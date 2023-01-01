echo "$2+$3 @gpu $1"
echo "base_configs: | model $2 | dataset $3 | lr $4 | ws $5"
echo "model_configs: | L2_reg $6 | LAMBDA_orth $7 | LAMBDA_TSS $8 | hidden_dim $9"

export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --epochs 30 \
  --dataset $3 \
  --lr $4 \
  --window_size $5 \
  --exp_id $2_$3_lr_$4_ws_$5_l2_$6_orth_$7_tss_$8_hidden_$9 \
  $2 \
  --L2_reg $6 \
  --LAMBDA_orth $7 \
  --LAMBDA_TSS $8 \
  --hidden_dim $9