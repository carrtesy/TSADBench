echo "$2+$3 @gpu $1"
echo "base_configs: | model $2 | dataset $3 | lr $4"
echo "model_configs: | hd $5 | zd $6 | nl $7 | beta $8 |"
export CUDA_VISIBLE_DEVICES=$1;
python main.py \
  --exp_id $2_$3_lr_$4_hd_$5_zd_$6_nl_$7_beta_$8 \
  --dataset $3\
  --lr $4\
  $2 \
  --hidden_dim $5 \
  --z_dim $6 \
  --n_layers $7\
  --beta $8