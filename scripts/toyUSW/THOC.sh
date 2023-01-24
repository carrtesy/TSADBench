export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=toyUSW \
exp_id=default \
model=THOC \
model.L2_reg=1.0 \
model.LAMBDA_orth=1.0 \
model.LAMBDA_TSS=1.0