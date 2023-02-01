export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=PSM \
scaler=minmax \
exp_id=default \
model=USAD