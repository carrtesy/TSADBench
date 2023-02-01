export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=SMAP \
exp_id=default \
model=LSTMEncDec