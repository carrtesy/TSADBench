export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=toyUSW \
window_size=100 \
exp_id=default \
model=LSTMEncDec