export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-MUL \
exp_id=default \
window_size=25 \
model=LSTMEncDec