export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=toyUSW \
exp_id=default \
scaler=minmax \
window_size=200 \
model=USAD \
model.latent_dim=40