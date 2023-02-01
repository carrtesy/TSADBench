export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=SMD \
exp_id=default \
scaler=minmax \
window_size=5 \
epochs=250 \
model=USAD \
model.dsr=5 \
model.latent_dim=38