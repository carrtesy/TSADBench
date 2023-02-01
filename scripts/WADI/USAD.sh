export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=WADI \
scaler=minmax \
exp_id=default \
window_size=10 \
epochs=70 \
model=USAD \
model.dsr=5 \
model.latent_dim=100