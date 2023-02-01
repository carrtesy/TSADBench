export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-UNI \
exp_id=default \
scaler=minmax \
window_size=25 \
model=USAD \
model.latent_dim=40