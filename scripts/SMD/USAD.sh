export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=SMD \
exp_id=default \
model=USAD \
model.latent_dim=40