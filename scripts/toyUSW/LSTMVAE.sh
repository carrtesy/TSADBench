export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=toyUSW \
exp_id=default \
model=LSTMVAE \
model.hidden_dim=32 \
model.z_dim=3 \
model.n_layers=2 \
model.beta=1e-04