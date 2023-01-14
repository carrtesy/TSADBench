export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-MUL \
exp_id=default \
model=OmniAnomaly \
model.hidden_dim=128 \
model.z_dim=128 \
model.dense_dim=128 \
model.beta=1e-06