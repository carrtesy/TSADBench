export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-MUL \
exp_id=default \
model=DAGMM \
model.latent_dim=3 \
model.lambda_energy=0.1 \
model.lambda_cov_diag=0.005