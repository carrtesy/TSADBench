export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-MUL \
exp_id=default \
model=LSTMVAE \
model.beta=1e-04