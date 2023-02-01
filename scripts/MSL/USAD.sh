# configuration from:
# USAD paper (https://dl.acm.org/doi/pdf/10.1145/3394486.3403392)
# Appendix 3
export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=MSL \
epochs=250 \
window_size=5 \
scaler=minmax \
exp_id=default \
model=USAD \
model.dsr=5 \
model.latent_dim=33