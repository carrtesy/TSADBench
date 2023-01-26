# configuration from:
# https://github.com/thuml/Anomaly-Transformer/blob/main/scripts/SMAP.sh
export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=SMAP \
exp_id=default \
epochs=3 \
batch_size=256 \
eval_batch_size=256 \
window_size=100 \
stride=100 \
model=AnomalyTransformer \
model.anomaly_ratio=1.0