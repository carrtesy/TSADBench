# configuration from:
# https://github.com/thuml/Anomaly-Transformer/blob/main/scripts/MSL.sh
export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=MSL \
epochs=3 \
batch_size=256 \
window_size=100 \
exp_id=default \
model=AnomalyTransformer \
model.anomaly_ratio=1.0