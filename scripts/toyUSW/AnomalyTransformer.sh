export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=toyUSW \
exp_id=default \
epochs=5 \
batch_size=256 \
eval_batch_size=256 \
window_size=200 \
stride=200 \
model=AnomalyTransformer \
model.anomaly_ratio=1.0