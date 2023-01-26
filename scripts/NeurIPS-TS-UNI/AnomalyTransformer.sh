export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-UNI \
exp_id=default \
epochs=5 \
batch_size=256 \
eval_batch_size=256 \
window_size=25 \
stride=25 \
model=AnomalyTransformer \
model.anomaly_ratio=1.0