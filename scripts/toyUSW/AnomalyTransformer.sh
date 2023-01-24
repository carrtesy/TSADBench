export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=toyUSW \
exp_id=default \
epochs=5 \
window_size=100 \
stride=100 \
model=AnomalyTransformer \
model.anomaly_ratio=4.0 \
model.k=3 \
model.temperature=50 \
model.e_layers=3