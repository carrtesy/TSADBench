export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=NeurIPS-TS-MUL \
exp_id=default \
model=AnomalyTransformer \
model.anomaly_ratio=4.0 \
model.k=3 \
model.temperature=50 \
model.e_layers=3