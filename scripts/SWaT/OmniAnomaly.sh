export CUDA_VISIBLE_DEVICES=$1
python main.py \
dataset=SWaT \
exp_id=default \
model=OmniAnomaly \
model.beta=1e-06