export CUDA_VISIBLE_DEVICES=$1
python main.py dataset=toyUSW exp_id=default model=LSTMEncDec model.latent_dim=128