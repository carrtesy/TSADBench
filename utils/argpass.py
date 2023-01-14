import argparse
import os
import torch
from omegaconf import OmegaConf
from easydict import EasyDict as edict

def prepare_arguments(cfg):
    '''
    get input arguments
    :return: args
    '''
    # args get from hydra -> easydict
    args = OmegaConf.to_container(cfg)
    args = edict(args)
    args = configure_exp_id(args)

    # save paths config & make dir
    args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
    args.logging_path = os.path.join(args.logs, f"{args.exp_id}")
    args.output_path = os.path.join(args.outputs, f"{args.exp_id}")

    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.logging_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    args.home_dir = "."
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

def configure_exp_id(args):
    '''
    Configure exp_id for hyperparameters manually.
    subject to change when wandb supports yaml variable references feature:
    https://github.com/wandb/wandb/issues/3707
    '''
    if args.exp_id == "default":
        model = args.model.name
        dataset = args.dataset
        new_exp_id = f"{model}_{dataset}"
        if model == "OCSVM":
            pass
        elif model == "IsolationForest":
            new_exp_id += f"_n_estimator_{args.model.n_estimators}"
        elif model == "LOF":
            pass
        elif model == "LSTMEncDec":
            new_exp_id += f"_latent_dim_{args.model.latent_dim}"
            new_exp_id += f"_num_layers_{args.model.num_layers}"
            new_exp_id += f"_dropout_{args.model.dropout}"
        elif model == "LSTMVAE":
            new_exp_id += f"_hidden_dim_{args.model.hidden_dim}"
            new_exp_id += f"_z_dim_{args.model.z_dim}"
            new_exp_id += f"_n_layers_{args.model.n_layers}"
            new_exp_id += f"_beta_{args.model.beta}"
        elif model == "USAD":
            new_exp_id += f"_latent_dim_{args.model.latent_dim}"
            new_exp_id += f"_alpha_{args.model.alpha}"
            new_exp_id += f"_beta_{args.model.beta}"
            new_exp_id += f"_dsr_{args.model.dsr}"
        elif model == "OmniAnomaly":
            new_exp_id += f"_hidden_dim_{args.model.hidden_dim}"
            new_exp_id += f"_z_dim_{args.model.z_dim}"
            new_exp_id += f"_dense_dim_{args.model.dense_dim}"
            new_exp_id += f"_beta_{args.model.beta}"
        elif model == "DeepSVDD":
            pass
        elif model == "DAGMM":
            new_exp_id += f"_gmm_k_{args.model.gmm_k}"
            new_exp_id += f"_latent_dim_{args.model.latent_dim}"
            new_exp_id += f"_lambda_energy_{args.model.lambda_energy}"
            new_exp_id += f"_lambda_cov_diag_{args.model.lambda_cov_diag}"
            new_exp_id += f"_grad_clip_{args.model.grad_clip}"
        elif model == "THOC":
            new_exp_id += f"_hidden_dim_{args.model.hidden_dim}"
            new_exp_id += f"_L2_reg_{args.model.L2_reg}"
            new_exp_id += f"_LAMBDA_orth_{args.model.LAMBDA_orth}"
            new_exp_id += f"_LAMBDA_TSS_{args.model.LAMBDA_TSS}"
        elif model == "AnomalyTransformer":
            new_exp_id += f"_anomaly_ratio_{args.model.anomaly_ratio}"
            new_exp_id += f"_k_{args.model.k}"
            new_exp_id += f"_temperature_{args.model.temperature}"
            new_exp_id += f"_e_layers_{args.model.e_layers}"
        args.exp_id = new_exp_id

    return args


def EDA_prep_arguments(window_size=96, dataset="SWaT", scaler="std"):
    return edict({
        "home_dir": ".",
        "window_size": window_size,
        "stride": 1,
        "dataset": dataset,
        "batch_size": 64,
        "eval_batch_size": 64*3,
        "scaler": scaler,
        "window_anomaly": False,
    })

'''
Saved for future use.
### MAE
MAE_parser = subparser.add_parser("MAE")
MAE_parser.add_argument("--enc_dim", type=int, required=True, default=256, help=f"Encoder hidden dim")
MAE_parser.add_argument("--dec_dim", type=int, required=True, default=256, help=f"Decoder hidden dim")
MAE_parser.add_argument("--enc_num_layers", type=int, default=3, help=f"Encoder num layers")
MAE_parser.add_argument("--dec_num_layers", type=int, default=3, help=f"Decoder num layers")
MAE_parser.add_argument("--mask_ratio", type=float, default=0.1)
MAE_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")
MAE_parser.add_argument("--eval_samples", type=int, default=3)

### VQVAE
VQVAE_parser = subparser.add_parser("VQVAE")

VQVAE_parser.add_argument("--n_embeddings", type=int, required=True, default=40,
                          help=f"Encoder, decoder hidden dim")
VQVAE_parser.add_argument("--embedding_dim", type=int, required=True, default=40,
                          help=f"Encoder, decoder hidden dim")
VQVAE_parser.add_argument("--beta", type=float, required=True, default=0.25, help=f"Encoder, decoder hidden dim")

### MADE
MADE_parser = subparser.add_parser("MADE")
MADE_parser.add_argument("--latent_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
MADE_parser.add_argument("--depth", type=int, required=True, default=6, help=f"Encoder, decoder hidden dim")
MADE_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

### ANP
ANP_parser = subparser.add_parser("ANP")
ANP_parser.add_argument("--dim_hid", type=int, required=True, default=256, help=f"Encoder hidden dim")
ANP_parser.add_argument("--dim_lat", type=int, required=True, default=256, help=f"Decoder hidden dim")
ANP_parser.add_argument("--mask_ratio", type=float, default=0.1)
ANP_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")
ANP_parser.add_argument("--eval_samples", type=int, default=3)
ANP_parser.add_argument("--masking_mode", type=str, default="continuous")
'''
