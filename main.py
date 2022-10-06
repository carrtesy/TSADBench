######################################################
#                       _oo0oo_                      #
#                      o8888888o                     #
#                      88" . "88                     #
#                      (| -_- |)                     #
#                      0\  =  /0                     #
#                    ___/`---'\___                   #
#                  .' \\|     |// '.                 #
#                 / \\|||  :  |||// \                #
#                / _||||| -:- |||||- \               #
#               |   | \\\  -  /// |   |              #
#               | \_|  ''\---/''  |_/ |              #
#               \  .-\__  '-'  ___/-. /              #
#             ___'. .'  /--.--\  `. .'___            #
#          ."" '<  `.___\_<|>_/___.' >' "".          #
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |        #
#         \  \ `_.   \_ __\ /__ _/   .-` /  /        #
#     =====`-.____`.___ \_____/___.-`___.-'=====     #
#                       `=---='                      #
#                                                    #
#                                                    #
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    #
#                                                    #
#        Buddha Bless:   "No Bugs in my code"        #
#                                                    #
######################################################


import argparse
import torch
import os
import numpy as np

from tqdm import tqdm
from data.load_data import DataFactory
from Exp.Trainer import *
from Exp.Baselines import *
from Exp.MAE_Trainer import *
from Exp.VQVAE_Trainer import *
from Exp.ANP_Trainer import *

from utils.tools import SEED_everything
SEED_everything(42)

# 1. Argparse
print("=" * 30)
print(f"Parsing arguments...")
parser = argparse.ArgumentParser(description='[TSAD] Time Series Anomaly Detection')

### data
parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
parser.add_argument("--eval_batch_size", type=int, required=False, default=64*3, help=f"Batch size")
parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
parser.add_argument("--window_size", type=int, required=False, default=12, help=f"window size")
parser.add_argument("--epochs", type=int, required=False, default=30, help=f"epochs to run")
parser.add_argument("--load_pretrained", action="store_true", help=f"whether to load pretrained version")
parser.add_argument("--exp_id", type=str, default="test")
parser.add_argument("--scaler", type=str, default="std")

### save
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--outputs", type=str, default="./outputs")

## subparser of models
subparser = parser.add_subparsers(dest='model')

### USAD
USAD_parser = subparser.add_parser("USAD")
USAD_parser.add_argument("--latent_dim", type=int, required=True, default=128, help=f"Encoder, decoder hidden dim")
USAD_parser.add_argument("--alpha", type=float, required=False, default=0.1, help=f"alpha")
USAD_parser.add_argument("--beta", type=float, required=False, default=0.9, help=f"beta")
USAD_parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

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

VQVAE_parser.add_argument("--n_embeddings", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
VQVAE_parser.add_argument("--embedding_dim", type=int, required=True, default=40, help=f"Encoder, decoder hidden dim")
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


args = parser.parse_args()
args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
args.output_path = os.path.join(args.outputs, f"{args.exp_id}")
args.home_dir = "."
os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args)
print("done.")

# 2. Data
print("=" * 30)
print(f"Preparing {args.dataset} dataset...")
datafactory = DataFactory(args)
train_dataset, train_loader, test_dataset, test_loader = datafactory()
args.num_channels = train_dataset.x.shape[1]


# 3. Model
print("=" * 30)
print(f"Preparing {args.model} Trainer...")
Trainers = {
    "USAD": USAD_Trainer,
    "MAE": MAE_Trainer,
    "Isolation_Forest": IsolationForest_Trainer,
    "VQVAE": VQVAE_Trainer,
    "ANP": ANP_Trainer,
    #"MADE": MADE_Trainer,
}
trainer = Trainers[args.model](args)

# 4. train and infer
print("=" * 30)
print(f"Preparing {args.model} Training...")
epochs = tqdm(range(args.epochs))
best_result = None
for epoch in epochs:
    # train
    train_stats = trainer.train(train_dataset, train_loader)
    print(f"train_stats: {train_stats}")
    trainer.checkpoint(os.path.join(args.checkpoint_path, f"epoch{epoch}.pth"))

    # infer
    '''
    result = trainer.infer(test_dataset, test_loader)
    p, r, f1 = result["p"], result["r"], result["f1"]
    if best_result is None or f1 > best_result["f1"]:
        print(f"Saving best results @epoch{epoch} => f1 score:{f1} | precision:{p}, recall:{r}")
        best_result = result
        trainer.checkpoint(os.path.join(args.checkpoint_path, f"best.pth"))
    else:
        print(f"@epoch{epoch} => f1 score:{f1} | precision:{p}, recall:{r}")
    '''
