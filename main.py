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
from utils.parser import prepare_arguments
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
print(f"Preparing arguments...")
parser = argparse.ArgumentParser(description='[TSAD] Time Series Anomaly Detection')
args = prepare_arguments(parser)
os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)
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
    "OCSVM": OCSVM_Trainer,
    "IsolationForest": IsolationForest_Trainer,
    "LOF": LOF_Trainer,
    "USAD": USAD_Trainer,
    #"MAE": MAE_Trainer,
    #"VQVAE": VQVAE_Trainer,
    #"ANP": ANP_Trainer,
}
trainer = Trainers[args.model](args)

# 4. train
print("=" * 30)
print(f"Preparing {args.model} Training...")


if args.epochs > 0:
    epochs = tqdm(range(args.epochs))
    best_train_stats = None
    for epoch in epochs:
        # train
        train_stats = trainer.train(train_dataset, train_loader)
        print(f"train_stats: {train_stats}")
        trainer.checkpoint(os.path.join(args.checkpoint_path, f"epoch{epoch}.pth"))

        if args.eval_every_epoch:
            result = trainer.infer(test_dataset, test_loader)
            print(f"=== Result @epoch{epoch} ===")
            for key in result:
                print(f"{key}: {result[key]}")

        if best_train_stats is None or train_stats < best_train_stats:
            print(f"Saving best results @epoch{epoch}")
            trainer.checkpoint(os.path.join(args.checkpoint_path, f"best.pth"))
            best_train_stats = train_stats
else:
    trainer.train(train_dataset, train_loader)
    trainer.checkpoint(os.path.join(args.checkpoint_path, f"best.pth"))

# 5. infer
print("=" * 30)
print(f"Loading from best path...")
trainer.load(os.path.join(args.checkpoint_path, f"best.pth"))
result = trainer.infer(test_dataset, test_loader)

print(f"=== Final Result ===")
for key in result:
    print(f"{key}: {result[key]}")