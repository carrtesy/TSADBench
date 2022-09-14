import argparse
import torch
import os
import numpy as np

from tqdm import tqdm
from data.load_data import load_data, load_anomaly_intervals
from data.dataset import data_provider
from Exp.Trainer import USAD_Trainer
from metrics import get_statistics

from utils import SEED_everything
SEED_everything(42)

# 1. Argparse
print("=" * 30)
print(f"Parsing arguments...")
parser = argparse.ArgumentParser(description='[TSAD] Time Series Anomaly Detection')

### data
parser.add_argument("--dataset", type=str, required=True, default="SWaT", help=f"Dataset name")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
parser.add_argument("--window_size", type=int, required=False, default=12, help=f"window size")
parser.add_argument("--epochs", type=int, required=False, default=30, help=f"epochs to run")
parser.add_argument("--load_pretrained", action="store_true", help=f"whether to load pretrained version")
parser.add_argument("--exp_id", type=str, default="test")

### save
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--outputs", type=str, default="./outputs")

## subparser of models
subparser = parser.add_subparsers(dest='model')

### USAD
USAD_parser = subparser.add_parser("USAD")
USAD_parser.add_argument("--latent_dim", type=int, required=True, default=128, help=f"Encoder, decoder hidden dim")
USAD_parser.add_argument("--alpha", type=float, required=False, default=0.1, help=f"Encoder, decoder hidden dim")
USAD_parser.add_argument("--beta", type=float, required=False, default=0.9, help=f"Encoder, decoder hidden dim")

args = parser.parse_args()
args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
args.output_path = os.path.join(args.outputs, f"{args.exp_id}")
os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args)
print("done.")

# 2. Data
print("=" * 30)
print(f"Preparing {args.dataset} dataset...")
train_x, train_y, test_x, test_y = load_data(args.dataset)

args.input_dim = train_x.shape[1]

train_dataset, train_loader, test_dataset, test_loader = data_provider(train_x, train_y, test_x, test_y, window_size=args.window_size, dataset_type=args.dataset)

# 3. Model

print("=" * 30)
print(f"Preparing {args.model} Trainer...")
Trainers = {
    "USAD": USAD_Trainer,
}
trainer = Trainers[args.model](args)

# 4. train
print("=" * 30)
print(f"Preparing {args.model} Training...")
epochs = tqdm(range(args.epochs))
for epoch in epochs:
    # train
    train_loss = trainer.train(train_loader)
    epochs.set_postfix({"train_loss": train_loss})
    trainer.checkpoint(os.path.join(args.checkpoint_path, f"epoch{epoch}.pth"))

    # infer
    outputs = trainer.infer(test_loader)
    trainer.save_dictionary(outputs, os.path.join(args.output_path, f"epoch{epoch}_outputs.pkl"))
    anomaly_scores = outputs["anomaly_scores"]

    # calculate best f1
    result = trainer.get_best_f1(y=test_dataset.y, anomaly_scores=anomaly_scores)
    print(result)

# 5. save output
anomaly_scores = trainer.infer(test_loader)
print(anomaly_scores.shape)
with open(os.path.join(args.output_path, "anomaly_scores.npy"), "wb") as f:
    np.save(f, anomaly_scores)