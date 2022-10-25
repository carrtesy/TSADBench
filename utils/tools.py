import torch
import numpy as np
import random
import argparse
import pandas as pd

def SEED_everything(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def read_xlsx_and_convert_to_csv(path):
    excelFile = pd.read_excel(path, skiprows=[0])
    filename = path[:-5]
    excelFile.to_csv(f"{filename}.csv", index=None, header=True)


if __name__ == "__main__":
    print("*")
    read_xlsx_and_convert_to_csv("../data/SWaT/SWaT_Dataset_Attack_v0.xlsx")
    print("*")
    read_xlsx_and_convert_to_csv("../data/SWaT/SWaT_Dataset_Normal_v0.xlsx")
    print("*")
    read_xlsx_and_convert_to_csv("../data/SWaT/SWaT_Dataset_Normal_v1.xlsx")
    print("*")