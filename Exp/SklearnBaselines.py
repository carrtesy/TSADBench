# Trainer
from Exp.SklearnTrainer import SklearnModelTrainer

# models
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# utils
from utils.metrics import get_statistics

# others
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
import pickle

class LOF_Trainer(SklearnModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(LOF_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = LocalOutlierFactor(
            novelty=True,
        )

class IsolationForest_Trainer(SklearnModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(IsolationForest_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = IsolationForest(
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            verbose=args.verbose,
        )

class OCSVM_Trainer(SklearnModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(OCSVM_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = OneClassSVM()