# Trainer
from Exp.SklearnTrainer import SklearnModelTrainer

# models
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class LOF_Trainer(SklearnModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(LOF_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = LocalOutlierFactor(
            novelty=True,
            contamination=self.args.model.contamination,
            n_jobs=self.args.model.n_jobs,
        )


class IsolationForest_Trainer(SklearnModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(IsolationForest_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = IsolationForest(
            n_estimators=self.args.model.n_estimators,
            contamination=self.args.model.contamination,
            n_jobs=self.args.model.n_jobs,
            random_state=self.args.model.random_state,
            verbose=self.args.model.verbose,
        )


class OCSVM_Trainer(SklearnModelTrainer):
    def __init__(self, args, logger, train_loader, test_loader):
        super(OCSVM_Trainer, self).__init__(args, logger, train_loader, test_loader)
        self.model = OneClassSVM(
            max_iter=self.args.model.max_iter,
            nu=self.args.model.nu,
            verbose=True,
        )