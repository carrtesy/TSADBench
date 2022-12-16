import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing

class DataFactory:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.home_dir = args.home_dir
        self.logger.info(f"current location: {os.getcwd()}")
        self.logger.info(f"home dir: {args.home_dir}")

        self.dataset_fn_dict = {
            "toyUSW": self.load_toyUSW,
            "NeurIPS-TS-UNI": self.load_NeurIPS_TS_UNI,
            "NeurIPS-TS-MUL": self.load_NeurIPS_TS_MUL,
            "SWaT": self.load_SWaT,
            "WADI": self.load_WADI,
            "SMD": self.load_SMD,
            "PSM": self.load_PSM,
            "SMAP": self.load_SMAP,
            "MSL": self.load_MSL,
        }
        self.datasets = {
            "toyUSW": TSADStandardDataset,
            "NeurIPS-TS-UNI": TSADStandardDataset,
            "NeurIPS-TS-MUL": TSADStandardDataset,
            "SWaT": TSADStandardDataset,
            "WADI": TSADStandardDataset,
            "SMD": TSADStandardDataset,
            "PSM": TSADStandardDataset,
            "SMAP": TSADStandardDataset,
            "MSL": TSADStandardDataset,
        }

        self.transforms = {
            "minmax": preprocessing.MinMaxScaler(),
            "std": preprocessing.StandardScaler(),
        }

    def __call__(self):
        self.logger.info(f"Preparing {self.args.dataset} ...")
        train_x, train_y, test_x, test_y = self.load()
        self.logger.info(
            f"train: X - {train_x.shape}, y - {train_y.shape} " +
            f"test: X - {test_x.shape}, y - {test_y.shape}"
        )
        self.logger.info(f"Complete.")

        self.logger.info(f"Preparing dataloader...")
        train_dataset, train_loader, test_dataset, test_loader = self.prepare(
            train_x, train_y, test_x, test_y,
            window_size=self.args.window_size,
            stride=self.args.stride,
            dataset_type=self.args.dataset,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            train_shuffle=True,
            test_shuffle=False,
            scaler=self.args.scaler,
            window_anomaly=self.args.window_anomaly
        )

        sample_X, sample_y = next(iter(train_loader))
        self.logger.info(f"total train dataset- {len(train_loader)}, "
                         f"batch_X - {sample_X.shape}, "
                         f"batch_y - {sample_y.shape}")

        sample_X, sample_y = next(iter(test_loader))
        self.logger.info(f"total test dataset- {len(test_loader)}, "
                         f"batch_X - {sample_X.shape}, "
                         f"batch_y - {sample_y.shape}")
        self.logger.info(f"Complete.")

        return train_dataset, train_loader, test_dataset, test_loader

    def load(self):
        return self.dataset_fn_dict[self.args.dataset](self.home_dir)

    def prepare(self, train_x, train_y, test_x, test_y,
                window_size,
                stride,
                dataset_type,
                batch_size,
                eval_batch_size,
                train_shuffle,
                test_shuffle,
                scaler,
                window_anomaly,
                ):

        transform = self.transforms[scaler]
        train_dataset = self.datasets[dataset_type](train_x, train_y,
                                                    flag="train", transform=transform,
                                                    window_size=window_size,
                                                    stride=stride,
                                                    window_anomaly=window_anomaly,
                                                    )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
        )

        transform = train_dataset.transform
        test_dataset = self.datasets[dataset_type](test_x, test_y,
                                                   flag="test", transform=transform,
                                                   window_size=window_size,
                                                   stride=stride,
                                                   window_anomaly=window_anomaly,
                                                   )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=eval_batch_size,
            shuffle=test_shuffle,
        )


        return train_dataset, train_dataloader, test_dataset, test_dataloader

    @staticmethod
    def load_toyUSW(home_dir="."):
        base_dir = "data/toyUSW"
        with open(os.path.join(home_dir, base_dir, "train.npy"), 'rb') as f:
            train_X = np.load(f)
        with open(os.path.join(home_dir, base_dir, "train_label.npy"), 'rb') as f:
            train_y = np.load(f)

        with open(os.path.join(home_dir, base_dir, "test.npy"), 'rb') as f:
            test_X = np.load(f)
        with open(os.path.join(home_dir, base_dir, "test_label.npy"), 'rb') as f:
            test_y = np.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_NeurIPS_TS_UNI(home_dir="."):
        base_dir = "data/NeurIPS-TS"
        normal = pd.read_csv(os.path.join(home_dir, base_dir, "nts_uni_normal.csv"))
        abnormal = pd.read_csv(os.path.join(home_dir, base_dir, "nts_uni_abnormal.csv"))

        train_X, train_y = normal.values[:, :-1], normal.values[:, -1]
        test_X, test_y = abnormal.values[:, :-1], abnormal.values[:, -1]

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_NeurIPS_TS_MUL(home_dir="."):

        base_dir = "data/NeurIPS-TS"
        normal = pd.read_csv(os.path.join(home_dir, base_dir, "nts_mul_normal.csv"))
        abnormal = pd.read_csv(os.path.join(home_dir, base_dir, "nts_mul_abnormal.csv"))

        train_X, train_y = normal.values[:, :-1], normal.values[:, -1]
        test_X, test_y = abnormal.values[:, :-1], abnormal.values[:, -1]

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y


    @staticmethod
    def load_SWaT(home_dir="."):
        base_dir = "data/SWaT"
        SWAT_TRAIN_PATH = os.path.join(home_dir, base_dir, 'SWaT_Dataset_Normal_v0.csv')
        SWAT_TEST_PATH = os.path.join(home_dir, base_dir, 'SWaT_Dataset_Attack_v0.csv')
        df_train = pd.read_csv(SWAT_TRAIN_PATH, index_col=0)
        df_test = pd.read_csv(SWAT_TEST_PATH, index_col=0)

        def process_df(df):
            for var_index in [item for item in df.columns if item != 'Normal/Attack']:
                df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
            df.reset_index(drop=True, inplace=True)
            df.fillna(method='ffill', inplace=True)

            x = df.values[:, :-1].astype(np.float32)
            y = (df['Normal/Attack'] == 'Attack').to_numpy().astype(int)
            return x, y

        train_X, train_y = process_df(df_train)
        test_X, test_y = process_df(df_test)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_WADI(home_dir="."):
        base_dir = "data/WADI"
        WADI_TRAIN_PATH = os.path.join(home_dir, base_dir, 'WADI_14days_new.csv')
        WADI_TEST_PATH = os.path.join(home_dir, base_dir, 'WADI_attackdataLABLE.csv')
        df_train = pd.read_csv(WADI_TRAIN_PATH, index_col=0)
        df_test = pd.read_csv(WADI_TEST_PATH, index_col=0, skiprows=[0]) # apply skiprow or erase the first row

        # df_train
        for col in df_train.columns:
            if col.strip() not in ["Date", "Time"]:
                df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        df_train.fillna(method='ffill', inplace=True)
        df_train.fillna(method='bfill', inplace=True)
        df_train.dropna(axis='columns', inplace=True) # drop null columns

        train_X = df_train.values[:, 2:].astype(np.float32) # col0: Date, col1: Time
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)

        # df_test
        for col in df_test.columns:
            if col.strip() not in ["Date", "Time"]:
                df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
        df_test.fillna(method='ffill', inplace=True)
        df_test.fillna(method='bfill', inplace=True)
        df_test.dropna(axis='columns', inplace=True) # drop null columns

        test_X = df_test.values[:, 2:-1].astype(np.float32) # col0: Date, col1: Time, col[-1]: label
        test_y = (df_test.values[:, -1] == -1).astype(int)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_SMD(home_dir="."):
        base_dir = "data/SMD"
        with open(os.path.join(home_dir, base_dir, "SMD_train.pkl"), 'rb') as f:
            train_X = pickle.load(f)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        with open(os.path.join(home_dir, base_dir, "SMD_test.pkl"), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(home_dir, base_dir, "SMD_test_label.pkl"), 'rb') as f:
            test_y = pickle.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_PSM(home_dir="."):
        PSM_PATH = os.path.join(home_dir, "data", "PSM")
        df_train_X = pd.read_csv(os.path.join(PSM_PATH, "train.csv"), index_col=0)
        df_test_X = pd.read_csv(os.path.join(PSM_PATH, "test.csv"), index_col=0)
        train_X = df_train_X.values.astype(np.float32)
        test_X = df_test_X.values.astype(np.float32)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        df_test_y = pd.read_csv(os.path.join(PSM_PATH, "test_label.csv"), index_col=0)
        test_y = df_test_y.values.astype(int).reshape(-1)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_SMAP(home_dir="."):
        base_dir = "data/SMAP"
        with open(os.path.join(home_dir, base_dir, "SMAP_train.pkl"), 'rb') as f:
            train_X = pickle.load(f)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        with open(os.path.join(home_dir, base_dir, "SMAP_test.pkl"), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(home_dir, base_dir, "SMAP_test_label.pkl"), 'rb') as f:
            test_y = pickle.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)

        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_MSL(home_dir="."):
        base_dir = "data/MSL"
        with open(os.path.join(home_dir, base_dir, "MSL_train.pkl"), 'rb') as f:
            train_X = pickle.load(f)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        with open(os.path.join(home_dir, base_dir, "MSL_test.pkl"), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(home_dir, base_dir, "MSL_test_label.pkl"), 'rb') as f:
            test_y = pickle.load(f)

        train_X, test_X = train_X.astype(np.float32), test_X.astype(np.float32)
        train_y, test_y = train_y.astype(int), test_y.astype(int)
        return train_X, train_y, test_X, test_y

    @staticmethod
    def visualize_dataset(train_X, train_y, test_X, test_y, dataset_name, feature_idx):
        '''
        :param dataset_name
        :param feature_idx
        '''

        F = feature_idx
        # plot normal
        plt.figure(figsize=(20, 6))
        plt.plot(train_X[:, F])
        plt.title(f"{dataset_name} train data, feature {F}")
        plt.show()

        # plot abnormal
        plt.figure(figsize=(20, 6))
        plt.plot(test_X[:, F])
        plt.title(f"{dataset_name} test data, feature {F}")
        s, e = None, None
        for i in range(len(test_X)):
            if test_y[i] == 1 and s is None:
                s = i
            elif test_y[i] == 0 and s is not None:
                e = i - 1
                if (e - s) > 0:
                    plt.axvspan(s, e, facecolor='red', alpha=0.5)
                else:
                    plt.plot(s, test_X[s, F], 'ro')
                s, e = None, None
        plt.show()


class TSADStandardDataset(Dataset):
    def __init__(self, x, y, flag, transform, window_size, stride, window_anomaly):
        super().__init__()
        self.transform = transform
        self.len = (x.shape[0] - window_size) // stride + 1
        self.window_size = window_size
        self.stride = stride
        self.window_anomaly = window_anomaly

        x, y = x[:self.len*self.window_size], y[:self.len*self.window_size]
        self.x = self.transform.fit_transform(x) if flag == "train" else self.transform.transform(x)
        self.y = y

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        _idx = idx * self.stride
        label = self.y[_idx:_idx+self.window_size]
        X, y = self.x[_idx:_idx+self.window_size], (1 in label) if self.window_anomaly else label
        return X, y

if __name__ == "__main__":
    datafactory = DataFactory()
    train_X, train_y, test_X, test_y = datafactory.load_MSL("..")
    # train_X, train_y, test_X, test_y = datafactory.load_WADI("..")
    # train_X, train_y, test_X, test_y = datafactory.load_NeurIPS_TS("..")