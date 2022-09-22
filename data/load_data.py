import pandas as pd
import numpy as np
import os
import pickle

from torch.utils.data import DataLoader, Dataset
from sklearn import preprocessing

class DataFactory:
    def __init__(self, args):
        self.args = args
        self.home_dir = args.home_dir
        print(f"current location: {os.getcwd()}")
        print(f"home dir: {args.home_dir}")

        self.dataset_fn_dict = {
            "SWaT": self.load_SWaT,
            "SMAP": self.load_SMAP,
        }
        self.datasets = {
            "SWaT": TSADStandardDataset,
            "SMAP": TSADStandardDataset,
        }

        self.transforms = {
            "minmax": preprocessing.MinMaxScaler(),
            "std": preprocessing.StandardScaler(),
        }

    def __call__(self):
        train_x, train_y, test_x, test_y = self.load()
        train_dataset, train_loader, test_dataset, test_loader = self.prepare(train_x, train_y, test_x, test_y,
                                                                              window_size=self.args.window_size,
                                                                              dataset_type=self.args.dataset,
                                                                              batch_size=self.args.batch_size,
                                                                              shuffle=False,
                                                                              scaler=self.args.scaler)
        return train_dataset, train_loader, test_dataset, test_loader

    def load(self):
        return self.dataset_fn_dict[self.args.dataset](self.home_dir)

    def prepare(self, train_x, train_y, test_x, test_y,
                window_size,
                dataset_type,
                batch_size,
                shuffle,
                scaler):

        transform = self.transforms[scaler]
        train_dataset = self.datasets[dataset_type](train_x, train_y,
                                                    flag="train", transform=transform,
                                                    window_size=window_size)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        transform = train_dataset.transform
        test_dataset = self.datasets[dataset_type](test_x, test_y,
                                                   flag="test", transform=transform,
                                                   window_size=window_size)
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        return train_dataset, train_dataloader, test_dataset, test_dataloader

    @staticmethod
    def load_SWaT(home_dir):
        print("Reading SWaT...")
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

        print(f"train: X - {train_X.shape}, y - {train_y.shape}")
        print(f"test: X - {test_X.shape}, y - {test_y.shape}")
        print("Loading complete.")
        return train_X, train_y, test_X, test_y

    @staticmethod
    def load_SMAP(home_dir):
        print("Reading SMAP...")

        base_dir = "data/SMAP"
        with open(os.path.join(home_dir, base_dir, "SMAP_train.pkl"), 'rb') as f:
            train_X = pickle.load(f)
        T, C = train_X.shape
        train_y = np.zeros((T,), dtype=int)
        with open(os.path.join(home_dir, base_dir, "SMAP_test.pkl"), 'rb') as f:
            test_X = pickle.load(f)
        with open(os.path.join(home_dir, base_dir, "SMAP_test_label.pkl"), 'rb') as f:
            test_y = pickle.load(f)

        print(f"train: X - {train_X.shape}, y - {train_y.shape}")
        print(f"test: X - {test_X.shape}, y - {test_y.shape}")
        print("Loading complete.")
        return train_X, train_y, test_X, test_y

class TSADStandardDataset(Dataset):
    def __init__(self, x, y, flag, transform, window_size=1):
        super().__init__()
        self.transform = transform
        self.x = self.transform.fit_transform(x) if flag == "train" else self.transform.transform(x)
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return self.x.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        return self.x[idx:idx+self.window_size], self.y[idx:idx+self.window_size]