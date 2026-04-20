import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from torch.utils.data import WeightedRandomSampler


class BaseDataset(Dataset):
    def __init__(self, frame, data_path, max_len=1000, use_tabular=0, tabular_features=None):
        self.frame = frame.reset_index(drop=True)
        self.max_len = max_len
        self.data_path = data_path
        self.use_tabular = use_tabular
        self.tabular_features = tabular_features

    def __len__(self):
        return len(self.frame)

    def _load_signal(self, record_path):
        signal, meta = wfdb.rdsamp(str(record_path))
        signal = signal.astype(np.float32)

        if signal.shape[0] > self.max_len:
            signal = signal[:self.max_len]
        elif signal.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - signal.shape[0], signal.shape[1]), dtype=np.float32)
            signal = np.vstack([signal, pad])

        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True) + 1e-6
        signal = (signal - mean) / std

        signal = np.transpose(signal, (1, 0))
        return signal

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        x = self._load_signal(self.data_path+ row["filename_hr"])
        y = np.float32(row["has_mi"])
        if self.use_tabular:
            return torch.tensor(x), torch.tensor(row[self.tabular_features].values.astype(np.float32)), torch.tensor(y)
        return torch.tensor(x), torch.tensor(y)
    
    def get_full_target(self):
        return self.frame['has_mi']
    
class Dataset():
    def __init__(self, data_path, use_data, batch_size, use_tabular=0, tabular_features=None):
        self.data_path = data_path
        self.use_tabular = use_tabular
        self.tabular_features = tabular_features
        self.df = pd.read_csv(data_path + "ptbxl_database.csv")
        self._preprocess()
        self.data = self.df[self.df['mi_or_norm']]
        

        if use_data == "F":
            self.data = self.data[self.data['sex'] == 1]
        elif use_data == "M":
            self.data = self.data[self.data['sex'] == 0]

        self.train = self.data[self.data['strat_fold'] < 9]
        self.val = self.data[self.data['strat_fold'] == 9]
        self.test = self.data[self.data['strat_fold'] == 10] 

        self.train_ds = BaseDataset(self.train, self.data_path, use_tabular=use_tabular, tabular_features=tabular_features)
        self.val_ds = BaseDataset(self.val, self.data_path, use_tabular=use_tabular, tabular_features=tabular_features)
        self.test_ds = BaseDataset(self.test, self.data_path, use_tabular=use_tabular, tabular_features=tabular_features)

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, sampler=self._make_sampler(self.train_ds), num_workers=0)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, sampler=self._make_sampler(self.val_ds), num_workers=0)
        self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, sampler=self._make_sampler(self.test_ds), num_workers=0)  


    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
    
    def _make_sampler(self, dataset):
        labels = dataset.get_full_target().values
        class_counts = np.bincount(labels.astype(int))
        weights = 1.0 / class_counts[labels.astype(int)]
        return WeightedRandomSampler(weights, len(weights))


    def _preprocess(self):
        self.df["scp_codes"] = self.df["scp_codes"].apply(ast.literal_eval)
        agg_df = pd.read_csv(self.data_path + "/scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        self.df['diagnostic_superclass'] = self.df["scp_codes"].apply(aggregate_diagnostic)

        if self.use_tabular:
            self.df.dropna(subset=self.tabular_features, inplace=True)
            self.df[self.tabular_features].astype(np.float32)

        def has_mi(diagnostic_superclass):
            return "MI" in diagnostic_superclass

        def has_mi_or_healthy(diagnostic_superclass):
            return "MI" in diagnostic_superclass or "NORM" in diagnostic_superclass

        self.df["has_mi"] = self.df["diagnostic_superclass"].apply(has_mi)
        self.df["mi_or_norm"] = self.df["diagnostic_superclass"].apply(has_mi_or_healthy)
   