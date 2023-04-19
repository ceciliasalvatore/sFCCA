import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import cfg

class Dataset:
    def __init__(self, train_size=None):
        self.data = pd.read_csv(cfg.data_dir)
        self.label_column = self.data.columns[-1]
        self.feature_columns = self.data.columns[:-1]
        self.classes_ = self.data[self.label_column].unique()

        scaler = MinMaxScaler()
        self.data[self.feature_columns] = scaler.fit_transform(self.data[self.feature_columns])

        if train_size is not None:
            self.data_ts = self.data.sample(n=(len(self.data)-train_size),random_state=cfg.seed)
            self.data = self.data.drop(index=self.data_ts.index)

    def get_x(self):
        return self.data[self.feature_columns]

    def get_y(self):
        return self.data[self.label_column]

    def get_x_ts(self):
        try:
            return self.data_ts[self.feature_columns]
        except:
            warnings.warn('A separate test set was not extracted')
            return None

    def get_y_ts(self):
        try:
            return self.data_ts[self.label_column]
        except:
            warnings.warn('A separate test set was not extracted')
        return None

    @staticmethod
    def GetTollerance(x):
        eps = pd.Series(index=x.columns,dtype=float)
        for i in x.columns:
            try:
                eps[i] = max(1.e-4, np.min(np.diff(np.unique(x[i]))) / 2)
            except:
                eps[i] = 1.e-4
        return eps




