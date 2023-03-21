import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import cfg

class Dataset:
    def __init__(self, test_size=0.2):
        df = pd.read_csv(cfg.data_dir)
        self.label_column = df.columns[-1]
        self.feature_columns = df.columns[:-1]
        self.classes_ = df[self.label_column].unique()

        """if os.path.exists(cfg.get_filename('train')) and os.path.exists(cfg.get_filename('test')):
            self.data_tr = pd.read_csv(cfg.get_filename('train'))
            self.data_ts = pd.read_csv(cfg.get_filename('test'))
        else:"""
        self.data_ts = df.sample(frac=test_size, random_state=cfg.seed)
        self.data_tr = df.drop(index=self.data_ts.index)

        scaler = MinMaxScaler()
        self.data_tr[self.feature_columns] = scaler.fit_transform(self.data_tr[self.feature_columns])
        self.data_ts[self.feature_columns] = scaler.transform(self.data_ts[self.feature_columns])

        #self.data_tr.to_csv(cfg.get_filename('train'), index=False)
        #self.data_ts.to_csv(cfg.get_filename('test'), index=False)
            #self.y_ts = self.y_ts[np.all(self.x_ts>=-100, axis=1)]
            #self.x_ts = self.x_ts[np.all(self.x_ts>=-100, axis=1),:]
            #self.y_ts = self.y_ts[np.all(self.x_ts<=100, axis=1)]
            #self.x_ts = self.x_ts[np.all(self.x_ts<=100, axis=1),:]

        #self.lb = np.min(self.x_tr,axis=0).tolist()
        #self.ub = np.max(self.x_tr,axis=0).tolist()

    def get_x_tr(self):
        return self.data_tr[self.feature_columns]

    def get_x_ts(self):
        return self.data_ts[self.feature_columns]

    def get_y_tr(self):
        return self.data_tr[self.label_column]

    def get_y_ts(self):
        return self.data_ts[self.label_column]

    @staticmethod
    def GetTollerance(x):
        eps = pd.Series(index=x.columns,dtype=float)
        for i in x.columns:
            try:
                eps[i] = max(1.e-4, np.min(np.diff(np.unique(x[i]))) / 2)
            except:
                eps[i] = 1.e-4
        return eps




