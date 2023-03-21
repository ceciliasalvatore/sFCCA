import warnings
import pandas as pd
import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, KFold
from scipy.spatial import distance_matrix
from gosdt.model.threshold_guess import compute_thresholds

from dataset import Dataset
from CounterfactualAnalysis.counterfactualExplanations import CounterfactualExplanation

class Discretizer:
    def fit(self, x, y):
        raise NotImplementedError

    def transform(self, x, y, tao=None):
        if tao is None:
            tao = self.tao
        x_discretized = tao.apply(lambda t: (x[t['Feature']]<=t['Threshold']).astype(int),axis=1).T
        x_discretized.columns = tao.apply(lambda t: f"{t['Feature']}<={t['Threshold']}", axis=1)
        return x_discretized, y

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x, y)

    def compression_rate(self, x, y, tao=None):
        if tao is None:
            tao = self.tao
        x_discr, y_discr = self.transform(x, y, tao)
        u = np.sort(np.unique(x_discr,axis=0,return_index=True)[1])
        compression_rate = 1 - len(u)/len(x)
        return compression_rate

    def inconsistency_rate(self, x, y, tao=None):
        if tao is None:
            tao = self.tao
        x_discr, y_discr = self.transform(x, y, tao)
        u = np.sort(np.unique(x_discr,axis=0,return_index=True)[1])
        count = 0
        for i in range(len(x_discr)):
            if i in u:
                si = np.where((x_discr==x_discr.iloc[i]).all(axis=1))[0]
                yy, cc = np.unique(y.iloc[si], return_counts=True)
                if len(yy) > 1:
                    count += np.min(cc)
        inconsistency_rate = count / len(x_discr)
        return inconsistency_rate

class TotalDiscretizer(Discretizer):
    def fit(self, x, y):
        self.tao = pd.DataFrame(columns=['Feature','Threshold'])
        for i in x.columns:
            self.tao = pd.concat((self.tao,self.getAllThresholds(x, i)))

    def getAllThresholds(self, x, c):
        tao_c = pd.DataFrame(columns=['Feature','Threshold'])
        x_c = np.sort(np.unique((x[c])))
        tao_c['Threshold'] = x_c[:-1] + (x_c[1:] - x_c[:-1]) / 2
        tao_c['Feature'] = c
        return tao_c

class FCCA(Discretizer):
    def __init__(self, estimator, p0=0.5, p1=1, p2=1, lambda0=0.1, lambda1=1, lambda2=0.0, compress=False, Q=None):
        super().__init__()
        self.estimator = estimator
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.compress = compress
        self.Q = Q

    def fit(self, x, y, x_ts=None, y_ts=None):
        self.estimator.fit(x, y)
        eps = Dataset.GetTollerance(x)
        if x_ts is not None and y_ts is not None:
            x0, y0 = self.getRelevant(x, y)
        else:
            x0, y0 = self.getRelevant(x, y)
        xCE, yCE = self.getCounterfactualExplanations(x0, y0, eps)
        self.tao = self.getCounterfactualThresholds(x0, xCE, eps)
        if self.compress:
            self.tao = self.compressThresholds(self.tao)
        if self.Q is not None:
            self.chooseQ(x, y)

    def chooseQ(self, x, y, estimator=DecisionTreeClassifier(max_depth=4), split=None, tradeoff=0.5):
        levels = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
        if self.Q=='KFold':
            if split is None:
                split = list(KFold(self.k).split(x, y))
            else:
                split = list(split)
            best_score = -1
            self.Q_detected = None
            for l in levels:
                tao_l = self.selectThresholds(l)
                average_score = 0.0
                for train_idx, val_idx in split:
                    x_tr_l, y_tr_l = self.transform(x.iloc[train_idx], y.iloc[train_idx], tao_l)
                    x_val_l, y_val_l = self.transform(x.iloc[val_idx], y.iloc[val_idx], tao_l)
                    estimator.fit(x_tr_l, y_tr_l)
                    average_score += accuracy_score(y_val_l, estimator.predict(x_val_l))
                average_score /= len(list(split))
                if best_score < average_score:
                    best_score = average_score
                    self.Q_detected = l
        elif self.Q=='Tradeoff':
            best_tradeoff = -1
            self.Q_detected = None
            for l in levels:
                tao_l = self.selectThresholds(l)
                tradeoff_l = (1-tradeoff)*self.compression_rate(x, y, tao_l) - tradeoff * self.inconsistency_rate(x, y, tao_l)
                if best_tradeoff < tradeoff_l:
                    best_tradeoff = tradeoff_l
                    self.Q_detected = l
        elif isinstance(self.Q, float):
            self.Q_detected = self.Q
        else:
            raise Exception(f'Unknown strategy for selecting thresholds ({self.Q})')

        self.tao = self.selectThresholds(self.Q_detected)

    def selectThresholds(self, Q):
        threshold_importance = np.quantile(self.tao['Count'], Q)
        return self.tao[self.tao['Count']>=threshold_importance]

    def getCounterfactualThresholds(self, x0, xCE, eps):
        xCE = xCE.loc[x0.index]
        difference = (x0 - xCE).abs()

        tao = pd.DataFrame(columns=['Feature', 'Threshold', 'Count'])
        for i in x0.columns:
            tao_i = pd.DataFrame((xCE[i][difference[i]>eps[i]]).tolist(), columns=['Threshold'])
            tao_i['Feature'] = i
            tao_i['Count'] = 1
            """tao_i = pd.DataFrame(xCE[i][difference[i]>eps[i]].sort_values())
            tao_i['flag'] = np.nan
            mask = (tao_i[i].diff().isna()) | (tao_i[i].diff() > 0.01)
            tao_i['flag'][mask] = np.arange(len(tao_i[mask]))
            tao_i['flag'] = tao_i['flag'].fillna(method='ffill')
            tao_i = tao_i.groupby('flag').agg(Threshold=(i,'mean'),Count=(i,'count'))
            tao_i['Feature'] = i"""
            tao = pd.concat((tao,tao_i))
        tao.index = np.arange(len(tao))

        return tao

    def compressThresholds(self, thresholds, tollerance=0.01):

        thresholds['flag'] = np.nan
        thresholds = thresholds.sort_values(['Feature', 'Threshold'])
        for f in thresholds['Feature'].unique():
            thresholds_f = thresholds[thresholds['Feature'] == f]
            mask = (thresholds_f['Threshold'].diff().isna()) | (thresholds_f['Threshold'].diff() > tollerance)
            thresholds_f.loc[mask,'flag'] = np.arange(mask.sum())
            thresholds.loc[thresholds['Feature']==f,'flag'] = thresholds_f['flag'].fillna(method='ffill')

        thresholds = thresholds.groupby(['Feature','flag']).agg({'Threshold':'mean','Count':'count'}).reset_index().drop(columns=['flag'])

        return thresholds

    def getCounterfactualExplanations(self, x0, y0, eps):
        solver = CounterfactualExplanation(self.estimator, lambda0=self.lambda0, lambda1=self.lambda1, lambda2=self.lambda2, eps=eps)
        xCE, yCE = solver.compute(x0, y0)
        """def random_perturbation(x):
            x = x.copy()
            x[np.random.choice(x.index, 3)] += (np.random.random(3) * 2 - 1)*0.3
            x[x<0]=0
            x[x>1]=1
            return x
        xCE = x0.apply(random_perturbation,axis=1)
        yCE = y0.apply(lambda x:self.counterfactual_labels[x])"""
        return xCE, yCE

    def getRelevant(self, x, y):
        try:
            index = np.where((self.estimator.predict(x) == y) &
                            (np.max(self.estimator.predict_proba(x), axis=1) >= self.p0) &
                            (np.max(self.estimator.predict_proba(x), axis=1) <= self.p1))[0]
        except:
            warnings.warn(f"Disabling probability control in FCCA.getRelevant for estimator of class {self.estimator.__class__}")
            index = np.where((self.estimator.predict(x)==y))
        x_relevant = x.iloc[index]
        y_relevant = y.iloc[index]
        if self.p2 < 1:
            x_k = {}
            y_k = {}
            for k in y_relevant.unique():
                x_k[k] = x_relevant[y_relevant==k]
                y_k[k] = y_relevant[y_relevant==k]
                centers = KMeans(n_clusters=int(len(x_k[k]) * self.p2), n_init=100).fit(x_k[k]).cluster_centers_
                closest = np.argsort(distance_matrix(centers, x_k[k]))[:, 0]
                x_k[k] = x_k[k].iloc[closest]
                y_k[k] = y_k[k].iloc[closest]
            x_relevant = pd.concat(tuple(x_k[k] for k in x_k.keys()))
            y_relevant = pd.concat(tuple(y_k[k] for k in y_k.keys()))
        return x_relevant, y_relevant

class FCCA_KFold(FCCA):
    def __init__(self, estimator, k, p0=0.5, p1=1, p2=1, lambda0=0.1, lambda1=1, lambda2=0.0, compress=True, Q='KFold'):
        super().__init__(estimator, p0, p1, p2, lambda0, lambda1, lambda2, compress, Q=Q)
        self.k = k

    def fit(self, x, y):
        self.estimator.fit(x, y)
        print(self.estimator.best_estimator_)

        folds = KFold(self.k)
        compressors = {}
        for i, (train_idx, val_idx) in enumerate(folds.split(x, y)):
            x_tr = x.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_val = y.iloc[val_idx]
            if isinstance(self.estimator, GridSearchCV):
                estimator = copy.deepcopy(self.estimator.best_estimator_)
            else:
                estimator = copy.deepcopy(self.estimator)
            compressors[i] = FCCA(estimator, p1=self.p1, p2=self.p2, lambda0=self.lambda0, lambda1=self.lambda1, compress=False, Q=None)
            compressors[i].fit(x_tr, y_tr, x_val, y_val)

        self.tao = compressors[0].tao
        for i in range(1,self.k):
            self.tao = pd.concat((self.tao, compressors[i].tao))

        if self.compress:
            self.tao = self.compressThresholds(self.tao)

        if self.Q is not None:
            self.chooseQ(x, y, split=folds.split(x, y))

class GTRE(Discretizer):
    def fit(self, x, y):
        _,_,self.tao,_ = compute_thresholds(x, y, 100, 1)
        self.tao = pd.DataFrame(data={'Feature':[self.tao[i].split('<=')[0] for i in range(len(self.tao))],
                                      'Threshold':[float(self.tao[i].split('<=')[1]) for i in range(len(self.tao))]})