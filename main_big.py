import warnings
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from gosdt.model.gosdt import GOSDT
from matplotlib import pyplot as plt

from dataset import Dataset
from discretize import NoDiscretization, FCCA, TotalDiscretizer, GTRE
from performance import Performance
from config import cfg

if __name__ == '__main__':
    datasets = ['boston', 'arrhythmia', 'ionosphere', 'magic', 'particle', 'vehicle']
    datasets = ['magic','particle','vehicle']

    for name in datasets:
        cfg.set_name(name)
        dataset = Dataset()

        if cfg.logger:
            print(f"---- {name} ----", file=open(cfg.get_filename('logger'), mode='w'))
            if cfg.load_thresholds == False:
                print(f"p, t", file=open(cfg.get_filename('counterfactuals_time'), mode='w'))

        for seed in np.arange(100,110):
            cfg.seed = seed
            np.random.seed = cfg.seed


            if cfg.logger:
                print(f"---- {name}-{cfg.seed} ----", file=open(cfg.get_filename('logger'), mode='a'))
            x_tr = dataset.get_x().sample(n=1000)
            y_tr = dataset.get_y().loc[x_tr.index]

            x_ts = dataset.get_x().drop(x_tr.index)
            y_ts = dataset.get_y().drop(x_tr.index)

            model = GridSearchCV(RandomForestClassifier(n_estimators=30, random_state=cfg.seed), param_grid={'max_depth': [3, 4, 6, 8, 10]}, cv=cfg.k)
            model.fit(x_tr, y_tr)
            print(model.cv_results_['mean_test_score'], file=open(cfg.get_filename('logger'), mode='a'))
            print(f"Best Random Forest {model.best_params_}", file=open(cfg.get_filename('logger'), mode='a'))
            cfg.rf_depth = model.best_params_['max_depth']

            discretizer = FCCA(estimator=RandomForestClassifier(n_estimators=cfg.rf_nestimators, max_depth=cfg.rf_depth, random_state=cfg.seed))
            discretizer.estimator.fit(x_tr, y_tr)
            x0, y0 = discretizer.getRelevant(x_tr, y_tr)

            print(f"Number of Counterfactuals {len(x0)}", file=open(cfg.get_filename('logger'),mode='a'))

            print(f'Train accuracy {accuracy_score(y_tr, discretizer.estimator.predict(x_tr))}', file=open(cfg.get_filename('logger'),mode='a'))
            print(f'Validation accuracy {accuracy_score(y_ts, discretizer.estimator.predict(x_ts))}', file=open(cfg.get_filename('logger'),mode='a'))

