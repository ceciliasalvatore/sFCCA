import warnings
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from gosdt.model.gosdt import GOSDT
from matplotlib import pyplot as plt

from dataset import Dataset
from discretize import FCCA, TotalDiscretizer, GTRE
from performance import Performance
from config import cfg

if __name__ == '__main__':
    datasets = ['boston', 'arrhythmia', 'ionosphere', 'magic', 'particle', 'vehicle']
    datasets = ['boston']

    q_list = [0, 0.5, 0.6, 0.7, 0.8, 0.95, 0.97, 0.98, 0.99]

    for name in datasets:
        np.random.seed = cfg.seed
        cfg.set_name(name)
        dataset = Dataset()

        if cfg.logger:
            print(f"---- {name}-{cfg.seed} ----", file=open(cfg.get_filename('logger'), mode='w'))
            if cfg.load_thresholds==False:
                print(f"p, t", file=open(cfg.get_filename('counterfactuals_time'), mode='w'))

        model = GridSearchCV(RandomForestClassifier(n_estimators=cfg.rf_nestimators, random_state=cfg.seed), param_grid={'max_depth':[3,4,6,8]})
        model.fit(dataset.get_x(), dataset.get_y())
        cfg.rf_depth = model.best_params_['max_depth']

        discretizers = {'total': TotalDiscretizer(), 'gtre': GTRE(), 'fcca': FCCA(RandomForestClassifier(max_depth=cfg.rf_depth, n_estimators=cfg.rf_nestimators, random_state=cfg.seed), p1=cfg.p1, p2=cfg.p2, lambda0=cfg.lambda0, lambda1=cfg.lambda1, compress=True, Q=None)}
        if name != 'boston' and name != 'ionosphere':
            discretizers = {'gtre': GTRE(), 'fcca': FCCA(RandomForestClassifier(max_depth=cfg.rf_depth, n_estimators=cfg.rf_nestimators, random_state=cfg.seed), p1=cfg.p1, p2=cfg.p2, lambda0=cfg.lambda0, lambda1=cfg.lambda1, compress=True, Q=None)}

        # if name == 'arrhythmia' or name == 'magic':
        #    discretizers = {'gtre':GTRE(), 'fcca':FCCA_KFold(GridSearchCV(RandomForestClassifier(n_estimators=50, random_state=cfg.seed), {'max_depth':[3,4,6,8,10]}), k=cfg.k, p1=0.7, p2=0.2, lambda0=0.1, lambda1=1, compress=True, Q=None)}
        models = {'cart': DecisionTreeClassifier(max_depth=cfg.dt_depth), 'gosdt': GOSDT({'regularization': 1 / ((cfg.k - 1) * dataset.get_x().shape[0] / (cfg.k)), 'depth_budget': cfg.dt_depth+1, 'time_limit': cfg.gosdt_timelimit})}

        performance = {}

        kfold = KFold(n_splits=cfg.k)

        for d in discretizers.keys():
            for m in models.keys():
                for i in range(cfg.k):
                    performance[(d,m,i)] = Performance()

        for i, (train_idx, val_idx) in enumerate(kfold.split(dataset.get_x(), dataset.get_y())):
            cfg.fold = i
            x_tr = dataset.get_x().iloc[train_idx]
            y_tr = dataset.get_y().iloc[train_idx]
            x_val = dataset.get_x().iloc[val_idx]
            y_val = dataset.get_y().iloc[val_idx]
            for d in discretizers.keys():

                x_tr_discr, y_tr_discr = discretizers[d].fit_transform(x_tr, y_tr)
                x_val_discr, y_val_discr = discretizers[d].transform(x_val, y_val)

                if cfg.logger:
                    if d == 'fcca':
                        print(f"{d} fold {i}: {discretizers[d].estimator.__class__} target model with {discretizers[d].estimator.n_estimators} estimators and max_depth {discretizers[d].estimator.max_depth}", file=open(cfg.get_filename('logger'), mode='a'))
                        print(f"{d} fold {i}: Target accuracy train set: {accuracy_score(y_tr, discretizers[d].estimator.predict(x_tr))}", file=open(cfg.get_filename('logger'), mode='a'))
                        print(f"{d} fold {i}: Target accuracy validation set: {accuracy_score(y_val, discretizers[d].estimator.predict(x_val))}", file=open(cfg.get_filename('logger'), mode='a'))
                    print(f"{d} fold {i}: {x_tr_discr.shape[1]} thresholds", file=open(cfg.get_filename('logger'), mode='a'))

                for m in models.keys():
                    # try:
                    model = copy.deepcopy(models[m])
                    model.fit(x_tr_discr, y_tr_discr)

                    if cfg.logger:
                        print(f"{d}-{m} fold {i}", file=open(cfg.get_filename('logger'),mode='a'))
                        if isinstance(model, DecisionTreeClassifier):
                            print(tree.export_text(model, feature_names=x_tr_discr.columns.to_list()), file=open(cfg.get_filename('logger'),mode='a'))
                        if isinstance(model, GOSDT):
                            print(f"{model.tree}", file=open(cfg.get_filename('logger'),mode='a'))
                            if model.timeout:
                                print(f"{d}-{m} fold {i} reached timeout", file=open(cfg.get_filename('logger'),mode='a'))
                                warnings.warn("GOSDT reached timeout")

                    print(f"{d}-{m} fold {i}: accuracy train set {accuracy_score(y_tr_discr, model.predict(x_tr_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                    print(f"{d}-{m} fold {i}: accuracy validation set {accuracy_score(y_val_discr, model.predict(x_val_discr))}", file=open(cfg.get_filename('logger'), mode='a'))

                    performance[(d, m, i)].accuracy = accuracy_score(y_val_discr, model.predict(x_val_discr))

                    for q in q_list:
                        if d == 'fcca':
                            tao_q = discretizers[d].selectThresholds(q)
                            performance[(d, m, i)].compression[q] = discretizers[d].compression_rate(x_val, y_val, tao_q)
                            performance[(d, m, i)].inconsistency[q] = discretizers[d].inconsistency_rate(x_val, y_val, tao_q)
                        else:
                            performance[(d, m, i)].compression[q] = discretizers[d].compression_rate(x_val, y_val)
                            performance[(d, m, i)].inconsistency[q] = discretizers[d].inconsistency_rate(x_val, y_val)
                    if isinstance(model, GOSDT):
                        features = model.tree.features()
                    else:
                        features = model.feature_names_in_[model.tree_.feature[model.tree_.feature >= 0]]
                    features = np.unique([f.split('<')[0] for f in features])
                    performance[(d, m, i)].n_features = len(features)
                    performance[(d, m, i)].n_thresholds = x_tr_discr.shape[1]
                # except:
                #    warnings.warn(f"{m} could not be trained for discretization strategy {d}")

        fig = plt.figure()
        x = [f'{d}_{m}' for d in discretizers.keys() for m in models.keys()]
        y = [np.mean([performance[d,m,i].accuracy for i in range(cfg.k)]) for d in discretizers.keys() for m in models.keys()]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
        color_map = dict(zip(x,colors))
        plt.bar(x, y, color=colors)
        plt.ylim((max(0, np.min(y) - 0.01), min(1, np.max(y) + 0.01)))
        plt.title(f"{name} - Accuracy")
        plt.savefig(cfg.get_filename('accuracy', 'png'))
        plt.close(fig)

        fig = plt.figure()
        for d in discretizers.keys():
            m = list(models.keys())[0]
            y = []
            for q in q_list:
                y.append(np.mean([performance[d,m,i].compression[q] for i in range(cfg.k)]))
            plt.plot(q_list, y, color=color_map[f'{d}_{m}'], marker='o', linewidth=3, label=d)
        plt.legend(loc="upper left")
        plt.ylim(np.min(y)-0.1, np.max(y)+0.1)
        plt.title(f"{name} - Compression")
        plt.savefig(cfg.get_filename('compression', 'png'))
        plt.close(fig)

        fig = plt.figure()
        for d in discretizers.keys():
            m = list(models.keys())[0]
            y = []
            for q in q_list:
                y.append(np.mean([performance[d, m, i].inconsistency[q] for i in range(cfg.k)]))
            plt.plot(q_list, y, color=color_map[f'{d}_{m}'], marker='o', linewidth=3, label=d)
        plt.legend(loc="upper left")
        plt.ylim(np.min(y)-0.1, np.max(y)+0.1)
        plt.title(f"{name} - Inconsistency")
        plt.savefig(cfg.get_filename('inconsistency', 'png'))
        plt.close(fig)

        fig = plt.figure()
        y = [np.mean([performance[d,m,i].n_features for i in range(cfg.k)]) for d in discretizers.keys() for m in models.keys()]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
        plt.bar(x, y, color=colors)
        plt.ylim((max(0, np.min(y) - 1), np.max(y) + 1))
        plt.title(f"{name} - Number of features")
        plt.savefig(cfg.get_filename('n_features', 'png'))
        plt.close(fig)

        fig = plt.figure()
        y = [np.mean([performance[d,m,i].n_thresholds for i in range(cfg.k)]) for d in discretizers.keys() for m in models.keys()]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
        plt.bar(x, y, color=colors)
        plt.ylim((max(0, np.min(y) - 100), np.max(y) + 100))
        plt.title(f"{name} - Number of thresholds")
        plt.savefig(cfg.get_filename('n_thresholds', 'png'))
        plt.close(fig)
