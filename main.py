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
from plots import plot
from config import cfg

if __name__ == '__main__':
    datasets = ['boston', 'arrhythmia', 'ionosphere', 'magic', 'particle', 'vehicle']
    datasets = ['magic', 'particle', 'vehicle']

    for name in datasets:
        np.random.seed = cfg.seed
        cfg.set_name(name)
        dataset = Dataset(cfg.train_size)

        if cfg.logger:
            print(f"---- {name}-{cfg.seed} ----", file=open(cfg.get_filename('logger'), mode='w'))
            if cfg.load_thresholds==False:
                print(f"p, t", file=open(cfg.get_filename('counterfactuals_time'), mode='w'))

        model = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=cfg.seed), param_grid={'max_depth':[3,4,6,8,10]}, cv=cfg.k)
        model.fit(dataset.get_x(), dataset.get_y())
        cfg.rf_depth = model.best_params_['max_depth']

        if name == 'boston':
            cfg.rf_depth = 3
        if name == 'arrhythmia':
            cfg.rf_depth = 6
        if cfg.rf_depth > 6:
            cfg.rf_nestimators = 30
            #cfg.dt_depth = 4
        if cfg.rf_depth > 4:
            cfg.regularization_factor = 1
        else:
            cfg.regularization_factor = 10

        if cfg.logger:
            print(f"cv scores: {model.cv_results_['mean_test_score']}", file=open(cfg.get_filename('logger'), mode='a'))
            print(f"Best target model: {cfg.rf_depth}", file=open(cfg.get_filename('logger'), mode='a'))
        discretizers = ['continuous', 'gtre', 'fcca']
        #if name != 'boston' and name != 'ionosphere':
        #    discretizers = ['gtre', 'fcca']
           # {'total': TotalDiscretizer(), 'gtre': GTRE(), 'fcca': FCCA(RandomForestClassifier(max_depth=cfg.rf_depth, n_estimators=cfg.rf_nestimators, random_state=cfg.seed), p1=cfg.p1, p2=cfg.p2, lambda0=cfg.lambda0, lambda1=cfg.lambda1, compress=True, Q=None)}
        #if name != 'boston' and name != 'ionosphere':
        #    discretizers = {'gtre': GTRE(), 'fcca': FCCA(RandomForestClassifier(max_depth=cfg.rf_depth, n_estimators=cfg.rf_nestimators, random_state=cfg.seed), p1=cfg.p1, p2=cfg.p2, lambda0=cfg.lambda0, lambda1=cfg.lambda1, compress=True, Q=None)}

        # if name == 'arrhythmia' or name == 'magic':
        #    discretizers = {'gtre':GTRE(), 'fcca':FCCA_KFold(GridSearchCV(RandomForestClassifier(n_estimators=50, random_state=cfg.seed), {'max_depth':[3,4,6,8,10]}), k=cfg.k, p1=0.7, p2=0.2, lambda0=0.1, lambda1=1, compress=True, Q=None)}
        models = ['cart', 'gosdt']
        #models = {'cart': DecisionTreeClassifier(max_depth=cfg.dt_depth), 'gosdt': GOSDT({'regularization': 1 / ((cfg.k - 1) * dataset.get_x().shape[0] / (cfg.k)), 'depth_budget': cfg.dt_depth+1, 'time_limit': cfg.gosdt_timelimit, 'verbose':True})}

        performance = {}

        kfold = StratifiedKFold(n_splits=cfg.k)

        for i, (train_idx, val_idx) in enumerate(kfold.split(dataset.get_x(), dataset.get_y())):
            cfg.fold = i
            x_tr = dataset.get_x().iloc[train_idx]
            y_tr = dataset.get_y().iloc[train_idx]
            x_val = dataset.get_x().iloc[val_idx]
            y_val = dataset.get_y().iloc[val_idx]
            for d in discretizers:
                if d == 'continuous':
                    discretizer = NoDiscretization()
                elif d == 'total':
                    discretizer = TotalDiscretizer()
                elif d == 'gtre':
                    discretizer = GTRE()
                elif d == 'fcca':
                    discretizer = FCCA(RandomForestClassifier(max_depth=cfg.rf_depth, n_estimators=cfg.rf_nestimators, random_state=cfg.seed), p1=cfg.p1, p2=cfg.p2, lambda0=cfg.lambda0, lambda1=cfg.lambda1, compress=True, Q=cfg.Q)
                    #estimator = GridSearchCV(RandomForestClassifier(n_estimators=cfg.rf_nestimators, random_state=cfg.seed), param_grid={'max_depth':[3,4,6,8,10]}, cv=cfg.k)
                    #discretizer = FCCA(estimator, p1=cfg.p1, p2=cfg.p2, lambda0=cfg.lambda0, lambda1=cfg.lambda1, compress=True, Q=cfg.Q)
                else:
                    raise NotImplementedError(f'Unknown discretization strategy {d}')
                x_tr_discr, y_tr_discr = discretizer.fit_transform(x_tr, y_tr)
                x_val_discr, y_val_discr = discretizer.transform(x_val, y_val)

                if cfg.logger:
                    if d == 'fcca':
                        print(f"{d} fold {i}: {discretizer.estimator.__class__} target model with {discretizer.estimator.n_estimators} estimators and max_depth {discretizer.estimator.max_depth}", file=open(cfg.get_filename('logger'), mode='a'))
                        print(f"{d} fold{i}: {np.unique(y_tr_discr,return_counts=True)[1]}, {np.unique(y_val_discr,return_counts=True)[1]}")
                        print(f"{d} fold {i}: Target accuracy train set: {accuracy_score(y_tr, discretizer.estimator.predict(x_tr))}", file=open(cfg.get_filename('logger'), mode='a'))
                        print(f"{d} fold {i}: Target accuracy validation set: {accuracy_score(y_val, discretizer.estimator.predict(x_val))}", file=open(cfg.get_filename('logger'), mode='a'))
                    print(f"{d} fold {i}: {x_tr_discr.shape[1]} thresholds", file=open(cfg.get_filename('logger'), mode='a'))

                for m in models:
                    if m == 'gosdt' and (d=='continuous' or d=='total'):
                        break

                    if m == 'cart':
                        model = DecisionTreeClassifier(max_depth=cfg.dt_depth)
                    elif m == 'gosdt':
                        model = GOSDT({'regularization': cfg.regularization_factor/len(x_tr_discr), 'depth_budget': cfg.dt_depth+1, 'time_limit': cfg.gosdt_timelimit, 'verbose':True})
                    else:
                        raise NotImplementedError(f'Unknown model {m}')

                    if (d,m) not in performance.keys():
                        performance[(d,m)] = Performance()

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
                            else:
                                print(f"{d}-{m} fold {i} GOSDT completed in {model.time} s", file=open(cfg.get_filename('logger'),mode='a'))


                    print(f"{d}-{m} fold {i}: accuracy train set {accuracy_score(y_tr_discr, model.predict(x_tr_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                    print(confusion_matrix(y_tr_discr, model.predict(x_tr_discr)), file=open(cfg.get_filename('logger'), mode='a'))
                    print(f"{d}-{m} fold {i}: accuracy validation set {accuracy_score(y_val_discr, model.predict(x_val_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                    print(confusion_matrix(y_val_discr, model.predict(x_val_discr)), file=open(cfg.get_filename('logger'), mode='a'))

                    performance[(d,m)].discretizer[i] = discretizer
                    performance[(d,m)].model[i] = model
                    performance[(d, m)].accuracy[i] = accuracy_score(y_val_discr, model.predict(x_val_discr))

                    performance[(d,m)].compression[i] = {}
                    performance[(d,m)].inconsistency[i] = {}

                    for q in cfg.q_list:
                        if d == 'fcca':
                            tao_q = discretizer.selectThresholds(q)
                            performance[(d, m)].compression[i][q] = discretizer.compression_rate(x_val, y_val, tao_q)
                            performance[(d, m)].inconsistency[i][q] = discretizer.inconsistency_rate(x_val, y_val, tao_q)
                        else:
                            performance[(d, m)].compression[i][q] = discretizer.compression_rate(x_val, y_val)
                            performance[(d, m)].inconsistency[i][q] = discretizer.inconsistency_rate(x_val, y_val)
                    if isinstance(model, GOSDT):
                        features = model.tree.features()
                    else:
                        features = model.feature_names_in_[model.tree_.feature[model.tree_.feature >= 0]]
                    features = np.unique([f.split('<')[0] for f in features])
                    performance[(d, m)].n_features[i] = len(features)
                    performance[(d, m)].n_thresholds[i] = x_tr_discr.shape[1]

        plot(performance, 'accuracy')

        try:
            performance_test = {}
            x_ts = dataset.get_x_ts()
            y_ts = dataset.get_y_ts()
            for (d,m) in performance.keys():
                performance_test[(d,m)] = Performance()
                for i in performance[(d,m)].model.keys():
                    x_ts_discr, y_ts_discr = performance[(d,m)].discretizer[i].transform(x_ts, y_ts)
                    performance_test[(d,m)].accuracy[i] = accuracy_score(y_ts_discr, performance[(d,m)].model[i].predict(x_ts_discr))
            plot(performance_test, 'accuracy_test')
        except:
            pass