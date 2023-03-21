import warnings
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from gosdt.model.gosdt import GOSDT
from matplotlib import pyplot as plt

from dataset import Dataset
from discretize import FCCA_KFold, TotalDiscretizer, GTRE
from performance import Performance
from config import cfg

if __name__ == '__main__':
    datasets = ['boston','arrhythmia','ionosphere','magic','particle','vehicle']
    datasets = ['magic']

    for name in datasets:
        for seed in [100]:
            cfg.seed = seed
            np.random.seed = cfg.seed
            cfg.set_name(name)
            dataset = Dataset()

            discretizers = {'total':TotalDiscretizer(), 'gtre':GTRE(), 'fcca':FCCA_KFold(GridSearchCV(RandomForestClassifier(n_estimators=50, random_state=cfg.seed), {'max_depth':[3,4,6,8,10]}), k=cfg.k, p1=1, p2=1, lambda0=0.1, lambda1=1, compress=True, Q=None)}
            if name == 'arrhythmia' or name == 'magic':
                discretizers = {'gtre':GTRE(), 'fcca':FCCA_KFold(GridSearchCV(RandomForestClassifier(n_estimators=50, random_state=cfg.seed), {'max_depth':[3,4,6,8,10]}), k=cfg.k, p1=0.7, p2=0.2, lambda0=0.1, lambda1=1, compress=True, Q=None)}
            models = {'cart':DecisionTreeClassifier(max_depth=3), 'gosdt':GOSDT({'regularization':1/dataset.get_x_tr().shape[0], 'depth_budget': 3, 'time_limit': 10*60})}

            performance = {}
            for d in discretizers.keys():
                x_tr_discr, y_tr_discr = discretizers[d].fit_transform(dataset.get_x_tr(), dataset.get_y_tr())
                x_ts_discr, y_ts_discr = discretizers[d].transform(dataset.get_x_ts(), dataset.get_y_ts())
                print(f"Compression rate {discretizers[d].compression_rate(dataset.get_x_ts(), dataset.get_y_ts())}")
                print(f"Inconsistency rate {discretizers[d].inconsistency_rate(dataset.get_x_ts(), dataset.get_y_ts())}")

                for m in models.keys():
                    performance[(d, m)] = Performance()

                    try:
                        model = copy.deepcopy(models[m])
                        model.fit(x_tr_discr, y_tr_discr)
                        print(f"{accuracy_score(y_ts_discr, model.predict(x_ts_discr))}")
                        performance[(d,m)].accuracy = accuracy_score(y_ts_discr, model.predict(x_ts_discr))
                        performance[(d,m)].compression = discretizers[d].compression_rate(dataset.get_x_ts(), dataset.get_y_ts())
                        performance[(d,m)].inconsistency = discretizers[d].inconsistency_rate(dataset.get_x_ts(), dataset.get_y_ts())
                        if isinstance(model, GOSDT):
                            features = model.tree.features()
                        else:
                            features = model.feature_names_in_[model.tree_.feature[model.tree_.feature >= 0]]
                        performance[(d,m)].features = np.unique([f.split('<')[0] for f in features])
                        performance[(d,m)].n_features = len(performance[(d,m)].features)
                        performance[(d,m)].n_thresholds = x_tr_discr.shape[1]
                    except:
                        warnings.warn(f"{m} could not be trained for discretization strategy {d}")
            fig = plt.figure()
            x = [f'{k[0]}_{k[1]}' for k in performance.keys()]
            y = [performance[k].accuracy for k in performance.keys()]
            colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
            plt.bar(x, y, color=colors)
            plt.ylim((max(0,np.min(y)-0.01), min(1,np.max(y)+0.01)))
            plt.title(f"{name} - Accuracy")
            plt.savefig(cfg.get_filename('accuracy','png'))
            plt.close(fig)

            fig = plt.figure()
            y = [performance[k].compression for k in performance.keys()]
            plt.bar(x, y, color=colors)
            plt.ylim((max(0,np.min(y)-0.01), min(1,np.max(y)+0.01)))
            plt.title(f"{name} - Compression")
            plt.savefig(cfg.get_filename('compression','png'))
            plt.close(fig)

            fig = plt.figure()
            y = [performance[k].inconsistency for k in performance.keys()]
            plt.bar(x, y, color=colors)
            plt.ylim((max(0,np.min(y)-0.01), min(1,np.max(y)+0.01)))
            plt.title(f"{name} - Inconsistency")
            plt.savefig(cfg.get_filename('inconsistency','png'))
            plt.close(fig)

            fig = plt.figure()
            y = [performance[k].n_features for k in performance.keys()]
            plt.bar(x, y, color=colors)
            plt.ylim((max(0,np.min(y)-1), np.max(y)+1))
            plt.title(f"{name} - Number of Features Used")
            plt.savefig(cfg.get_filename('n_features','png'))
            plt.close(fig)

            fig = plt.figure()
            y = [performance[k].n_thresholds for k in performance.keys()]
            plt.bar(x, y, color=colors)
            plt.ylim((max(0,np.min(y)-1), np.max(y)+1))
            plt.title(f"{name} - Number of Thresholds")
            plt.savefig(cfg.get_filename('n_thresholds','png'))
            plt.close(fig)
