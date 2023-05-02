import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from gosdt.model.gosdt import GOSDT

from performance import Performance
from dataset import Dataset
from discretize import FCCA
from config import cfg

if __name__ == '__main__':
    datasets = ['ionosphere']

    for name in datasets:
        np.random.seed = cfg.seed
        cfg.set_name(name)
        dataset = Dataset(cfg.train_size)

        if cfg.logger:
            print(f"---- {name}-{cfg.seed} ----", file=open(cfg.get_filename('logger'), mode='w'))
            if cfg.load_thresholds==False:
                print(f"p, t", file=open(cfg.get_filename('counterfactuals_time'), mode='w'))

        model = GridSearchCV(GradientBoostingClassifier(n_estimators=100, random_state=cfg.seed, learning_rate=0.1), param_grid={'max_depth': [1,2,3,4]}, cv=cfg.k)
        model.fit(dataset.get_x(), dataset.get_y())
        print("Gradient Boosting")
        print(model.cv_results_['mean_test_score'])

        model = GridSearchCV(RandomForestClassifier(random_state=cfg.seed), param_grid=[{'n_estimators': [100], 'max_depth': [3,4]}, {'n_estimators': [50], 'max_depth': [6]}], cv=cfg.k)
        model.fit(dataset.get_x(), dataset.get_y())
        print("Random Forest")
        rf_depth = model.best_params_['max_depth']
        rf_nestimators = model.best_params_['n_estimators']
        print(model.cv_results_['mean_test_score'])

        model = GridSearchCV(LinearSVC(), param_grid={'C': [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1]}, cv=cfg.k)
        model.fit(dataset.get_x(), dataset.get_y())
        C = model.best_params_['C']
        print("SVM")
        print(model.cv_results_['mean_test_score'])

        targets = ['GradientBoosting', 'RandomForest', 'SVM']
        performance = {}

        q_list = [0, 0.7, 0.8, 0.9, 0.95]
        kfold = StratifiedKFold(n_splits=cfg.k)

        for i, (train_idx, val_idx) in enumerate(kfold.split(dataset.get_x(), dataset.get_y())):
            x_tr = dataset.get_x().iloc[train_idx]
            y_tr = dataset.get_y().iloc[train_idx]
            x_val = dataset.get_x().iloc[val_idx]
            y_val = dataset.get_y().iloc[val_idx]

            for t in targets:
                if t == 'GradientBoosting':
                    target = GradientBoostingClassifier(max_depth=1, n_estimators=100, random_state=cfg.seed,
                                                        learning_rate=0.1)
                elif t == 'RandomForest':
                    target = RandomForestClassifier(max_depth=rf_depth, n_estimators=rf_nestimators,
                                                    random_state=cfg.seed)
                elif t == 'SVM':
                    target = LinearSVC(C=C)
                print(target)

                discretizer = FCCA(target, p1=cfg.p1, lambda0=cfg.lambda0, lambda1=cfg.lambda1)
                discretizer.fit(x_tr, y_tr)

                for Q in q_list:
                    if (t,Q) not in performance.keys():
                        performance[t, Q] = Performance()

                    tao_q = discretizer.selectThresholds(Q)
                    x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr, tao_q)
                    x_val_discr, y_val_discr = discretizer.transform(x_val, y_val, tao_q)
                    model = GOSDT({'regularization': cfg.regularization_factor / len(x_tr_discr),
                                    'depth_budget': cfg.dt_depth + 1, 'time_limit': cfg.gosdt_timelimit})
                    model.fit(x_tr_discr, y_tr_discr)
                    performance[t, Q].discretizer[i] = discretizer
                    performance[t, Q].model[i] = model
                    performance[t, Q].accuracy[i] = accuracy_score(y_val_discr, model.predict(x_val_discr))
                    performance[t, Q].compression[i] = discretizer.compression_rate(x_val, y_val, tao_q)
                    performance[t, Q].inconsistency[i] = discretizer.inconsistency_rate(x_val, y_val, tao_q)

                    features = model.tree.features()
                    features = np.unique([f.split('<')[0] for f in features])
                    performance[t, Q].n_features[i] = len(features)
                    performance[t, Q].n_thresholds[i] = x_tr_discr.shape[1]

        colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))
        color_map = dict(zip(targets, colors))

        types = ['accuracy', 'compression', 'inconsistency', 'n_features']
        for type_ in types:
            fig = plt.figure()
            for t in targets:
                x = []
                for Q in q_list:
                    x.append(np.mean(list(performance[t, Q].__getattribute__(type_).values())))
                plt.plot(q_list, x, color=color_map[t], label=t, linewidth='3', markersize='15', marker='.')
            plt.xlabel('Q')
            plt.legend()
            plt.title(f"{cfg.name} - {type_}")
            plt.savefig(cfg.get_filename(type_, 'png'))
            plt.close(fig)

        for t in targets:
            for Q in [0, 0.7]:
                ThresholdsFolds = pd.DataFrame(index=dataset.feature_columns, columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=0)
                for fold in performance[t,0].discretizer.keys():
                    Thresholds = performance[t,0].discretizer[fold].selectThresholds(Q).copy()
                    if 'Count' not in Thresholds.columns:
                        Thresholds['Count'] = 1
                    Thresholds['Threshold'] = np.floor(Thresholds['Threshold'].astype(float) * 10) / 10
                    Thresholds = Thresholds.groupby(['Feature', 'Threshold']).sum().reset_index()
                    Thresholds = Thresholds.reset_index().pivot(index='Feature', columns='Threshold', values='Count')
                    Thresholds = Thresholds.replace(np.nan, 0)
                    ThresholdsFolds.update(Thresholds)
                ThresholdsFolds = ThresholdsFolds / np.max(ThresholdsFolds.to_numpy())
                ThresholdsFolds[ThresholdsFolds > 0] = 1
                sns.heatmap(ThresholdsFolds, cmap="YlOrBr", cbar=False)
                plt.ylabel('Features')
                plt.xlabel('Thresholds')
                title = f'FCCA Q={Q} with {t}'
                plt.title(title)
                plt.yticks([])
                plt.savefig(cfg.get_filename(title, 'png'))
                plt.close(fig)
