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
    datasets = ['boston']

    for name in datasets:
        np.random.seed = cfg.seed
        cfg.set_name(name)
        dataset = Dataset(cfg.train_size)

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
                    if t not in performance.keys():
                        performance[t] = {}
                    if Q not in performance[t].keys():
                        performance[t][Q] = Performance()

                    tao_q = discretizer.selectThresholds(Q)
                    x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr, tao_q)
                    x_val_discr, y_val_discr = discretizer.transform(x_val, y_val, tao_q)
                    model = GOSDT({'regularization': cfg.regularization_factor / len(x_tr_discr),
                                    'depth_budget': cfg.dt_depth + 1, 'time_limit': cfg.gosdt_timelimit})
                    model.fit(x_tr_discr, y_tr_discr)
                    performance[t][Q].discretizer[i] = discretizer
                    performance[t][Q].model[i] = model
                    performance[t][Q].accuracy[i] = accuracy_score(y_val_discr, model.predict(x_val_discr))
                    performance[t][Q].compression[i] = discretizer.compression_rate(x_val, y_val, tao_q)
                    performance[t][Q].inconsistency[i] = discretizer.inconsistency_rate(x_val, y_val, tao_q)

                    features = model.tree.features()
                    features = np.unique([f.split('<')[0] for f in features])
                    performance[t][Q].n_features[i] = len(features)
                    performance[t][Q].n_thresholds[i] = x_tr_discr.shape[1]

        colors = plt.cm.rainbow(np.linspace(0, 1, len(targets)))
        color_map = dict(zip(targets, colors))

        types = ['accuracy', 'compression', 'inconsistency', 'n_features']
        for type_ in types:
            fig = plt.figure()
            for t in targets:
                x = []
                for Q in q_list:
                    x.append(np.mean(list(performance[t][Q].__getattribute__(type_).values())))
                plt.plot(q_list, x, color=color_map[t], label=t, linewidth='3', markersize='15', marker='.')
            plt.xlabel('Q')
            plt.legend()
            plt.title(f"{cfg.name} - {type_}")
            plt.savefig(cfg.get_filename(type_, 'png'))
            plt.close(fig)

        for t in targets:
            ThresholdsFolds = pd.DataFrame(index=dataset.feature_columns, columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=np.nan)
            vmap = {'N':-1}
            i=0
            for Q in q_list:
                vmap[Q]=i
                i+=1

            for Q in q_list:
                for fold in performance[t][Q].discretizer.keys():
                    Thresholds = performance[t][Q].discretizer[fold].selectThresholds(Q).copy()
                    Thresholds['Value'] = vmap[Q]
                    Thresholds['Threshold'] = np.floor(Thresholds['Threshold'].astype(float) * 10) / 10
                    Thresholds = Thresholds.groupby(['Feature', 'Threshold'])['Value'].min().reset_index()
                    Thresholds = Thresholds.reset_index().pivot(index='Feature', columns='Threshold', values='Value')
                    ThresholdsFolds.update(Thresholds)
            ThresholdsFolds = ThresholdsFolds.replace(np.nan, -1)

            cmap = sns.color_palette("YlOrBr", len(vmap)+1)
            cmap.pop(1)

            if len(dataset.feature_columns) >= 50:
                figsize = (6.4, 4.8 * 1.5)
            else:
                figsize = (6.4, 4.8)
            fig = plt.figure(figsize=figsize)
            ax = sns.heatmap(ThresholdsFolds, cmap=cmap, vmin=-1.5, vmax=i-0.5)
            # Get the colorbar object from the Seaborn heatmap
            colorbar = ax.collections[0].colorbar
            # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
            colorbar.set_ticks(list(vmap.values()))
            colorbar.set_ticklabels(vmap.keys())

            plt.ylabel('Features')
            plt.xlabel('Thresholds')
            title = f'FCCA {t}'
            plt.title(f'{cfg.name} - {title}')
            plt.yticks([])
            plt.savefig(cfg.get_filename(title, 'png'))
            plt.close(fig)
