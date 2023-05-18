import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from gosdt.model.gosdt import GOSDT
import time

from performance import Performance
from dataset import Dataset
from discretize import FCCA, NoDiscretization, TotalDiscretizer, GTRE
from config import cfg

if __name__ == '__main__':
    datasets = ['boston', 'arrhythmia', 'ionosphere', 'magic', 'particle', 'vehicle']

    for name in datasets:
        np.random.seed = cfg.seed
        cfg.set_name(name)
        dataset = Dataset(cfg.train_size)

        if cfg.logger:
            print(f"---- {name}-{cfg.seed} ----", file=open(cfg.get_filename('logger'), mode='w'))

        if cfg.target_model == GradientBoostingClassifier:
            model = GridSearchCV(GradientBoostingClassifier(n_estimators=cfg.target_nestimators,learning_rate=0.1,random_state=cfg.seed),param_grid={'max_depth':[1,2,3]},cv=cfg.k)
            model.fit(dataset.get_x(),dataset.get_y())
            if cfg.target_cross_validation:
                cfg.target_depth = model.best_params_['max_depth']

        if cfg.target_model == RandomForestClassifier:
            model = GridSearchCV(RandomForestClassifier(n_estimators=cfg.target_nestimators, random_state=cfg.seed), param_grid=[{'n_estimators': [100], 'max_depth': [3,4]}, {'n_estimators': [50], 'max_depth': [6]}], cv=cfg.k)
            model.fit(dataset.get_x(), dataset.get_y())
            if cfg.target_cross_validation:
                cfg.target_depth = model.best_params_['max_depth']
                cfg.target_nestimators = model.best_params_['n_estimators']
        if cfg.target_model == LinearSVC:
            model = GridSearchCV(LinearSVC(), param_grid={'C': [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1]}, cv=cfg.k)
            model.fit(dataset.get_x(), dataset.get_y())
            if cfg.target_cross_validation:
                cfg.target_C = model.best_params_['C']
        if cfg.logger:
            print(f"cv scores: {model.cv_results_['mean_test_score']}", file=open(cfg.get_filename('logger'), mode='a'))
            print(f"Target model: {cfg.target_depth}", file=open(cfg.get_filename('logger'), mode='a'))

        discretizers = ['continuous', 'gtre', 'fcca']

        q_list = [0, 0.7, 0.8, 0.9, 0.95]
        models = ['cart', 'gosdt']

        performance = {}

        kfold = StratifiedKFold(n_splits=cfg.k)

        for i, (train_idx, val_idx) in enumerate(kfold.split(dataset.get_x(), dataset.get_y())):
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
                    if cfg.target_model == GradientBoostingClassifier:
                        discretizer = GTRE(max_depth=cfg.target_depth, n_estimators=cfg.target_nestimators)
                    else:
                        discretizer = GTRE(max_depth=1, n_estimators=100)
                elif d == 'fcca':
                    if cfg.target_model == GradientBoostingClassifier:
                        discretizer = FCCA(GradientBoostingClassifier(max_depth=cfg.target_depth, n_estimators=cfg.target_nestimators, random_state=cfg.seed, learning_rate=0.1), p1=cfg.p1, lambda0=cfg.lambda0, lambda1=cfg.lambda1, lambda2=cfg.lambda2)
                    elif cfg.target_model == RandomForestClassifier:
                        discretizer = FCCA(RandomForestClassifier(max_depth=cfg.target_depth, n_estimators=cfg.target_nestimators, random_state=cfg.seed), p1=cfg.p1, lambda0=cfg.lambda0, lambda1=cfg.lambda1, lambda2=cfg.lambda2)
                    elif cfg.target_model == LinearSVC:
                        discretizer = FCCA(LinearSVC(C=cfg.target_C), p1=cfg.p1, lambda0=cfg.lambda0, lambda1=cfg.lambda1, lambda2=cfg.lambda2)
                    else:
                        raise NotImplementedError(f'Unknown discretizer model {cfg.target_model}')
                t0 = time.time()
                discretizer.fit(x_tr, y_tr)
                print(f"{d} fold {i}: discretizer fitted in {time.time()-t0} seconds", file=open(cfg.get_filename('logger'), mode='a'))
                if d == 'fcca':
                    if cfg.logger:
                        print(f"{d} fold {i}: {discretizer.estimator.__class__} target model", file=open(cfg.get_filename('logger'), mode='a'))
                        print(f"{d} fold {i}: Target accuracy train set: {accuracy_score(y_tr, discretizer.estimator.predict(x_tr))}", file=open(cfg.get_filename('logger'), mode='a'))
                        print(f"{d} fold {i}: Target accuracy validation set: {accuracy_score(y_val, discretizer.estimator.predict(x_val))}", file=open(cfg.get_filename('logger'), mode='a'))

                    for Q in q_list:
                        tao_q = discretizer.selectThresholds(Q)
                        x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr, tao_q)
                        x_val_discr, y_val_discr = discretizer.transform(x_val, y_val, tao_q)

                        if cfg.logger:
                            correctly_classified = np.where(discretizer.estimator.predict(x_tr)==y_tr)[0]
                            print(f"{d} fold {i} Q {Q}: inconsistency training set {discretizer.inconsistency_rate(x_tr.iloc[correctly_classified], y_tr.iloc[correctly_classified], tao_q)}", file=open(cfg.get_filename('logger'), mode='a'))
                            print(f"{d} fold {i} Q {Q}: {x_tr_discr.shape[1]} thresholds", file=open(cfg.get_filename('logger'), mode='a'))

                        for m in models:
                            if m == 'gosdt' and (d == 'continuous' or d == 'total'):
                                break
                            if m == 'cart':
                                model = DecisionTreeClassifier(max_depth=cfg.dt_depth)
                            elif m == 'gosdt':
                                model = GOSDT({'regularization': cfg.regularization_factor / len(x_tr_discr), 'depth_budget': cfg.dt_depth + 1, 'time_limit': cfg.gosdt_timelimit})
                            else:
                                raise NotImplementedError(f'Unknown model {m}')

                            if (d, m) not in performance.keys():
                                performance[(d, m)] = {}
                            if Q not in performance[(d,m)]:
                                performance[(d,m)][Q] = Performance()

                            model.fit(x_tr_discr, y_tr_discr)

                            if cfg.logger:
                                if isinstance(model, DecisionTreeClassifier):
                                    print(tree.export_text(model, feature_names=x_tr_discr.columns.to_list()), file=open(cfg.get_filename('logger'), mode='a'))
                                if isinstance(model, GOSDT):
                                    print(f"{model.tree}", file=open(cfg.get_filename('logger'), mode='a'))
                                    if model.timeout:
                                        print(f"{d}-{m} fold {i} Q {Q}: GOSDT reached timeout", file=open(cfg.get_filename('logger'), mode='a'))
                                        warnings.warn("GOSDT reached timeout")
                                    else:
                                        print(f"{d}-{m} fold {i} Q {Q}: GOSDT completed in {model.time} s", file=open(cfg.get_filename('logger'), mode='a'))
                                print(f"{d}-{m} fold {i} Q {Q}: accuracy train set {accuracy_score(y_tr_discr, model.predict(x_tr_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                                print(confusion_matrix(y_tr_discr, model.predict(x_tr_discr)), file=open(cfg.get_filename('logger'), mode='a'))
                                print(f"{d}-{m} fold {i} Q {Q}: accuracy validation set {accuracy_score(y_val_discr, model.predict(x_val_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                                print(confusion_matrix(y_val_discr, model.predict(x_val_discr)), file=open(cfg.get_filename('logger'), mode='a'))

                            performance[d, m][Q].discretizer[i] = discretizer
                            performance[d, m][Q].model[i] = model
                            performance[d, m][Q].accuracy[i] = accuracy_score(y_val_discr, model.predict(x_val_discr))
                            performance[d, m][Q].compression[i] = discretizer.compression_rate(x_val, y_val, tao_q)
                            performance[d, m][Q].inconsistency[i] = discretizer.inconsistency_rate(x_val, y_val, tao_q)

                            if isinstance(model, GOSDT):
                                features = model.tree.features()
                            else:
                                features = model.feature_names_in_[model.tree_.feature[model.tree_.feature >= 0]]
                            features = np.unique([f.split('<')[0] for f in features])
                            performance[d, m][Q].n_features[i] = len(features)
                            performance[d, m][Q].n_thresholds[i] = x_tr_discr.shape[1]
                else:
                    x_tr_discr, y_tr_discr = discretizer.transform(x_tr, y_tr)
                    x_val_discr, y_val_discr = discretizer.transform(x_val, y_val)

                    if cfg.logger:
                        print(f"{d} fold {i}: {x_tr_discr.shape[1]} thresholds", file=open(cfg.get_filename('logger'), mode='a'))

                    for m in models:
                        if m == 'gosdt' and (d == 'continuous' or d == 'total'):
                            break
                        if m == 'cart':
                            model = DecisionTreeClassifier(max_depth=cfg.dt_depth)
                        elif m == 'gosdt':
                            model = GOSDT({'regularization': cfg.regularization_factor / len(x_tr_discr), 'depth_budget': cfg.dt_depth + 1, 'time_limit': cfg.gosdt_timelimit})
                        else:
                            raise NotImplementedError(f'Unknown model {m}')

                        if (d, m) not in performance.keys():
                            performance[(d, m)] = Performance()

                        model.fit(x_tr_discr, y_tr_discr)

                        if cfg.logger:
                            if isinstance(model, DecisionTreeClassifier):
                                print(tree.export_text(model, feature_names=x_tr_discr.columns.to_list()), file=open(cfg.get_filename('logger'), mode='a'))
                            if isinstance(model, GOSDT):
                                print(f"{model.tree}", file=open(cfg.get_filename('logger'), mode='a'))
                                if model.timeout:
                                    print(f"{d}-{m} fold {i}: GOSDT reached timeout", file=open(cfg.get_filename('logger'), mode='a'))
                                    warnings.warn("GOSDT reached timeout")
                                else:
                                    print(f"{d}-{m} fold {i}: GOSDT completed in {model.time} s", file=open(cfg.get_filename('logger'), mode='a'))
                            print(f"{d}-{m} fold {i}: accuracy train set {accuracy_score(y_tr_discr, model.predict(x_tr_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                            print(confusion_matrix(y_tr_discr, model.predict(x_tr_discr)), file=open(cfg.get_filename('logger'), mode='a'))
                            print(f"{d}-{m} fold {i}: accuracy validation set {accuracy_score(y_val_discr, model.predict(x_val_discr))}", file=open(cfg.get_filename('logger'), mode='a'))
                            print(confusion_matrix(y_val_discr, model.predict(x_val_discr)), file=open(cfg.get_filename('logger'), mode='a'))

                        performance[d, m].discretizer[i] = discretizer
                        performance[d, m].model[i] = model
                        performance[d, m].accuracy[i] = accuracy_score(y_val_discr, model.predict(x_val_discr))
                        performance[d, m].compression[i] = discretizer.compression_rate(x_val, y_val)
                        performance[d, m].inconsistency[i] = discretizer.inconsistency_rate(x_val, y_val)

                        if isinstance(model, GOSDT):
                            features = model.tree.features()
                        else:
                            features = model.feature_names_in_[model.tree_.feature[model.tree_.feature >= 0]]
                        features = np.unique([f.split('<')[0] for f in features])
                        performance[d, m].n_features[i] = len(features)
                        performance[d, m].n_thresholds[i] = x_tr_discr.shape[1]


        colors = plt.cm.rainbow(np.linspace(0, 1, len(performance.keys())))
        color_map = dict(zip(performance.keys(), colors))

        types = ['accuracy', 'n_features']

        for type_ in types:
            fig = plt.figure()
            for (d,m) in performance.keys():
                if d == 'fcca':
                    x = []
                    for Q in performance[d,m].keys():
                        x.append(np.mean(list(performance[d,m][Q].__getattribute__(type_).values())))
                    plt.plot(performance[d,m].keys(), x, color=color_map[d,m], label=f'{d}_{m}', linewidth='3', markersize='15', marker='.')
                else:
                    y = np.mean(list(performance[d,m].__getattribute__(type_).values()))
                    plt.plot([0, 1], [y, y], color=color_map[d,m], label=f'{d}_{m}', linewidth='3')
            plt.xlabel('Q')
            plt.legend()
            plt.title(f"{cfg.name} - {type_}")
            plt.savefig(cfg.get_filename(type_, 'png'))
            plt.close(fig)

        types = ['compression','inconsistency']
        for type_ in types:
            fig = plt.figure()
            for (d, m) in performance.keys():
                if m == 'cart':
                    if d == 'fcca':
                        x = []
                        for Q in performance[d, m].keys():
                            x.append(np.mean(list(performance[d, m][Q].__getattribute__(type_).values())))
                        plt.plot(performance[d, m].keys(), x, color=color_map[d, m], label=f'{d}', linewidth='3',
                                 markersize='15', marker='.')
                    else:
                        y = np.mean(list(performance[d, m].__getattribute__(type_).values()))
                        plt.plot([0, 1], [y, y], color=color_map[d, m], label=f'{d}', linewidth='3')
            plt.xlabel('Q')
            plt.legend()
            plt.title(f"{cfg.name} - {type_}")
            plt.savefig(cfg.get_filename(type_, 'png'))
            plt.close(fig)

        try:
            performance_test = {}
            x_ts = dataset.get_x_ts()
            y_ts = dataset.get_y_ts()
            for (d,m) in performance.keys():
                if d == 'fcca':
                    performance_test[d, m] = {}
                    for Q in performance[d,m].keys():
                        performance_test[d,m][Q] = Performance()
                        for i in performance[(d, m)][Q].model.keys():
                            if cfg.logger:
                                print(f'{d}-{m} fold {i} Q {Q}: accuracy Target on test set: {accuracy_score(y_ts, performance[d,m][Q].discretizer[i].estimator.predict(x_ts))}', file=open(cfg.get_filename('logger'), mode='a'))
                            tao_q = performance[(d, m)][Q].discretizer[i].selectThresholds(Q)
                            x_ts_discr, y_ts_discr = performance[d,m][Q].discretizer[i].transform(x_ts, y_ts, tao_q)
                            performance_test[(d,m)][Q].accuracy[i] = accuracy_score(y_ts_discr, performance[(d,m)][Q].model[i].predict(x_ts_discr))
                            if cfg.logger:
                                print(f'{d}-{m} fold {i} Q {Q}: accuracy on test set: {performance_test[(d,m)][Q].accuracy[i]}', file=open(cfg.get_filename('logger'), mode='a'))
                else:
                    performance_test[(d,m)] = Performance()
                    for i in performance[(d,m)].model.keys():
                        x_ts_discr, y_ts_discr = performance[(d,m)].discretizer[i].transform(x_ts, y_ts)
                        performance_test[(d,m)].accuracy[i] = accuracy_score(y_ts_discr, performance[(d,m)].model[i].predict(x_ts_discr))

            fig = plt.figure()
            for (d,m) in performance_test.keys():
                if d == 'fcca':
                    x = []
                    for Q in performance_test[d,m].keys():
                        x.append(np.mean(list(performance_test[d,m][Q].accuracy.values())))
                    plt.plot(performance_test[d,m].keys(), x, color=color_map[d,m], label=f'{d}_{m}', linewidth='3', markersize='15', marker='.')
                else:
                    y = np.mean(list(performance_test[d,m].accuracy.values()))
                    plt.plot([0, 1], [y, y], color=color_map[d,m], label=f'{d}_{m}', linewidth='3')
            plt.xlabel('Q')
            plt.legend()
            plt.title(f"{cfg.name} - accuracy test")
            plt.savefig(cfg.get_filename(f'accuracy test', 'png'))
            plt.close(fig)
        except:
            pass

        if ('gtre','cart') in performance.keys():
            vmap = {'N':-1, 'Y':1}
            ThresholdsFolds = pd.DataFrame(index=dataset.feature_columns, columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=np.nan)
            for fold in performance['gtre','cart'].discretizer.keys():
                Thresholds = performance['gtre','cart'].discretizer[fold].tao.copy()
                Thresholds['Value'] = vmap['Y']
                Thresholds['Threshold'] = np.floor(Thresholds['Threshold'].astype(float) * 10) / 10
                Thresholds = Thresholds.groupby(['Feature', 'Threshold'])['Value'].min().reset_index()
                Thresholds = Thresholds.reset_index().pivot(index='Feature', columns='Threshold', values='Value')
                ThresholdsFolds.update(Thresholds)
            if len(dataset.feature_columns)>=50:
                figsize = (6.4, 4.8 * 1.5)
            else:
                figsize = (6.4, 4.8)
            fig = plt.figure(figsize=figsize)
            ThresholdsFolds = ThresholdsFolds.replace(np.nan, vmap['N'])
            cmap = sns.color_palette("YlOrBr", len(vmap)+len(q_list))
            for j in range(len(q_list)):
                cmap.pop(1)
            ax = sns.heatmap(ThresholdsFolds, cmap=cmap, vmin=-1.5, vmax=1.5)
            colorbar = ax.collections[0].colorbar
            # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
            colorbar.set_ticks(list(vmap.values()))
            colorbar.set_ticklabels(vmap.keys())
            plt.ylabel('Features')
            plt.xlabel('Thresholds')
            title = 'GTRE'
            plt.title(f'{cfg.name} - {title}')
            plt.yticks([])
            plt.savefig(cfg.get_filename(title, 'png'))
            plt.close(fig)

        """if ('fcca','cart') in performance.keys():
            for Q in [0, 0.7]:
                ThresholdsFolds = pd.DataFrame(index=dataset.feature_columns, columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=0)
                for fold in performance['fcca','cart'][0].discretizer.keys():
                    Thresholds = performance['fcca','cart'][0].discretizer[fold].selectThresholds(Q).copy()
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
                if cfg.target_model == GradientBoostingClassifier:
                    t = 'GradientBoosting'
                if cfg.target_model == RandomForestClassifier:
                    t = 'RandomForest'
                if cfg.target_model == LinearSVC:
                    t = 'SVM'
                title = f'FCCA Q={Q} with {t}'
                plt.title(f'{cfg.name} - {title}')
                plt.yticks([])
                plt.savefig(cfg.get_filename(title, 'png'))
                plt.close(fig)"""

        if ('fcca','cart') in performance.keys():
            ThresholdsFolds = pd.DataFrame(index=dataset.feature_columns, columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=np.nan)
            vmap = {'N':-1}
            i=0
            for Q in q_list:
                vmap[Q]=i
                i+=1

            for Q in q_list:
                for fold in performance['fcca', 'cart'][Q].discretizer.keys():
                    Thresholds = performance['fcca', 'cart'][Q].discretizer[fold].selectThresholds(Q).copy()
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
            if cfg.target_model == GradientBoostingClassifier:
                t = 'GradientBoosting'
            if cfg.target_model == RandomForestClassifier:
                t = 'RandomForest'
            if cfg.target_model == LinearSVC:
                t = 'SVM'
            title = f'FCCA {t}'
            plt.title(f'{cfg.name} - {title}')
            plt.yticks([])
            plt.savefig(cfg.get_filename(title, 'png'))
            plt.close(fig)