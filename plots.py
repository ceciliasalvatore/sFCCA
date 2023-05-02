import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC

from config import cfg

def plot(performance, type_, title=None):
    types = ['continuous_cart', 'gtre_cart', 'gtre_gosdt', 'fcca_cart', 'fcca_gosdt']#, 'fcca_0_cart', 'fcca_0_gosdt', 'fcca_0.5_cart', 'fcca_0.5_gosdt', 'fcca_0.6_cart', 'fcca_0.6_gosdt', 'fcca_0.7_cart', 'fcca_0.7_gosdt', 'fcca_0.8_cart', 'fcca_0.8_gosdt', 'fcca_0.9_cart', 'fcca_0.9_gosdt', 'fcca_0.95_cart', 'fcca_0.95_gosdt']
    colors = plt.cm.rainbow(np.linspace(0, 1, len(types)))
    color_map = dict(zip(types, colors))

    if title is None:
        title = type_

    if type_ == 'accuracy' or type_ == 'n_features':
        fig = plt.figure()
        for t in types:
            keys = [k for k in performance.keys() if t.split('_')[0] in k[0] and t.split('_')[1] in k[1]]
            if len(keys)>1:
                q = [float(k[0].split('_')[1]) for k in keys]
                y = [np.mean(list(performance[k].__getattribute__(type_).values())) for k in keys]
                plt.plot(q, y, color=color_map[t], label=t, linewidth='3', markersize='15', marker='.')
            else:
                y = np.mean(list(performance[keys[0]].__getattribute__(type_).values()))
                plt.plot([0, 1], [y, y], color=color_map[t], label=t, linewidth='3')
        plt.xlabel('Q')
        plt.legend()
        plt.title(f"{cfg.name} - {title}")
        plt.savefig(cfg.get_filename(title, 'png'))
        plt.close(fig)
    elif type_ == 'compression' or type_ == 'inconsistency':
        fig = plt.figure()
        for t in types:
            if t.split('_')[1] == 'cart':
                keys = [k for k in performance.keys() if t.split('_')[0] in k[0] and t.split('_')[1] in k[1]]
                if len(keys)>1:
                    q = [float(k[0].split('_')[1]) for k in keys]
                    y = [np.mean(list(performance[k].__getattribute__(type_).values())) for k in keys]
                    plt.plot(q, y, color=color_map[t], label=t.split('_')[0], linewidth='3', markersize='15', marker='.')
                else:
                    y = np.mean(list(performance[keys[0]].__getattribute__(type_).values()))
                    plt.plot([0, 1], [y, y], color=color_map[t], label=t.split('_')[0], linewidth='3')
        plt.legend()
        plt.xlabel('Q')
        plt.title(f"{cfg.name} - {title}")
        plt.savefig(cfg.get_filename(title, 'png'))
        plt.close(fig)
    elif type_ == 'thresholds':
        keys = ['fcca_0','fcca_0.7', 'gtre']
        features_names = performance['continuous', 'cart'].model[0].feature_names_in_
        for i in keys:
            fig = plt.figure()
            ThresholdsFolds = pd.DataFrame(index=features_names, columns=np.round(np.arange(0, 1, 0.1), decimals=2), data=0)
            for fold in performance[i,'cart'].discretizer.keys():
                Thresholds = performance[i, 'cart'].discretizer[fold].tao.copy()
                if 'Count' not in Thresholds.columns:
                    Thresholds['Count'] = 1
                Thresholds['Threshold'] = np.floor(Thresholds['Threshold'].astype(float) * 10) / 10
                Thresholds = Thresholds.groupby(['Feature', 'Threshold']).sum().reset_index()
                Thresholds = Thresholds.reset_index().pivot(index='Feature', columns='Threshold', values='Count')
                Thresholds = Thresholds.replace(np.nan, 0)
                ThresholdsFolds.update(Thresholds)
            ThresholdsFolds = ThresholdsFolds / np.max(ThresholdsFolds.to_numpy())
            ThresholdsFolds[ThresholdsFolds>0]=1
            sns.heatmap(ThresholdsFolds, cmap="YlOrBr", cbar=False)
            plt.ylabel('Features')
            plt.xlabel('Thresholds')
            if 'fcca' in i:
                if cfg.target_model == GradientBoostingClassifier:
                    title = f'FCCA Q={i.split("_")[1]} with GradientBoosting'
                if cfg.target_model == RandomForestClassifier:
                    title = f'FCCA Q={i.split("_")[1]} with RandomForest'
                if cfg.target_model == LinearSVC:
                    title = f'FCCA Q={i.split("_")[1]} with Linear SVM'
            else:
                title = 'GTRE'
            plt.title(f'{cfg.name} - {title}')
            plt.yticks(np.arange(len(features_names)) + .5, features_names, rotation=0)
            if cfg.name!='boston':
                plt.yticks([])
            plt.savefig(cfg.get_filename(title, 'png'))
            plt.close(fig)
