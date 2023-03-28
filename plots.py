import numpy as np
from matplotlib import pyplot as plt

from config import cfg

def plot(performance, title):
    fig = plt.figure()
    x = [f'{d}_{m}' for (d, m) in performance.keys()]
    y = [np.mean(list(performance[d, m].accuracy.values())) for (d, m) in performance.keys()]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    color_map = dict(zip(x, colors))
    plt.bar(x, y, color=colors)
    plt.ylim((max(0, np.min(y) - 0.01), min(1, np.max(y) + 0.01)))
    plt.title(f"{cfg.name} - {title}")
    plt.savefig(cfg.get_filename(title, 'png'))
    plt.close(fig)

    """fig = plt.figure()
    for d in discretizers:
        m = 'cart'
        y = []
        for q in q_list:
            y.append(np.mean([performance[d, m].compression[i][q] for i in range(cfg.k)]))
        plt.plot(q_list, y, color=color_map[f'{d}_{m}'], marker='o', linewidth=3, label=d)
    plt.legend(loc="upper left")
    plt.ylim(np.min(y) - 0.1, np.max(y) + 0.1)
    plt.title(f"{name} - Compression")
    plt.savefig(cfg.get_filename('compression', 'png'))
    plt.close(fig)

    fig = plt.figure()
    for d in discretizers:
        m = 'cart'
        y = []
        for q in q_list:
            y.append(np.mean([performance[d, m].inconsistency[i][q] for i in range(cfg.k)]))
        plt.plot(q_list, y, color=color_map[f'{d}_{m}'], marker='o', linewidth=3, label=d)
    plt.legend(loc="upper left")
    plt.ylim(np.min(y) - 0.1, np.max(y) + 0.1)
    plt.title(f"{name} - Inconsistency")
    plt.savefig(cfg.get_filename('inconsistency', 'png'))
    plt.close(fig)"""

    """fig = plt.figure()
    y = [np.mean(list(performance[d, m].n_features.values())) for d, m in performance.keys()]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    plt.bar(x, y, color=colors)
    plt.ylim((max(0, np.min(y) - 1), np.max(y) + 1))
    plt.title(f"{name} - Number of features")
    plt.savefig(cfg.get_filename('n_features', 'png'))
    plt.close(fig)

    fig = plt.figure()
    y = [np.mean(list(performance[d, m].n_thresholds.values())) for d, m in performance.keys()]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(x)))
    plt.bar(x, y, color=colors)
    plt.ylim((max(0, np.min(y) - 1), np.max(y) + 1))
    plt.title(f"{name} - Number of thresholds")
    plt.savefig(cfg.get_filename('n_thresholds', 'png'))
    plt.close(fig)"""
