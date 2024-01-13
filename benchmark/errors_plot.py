import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FILES = [
    ('results_random.csv', 'Random'),
    ('results_node2vec.csv', 'Node2Vec'),
    ('results_lap_highest.csv', r'Laplacian (top $k$)'),
    ('results_lap_smallest.csv', r'Laplacian (bottom $k$)'),
    ('results_sgc.csv', 'SGC'),
    ('results_rprop.csv', 'RProp'),
    ('results_rprop_fix.csv', 'RProp (fixed)')
]

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
MARKERS = ['o', 'v', 's', 'd', 'p', 'P', '*', 'X', 'h', 'H', '+', 'x', 'D', '|', '_', '<', '>', '^']

PLOT_PARAMS = {
    'figure.figsize': (6.5, 3),
    'figure.dpi': 72,
    'font.size': 9,
    'font.family': 'serif',
    'font.serif': 'cm',
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'grid.alpha': 0.42,
    'grid.linewidth': 0.5,
    'legend.title_fontsize': 7,
    'legend.fontsize': 7,
    'lines.linewidth': 1.,
    'lines.markersize': 2,
    'text.usetex': True,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
}


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    x = np.random.normal(size=9) * 0.1
    _, ax = plt.subplots()

    for i, (file, _) in enumerate(FILES):
        df = pd.read_csv(file)
        y = 100 * df['d_loss'] / df['n_edges']

        mean, std = np.mean(y), np.std(y)
        ax.errorbar(i, mean, yerr=std, fmt='o', c='gray')

        for j, point in enumerate(y):
            ax.scatter(i + x[j], point, c=COLORS[i], marker=MARKERS[j % len(MARKERS)], s=8)

    ax.set_xticks([i for i in range(len(FILES))])
    ax.set_xticklabels([label for _, label in FILES], rotation=45, ha='right')
    ax.set_ylabel('Average error (\%)')
    ax.spines[['right', 'top']].set_visible(False)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results.pdf')
    plt.show()
