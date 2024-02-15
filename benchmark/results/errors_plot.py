import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import GRAPHS


FILES = [
    ('errors_iterative.csv', "Our method"),
    ('errors_random.csv', 'Random'),
    ('errors_node2vec.csv', 'Node2Vec'),
    ('errors_lap_highest.csv', r'Laplacian (top $k$)'),
    ('errors_lap_smallest.csv', r'Laplacian (bottom $k$)'),
    ('errors_sgc.csv', r'SGC$^{*}$'),
    ('errors_rprop.csv', 'RProp'),
    ('errors_rprop_fix.csv', 'RProp (fixed)')
]

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']


if __name__ == '__main__':
    y = np.random.normal(size=len(GRAPHS), scale=0.1)
    _, ax = plt.subplots()

    for i, (file, _) in enumerate(FILES):
        df = pd.read_csv(file)
        x = 100 * df['d_loss'] / df['n_edges']

        mean, std = np.mean(x), np.std(x)
        mask = x < 10

        ax.errorbar(mean, i, xerr=std, fmt='x', c='gray', ms=6, zorder=-1)
        ax.scatter(x[mask], (i + y)[mask], c=COLORS[i], s=8)

    ax.set_yticks([i for i in range(len(FILES))])
    ax.set_yticklabels([label for _, label in FILES])
    ax.set_xlabel('Average error (\%)')
    ax.set_xlim(left=0)
    ax.spines[['right', 'top']].set_visible(False)
    plt.grid(axis='x', alpha=0.5)
    plt.tight_layout()
    plt.savefig('errors_results.pdf')
    plt.show()
