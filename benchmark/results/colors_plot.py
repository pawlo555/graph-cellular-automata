import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark import GRAPHS


if __name__ == '__main__':
    rprop_df = pd.read_csv('colors_rprop_fix.csv')
    greedy_df = pd.read_csv('colors_greedy.csv')

    graphs = list(GRAPHS.keys())
    best_known = rprop_df['best_known']
    greedy = greedy_df['k']
    rprop = rprop_df['k']

    bar_width = 0.25
    index = np.arange(len(graphs))

    _, ax = plt.subplots()

    ax.bar(index, best_known, width=bar_width, label='Best Known', color='C2')
    ax.bar(index + bar_width, greedy, width=bar_width, label='Greedy', color='C1')
    ax.bar(index + 2 * bar_width, rprop, width=bar_width, label='Our method', color='C0')

    ax.set_ylabel('Number of colors')
    ax.set_yscale('log')
    ax.set_xticks(index + bar_width, graphs, rotation=90)
    ax.spines[['right', 'top']].set_visible(False)
    plt.grid(axis='y', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('colors_results.pdf')
    plt.show()
