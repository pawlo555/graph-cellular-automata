from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from benchmark import GRAPHS, col_to_graph
from gradient_coloring.training import graph_coloring, iterate_graph_coloring


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--max_iter', type=int, default=600, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args.add_argument('--output', type=str, default='../benchmark/results/colors.csv', help='Output file')
    args = args.parse_args()

    with open(args.output, 'w') as file:
        file.write('graph,n_nodes,n_edges,best_known,k\n')

    for filename, colors_num in tqdm(GRAPHS.items()):
        graph = col_to_graph(f'../benchmark/{filename}.col')
        k = colors_num

        while True:
            print(f'Trial for {filename} with {k} colors')
            embedding, d_loss_history, _ = iterate_graph_coloring(
                graph=graph,
                min_k=max(k-20, 2),
                max_k=1000,
                max_iter=args.max_iter,
                lr=args.lr,
                verbose=False,
                use_model=False,
                fix_errors=True
            )
            true_k = np.max(embedding)+1

            if d_loss_history[-1] == 0:
                break

            k += 1

        with open(args.output, 'a') as file:
            file.write(f'{filename},{len(graph.nodes)},{len(graph.edges)},{colors_num},{true_k}\n')
