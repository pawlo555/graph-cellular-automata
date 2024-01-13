from argparse import ArgumentParser

import networkx as nx
from tqdm import tqdm

from gradient_coloring.training import graph_coloring


GRAPHS = {
    'DSJC125.1': 5,
    'DSJC125.5': 17,
    'DSJC125.9': 44,
    'DSJC250.1': 8,
    'DSJC250.5': 28,
    'DSJC250.9': 72,
    'DSJC500.1': 12,
    'DSJC500.5': 48,
    # 'DSJC500.9': 126,
    'DSJC1000.1': 20,
    'DSJC1000.5': 87,
    # 'DSJC1000.9': 223,
    'DSJR500.1': 12,
    # 'DSJR500.5': 126,
    'anna': 11,
    'david': 11,
    # 'homer': 13,
    'huck': 11,
    # 'jean': 10,
    'games120': 9
}


def col_to_graph(filename):
    G = nx.Graph()

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) == 0 or parts[0] != 'e':
                continue

            _, node1, node2 = parts
            G.add_edge(int(node1) - 1, int(node2) - 1)

    return G


def main():
    args = ArgumentParser()
    args.add_argument('--max_iter', type=int, default=300, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args.add_argument('--output', type=str, default='results.csv', help='Output file')
    args = args.parse_args()

    with open(args.output, 'w') as file:
        file.write('graph,n_nodes,n_edges,colors_num,d_loss\n')

    for filename, colors_num in tqdm(GRAPHS.items()):
        graph = col_to_graph(f'../benchmark/{filename}.col')
        _, d_loss_history, _ = graph_coloring(
            graph=graph,
            k=colors_num,
            max_iter=args.max_iter,
            lr=args.lr,
            verbose=True,
            use_model=False,
            fix_errors=True
        )

        with open(args.output, 'a') as file:
            file.write(f'{filename},{len(graph.nodes)},{len(graph.edges)},{colors_num},{d_loss_history[-1]}\n')


if __name__ == '__main__':
    main()
