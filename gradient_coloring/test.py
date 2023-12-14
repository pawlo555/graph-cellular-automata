from argparse import ArgumentParser

import networkx as nx
from tqdm import tqdm

from gradient_coloring.training import train_with_restarts


def generate_graphs(n, nodes, edges, seed):
    for _ in range(n):
        graph = nx.gnm_random_graph(nodes, edges, seed)
        colors_num = max(nx.greedy_color(graph).values()) + 1
        yield graph, colors_num


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n', type=int, default=10, help='Number of graphs to generate')
    args.add_argument('--nodes', type=int, default=30, help='Number of nodes')
    args.add_argument('--edges', type=int, default=80, help='Number of edges')
    args.add_argument('--seed', type=int, default=42, help='Random seed for graph generation')
    args.add_argument('--restarts', type=int, default=10, help='Number of restarts')
    args.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.2, help='Learning rate')
    args = args.parse_args()

    errors = []
    mistakes = []

    for graph, colors_num in tqdm(generate_graphs(args.n, args.nodes, args.edges, args.seed)):
        _, d_loss_history, _ = train_with_restarts(
            graph=graph,
            k=colors_num,
            max_iter=args.max_iter,
            lr=args.lr,
            restarts=args.restarts,
            verbose=False
        )

        errors.append(d_loss_history[-1] / args.edges)
        mistakes.append(d_loss_history[-1])

    print(f'nodes = {args.nodes}, edges = {args.edges}, error = {sum(errors) / len(errors)}, mistakes = {sum(mistakes) / (args.n * args.edges)}')
