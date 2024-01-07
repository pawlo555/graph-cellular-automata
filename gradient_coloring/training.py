import math
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def discrete_loss(embeddings, graph):
    return sum(embeddings[i].argmax() == embeddings[j].argmax() for i, j in graph.edges)


def continuous_loss(embeddings, graph):
    return sum(torch.dot(embeddings[i], embeddings[j]) for i, j in graph.edges)


def graph_coloring(graph, k, max_iter, lr, verbose):
    laplacian = nx.laplacian_matrix(graph).toarray()
    _, eigenvectors = np.linalg.eigh(laplacian)
    embeddings = torch.tensor(eigenvectors[:, -k:], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.AdamW([embeddings], lr=lr)
    softmax = torch.nn.Softmax(dim=1)

    discrete_loss_history = []
    continuous_loss_history = []
    step = 0
    d_loss = math.inf

    while d_loss != 0 and step < max_iter:
        soft_embeddings = softmax(embeddings)

        optimizer.zero_grad()
        c_loss = continuous_loss(soft_embeddings, graph)
        c_loss.backward()
        optimizer.step()

        with torch.no_grad():
            d_loss = discrete_loss(soft_embeddings, graph).numpy()
            c_loss = c_loss.numpy()

        discrete_loss_history.append(d_loss)
        continuous_loss_history.append(c_loss)

        if step % 100 == 0 and verbose:
            print(f"Step {step} - discrete loss: {d_loss}, continuous loss: {c_loss}")

        step += 1

    return embeddings.argmax(dim=1).numpy(), discrete_loss_history, continuous_loss_history


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n', type=int, default=100, help='Number of nodes')
    args.add_argument('--m', type=int, default=200, help='Number of edges')
    args.add_argument('--seed', type=int, default=42, help='Random seed for graph generation')
    args.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.3, help='Learning rate')
    args = args.parse_args()

    G = nx.gnm_random_graph(args.n, args.m, args.seed)
    layout = nx.spring_layout(G)

    greedy_colors_dict = nx.greedy_color(G)
    greedy_colors = [greedy_colors_dict[key] for key in sorted(greedy_colors_dict.keys())]

    best_colors, d_loss_history, c_loss_history = graph_coloring(
        graph=G,
        k=max(greedy_colors) + 1,
        max_iter=args.max_iter,
        lr=args.lr,
        verbose=True
    )

    plt.plot(d_loss_history, label='Discrete loss')
    plt.plot(c_loss_history, label='Continuous loss')
    plt.xscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.title("Greedy coloring")
    nx.draw(G, pos=layout, node_color=greedy_colors, node_size=100)
    plt.show()

    plt.title("Gradient coloring")
    nx.draw(G, pos=layout, node_color=best_colors, node_size=100)
    plt.show()
