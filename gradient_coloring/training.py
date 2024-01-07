import math
import time
from argparse import ArgumentParser

from torch_geometric.nn.conv import SGConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def discrete_loss(embeddings, edge_index):
    return torch.sum(embeddings[edge_index[0]].argmax(dim=1) == embeddings[edge_index[1]].argmax(dim=1))


def continuous_loss(embeddings, edge_index):
    return torch.sum(embeddings[edge_index[0]] * embeddings[edge_index[1]])


def prepare_embeddings(graph, k) -> torch.Tensor:
    laplacian = nx.laplacian_matrix(graph).toarray()
    _, eigenvectors = np.linalg.eigh(laplacian)
    return torch.tensor(eigenvectors[:, -k:], dtype=torch.float32, requires_grad=True)


def graph_coloring(graph, k, max_iter, lr, verbose):
    embeddings = prepare_embeddings(graph, k)
    #embeddings = torch.randn(graph.number_of_nodes(), k)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()

    softmax = torch.nn.Softmax(dim=1)
    conv = SGConv(
        in_channels=embeddings.shape[-1],
        out_channels=embeddings.shape[-1],
        K=1,  # setting bigger didn't drastically change anything
        cached=True,
    )
    params = list(conv.parameters())
    params.append(embeddings)

    optimizer = torch.optim.AdamW(params, lr=lr)

    discrete_loss_history = []
    continuous_loss_history = []
    step = 0
    d_loss = math.inf

    while d_loss != 0 and step < max_iter:
        x = conv(torch.nn.functional.sigmoid(embeddings), edge_index)  # during experiments sigmoid here seems to help
        # here adding x to embeddings not passing x is crucial for results
        x = softmax(embeddings + x)

        optimizer.zero_grad()
        c_loss = continuous_loss(x, edge_index)
        c_loss.backward()
        optimizer.step()

        with torch.no_grad():
            d_loss = discrete_loss(x, edge_index).numpy()
            c_loss = c_loss.numpy()

        discrete_loss_history.append(d_loss)
        continuous_loss_history.append(c_loss)

        if step % 100 == 0 and verbose:
            print(f"Step {step} - discrete loss: {d_loss}, continuous loss: {c_loss}")

        step += 1

    return x.argmax(dim=1).numpy(), discrete_loss_history, continuous_loss_history


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n', type=int, default=30000, help='Number of nodes')
    args.add_argument('--m', type=int, default=80000, help='Number of edges')
    args.add_argument('--seed', type=int, default=42, help='Random seed for graph generation')
    args.add_argument('--max_iter', type=int, default=20000, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    args = args.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    G = nx.gnm_random_graph(args.n, args.m, args.seed)
    #layout = nx.spring_layout(G)
    start = time.time()
    greedy_colors_dict = nx.greedy_color(G)
    greedy_time = time.time() - start
    greedy_colors = [greedy_colors_dict[key] for key in sorted(greedy_colors_dict.keys())]
    print("Starting")
    start = time.time()
    best_colors, d_loss_history, c_loss_history = graph_coloring(
        graph=G,
        k=max(greedy_colors) + 1,
        max_iter=args.max_iter,
        lr=args.lr,
        verbose=True,
    )
    total_time = time.time() - start
    print(f"Final result for {args.n} nodes, {args.m} edges:")
    print(f"Epochs: {len(c_loss_history)}")
    print(f"Discrete loss: {d_loss_history[-1]}")
    print(f"Continuous loss: {c_loss_history[-1]}")
    print(f"Total time: {total_time:0.2f} s")
    print(f"Greedy colouring time: {greedy_time:0.2f} s")
    print(f"Colors used: {max(greedy_colors) + 1}")

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

    plt.title(f"Gradient coloring")
    nx.draw(G, pos=layout, node_color=best_colors, node_size=100)
    plt.show()
