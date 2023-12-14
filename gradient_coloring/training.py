from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import torch


def discrete_loss(embeddings, graph):
    return sum(embeddings[i].argmax() == embeddings[j].argmax() for i, j in graph.edges)


def continuous_loss(embeddings, graph):
    return sum(torch.dot(embeddings[i], embeddings[j]) for i, j in graph.edges)


def graph_coloring(graph, k, max_iter, lr, verbose):
    embeddings = torch.rand((len(graph), k), requires_grad=True)
    optimizer = torch.optim.SGD([embeddings], lr=lr)
    softmax = torch.nn.Softmax(dim=1)

    discrete_loss_history = []
    continuous_loss_history = []
    step = 0

    while True:
        soft_embeddings = softmax(embeddings)

        optimizer.zero_grad()
        c_loss = continuous_loss(soft_embeddings, graph)
        c_loss.backward()
        optimizer.step()

        d_loss = discrete_loss(soft_embeddings, graph).detach().numpy()
        c_loss = c_loss.detach().numpy()

        discrete_loss_history.append(d_loss)
        continuous_loss_history.append(c_loss)

        if step % 100 == 0 and verbose:
            print(f"Step {step} - discrete loss: {d_loss}, continuous loss: {c_loss}")

        if d_loss == 0 or step >= max_iter:
            break

        step += 1

    return embeddings.argmax(dim=1).numpy(), discrete_loss_history, continuous_loss_history


def train_with_restarts(graph, k, max_iter, lr, restarts, verbose):
    best_colors = None
    best_loss = float('inf')

    for _ in range(restarts):
        network_colors, d_loss_history, c_loss_history = graph_coloring(graph, k, max_iter, lr, verbose)

        if d_loss_history[-1] < best_loss:
            best_loss = d_loss_history[-1]
            best_colors = network_colors

        if d_loss_history[-1] == 0:
            break

    return best_colors, d_loss_history, c_loss_history


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n', type=int, default=30, help='Number of nodes')
    args.add_argument('--m', type=int, default=80, help='Number of edges')
    args.add_argument('--seed', type=int, default=42, help='Random seed for graph generation')
    args.add_argument('--restarts', type=int, default=10, help='Number of restarts')
    args.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    args = args.parse_args()

    G = nx.gnm_random_graph(args.n, args.m, args.seed)
    layout = nx.spring_layout(G)

    greedy_colors_dict = nx.greedy_color(G)
    greedy_colors = [greedy_colors_dict[key] for key in sorted(greedy_colors_dict.keys())]

    best_colors, d_loss_history, c_loss_history = train_with_restarts(
        graph=G,
        k=max(greedy_colors) + 1,
        max_iter=args.max_iter,
        lr=args.lr,
        restarts=args.restarts,
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
