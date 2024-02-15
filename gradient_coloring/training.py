import math
import time
import random
from argparse import ArgumentParser

from torch_geometric.nn.conv import SGConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def discrete_loss(embeddings, edge_index):
    return torch.count_nonzero(torch.eq(embeddings[edge_index[0]].argmax(dim=1),
                                        embeddings[edge_index[1]].argmax(dim=1)))


def continuous_loss(embeddings, edge_index):
    return torch.sum(embeddings[edge_index[0]] * embeddings[edge_index[1]])


def prepare_embeddings(graph, k) -> torch.Tensor:
    laplacian = nx.laplacian_matrix(graph).toarray()
    _, eigenvectors = np.linalg.eigh(laplacian)
    return torch.tensor(eigenvectors[:, -k:], dtype=torch.float32, requires_grad=True)


def process_model(embeddings, edge_index, conv, softmax):
    if conv is not None:
        # during experiments sigmoid here seems to help
        x = conv(torch.nn.functional.sigmoid(embeddings), edge_index)
        # here adding x to embeddings not passing x is crucial for results
        return softmax(embeddings + x)
    else:
        return softmax(embeddings)


def iterate_graph_coloring(graph, min_k, max_k, max_iter, lr, verbose, use_model, fix_errors):
    embeddings = prepare_embeddings(graph, min_k)
    all_discrete_loss_history = []
    all_continuous_loss_history = []
    colors = None

    for k in range(min_k, max_k+1):
        if verbose:
            print(f"Coloring for {k} colors")
        colors, discrete_loss_history, continuous_loss_history = graph_coloring(graph, k, max_iter, lr, verbose,
                                                                                use_model, fix_errors, embeddings)
        all_continuous_loss_history += continuous_loss_history
        all_discrete_loss_history += discrete_loss_history
        if discrete_loss_history[-1] == 0:
            return colors, all_discrete_loss_history, all_continuous_loss_history
        embeddings = torch.nn.functional.one_hot(torch.tensor(colors, dtype=torch.int64), num_classes=k+1).to(torch.float32)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()

    return colors, all_discrete_loss_history, all_continuous_loss_history


def graph_coloring(graph, k, max_iter, lr, verbose, use_model, fix_errors, embeddings=None):
    if embeddings is None:
        embeddings = prepare_embeddings(graph, k)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()

    if use_model:
        conv = SGConv(
            in_channels=embeddings.shape[-1],
            out_channels=embeddings.shape[-1],
            K=1,  # setting bigger didn't drastically change anything
            cached=True,
        )
        params = list(conv.parameters())
    else:
        conv = None
        params = []
    params.append(embeddings)

    softmax = torch.nn.Softmax(dim=-1)
    optimizer = torch.optim.Rprop(params, lr=lr)

    discrete_loss_history = []
    continuous_loss_history = []

    step = 0
    d_loss = math.inf

    while d_loss != 0 and step < max_iter:
        optimizer.zero_grad()
        x = process_model(embeddings, edge_index, conv, softmax)
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

    if fix_errors:
        if verbose:
            print("Fixing")

        with torch.no_grad():
            x = process_model(embeddings, edge_index, conv, softmax)
            errors = torch.eq(x[edge_index[0]].argmax(dim=1), x[edge_index[1]].argmax(dim=1))
            errors_indices = np.where(errors)[0]
            for error_index in errors_indices:
                for error_node in [edge_index[0, error_index], edge_index[1, error_index]]:
                    connecting_nodes_a = edge_index[1, edge_index[0] == error_node]
                    connecting_nodes_b = edge_index[0, edge_index[1] == error_node]
                    connecting_nodes = torch.cat((connecting_nodes_a, connecting_nodes_b))

                    values = x[connecting_nodes].argmax(dim=1)
                    possible_values = set(range(k)) - set(values.clone().detach().numpy())
                    if possible_values:
                        new_color = min(possible_values)
                        new_embedding = torch.nn.functional.one_hot(torch.tensor(new_color), k)
                        if verbose:
                            print(f"Found fix for {edge_index[0, error_index]} node: {new_color}")
                        x[error_node] = new_embedding
                        break

            c_loss = continuous_loss(x, edge_index).numpy()
            d_loss = discrete_loss(x, edge_index).numpy()

            discrete_loss_history.append(d_loss)
            continuous_loss_history.append(c_loss)

    return x.argmax(dim=1).numpy(), discrete_loss_history, continuous_loss_history


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--n', type=int, default=300, help='Number of nodes')
    args.add_argument('--m', type=int, default=2000, help='Number of edges')
    args.add_argument('--seed', type=int, default=42, help='Random seed for graph generation')
    args.add_argument('--max_iter', type=int, default=300, help='Maximum number of iterations')
    args.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args.add_argument('--use-model', action='store_true', default=False, help='Use SGC model')
    args.add_argument('--print-graph', action='store_true', default=False, help='Displaying graphs')
    args.add_argument('--fix-errors', action='store_true', default=False, help='Try to manually fix '
                                                                               'embeddings at the end of training')
    args = args.parse_args()

    random.seed(246)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    G = nx.gnm_random_graph(args.n, args.m, args.seed)
    start = time.time()
    greedy_colors_dict = nx.greedy_color(G)
    greedy_time = time.time() - start
    greedy_colors = [greedy_colors_dict[key] for key in sorted(greedy_colors_dict.keys())]
    print("Starting")
    start = time.time()
    best_colors, d_loss_history, c_loss_history = iterate_graph_coloring(
        graph=G,
        min_k=2,
        max_k=max(greedy_colors)+1,
        max_iter=args.max_iter,
        lr=args.lr,
        verbose=True,
        use_model=args.use_model,
        fix_errors=args.fix_errors
    )
    total_time = time.time() - start
    print(f"Final result for {args.n} nodes, {args.m} edges:")
    print(f"Epochs: {len(c_loss_history)}")
    print(f"Discrete loss: {d_loss_history[-1]}")
    print(f"Continuous loss: {c_loss_history[-1]}")
    print(f"Total time: {total_time:0.2f} s")
    print(f"Greedy colouring time: {greedy_time:0.2f} s")
    print(f"Greedy colors used: {max(greedy_colors) + 1}")
    print(f"Gradient colors: {max(best_colors)+1}")

    plt.plot(d_loss_history, label='Discrete loss')
    plt.plot(c_loss_history, label='Continuous loss')
    plt.xscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    if args.print_graph:
        layout = nx.spring_layout(G)
        plt.title("Greedy coloring")
        nx.draw(G, pos=layout, node_color=greedy_colors, node_size=100)
        plt.show()

        plt.title(f"Gradient coloring")
        nx.draw(G, pos=layout, node_color=best_colors, node_size=100)
        plt.show()
