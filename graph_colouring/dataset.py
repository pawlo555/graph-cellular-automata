import os
from os import listdir
from os.path import isfile, join

import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from torch_geometric.data import Dataset

np.random.seed(42)


def generate_random_graph(nodes, edges):
    G = nx.gnm_random_graph(nodes, edges, seed=42)

    node_colors_dict = nx.greedy_color(G)
    node_colors = []
    for key in sorted(node_colors_dict.keys()):
        node_colors.append(node_colors_dict[key])

    return G, node_colors


def create_pyg_example(graph, node_colors, num_colors, features_len=16):
    edge_list = list(graph.edges())
    edge_index_ab = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index_ba = torch.stack([edge_index_ab[1], edge_index_ab[0]], dim=0)
    edge_index = torch.concat([edge_index_ab, edge_index_ba], dim=-1)

    node_degree = dict(graph.degree)
    edge_weight_ab = torch.tensor([node_degree[ab] <= node_degree[ba] for ab, ba in edge_list], dtype=torch.float)
    edge_weight_ba = 1 - edge_weight_ab
    edge_weight = torch.cat([edge_weight_ab, edge_weight_ba], dim=-1)

    # Convert node colors to tensor
    node_features = torch.normal(0., 1., size=(len(node_colors), features_len))
    colors = torch.tensor(node_colors, dtype=torch.long).clone().detach()
    colors[:] = torch.max(colors)

    # Create a PyTorch Geometric data object
    data = Data(
        x=node_features,
        y=torch.nn.functional.one_hot(colors, num_colors),
        edge_index=edge_index,
        edge_weight=edge_weight,
        max_colours=colors
    )

    return data


class GraphColouringDataset(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.files = [join(root, f) for f in listdir(root) if isfile(join(root, f)) and "graph" in f]
        self.num_colors = torch.load(os.path.join(root, "meta.pt"))

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(join(self.root, f'graph_{idx}.pt'))
        return data


def create_graph_dataset(dataset_name: str, elements: int, min_nodes: int, max_nodes: int, min_edges: int,
                         max_edges: int):
    os.makedirs(dataset_name, exist_ok=True)
    graphs = []
    max_colors = 0
    for element in range(elements):
        num_nodes = random.randint(min_nodes, max_nodes)
        num_edges = random.randint(min_edges, max_edges)
        random_graph, colors = generate_random_graph(num_nodes, num_edges)
        graphs.append((random_graph, colors))
        num_colors = np.max(np.array(colors))
        if max_colors < num_colors:
            max_colors = num_colors
    for element in range(elements):
        random_graph, colors = graphs[element]
        data = create_pyg_example(random_graph, colors, max_colors+1)
        torch.save(data, os.path.join(dataset_name, f"graph_{element}.pt"))
    torch.save(max_colors+1, os.path.join(dataset_name, "meta.pt"))


def generating_graph_example():
    # Define the number of nodes and edges for the random graph
    num_nodes = 40
    num_edges = 80

    # Generate a random graph and node colors
    random_graph, colors = generate_random_graph(num_nodes, num_edges)

    # Create PyTorch Geometric dataset
    dataset = create_pyg_example(random_graph, colors, np.max(colors)+1)

    # Visualize the generated graph with colors (optional)
    nx.draw(random_graph, node_color=colors, with_labels=True)
    plt.show()

    # Example of accessing data from the dataset
    print("Node features (colors):", dataset.x)
    print("Edge indices:", dataset.edge_index)


if __name__ == '__main__':
    generating_graph_example()
    create_graph_dataset("basic_dataset", 10000, 50, 80, 80, 150)
    dataset = GraphColouringDataset("basic_dataset")
    print(len(dataset))
    print(dataset.num_colors)
