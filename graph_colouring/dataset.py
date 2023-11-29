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

    # Generate random colors for nodes (3 colors for demonstration)
    node_colors_dict = nx.greedy_color(G)
    node_colors = []
    for key in sorted(node_colors_dict.keys()):
        node_colors.append(node_colors_dict[key])

    return G, node_colors


def create_pyg_example(graph, node_colors):
    edge_list = list(graph.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Convert node colors to tensor
    node_features = torch.tensor(node_colors, dtype=torch.long)

    # Create a PyTorch Geometric data object
    data = Data(x=torch.nn.functional.one_hot(node_features), edge_index=edge_index, max_colours=2)

    return data


class GraphColouringDataset(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.files = [join(root, f) for f in listdir(root) if isfile(join(root, f)) and "graph" in f]
        self.num_colors = torch.load(os.path.join(root, "meta.pt"))

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(join(self.processed_dir, f'data_{idx}.pt'))
        return data


def create_graph_dataset(dataset_name: str, elements: int, min_nodes: int, max_nodes: int, min_edges: int,
                         max_edges: int):
    os.makedirs(dataset_name, exist_ok=True)
    max_colors = 0
    for element in range(elements):
        num_nodes = random.randint(min_nodes, max_nodes)
        num_edges = random.randint(min_edges, max_edges)
        random_graph, colors = generate_random_graph(num_nodes, num_edges)
        num_colors = np.max(colors)
        if max_colors < num_colors:
            max_colors = num_colors
        data = create_pyg_example(random_graph, colors)
        torch.save(data, os.path.join(dataset_name, f"graph_{element}.pt"))
    torch.save(max_colors, os.path.join(dataset_name, "meta.pt"))


def generating_graph_example():
    # Define the number of nodes and edges for the random graph
    num_nodes = 40
    num_edges = 80

    # Generate a random graph and node colors
    random_graph, colors = generate_random_graph(num_nodes, num_edges)

    # Create PyTorch Geometric dataset
    dataset = create_pyg_example(random_graph, colors)

    # Visualize the generated graph with colors (optional)
    nx.draw(random_graph, node_color=colors, with_labels=True)
    plt.show()

    # Example of accessing data from the dataset
    print("Node features (colors):", dataset.x)
    print("Edge indices:", dataset.edge_index)


if __name__ == '__main__':
    generating_graph_example()
    create_graph_dataset("basic_dataset", 1000, 50, 100, 200, 400)
    dataset = GraphColouringDataset("basic_dataset")
    print(len(dataset))
    print(dataset.num_colors)
