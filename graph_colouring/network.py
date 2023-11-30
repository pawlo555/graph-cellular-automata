import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

from graph_colouring.dataset import GraphColouringDataset


class GNNGraphColoring(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNGraphColoring, self).__init__()
        self.conv1 = GCNConv(1, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.arange(0, x.shape[0], dtype=torch.float)  # need to create sensible features, cannot use x because they are true colors
        x = torch.unsqueeze(x, dim=-1)
        x = x.to(torch.float)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index)

        return torch.nn.functional.softmax(x, dim=-1)


def show_results(dataset, model):
    pyg_graph = dataset.get(10)
    results = model(pyg_graph)
    print(results.shape)
    print(results)
    G = nx.Graph()

    G.add_nodes_from(list(range(pyg_graph.x.shape[0])))
    G.add_edges_from(pyg_graph.edge_index.cpu().numpy().T)
    true_colors = np.argmax(pyg_graph.x.cpu().numpy(), axis=-1)
    plt.title("Proper graph")
    nx.draw(G, node_color=true_colors)
    plt.show()

    network_colors = np.argmax(results.cpu().detach().numpy(), axis=-1)
    plt.title("Network coloring")
    nx.draw(G, node_color=network_colors)
    plt.show()


if __name__ == '__main__':
    dataset = GraphColouringDataset("basic_dataset")
    print(len(dataset))
    # Example usage:
    # Define the parameters for the model
    num_features = dataset.num_colors  # For simplicity, let's start with 1 feature per node
    hidden_channels = 8

    num_classes = dataset.num_colors  # Number of colors

    # Create an instance of the GNN model
    model = GNNGraphColoring(num_features, hidden_channels, num_classes)
    show_results(dataset, model)
