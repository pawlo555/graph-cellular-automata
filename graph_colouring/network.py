import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

from graph_colouring.dataset import GraphColouringDataset


class GNNGraphColoring(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_channels, num_classes):
        super(GNNGraphColoring, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.num_nodes = num_nodes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.to(torch.float)
        print(data.x.shape)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)

        return x


if __name__ == '__main__':
    dataset = GraphColouringDataset("basic_dataset")
    print(len(dataset))
    # Example usage:
    # Define the parameters for the model
    num_nodes = 10
    num_features = dataset.num_colors  # For simplicity, let's start with 1 feature per node
    hidden_channels = 16

    num_classes = dataset.num_colors  # Number of colors

    # Create an instance of the GNN model
    model = GNNGraphColoring(num_nodes, num_features, hidden_channels, num_classes)
    pyg_graph = dataset.get(10)
    results = model(pyg_graph)
    print(results.shape)

    G = nx.Graph()
    G.add_edges_from(pyg_graph.edge_index.cpu().numpy().T)
    true_colors = np.argmax(pyg_graph.x.cpu().numpy(), axis=-1)
    node_color = true_colors
    plt.title("Proper graph")
    nx.draw(G, node_color=true_colors)
    plt.show()

    network_colors = np.argmax(results.cpu().detach().numpy(), axis=-1)
    plt.title("Network coloring")
    nx.draw(G, node_color=network_colors)
    plt.show()
