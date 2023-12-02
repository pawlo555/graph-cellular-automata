import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN
import matplotlib.pyplot as plt
import numpy as np

from graph_colouring.dataset import GraphColouringDataset


class GNNGraphColoring(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers):
        super(GNNGraphColoring, self).__init__()
        self.gcn = GCN(num_features, hidden_channels, num_layers, num_classes, dropout=0.3, agg='sum', act='softmax')

    def forward(self, data):
        x = self.gcn(data.x, data.edge_index, data.edge_weight)
        return F.softmax(x, dim=-1)


def show_results(dataset, model):
    for i in range(20):
        pyg_graph = dataset.get(i)
        results = model(pyg_graph)
        print(results.shape)
        print(results)
        G = nx.Graph()

        G.add_nodes_from(list(range(pyg_graph.x.shape[0])))
        G.add_edges_from(pyg_graph.edge_index.cpu().numpy().T)
        true_colors = np.argmax(pyg_graph.x.cpu().numpy(), axis=-1)
        plt.title("Proper graph")
        nx.draw(G, node_color=true_colors, node_size=50)
        plt.savefig(f"proper_graph_{i}.png")
        plt.clf()

        network_colors = np.argmax(results.cpu().detach().numpy(), axis=-1)
        plt.title("Network coloring")
        nx.draw(G, node_color=network_colors, node_size=50)
        plt.savefig(f"network_graph_{i}.png")
        plt.clf()


if __name__ == '__main__':
    dataset = GraphColouringDataset("basic_dataset")
    print(len(dataset))
    # Example usage:
    # Define the parameters for the model
    num_features = 16
    hidden_channels = 64

    num_classes = dataset.num_colors  # Number of colors

    # Create an instance of the GNN model
    model = GNNGraphColoring(num_features, hidden_channels, num_classes, 3)
    show_results(dataset, model)
