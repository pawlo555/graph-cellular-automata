import torch
from torch_geometric.data import DataLoader, Data
from tqdm import tqdm

from graph_colouring.dataset import GraphColouringDataset
from graph_colouring.network import GNNGraphColoring, show_results
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Define a custom loss function
def loss_function(probabilities, edge_index, max_colors):
    used_more_colors_penalty = 0 # torch.sum(probabilities[..., max_colors:])

    right_coloring = torch.sum(probabilities[edge_index[0]] * probabilities[edge_index[1]])
    total_loss = right_coloring + used_more_colors_penalty

    return total_loss


# Training loop with tqdm
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc='Training')
    for data in pbar:
        data = data.to("cpu")
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, data.edge_index, data.max_colours)  # Using custom loss function
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'Loss': total_loss / len(train_loader.dataset)})
    return total_loss / len(train_loader.dataset)


# Evaluation loop with tqdm
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc='Testing')
    with torch.no_grad():
        for data in pbar:
            data = data.to("cpu")
            output = model(data)
            loss = loss_function(output, data.edge_index, data.max_colours)
            total_loss += loss.item()
            pbar.set_postfix({'Loss': total_loss / len(loader.dataset)})
    return total_loss / len(loader.dataset)


def train_model():
    # Load your dataset (e.g., a graph dataset from PyTorch Geometric)
    dataset = GraphColouringDataset("basic_dataset")

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Define train-test split and create data loaders

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    num_classes = dataset.num_colors  # Number of colors
    num_features = 16
    hidden_channels = 64
    num_layers = 2

    # Create an instance of the GNN model
    model = GNNGraphColoring(num_features, hidden_channels, num_classes, num_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = "cpu"
    model = model.to(device)

    # Training and evaluation iterations
    num_epochs = 10000
    train_losses = []
    test_accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(epoch)
        train_loss = train(model, train_loader, optimizer)
        train_losses.append(train_loss)
        test_acc = evaluate(model, test_loader)
        test_accuracies.append(test_acc)

    # Plotting training loss and test accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.xscale('log')
    plt.legend()
    plt.show()

    show_results(dataset, model)

    model.eval()
    torch.save(model.state_dict(), "model.pt")


def test_cost_function():
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[0, 1, 0], [0.9, 0, 0.1], [0, 0.2, 0.8]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    max_colors = 2
    loss_function(data.x, data.edge_index, max_colors)
    # x = torch.stack([data.x, data.x, data.x])
    # edge = torch.stack([data.edge_index, data.edge_index, data.edge_index])
    # colors = torch.tensor([max_colors, max_colors, max_colors])
    # loss_function(x,
    #               edge,
    #               colors
    #               )


if __name__ == '__main__':
    #test_cost_function()
    train_model()
