import matplotlib.pyplot as plt
import networkx as nx
import torch


def discrete_loss(embeddings, graph):
    return sum(embeddings[i].argmax() == embeddings[j].argmax() for i, j in graph.edges)


def continuous_loss(embeddings, graph):
    return sum(torch.dot(embeddings[i], embeddings[j]) for i, j in graph.edges)


def graph_coloring(graph, k):
    embeddings = torch.rand((len(graph), k), requires_grad=True)
    optimizer = torch.optim.SGD([embeddings], lr=0.1)
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

        if step % 100 == 0:
            print(f"Step {step} - discrete loss: {d_loss}, continuous loss: {c_loss}")

        if d_loss == 0 or step >= 1000:
            break

        step += 1

    return embeddings.argmax(dim=1).numpy(), discrete_loss_history, continuous_loss_history


if __name__ == '__main__':
    G = nx.gnm_random_graph(30, 80, seed=42)
    layout = nx.spring_layout(G)

    greedy_colors_dict = nx.greedy_color(G)
    greedy_colors = []

    for key in sorted(greedy_colors_dict.keys()):
        greedy_colors.append(greedy_colors_dict[key])

    for _ in range(10):
        network_colors, discrete_loss_history, continuous_loss_history = graph_coloring(G, max(greedy_colors) + 1)

        if discrete_loss_history[-1] == 0:
            break

    plt.plot(discrete_loss_history, label='Discrete loss')
    plt.plot(continuous_loss_history, label='Continuous loss')
    plt.xscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.title("Proper coloring")
    nx.draw(G, pos=layout, node_color=greedy_colors, node_size=100)
    plt.show()

    plt.title("Gradient coloring")
    nx.draw(G, pos=layout, node_color=network_colors, node_size=100)
    plt.show()
