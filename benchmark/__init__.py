import networkx as nx


GRAPHS = {
    'DSJC125.1': 5,
    'DSJC125.5': 17,
    'DSJC125.9': 44,
    'DSJC250.1': 8,
    'DSJC250.5': 28,
    'DSJC250.9': 72,
    'DSJC500.1': 12,
    'DSJC500.5': 48,
    'DSJC500.9': 126,
    'DSJC1000.1': 20,
    'DSJC1000.5': 87,
    'DSJC1000.9': 223,
    'DSJR500.1': 12,
    'DSJR500.1c': 85,
    'DSJR500.5': 126,
    'le450_5a': 5,
    'le450_5b': 5,
    'le450_5c': 5,
    'le450_5d': 5,
    'le450_15a': 15,
    'le450_15b': 15,
    'le450_15c': 15,
    'le450_15d': 15,
    'le450_25a': 25,
    'le450_25b': 25,
    'le450_25c': 25,
    'le450_25d': 25
}


def col_to_graph(filename):
    G = nx.Graph()

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) == 0 or parts[0] != 'e':
                continue

            _, node1, node2 = parts
            G.add_edge(int(node1) - 1, int(node2) - 1)

    return G
