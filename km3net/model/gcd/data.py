import pandas as pd
import torch
import networkx as nx
import torch_geometric as pg
import torch_geometric.utils as pgutils
import torch_geometric.data as pgdata
from sklearn.preprocessing import MinMaxScaler

def create_graph(X, y, w, scale=True):
    """
    In
    --
    X -> (n, m) ndarray, which will translate to a graph with n nodes and each
    node will be assigned the corresponding (m,) feature vector
    w -> (n**2-n,) ndarray, weights for the edges.
    y -> (n,) ndarray, node labels
    scale -> Bool, whether to scale X between [0, 1], defaults to True.

    Expects
    -------
    y to be same length as X (number of labels should be same as number of
    nodes)
    w to be same length as number of edges of the graph

    Out
    ---
    G -> torch_geometric.data, Graph with n nodes, G.x assigned X, G.y
    assigned to y and G.edge_attr assigned to w.
    """
    G = nx.complete_graph(len(X))
    G = pgutils.from_networkx(G)
    if scale:
        X = MinMaxScaler().fit_transform(X) # scale between [0,1]
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    w = torch.tensor(w, dtype=torch.float)
    G.x, G.y, G.edge_attr  = X, y, w

    return G

def print_stats(G):
    """
    In
    --
    G -> torch_geometric.data, Graph

    Out
    ---
    None
    """
    print(G)
    print(f'Number of nodes: {G.num_nodes}')
    print(f'Number of edges: {G.num_edges}')
    print(f'Average node degree: {G.num_edges / G.num_nodes:.2f}')
    print(f'Edge weight shape: {G.edge_attr.shape}')
    print(f'Contains isolated nodes: {G.contains_isolated_nodes()}')
    print(f'Contains self-loops: {G.contains_self_loops()}')
    print(f'Is undirected: {G.is_undirected()}')

def visualize(x, color):
    """
    In
    --
    x -> (n, m) tensor, node features
    color -> (n,) tensor, passed to matplotlib.pyplot.scatter

    Out
    ---
    None
    """
    z = TSNE(n_components=2).fit_transform(x.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color.detach().cpu().numpy(), cmap="Set2")
    plt.show()

