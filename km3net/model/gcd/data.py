import pandas as pd
import torch
import networkx as nx
import torch_geometric as pg
import torch_geometric.utils as pgutils
import torch_geometric.data as pgdata
from sklearn.preprocessing import MinMaxScaler

def create_graph(df_path, weight_path):
    """
    In
    --
    df_path -> Str, path to pandas dataframe of shape (n, m) which
    will translate to a graph with n nodes and each node will be
    assigned the corresponding (m,) feature vector
    weight_path -> Str, path to pandas dataframe containing the
    weights. Ideally this should point to the exploded dataframe where
    the last column is the label

    Out
    ---
    G -> torch_geometric.data, Graph with n nodes each assigned (m,)
    feature vector.
    """
    df = pd.read_csv(df_path, header=None)
    weight = pd.read_csv(weight_path, header=None)
    weight = weight.values[:, -1:].reshape(-1,)
    G = nx.complete_graph(len(df))
    G = pgutils.from_networkx(G)
    X = df.values[:, :4] # extract node feature matrix
    X = MinMaxScaler().fit_transform(X) # scale between [0,1]
    X = torch.tensor(X, dtype=torch.float)
    y = df.values[:, 4:5] # extract node labels
    y = torch.tensor(y, dtype=torch.float)
    weight = torch.tensor(weight, dtype=torch.float)
    G.x, G.y, G.edge_attr  = X, y, weight

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

