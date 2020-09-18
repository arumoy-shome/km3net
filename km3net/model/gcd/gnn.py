import torch.nn.functional as F
from torch.nn import Module, Linear
from torch_geometric.nn import GCNConv

class GNN(Module):
    def __init__(self, in_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_dim, 16)
        self.conv2 = GCNConv(16, 2)
        self.linear1 = Linear(2, 1)

    def forward(self, G):
        x, edge_index, edge_weight = G.x, G.edge_index, G.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        out = x.relu()
        out = self.linear1(out)
        out = out.sigmoid()

        return out, x

