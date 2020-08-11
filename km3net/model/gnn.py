import dgl
from torch.nn import Linear, ReLU, Sigmoid, Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from dgl.nn.pytorch import GraphConv

class GNN(Module):
    def __init__(self, in_dim, hidden_dim):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden1 = Linear(hidden_dim, 1)
        xavier_uniform_(self.hidden1.weight)
        self.act3 = Sigmoid()

    def forward(self, g):
        h = g.in_degrees().view(-1, 1)
        h = self.conv1(g, h)
        h = self.act1(h)
        h = self.conv2(g, h)
        h = self.act2(h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        hg = self.hidden1(hg)
        hg = self.act3(hg)

        return hg

