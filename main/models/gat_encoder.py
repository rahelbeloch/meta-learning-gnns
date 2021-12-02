import torch
import torch.nn.functional as func
from torch import nn
from torch_geometric.nn import GATv2Conv


# Using GATConv from torch geometric
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, merge='mean'):
        super(GATEncoder, self).__init__()

        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=num_heads, concat=merge == 'cat', dropout=0.1)

        # why 3 * hidden_dim??
        # self.conv2 = GATv2Conv(3*hidden_dim, hidden_dim, heads=num_heads, concat=merge == 'cat', dropout=0.1)

        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=merge == 'cat', dropout=0.1)

    def forward(self, sub_graphs):
        # TODO: check what this is
        # node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > self.node_drop
        # if self.training:
        #     x = node_mask.to(device) * x  # / (1 - self.node_drop)
        node_mask = None

        # Check if we can get edge weights
        features, edge_index = sub_graphs.ndata['feat'], torch.stack(sub_graphs.all_edges())
        # TODO: if features is sparse; calling float might make it un sparse
        features = self.conv1(features.float(), edge_index)
        features = func.relu(features)

        # TODO: add dropout
        # x = func.dropout(x, p=self.dropout, training=self.training)
        # TODO: add edge weight?

        features = self.conv2(features.float(), edge_index)

        return features, node_mask