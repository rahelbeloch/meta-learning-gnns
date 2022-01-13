import torch
import torch.nn.functional as func
from torch import nn


# Gat Layer from Phillips Tutorial
# (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html?highlight=graph%20attention#Graph-Attention)
# Extended with torch sparse vectors

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            in_features - Dimensionality of input features
            out_features - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()

        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert out_features % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            out_features = out_features // num_heads

        self.out_features = out_features

        # Submodules and parameters needed in the layer
        self.projection = nn.Linear(in_features, out_features * num_heads)

        # Parameters for each head
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edges, print_attn_probs=False):
        """
        Inputs:
            x - Batch of node features.
            edges - Batch of edges.
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging)
        """

        assert x.is_sparse, "Feature vector is not sparse!"

        device = x.device
        num_nodes = x.shape[0]

        # make dense because we want to iterate over it
        edges = edges.T
        assert not edges.is_sparse, "Edge index vector is sparse although it should not!"

        # either receiving all sub graphs as single batch or each sub graph one by one
        batch_size = 1

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(x).to_sparse()
        assert node_feats.is_sparse

        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        batch_idx = torch.zeros((edges.shape[0], 1)).to(torch.int64).to(device)
        edges = torch.cat((batch_idx, edges), dim=1)

        # Calculate attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]

        # need to be on the same device (GPU if available) for index select
        edge_indices_col = edge_indices_col.to(device)
        edge_indices_row = edge_indices_row.to(device)
        node_feats_flat = node_feats_flat.to(device)

        # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
        idx_select_1 = self.idx_select(node_feats_flat, edge_indices_row)
        idx_select_2 = self.idx_select(node_feats_flat, edge_indices_col)
        a_input = torch.cat([idx_select_1, idx_select_2], dim=-1)

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leaky_relu(attn_logits)
        # print(f'Attn logits shape {str(attn_logits.shape)}')

        adj_matrix = torch.zeros((num_nodes, num_nodes)).to(torch.int64).to(device)
        for i, edge in enumerate(edges):
            adj_matrix[edge[0], edge[1]] = 1

        # we consider the whole graph (containing multiple sub graphs) because we have as one batch
        adj_matrix = adj_matrix.unsqueeze(dim=0)

        # adj_matrix_shape = (1, num_nodes, num_nodes)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        # print(f'Attn matrix shape {str(attn_matrix.shape)}')

        head_mask = adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1
        # print(f'Head mask shape {str(head_mask.shape)}')
        #
        # print(f'Attn logits min {str(attn_logits.min())}')
        # print(f'Attn logits max {str(attn_logits.max())}')

        attn_matrix[head_mask] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = func.softmax(attn_matrix, dim=2)

        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)
        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats

    @staticmethod
    def idx_select(node_feats, edge_indices):
        return torch.index_select(input=node_feats, index=edge_indices, dim=0)

    def initialize_lin_layer(self, in_features):
        self.projection = nn.Linear(in_features, self.out_features * self.num_heads)
