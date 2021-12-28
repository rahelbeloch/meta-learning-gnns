import torch
import torch.nn.functional as func
from torch import nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Gat Layer from Phillips Tutorial
# (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html?highlight=graph%20attention#Graph-Attention)
# Extended with torch sparse vectors

class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()

        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Submodules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)

        # Parameters for each head
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))
        self.leaky_relu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, subgraph_batch, print_attn_probs=False):
        """
        Inputs:
            sub_graphs - Batch of sub graphs containing node features and edges.
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging)
        """

        node_feats = subgraph_batch.x
        num_nodes = subgraph_batch.num_nodes
        edge_index = subgraph_batch.edge_index

        assert node_feats.is_sparse, "Features vector is not sparse!"

        batch_size = 1

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # Create adjacency matrix from the edge list; MUST be on cuda if available!
        adj_matrix = torch.zeros((num_nodes, num_nodes)).to(torch.int64)
        for i, edge in enumerate(edge_index.T):
            adj_matrix[edge[0], edge[1]] = 1

        # we consider the whole graph (containing multiple sub graphs) because we have as one batch
        adj_matrix = adj_matrix.unsqueeze(dim=0)

        # Calculate attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]

        # need to be on the same device (GPU if available) for index select
        print(f"edge_indices_col is on device {edge_indices_col.device}.")
        print(f"edge_indices_row is on device {edge_indices_row.device}.")
        print(f"node_feats_flat is on device {node_feats_flat.device}.")

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

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        head_mask = adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1
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

    # def index_select_sparse(groups, mask_index):
    #
    #     index = groups._indices()
    #     newrowindex = -1
    #
    #     for ind in mask_index:
    #         try:
    #             newrowindex = newrowindex + 1
    #         except NameError:
    #             newrowindex = 0
    #
    #         keptindex = torch.squeeze((index[0] == ind).nonzero())
    #
    #         if len(keptindex.size()) == 0:
    #             # Get column values from mask, create new row idx
    #             try:
    #                 newidx = torch.cat((newidx, torch.tensor([newrowindex])), 0)
    #                 newcolval = torch.cat((newcolval, torch.tensor([index[1][keptindex.item()]])), 0)
    #             except NameError:
    #                 newidx = torch.tensor([newrowindex])
    #                 newcolval = torch.tensor([index[1][keptindex.item()]])
    #
    #         else:
    #             # Get column values from mask, create new row idx
    #             # Add newrowindex eee.size() time to list
    #             for i in range(list(keptindex.size())[0]):
    #                 try:
    #                     newidx = torch.cat((newidx, torch.tensor([newrowindex])), 0)
    #                     newcolval = torch.cat((newcolval, torch.tensor([index[1][keptindex.tolist()[i]]])), 0)
    #                 except NameError:
    #                     newidx = torch.tensor([newrowindex])
    #                     newcolval = torch.tensor([index[1][keptindex.tolist()[i]]])
    #
    #     groups = torch.sparse_coo_tensor(indices=torch.stack((newidx, newcolval), dim=0),
    #                                      values=torch.ones(newidx.shape[0], dtype=torch.float),
    #                                      size=(len(mask_index), groups.shape[1]))
    #     return groups