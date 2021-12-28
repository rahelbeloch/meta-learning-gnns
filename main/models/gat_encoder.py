import torch
import torch.nn.functional as func
from torch import nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Gat Layer from Phillips Tutorial
# (https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html?highlight=graph%20attention#Graph-Attention)
# Extended with torch sparse vectors

# class GATLayer(nn.Module):
#
#     def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
#         """
#         Inputs:
#             c_in - Dimensionality of input features
#             c_out - Dimensionality of output features
#             num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
#                         output features are equally split up over the heads if concat_heads=True.
#             concat_heads - If True, the output of the different heads is concatenated instead of averaged.
#             alpha - Negative slope of the LeakyReLU activation.
#         """
#         super().__init__()
#
#         self.num_heads = num_heads
#         self.concat_heads = concat_heads
#         if self.concat_heads:
#             assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
#             c_out = c_out // num_heads
#
#         # Submodules and parameters needed in the layer
#         self.projection = nn.Linear(c_in, c_out * num_heads)
#
#         # Parameters for each head
#         self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)).to_sparse()
#         self.leaky_relu = nn.LeakyReLU(alpha)
#
#         # Initialization from the original implementation
#         nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#     def forward(self, subgraph_batch, print_attn_probs=False):
#         """
#         Inputs:
#             sub_graphs - Batch of sub graphs containing node features and edges.
#             print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging)
#         """
#
#         node_feats = subgraph_batch.x
#         assert node_feats.is_sparse, "Feature vector is not sparse!"
#
#         num_nodes = subgraph_batch.num_nodes
#
#         # make dense because we want to iterate over it
#         edge_index = subgraph_batch.edge_index.T
#         assert not edge_index.is_sparse, "Edge index vector is sparse although it should not!"
#
#         # either receiving all sub graphs as single batch or each sub graph one by one
#         batch_size = 1
#
#         # Apply linear layer and sort nodes by head
#         node_feats = self.projection(node_feats).to_sparse()
#         assert node_feats.is_sparse
#
#         node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
#
#         batch_idx = torch.zeros((edge_index.shape[0], 1)).to(torch.int64).to(device)
#         edges = torch.cat((batch_idx, edge_index), dim=1)
#
#         # Calculate attention logits for every edge in the adjacency matrix
#         # Doing this on all possible combinations of nodes is very expensive
#         # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
#         # Returns indices where the adjacency matrix is not 0 => edges
#         node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
#         edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
#         edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
#
#         # need to be on the same device (GPU if available) for index select
#         edge_indices_col = edge_indices_col.to(device)
#         edge_indices_row = edge_indices_row.to(device)
#         node_feats_flat = node_feats_flat.to(device)
#
#         # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
#         idx_select_1 = self.idx_select(node_feats_flat, edge_indices_row)
#         idx_select_2 = self.idx_select(node_feats_flat, edge_indices_col)
#         a_input = torch.cat([idx_select_1, idx_select_2], dim=-1)
#
#         # Calculate attention MLP output (independent for each head)
#         attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
#         attn_logits = self.leaky_relu(attn_logits)
#         # print(f'Attn logits shape {str(attn_logits.shape)}')
#
#         adj_matrix = torch.zeros((num_nodes, num_nodes)).to(torch.int64).to(device)
#         for i, edge in enumerate(edge_index):
#             adj_matrix[edge[0], edge[1]] = 1
#
#         # we consider the whole graph (containing multiple sub graphs) because we have as one batch
#         adj_matrix = adj_matrix.unsqueeze(dim=0)
#
#         # adj_matrix_shape = (1, num_nodes, num_nodes)
#
#         # Map list of attention values back into a matrix
#         attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
#         # print(f'Attn matrix shape {str(attn_matrix.shape)}')
#
#         head_mask = adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1
#         # print(f'Head mask shape {str(head_mask.shape)}')
#         #
#         # print(f'Attn logits min {str(attn_logits.min())}')
#         # print(f'Attn logits max {str(attn_logits.max())}')
#
#         attn_matrix[head_mask] = attn_logits.reshape(-1)
#
#         # Weighted average of attention
#         attn_probs = func.softmax(attn_matrix, dim=2)
#
#         if print_attn_probs:
#             print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
#         node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)
#         # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
#         if self.concat_heads:
#             node_feats = node_feats.reshape(batch_size, num_nodes, -1)
#         else:
#             node_feats = node_feats.mean(dim=2)
#
#         return node_feats
#
#     @staticmethod
#     def idx_select(node_feats, edge_indices):
#         return torch.index_select(input=node_feats, index=edge_indices, dim=0)

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

        self.attention1 = SparseAttention(c_in, c_out, alpha)
        self.attention2 = SparseAttention(c_in, c_out, alpha)

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, subgraph_batch):
        """
        Inputs:
            sub_graphs - Batch of sub graphs containing node features and edges.
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging)
        """

        node_feats = subgraph_batch.x
        assert node_feats.is_sparse, "Feature vector is not sparse!"

        att_out1 = self.attention1(node_feats, subgraph_batch.edge_index, subgraph_batch.num_nodes)
        att_out2 = self.attention2(node_feats, subgraph_batch.edge_index, subgraph_batch.num_nodes)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            x = torch.cat([att_out1, att_out2], dim=1)
        else:
            x = torch.mean(torch.stack([att_out1, att_out2], dim=1), dim=1)

        return x


class SparseAttention(nn.Module):

    def __init__(self, c_in, c_out, alpha):
        super(SparseAttention, self).__init__()

        # linear projection parameters for head 1
        self.W = nn.Parameter(torch.zeros(size=(c_in, c_out)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * c_out)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, node_feats, edges, N):
        h = torch.mm(node_feats, self.W)
        # h: N x c_out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edges[0, :], :], h[edges[1, :], :]), dim=1).t()
        # edge: 2*c_out x E

        edge_e = torch.exp(-self.leaky_relu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e1: E

        edges = edges.to(device)
        edge_e = edge_e.to(device)

        e_row_sum = SpecialSpmmFunction.apply(edges, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=device))
        # e_row_sum: N x 1

        h_prime = SpecialSpmmFunction.apply(edges, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_row_sum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        return func.elu(h_prime)

    @staticmethod
    def idx_select(node_feats, edge_indices):
        return torch.index_select(input=node_feats, index=edge_indices, dim=0)

class SpecialSpmmFunction(torch.autograd.Function):
    """
    Special function for only sparse region backpropagation layer.
    """

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

# class LinearSparse(Module):
#     __constants__ = ['in_features', 'out_features']
#     in_features: int
#     out_features: int
#     weight: Tensor
#
#     def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
#         super(LinearSparse, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self) -> None:
#         init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
#
#         # make weights sparse
#         # self.weight = self.weight
#         # self.bias = self.bias.to_sparse()
#
#     def forward(self, input: Tensor) -> Tensor:
#         assert input.is_sparse
#         assert self.weight.is_sparse
#         assert self.bias.is_sparse
#         out = func.linear(input, self.weight, self.bias)
#         assert out.is_sparse
#         return out
#
#     def extra_repr(self) -> str:
#         return 'in_features={}, out_features={}, bias={}'.format(
#             self.in_features, self.out_features, self.bias is not None
#         )

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
