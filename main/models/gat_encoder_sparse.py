import torch
import torch.nn.functional as func
from torch import nn


class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=2, concat_heads=True, alpha=0.2):
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

        self.attentions = [SparseAttention(c_in, c_out, alpha) for _ in range(num_heads)]

        # register the attention layers so that lightning can find them
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, subgraph_batch):
        """
        Inputs:
            sub_graphs - Batch of sub graphs containing node features and edges.
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging)
        """

        node_feats = subgraph_batch.x.to(torch.float32)
        # assert node_feats.is_sparse, "Feature vector is not sparse!"

        # TODO: try to make more sparse, e.g. edge index etc.

        att_outs = [att(node_feats, subgraph_batch.edge_index, subgraph_batch.num_nodes) for att in self.attentions]

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            x = torch.cat(att_outs, dim=1)
        else:
            x = torch.mean(torch.stack(att_outs, dim=1), dim=1)
        return x


class SparseAttention(nn.Module):

    def __init__(self, c_in, c_out, alpha):
        super(SparseAttention, self).__init__()

        gain = nn.init.calculate_gain('leaky_relu')

        # linear projection parameters for head
        self.projection = nn.Linear(c_in, c_out)
        nn.init.xavier_uniform_(self.projection.weight.data, gain=gain)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * c_out)))
        nn.init.xavier_uniform_(self.a.data, gain=gain)

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, node_feats, edges, n):
        dv = node_feats.device

        # linear layer
        h = self.projection(node_feats)
        # h: N x c_out

        # TODO: fix this; h is getting nan at some point
        # assert not torch.isnan(h).any()
        if torch.isnan(h).any():
            h = self.nan_to_num(h)

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edges[0, :], :], h[edges[1, :], :]), dim=1).t()
        # edge: 2*c_out x E

        edge_e = torch.exp(-self.leaky_relu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e1: E

        edges = edges.to(dv)
        edge_e = edge_e.to(dv)

        e_row_sum = SparseMultiplyFunction.apply(edges, edge_e, torch.Size([n, n]), torch.ones(size=(n, 1), device=dv))
        # e_row_sum: N x 1

        h_prime = SparseMultiplyFunction.apply(edges, edge_e, torch.Size([n, n]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_row_sum)
        # h_prime: N x out

        # TODO: fix this; h_prime is getting nan at some point
        # assert not torch.isnan(h_prime).any()
        if torch.isnan(h_prime).any():
            h_prime = self.nan_to_num(h_prime)

        return func.elu(h_prime)

    @staticmethod
    def nan_to_num(x):
        """
        Creates masks for all values in the tensor x which are either nan, negative inf or inf. Then sets respective
        alternative values for these indices.
        """

        # TODO: make sure back propagation can still properly be done
        # because otherwise we get an error saying that isneginf/isinf/isnan
        # can not be used with tensors requiring grad
        with torch.no_grad():
            neg_inf_mask = torch.isneginf(x)
            inf_mask = torch.isinf(x)
            nan_mask = torch.isnan(x)

        x[neg_inf_mask] = -3.4028e+38
        x[inf_mask] = 3.4028e+38
        x[nan_mask] = 0
        return x

    @staticmethod
    def idx_select(node_feats, edge_indices):
        return torch.index_select(input=node_feats, index=edge_indices, dim=0)


class SparseMultiplyFunction(torch.autograd.Function):
    """
    A function which converts edge values and indices to sparse tensors and then applies a multiplication.
    Stores these tensors, and during backward, computes the gradients.
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
            # noinspection PyProtectedMember
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b
