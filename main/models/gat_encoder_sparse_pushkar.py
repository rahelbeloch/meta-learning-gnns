import torch
import torch.nn.functional as func
from torch import nn


class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer taken from the official PyTorch repository:
    https://github.com/Diego999/pyGAT/blob/similar_impl_tensorflow/layers.py
    """

    def __init__(self, in_features, out_features, dropout=0.6, attn_drop=0.6, alpha=0.2, concat=False):
        super(SparseGATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.attn_dropout = nn.Dropout(attn_drop)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.concat = concat
        self.linear = None

        # Constant projection
        # TODO: we don't project down on sth, constant projection
        # self.linear = nn.Linear(in_features, out_features, bias=False)
        self.initialize_lin_layer(in_features)

        # TODO: still initialize even if constant?
        # gain = nn.init.calculate_gain('leaky_relu')
        # nn.init.xavier_uniform_(self.linear.weight.data, gain=gain)

        # grad of the linear layer false --> will not be learned but instead constant projection
        self.linear.requires_grad_(False)

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def initialize_lin_layer(self, in_features):
        self.linear = nn.Linear(in_features, in_features, bias=False)

    def forward(self, x, edges):
        assert x.is_sparse

        # edges may not be sparse, because using a sparse vector as index for another vector does not work
        # (e.g. f_1[edges[0]] + f_2[edges[1]])
        assert not edges.is_sparse

        # initialize x to be simple weight matrix
        x = torch.mm(x, self.linear.weight.t())

        if torch.cuda.is_available():
            assert x.is_cuda
            assert edges.is_cuda

        seq = torch.transpose(x, 0, 1).unsqueeze(0)
        seq_fts = self.seq_transformation(seq)

        f_1 = self.f_1(seq_fts).squeeze()
        f_2 = self.f_2(seq_fts).squeeze()
        logits = f_1[edges[0]] + f_2[edges[1]]
        coefs = self.leaky_relu(logits).exp()  # E

        coef_sum = torch.zeros_like(x[:, 0]).index_add_(0, edges[0], coefs).view(-1, 1)
        coefs = self.attn_dropout(coefs)
        sparse_coefs = torch.sparse_coo_tensor(edges, coefs)
        seq_fts = self.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1))
        ret = torch.sparse.mm(sparse_coefs, seq_fts).div(coef_sum) + self.bias
        return func.elu(ret) if self.concat else ret
