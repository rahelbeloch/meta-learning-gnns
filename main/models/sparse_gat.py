import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGatNet(nn.Module):
    def __init__(self, model_params):
        super(SparseGatNet, self).__init__()

        self.in_dim = model_params["input_dim"]
        self.hid_dim = model_params["hid_dim"]
        self.fc_dim = model_params["fc_dim"]
        self.output_dim = model_params["output_dim"]
        self.n_heads = model_params["n_heads"]

        self.mask_p = model_params["node_mask_p"]
        self.dropout = model_params["dropout"]
        self.attn_dropout = model_params["attn_dropout"]

        self.mha_1 = nn.ModuleList(
            [
                SparseGATLayer(
                    in_features=self.in_dim,
                    out_features=self.hid_dim,
                    attn_drop=self.attn_dropout,
                    alpha=0.2,
                )
                for _ in range(self.n_heads)
            ]
        )

        self.non_lin_1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.mha_2 = nn.ModuleList(
            [
                SparseGATLayer(
                    in_features=self.n_heads * self.hid_dim,
                    out_features=self.hid_dim,
                    attn_drop=self.attn_dropout,
                    alpha=0.2,
                )
                for _ in range(self.n_heads)
            ]
        )

        self.mha_collator = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.n_heads * self.hid_dim, self.fc_dim, bias=True),
            nn.ReLU(),
        )

        self.classifier = self.get_classifier(self.output_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_classifier(self, num_classes: int = None):
        # Adapting the classifier dimensions
        if num_classes is None:
            num_classes = self.output_dim

        self.classifier = self.get_classifier(num_classes).to(self.device)

    def get_classifier(self, output_dim):
        return nn.Linear(self.fc_dim, output_dim, bias=True)

    def node_mask(self, x):
        if self.training:
            node_mask = torch.rand((x.shape[0], 1)).type_as(x)
            x = node_mask.ge(self.mask_p) * x

        return x

    def extract_features(self, x, edge_index):
        if torch.cuda.is_available():
            assert x.is_cuda
            assert edge_index.is_cuda

        x = self.node_mask(x)

        # For some reason, this seems to make training unstable
        # x = F.dropout(x, self.dropout, training=self.training)

        # Attention on input
        x = torch.cat([head(x, edge_index) for head in self.mha_1], dim=1)

        x = self.non_lin_1(x)

        x = torch.cat([head(x, edge_index) for head in self.mha_2], dim=1)

        # Concatenate using linear projection of large vector to smaller vector
        x = self.mha_collator(x)

        return x

    def forward(self, x, edge_index, mode=None):
        #! Deprecated argument: mode
        # Mode left here for legacy purposes, no longer serves a purpose
        # All dropout are now registered modules (i.e. model.eval())

        x = self.extract_features(x, edge_index)

        # Classification head
        logits = self.classifier(x)

        return logits


class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer taken from the official PyTorch repository:
    https://github.com/Diego999/pyGAT/blob/similar_impl_tensorflow/layers.py
    """

    def __init__(
        self,
        in_features,
        out_features,
        attn_drop=0.1,
        alpha=0.2,
    ):
        super(SparseGATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.attn_dropout = nn.Dropout(attn_drop)

        self.alpha = alpha
        self.linear, self.seq_transformation = None, None

        self.seq_transformation = nn.Conv1d(
            self.in_features, self.out_features, kernel_size=1, stride=1, bias=False
        )

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

        self.a_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.a_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edges):
        # This is GATv1: i.e. static attention
        # It can be made sparse by pushing the attention mechanism into the
        # concantenation. As pointed out by GATv2, this comes at a severe
        # expressiveness cost.

        # Unfortunately, I don't know how to make GATv2 sparse...

        # Linearly transform the input =========================================
        # 1 x in_features x num_nodes
        seq = torch.transpose(x, 0, 1).unsqueeze(0)

        # 1 x out_features x num_nodes
        seq = self.seq_transformation(seq)

        # Compute edge weights =================================================
        # num_nodes
        a_1 = self.a_1(seq).squeeze()
        a_2 = self.a_2(seq).squeeze()

        # num_nodes
        score = a_1[edges[0]] + a_2[edges[1]]

        score = self.leaky_relu(score).exp()

        # num_nodes x 1
        score_sum = torch.zeros_like(x[:, 0]).index_add_(0, edges[0], score).view(-1, 1)

        score = self.attn_dropout(score)

        # Compute and pass weighted messages ===================================
        # num_nodes x out_features
        score = torch.sparse_coo_tensor(edges, score)

        # num_nodes x out_features
        seq = torch.transpose(seq.squeeze(0), 0, 1)

        # num_nodes x out_features
        seq = torch.sparse.mm(score, seq).div(score_sum) + self.bias

        return seq
