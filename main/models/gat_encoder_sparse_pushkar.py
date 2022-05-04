import torch
import torch.nn.functional as func
from torch import nn
from torch_geometric.nn import GATConv


class GatNet(torch.nn.Module):
    """
    Graph Attention Networks
    https://arxiv.org/abs/1710.10903
    """

    def __init__(self, model_params):
        super(GatNet, self).__init__()
        self.n_heads = model_params["n_heads"]

        self.in_dim = model_params["input_dim"]

        self.output_dim = model_params["output_dim"]

        self.hid_dim = model_params["hid_dim"]
        self.feat_reduce_dim = model_params["feat_reduce_dim"]

        self.gat_dropout = model_params["gat_dropout"]
        self.lin_dropout = model_params["lin_dropout"]
        self.attn_dropout = model_params["attn_dropout"]

        # self.mask = model_params["mask_rate"]
        # self.attn = model_params.get("gat_attn", False)

        self.elu = nn.ELU()

        self.attentions = [SparseGATLayer(self.in_dim,
                                          self.hid_dim,
                                          self.feat_reduce_dim,
                                          dropout=self.gat_dropout,
                                          attn_drop=self.attn_dropout,
                                          concat=True) for _ in range(self.n_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        self.classifier = self.get_classifier()

    # def reset_classifier_dimensions(self, num_classes):
    #     # adapting the classifier dimensions
    #     self.classifier = self.get_classifier()

    def get_classifier(self):
        # Phillips implementation
        # return nn.Sequential(
        #     nn.Dropout(self.lin_dropout),
        #     nn.Linear(self.hid_dim * self.n_heads, num_classes)
        # )

        # Shans implementation
        # return nn.Sequential(nn.Dropout(self.lin_dropout),
        #                      nn.Linear(self.n_heads * self.hid_dim, self.feat_reduce_dim),
        #                      nn.ReLU(),
        #                      nn.Linear(self.feat_reduce_dim, num_classes))

        # Pushkar implementation
        return SparseGATLayer(self.hid_dim * self.n_heads, self.output_dim, self.feat_reduce_dim, dropout=self.gat_dropout,
                              attn_drop=self.attn_dropout, concat=False)

    def forward(self, x, edge_index, mode):
        x = func.dropout(x, self.gat_dropout, training=mode == 'train')
        if not x.is_sparse:
            x = x.to_sparse()

        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)

        if type(self.classifier) != SparseGATLayer:
            # linear classifier
            out = self.classifier(x)
        else:
            x = func.dropout(x, self.gat_dropout, training=mode == 'train')

            # attention out layer
            out = self.classifier(x, edge_index)

        # F1 is sensitive to threshold
        # area under the RC curve

        return out


class SparseGATLayer(nn.Module):
    """
    Sparse version GAT layer taken from the official PyTorch repository:
    https://github.com/Diego999/pyGAT/blob/similar_impl_tensorflow/layers.py
    """

    def __init__(self, in_features, out_features, feat_reduce_dim, dropout=0.6, attn_drop=0.6, alpha=0.2, concat=False):
        super(SparseGATLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.attn_dropout = nn.Dropout(attn_drop)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.concat = concat
        self.linear, self.seq_transformation = None, None

        # Constant projection to lower dimension: 256; compressing features
        self.linear = nn.Linear(in_features, feat_reduce_dim, bias=False)

        # grad of the linear layer false --> will not be learned but instead constant projection
        self.linear.requires_grad_(False)

        self.seq_transformation = nn.Conv1d(feat_reduce_dim, self.out_features, kernel_size=1, stride=1, bias=False)

        # initialization still matters
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear.weight.data, gain=gain)

        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edges):
        # assert x.is_sparse

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

        f_1 = self.f_1(seq_fts)
        f_2 = self.f_2(seq_fts)

        # TODO: generalize this in method
        # if there is a single data point, we loose all dimensions which should not be the case!
        f_1 = f_1.squeeze().unsqueeze(dim=0) if len(f_1.shape) == 0 else f_1.squeeze()
        f_2 = f_2.squeeze().unsqueeze(dim=0) if len(f_2.shape) == 0 else f_2.squeeze()

        logits = f_1[edges[0]] + f_2[edges[1]]
        coefs = self.leaky_relu(logits).exp()  # E
        coef_sum = torch.zeros_like(x[:, 0]).index_add_(0, edges[0], coefs).view(-1, 1)
        coefs = self.attn_dropout(coefs)
        sparse_coefs = torch.sparse_coo_tensor(edges, coefs)
        seq_fts = self.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1))
        ret = torch.sparse.mm(sparse_coefs, seq_fts).div(coef_sum) + self.bias

        return func.elu(ret) if self.concat else ret


class GraphNet(torch.nn.Module):
    def __init__(self, model_params):
        super(GraphNet, self).__init__()

        self.n_heads = model_params["n_heads"]

        self.in_dim = model_params["input_dim"]
        self.out_dim = model_params["output_dim"]
        self.hid_dim = model_params["hid_dim"]
        self.feat_reduce_dim = model_params["feat_reduce_dim"]

        self.gat_dropout = model_params["gat_dropout"]
        self.lin_dropout = model_params["lin_dropout"]
        self.attn_dropout = model_params["attn_dropout"]

        self.fc_dim = 64

        self.conv1 = GATConv(self.in_dim, self.hid_dim, heads=self.n_heads, concat=True, dropout=0.1)
        self.conv2 = GATConv(self.n_heads * self.hid_dim, self.hid_dim, heads=self.n_heads, concat=True, dropout=0.1)

        # Attention output layer or linear classifier

        # self.conv2 = GATConv(3*self.embed_dim, self.out_dim, heads=self.n_heads, concat=False, dropout=0.1)

        self.classifier = nn.Sequential(nn.Dropout(self.lin_dropout),
                                        nn.Linear(self.n_heads * self.hid_dim, self.fc_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.fc_dim, self.out_dim))

    def forward(self, x, edge_index, mode):
        # node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > self.node_drop
        # if self.training:
        #     x = node_mask.to(device) * x  # / (1 - self.node_drop)

        x = func.relu(self.conv1(x.float(), edge_index))
        x = func.dropout(x, p=self.attn_dropout, training=mode == 'train')
        x = self.conv2(x.float(), edge_index)
        out = self.classifier(x)
        # return out, node_mask
        return out
