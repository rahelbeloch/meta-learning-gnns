import torch
import torch.nn.functional as func
from torch import nn


class GatNetSparse(torch.nn.Module):
    """
    Graph Attention Networks
    https://arxiv.org/abs/1710.10903
    """

    def __init__(self, model_params):
        super(GatNetSparse, self).__init__()
        self.n_heads = model_params["n_heads"]

        self.in_dim = model_params["input_dim"]
        self.out_dim = model_params["output_dim"]
        self.hid_dim = model_params["hid_dim"]
        self.feat_reduce_dim = model_params["feat_reduce_dim"]

        # self.mask = model_params["mask_rate"]
        self.dropout = model_params.get("gat_dropout", 0.6)
        self.dropout_lin = model_params["dropout_lin"]

        # self.attn_drop = model_params.get("gat_mask", 0.6)
        # self.attn = model_params.get("gat_attn", False)
        # self.alpha = model_params.get("gat_alpha", 0.2)

        self.elu = nn.ELU()

        self.attentions = [
            SparseGATLayer(
                self.in_dim,
                self.hid_dim,
                self.feat_reduce_dim,
                self.dropout,
                # self.attn_drop,
                # self.alpha
            )
            for _ in range(self.n_heads)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)

        # self.out_att = SparseGATLayer(self.hid_dim * self.n_heads, self.out_dim, self.feat_reduce_dim, self.dropout,
        #                               # self.attn_drop, self.alpha
        #                               )

        self.classifier = self.get_classifier(self.out_dim)

    def reset_classifier_dimensions(self, num_classes):
        # adapting the classifier dimensions
        self.classifier = self.get_classifier(num_classes)

    def get_classifier(self, num_classes):
        # Phillips implementation
        return nn.Sequential(
            nn.Dropout(self.dropout_lin),
            nn.Linear(self.hid_dim * self.n_heads, num_classes)
        )

        # Shans implementation
        # self.classifier = nn.Sequential(nn.Dropout(config["dropout"]),
        #                                 nn.Linear(self.n_heads * self.hid_dim, self.fc_dim),
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.fc_dim, num_classes))

    def forward(self, x, edge_index, cl_mask, mode):
        # adj should be a sparse matrix

        # these 2 lines are only adding self loops; if I have them already, comment out
        # r = torch.arrange(adj.size(0)).repeat(2, 1).to(adj.device)
        # adj = adj + torch.sparse_coo_tensor(r, torch.ones(len(adj)).to(r))

        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = self.elu(x)

        x = x[cl_mask]
        out = self.classifier(x)

        return out


class GatNet(torch.nn.Module):

    def __init__(self, model_hparams):
        super(GatNet, self).__init__()

        self.dropout_lin = model_hparams['dropout_lin']
        self.dropout = model_hparams['dropout']
        self.hidden_dim = model_hparams['hid_dim']

        self.layer1 = SparseGATLayer(model_hparams['input_dim'], model_hparams['hid_dim'],
                                     model_hparams['feat_reduce_dim'], concat=model_hparams['concat'])

        # TODO: Check if we should really concatenate or not?
        # no multiple times hidden dim needed because Pushkar's version does not realy concatenate
        self.layer2 = SparseGATLayer(model_hparams['hid_dim'], model_hparams['hid_dim'],
                                     model_hparams['feat_reduce_dim'], concat=model_hparams['concat'])

        self.classifier = self.get_classifier(model_hparams['output_dim'])

    def reset_classifier_dimensions(self, num_classes):
        # adapting the classifier dimensions
        self.classifier = self.get_classifier(num_classes)

    def get_classifier(self, num_classes):
        # Phillips implementation
        return nn.Sequential(
            nn.Dropout(self.dropout_lin),
            nn.Linear(self.hidden_dim, num_classes)
        )

        # Shans implementation
        # self.classifier = nn.Sequential(nn.Dropout(config["dropout"]),
        #                                 nn.Linear(3 * self.embed_dim, self.fc_dim), 3 is number of heads
        #                                 nn.ReLU(),
        #                                 nn.Linear(self.fc_dim, config['n_classes']))

    def forward(self, x, edge_index, cl_node_mask, train=False):
        x = self.layer1(x, edge_index)

        # TODO: check if we need this (Pushkar's version is already doing this whe concatenating)
        # x = func.relu(x)
        # print(f'x is sparse after relu {x.is_sparse}')

        x = func.dropout(x, p=self.dropout, training=train)

        if not x.is_sparse:
            x = x.to_sparse()

        x = self.layer2(x, edge_index)

        x = x[cl_node_mask]

        out = self.classifier(x)
        return out


class GAT(nn.Module):
    """
    Graph Attention Networks
    https://arxiv.org/abs/1710.10903
    """

    def __init__(self, model_params):
        super(GAT, self).__init__()
        self.nheads = model_params["num_conv_layers"]
        self.nfeat = model_params["in_feature_dim"]
        self.nclass = model_params["output_dim"]
        self.nhid = model_params["embed_dim"]
        self.mask = model_params["mask_rate"]
        self.dropout = model_params.get("gat_dropout", 0.6)
        self.attn_drop = model_params.get("gat_mask", 0.6)
        self.attn = model_params.get("gat_attn", False)
        self.alpha = model_params.get("gat_alpha", 0.2)
        self.sparse = model_params.get("gat_sparse", False)
        self.elu = nn.ELU()
        gat_layer = layers.SparseGATLayer if self.sparse else layers.GATLayer
        self.attentions = [
            gat_layer(
                self.nfeat,
                self.nhid,
                self.dropout,
                self.attn_drop,
                self.alpha,
                self.attn,
            )
            for _ in range(self.nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)
        self.out_att = gat_layer(
            self.nhid * self.nheads,
            self.nclass,
            self.dropout,
            self.attn_drop,
            self.alpha,
            self.attn,
        )
        # Flag to return embeddings
        self.embeddings = False

    def forward(self, adj, x):

        # adj should be a sparse matrix

        if not self.sparse:
            adj = (
                adj.to_dense()
                    .fill_(-9e15)
                    .fill_diagonal_(0.0)
                    .index_put(tuple(adj._indices()), torch.tensor([0.0]).to(adj))
            )
        else:
            # these 2 lines are only adding self loops; if I have them already, comment out
            r = torch.arange(adj.size(0)).repeat(2, 1).to(adj.device)
            adj = adj + torch.sparse_coo_tensor(r, torch.ones(len(adj)).to(r))

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        if self.embeddings:
            return x
        x = self.elu(x)
        x = self.out_att(x, adj)
        return x


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

        # TODO: return sparse here?
        return func.elu(ret) if self.concat else ret
