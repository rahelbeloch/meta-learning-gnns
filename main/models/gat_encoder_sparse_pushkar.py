import torch
import torch.nn.functional as func
from torch import nn


class GatNet(torch.nn.Module):

    def __init__(self, model_hparams):
        super(GatNet, self).__init__()

        self.dropout_lin = model_hparams['dropout_lin']
        self.dropout = model_hparams['dropout']
        self.hidden_dim = model_hparams['hid_dim']

        self.layer1 = SparseGATLayer(model_hparams['input_dim'], model_hparams['hid_dim'],
                                     model_hparams['feat_reduce_dim'])

        in_layer2 = 2 * model_hparams['hid_dim']
        print(f'input size layer 2: {in_layer2}')

        self.layer2 = SparseGATLayer(in_layer2, model_hparams['hid_dim'],
                                     model_hparams['feat_reduce_dim'])

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

    def forward(self, x, edge_index, mode):
        x = self.layer1(x, edge_index)
        # print(f'x is sparse after layer 1 {x.is_sparse}')
        print(f'output size after layer 1: {x.shape}')

        # TODO: check if we need this
        # x = func.relu(x)
        # print(f'x is sparse after relu {x.is_sparse}')

        x = func.dropout(x, p=self.dropout, training=mode == 'train')

        # print(f'x is sparse after layer 2 {x.is_sparse}')
        if not x.is_sparse:
            x = x.to_sparse()
        print(f'output size after dropout: {x.shape}')
        x = self.layer2(x, edge_index)

        out = self.classifier(x)
        return out, x


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
