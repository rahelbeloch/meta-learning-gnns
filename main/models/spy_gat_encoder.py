import pytorch_lightning as pl
import torch
import torch.nn.functional as func
from torch import nn
from torch_geometric.data import Batch

from models.gat_base import get_classify_node_features
from models.train_utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class SpyGATLayer(pl.LightningModule):

    def __init__(self, model_hparams, optimizer_hparams, batch_size, checkpoint=None):
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

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        n_features = model_hparams['input_dim']
        n_hidden = model_hparams['cf_hid_dim']
        n_classes = len(model_hparams['class_weight'])
        n_heads = 2
        alpha = 0.2
        dropout = None

        self.attentions = [SpGraphAttentionLayer(n_features,
                                                 n_hidden,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(n_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(n_hidden * n_heads,
                                             n_classes,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

        self.loss_module = nn.CrossEntropyLoss(weight=model_hparams["class_weight"])

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer and enables training with different learning rates for encoder and classifier.
        Also initializes the learning rate scheduler.
        """

        lr = self.hparams.optimizer_hparams['lr']
        lr_cl = self.hparams.optimizer_hparams['lr_cl']
        if lr_cl < 0:  # classifier learning rate not specified
            lr_cl = lr

        # weight_decay_enc = self.hparams.optimizer_hparams["weight_decay_enc"]
        # weight_decay_cl = self.hparams.optimizer_hparams["weight_decay_cl"]
        # if weight_decay_cl < 0:  # classifier weight decay not specified
        #     weight_decay_cl = weight_decay_enc

        params = list(self.named_parameters())

        def is_encoder(n):
            return n.startswith('model')

        grouped_parameters = [
            {
                'params': [p for n, p in params if is_encoder(n)],
                'lr': lr  # ,
                # 'weight_decay': weight_decay_enc
            },
            {
                'params': [p for n, p in params if not is_encoder(n)],
                'lr': lr_cl  # ,
                # 'weight_decay': weight_decay_cl
            }
        ]

        optimizer = torch.optim.AdamW(grouped_parameters)

        # self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.hparams.optimizer_hparams['warmup'],
        #                                           max_iters=self.hparams.optimizer_hparams['max_iters'])

        return [optimizer], []

    def training_step(self, batch, batch_idx):

        sub_graphs, targets = batch
        predictions = self.forward(sub_graphs)

        loss = self.loss_module(predictions, targets)

        self.log_on_epoch('train_accuracy', accuracy(predictions, targets))
        self.log_on_epoch('train_f1_macro', f1(predictions, targets, average='macro'))
        self.log_on_epoch('train_f1_micro', f1(predictions, targets, average='micro'))
        self.log('train_loss', loss)

        # TODO: add scheduler
        # logging in optimizer step does not work, therefore here
        # self.log('lr_rate', self.lr_scheduler.get_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):

        sub_graphs, targets = batch
        out = self.forward(sub_graphs)
        predictions = self.classifier(out)

        self.log_on_epoch('val_accuracy', accuracy(predictions, targets))
        self.log_on_epoch('val_f1_macro', f1(predictions, targets, average='macro'))
        self.log_on_epoch('val_f1_micro', f1(predictions, targets, average='micro'))

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default logs it per epoch (weighted average over batches)
        sub_graphs, targets = batch
        out = self.forward(sub_graphs)
        predictions = self.classifier(out)

        self.log_on_epoch('test_accuracy', accuracy(predictions, targets))
        self.log_on_epoch('test_f1_macro', f1(predictions, targets, average='macro'))
        self.log_on_epoch('test_f1_micro', f1(predictions, targets, average='micro'))

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=self.hparams['batch_size'])

    def forward(self, sub_graphs):

        for g in sub_graphs:
            g.x = g.x.float().to_sparse()

        batch = Batch.from_data_list(sub_graphs)

        x = batch.x
        num_nodes = batch.num_nodes

        adj = torch.zeros((num_nodes, num_nodes)).to(torch.int64)
        for i, edge in enumerate(batch.edge_index.T):
            adj[edge[0], edge[1]] = 1

        # TODO
        # if self.dropout is not None:
        #     x = func.dropout(x, self.dropout, training=self.training)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = func.dropout(x, self.dropout, training=self.training)
        x = func.elu(self.out_att(x, adj))

        # we don't have the same nodes for every subgraph
        # we only want to classify the one center node
        feats = get_classify_node_features(sub_graphs, x)

        assert len(feats) == len(sub_graphs), "Nr of features returned does not equal nr. of classification nodes!"
        return feats


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=None, alpha=0.2, concat=False):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out

        # TODO
        # assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        edge = edge.to(dv)
        edge_e = edge_e.to(dv)

        e_row_sum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_row_sum: N x 1

        if self.dropout is not None:
            edge_e = self.dropout(edge_e)
            # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_row_sum)
        # h_prime: N x out

        # TODO
        # assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return func.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        assert indices.device == values.device, \
            f"backend of indices ({indices.device}) must match backend of values ({values.device})"
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropagation layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

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
