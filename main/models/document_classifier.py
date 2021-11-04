import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv


class DocumentClassifier(pl.LightningModule):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    def __init__(self, model_hparams, optimizer_hparams, checkpoint=None, transfer=False, h_search=False):
        """
        Args:
            model_hparams - Hyperparameters for the whole model, as dictionary.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        model_name = model_hparams['model']
        cf_hidden_dim = model_hparams['cf_hid_dim']
        num_classes = model_hparams['output_dim']

        if model_name == 'gat':
            self.model = GATEncoder(model_hparams['input_dim'], hidden_dim=cf_hidden_dim, num_heads=2)
            self.classifier = nn.Sequential(
                # TODO: maybe
                # nn.Dropout(config["dropout"]),
                nn.Linear(3 * cf_hidden_dim, cf_hidden_dim),
                nn.ReLU(),
                nn.Linear(cf_hidden_dim, num_classes))
        else:
            raise ValueError("Model type '%s' is not supported." % model_name)

        self.loss_module = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer and enables training with different learning rates for encoder and classifier.
        Also initializes the learning rate scheduler.
        """

        lr_enc = self.hparams.optimizer_hparams['lr_enc']
        lr_cl = self.hparams.optimizer_hparams['lr_cl']
        if lr_cl < 0:  # classifier learning rate not specified
            lr_cl = lr_enc

        weight_decay_enc = self.hparams.optimizer_hparams["weight_decay_enc"]
        weight_decay_cl = self.hparams.optimizer_hparams["weight_decay_cl"]
        if weight_decay_cl < 0:  # classifier weight decay not specified
            weight_decay_cl = weight_decay_enc

        params = list(self.named_parameters())

        def is_encoder(n):
            return n.startswith('model')

        grouped_parameters = [
            {
                'params': [p for n, p in params if is_encoder(n)],
                'lr': lr_enc,
                'weight_decay': weight_decay_enc
            },
            {
                'params': [p for n, p in params if not is_encoder(n)],
                'lr': lr_cl,
                'weight_decay': weight_decay_cl
            }
        ]

        optimizer = torch.optim.AdamW(grouped_parameters)

        # self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.hparams.optimizer_hparams['warmup'],
        #                                           max_iters=self.hparams.optimizer_hparams['max_iters'])

        return [optimizer], []

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()

    def training_step(self, batch, _):
        # , mode='train'
        sub_graphs, labels = batch

        out, node_mask = self.model(sub_graphs)

        predictions = self.classifier(out)
        loss = self.loss_module(predictions, labels)

        self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        # logging in optimizer step does not work, therefore here
        self.log('lr_rate', self.lr_scheduler.get_lr()[0])
        return loss

    # def training_step(self, batch, _):
    #     logits = self.model(batch, mode='train')
    #
    #     unbatched = dgl.unbatch(batch)
    #
    #     classification_mask = batch.ndata['classification_mask']
    #     labels = batch.ndata['label'][classification_mask]
    #     predictions = logits[classification_mask]
    #
    #     # predictions = self.classifier(out)
    #     loss = self.loss_module(predictions, labels)
    #
    #     self.log('train_accuracy', self.accuracy(predictions, labels).item(), on_step=False, on_epoch=True)
    #     self.log('train_loss', loss)
    #
    #     # logging in optimizer step does not work, therefore here
    #     # self.log('lr_rate', self.lr_scheduler.get_lr()[0])
    #     return loss

    def validation_step(self, batch, _):
        # By default logs it per epoch (weighted average over batches)
        logits = self.model(batch, mode='val')

        classification_mask = batch.ndata['classification_mask']
        labels = batch.ndata['label'][classification_mask]
        predictions = logits[classification_mask]

        self.log('val_accuracy', self.accuracy(predictions, labels))

    def test_step(self, batch, _):
        # By default logs it per epoch (weighted average over batches)
        out, labels = self.model(batch, mode=self.test_val_mode)
        # predictions = self.classifier(out)
        self.log('test_accuracy', self.accuracy(out, labels))


# class GATEncoder(nn.Module):
#
#     def __init__(self, c_in=10000, c_out=300, num_heads=1, concat_heads=True, alpha=0.2, hidden_dim=768):
#         """
#         Creates a GATEncoder object.
#         Args:
#             c_in (int): Input dimension of the document embeddings (input features). Defaults to 10000.
#             c_out (int): Input dimension of the word embeddings (output features). Defaults to 300.
#             num_heads (int, opt): Number of heads, i.e. attention mechanisms to apply in parallel.
#                                        The output features are equally split up over the heads if concat_heads=True
#             concat_heads (bool, opt): If True, the output of the different heads is concatenated instead of averaged.
#             alpha (float, opt) - Negative slope of the LeakyReLU activation.
#
#             hidden_dim (int, opt): hidden dimension that will be used by the encoder. Defaults to 768.
#         """
#         super(GATEncoder, self).__init__()
#
#         self.num_heads = num_heads
#
#         self.concat_heads = concat_heads
#         if self.concat_heads:
#             assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
#             c_out = c_out // num_heads
#
#         # Sub-modules and parameters needed in the layer
#         self.projection = nn.Linear(c_in, c_out * num_heads)
#         self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))  # One per head
#         self.leaky_relu = nn.LeakyReLU(alpha)
#
#         # Initialization from the original implementation
#         nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#     def forward(self, node_feats, adj_matrix):
#         """
#         Inputs:
#             node_feats - Input features of the node. Shape: [batch_size, c_in]
#             adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
#         """
#         batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
#
#         # Apply linear layer and sort nodes by head
#         node_feats = self.projection(node_feats)
#         node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)
#
#         # We need to calculate the attention logits for every edge in the adjacency matrix
#         # Doing this on all possible combinations of nodes is very expensive
#         # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
#         # Returns indices where the adjacency matrix is not 0 => edges
#         edges = adj_matrix.nonzero(as_tuple=False)
#
#         node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
#         edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
#         edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
#
#         # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
#         a_input = torch.cat([
#             torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
#             torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
#         ], dim=-1)
#
#         # Calculate attention MLP output (independent for each head)
#         attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
#         attn_logits = self.leakyrelu(attn_logits)
#
#         # Map list of attention values back into a matrix
#         attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
#         attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)
#
#         # Weighted average of attention
#         attn_probs = F.softmax(attn_matrix, dim=2)
#         node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)
#
#         # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
#         if self.concat_heads:
#             node_feats = node_feats.reshape(batch_size, num_nodes, -1)
#         else:
#             node_feats = node_feats.mean(dim=2)
#
#         return node_feats

# Using GATConv from torch geometric
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, merge='mean'):
        super(GATEncoder, self).__init__()

        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=num_heads, concat=merge == 'cat', dropout=0.1)
        self.conv2 = GATv2Conv(3 * hidden_dim, hidden_dim, heads=num_heads, concat=merge == 'cat', dropout=0.1)

    def forward(self, sub_graphs):
        # TODO: check what this is
        # node_mask = torch.FloatTensor(x.shape[0], 1).uniform_() > self.node_drop
        # if self.training:
        #     x = node_mask.to(device) * x  # / (1 - self.node_drop)
        node_mask = None

        # Check if we can get edge weights
        x, edge_index = sub_graphs.ndata['feat'], torch.stack(sub_graphs.all_edges())

        x = self.conv1(x.float(), edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # TODO: add edge weight?
        x = self.conv2(x.float(), edge_index)

        return x, node_mask

# Taken from: https://www.dgl.ai/blog/2019/02/17/gat.html (GAT Encoder in DGL)

# class GATEncoder(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, num_heads, merge='mean'):
#         super(GATEncoder, self).__init__()
#
#         self.merge = merge
#
#         # assert out_dim % num_heads == 0, "Number of output features must be a multiple of the count of heads."
#         # out_dim = out_dim // num_heads
#
#         self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, merge)
#
#         # Be aware that the input dimension is hidden_dim*num_heads since
#         # multiple head outputs are concatenated together. Also, only
#         # one attention head in the output layer.
#
#         if self.merge == 'cat':
#             self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, merge)
#         elif self.merge == 'mean':
#             self.layer2 = MultiHeadGATLayer(hidden_dim, out_dim, 1, merge)
#
#     def forward(self, graph, mode):
#         # graph.ndata is a dict containing keys: ['feat', 'label', 'train_mask', 'val_mask', '_ID']
#         node_feat = graph.ndata['feat'].float()
#
#         node_feat = self.layer1(graph, node_feat)
#         node_feat = F.elu(node_feat)
#         node_feat = self.layer2(graph, node_feat)
#
#         # return node_feat
#         return node_feat
#
#
# class MultiHeadGATLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, num_heads, merge):
#         super(MultiHeadGATLayer, self).__init__()
#         self.heads = nn.ModuleList()
#         for i in range(num_heads):
#             self.heads.append(GatLayer(in_dim, out_dim))
#         self.merge = merge
#
#     def forward(self, graph, h):
#         head_outs = [attn_head(graph, h) for attn_head in self.heads]
#         if self.merge == 'cat':
#             # concat on the output feature dimension (dim=1)
#             return torch.cat(head_outs, dim=1)
#         elif self.merge == 'mean':
#             # merge using average torch.mean(torch.stack(head_outs), dim=0)
#             # TODO: fix dimensions when using mean
#             # return torch.mean(torch.stack(head_outs))
#             return torch.mean(torch.stack(head_outs), dim=0)
#
#
# class GatLayer(nn.Module):
#
#     def __init__(self, in_dim, out_dim):
#         super(GatLayer, self).__init__()
#
#         # equation (1)
#         self.fc = nn.Linear(in_dim, out_dim, bias=False)
#
#         # equation (2)
#         self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """
#         Reinitialize learnable parameters.
#         """
#
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
#
#     def edge_attention(self, edges):
#         # edge UDF for equation (2)
#         z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
#         a = self.attn_fc(z2)
#         return {'e': F.leaky_relu(a)}
#
#     @staticmethod
#     def message_func(edges):
#         # message UDF for equation (3) & (4)
#         return {'z': edges.src['z'], 'e': edges.data['e']}
#
#     @staticmethod
#     def reduce_func(nodes):
#         # reduce UDF for equation (3) & (4)
#         # equation (3)
#         alpha = F.softmax(nodes.mailbox['e'], dim=1)
#         # equation (4)
#         h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
#         return {'h': h}
#
#     def forward(self, graph, h):
#         # equation (1)
#         z = self.fc(h)
#
#         graph.ndata['z'] = z
#
#         # equation (2)
#         graph.apply_edges(self.edge_attention)
#
#         # equation (3) & (4)
#         graph.update_all(self.message_func, self.reduce_func)
#
#         h = graph.ndata.pop('h')
#
#         return h
