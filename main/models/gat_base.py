import pytorch_lightning as pl
import torch
from torch import nn
# from models.gat_encoder import GATLayer
# from models.gat_encoder_sparse import GATLayer
from torch_geometric.data import Batch

from models.gat_encoder_sparse_pushkar import SparseGATLayer
from models.train_utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class GatBase(pl.LightningModule):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_hparams, optimizer_hparams, batch_size, checkpoint=None):
        """
        Args:
            model_hparams - Hyperparameters for the whole model, as dictionary.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        # self.model = GATLayer(in_features=model_hparams['input_dim'], out_features=model_hparams['hid_dim'])
        self.model = SparseGATLayer(in_features=model_hparams['input_dim'], out_features=model_hparams['hid_dim'])

        if checkpoint is not None:
            encoder = load_pretrained_encoder(checkpoint)
            self.model.load_state_dict(encoder)

        self.classifier = self.get_classifier(model_hparams['output_dim'])

        self.loss_module = nn.CrossEntropyLoss(weight=model_hparams["class_weight"])

    def reset_dimensions(self, num_classes, num_features):
        # adapting the classifier dimensions
        self.classifier = self.get_classifier(num_classes)

        # setting dimensions of constant projection
        self.model.initialize_first_layers(num_features)

    def get_classifier(self, num_classes):
        cf_hidden_dim = self.hparams['model_hparams']['hid_dim']
        return nn.Sequential(
            # TODO: maybe
            # nn.Dropout(config["dropout"]),
            # WHY??
            # nn.Linear(3 * cf_hidden_dim, cf_hidden_dim),
            nn.Linear(cf_hidden_dim, cf_hidden_dim),
            nn.ReLU(),
            nn.Linear(cf_hidden_dim, num_classes)
        )

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

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=self.hparams['batch_size'])

    def forward(self, sub_graphs, mode=None):

        # if mode == 'test' or mode == 'val':
        #     print(f"\nMode {mode}")
        #     num_nodes = [g.num_nodes for g in sub_graphs]
        #     print(f"Nums nodes: {str(num_nodes)}")

        # make a batch out of all sub graphs and push the batch through the model
        # [Data, Data, Data(x, y, ..)]
        x, edge_index = get_subgraph_batch(sub_graphs)
        feats = self.model(x, edge_index)
        feats = get_classify_node_features(sub_graphs, feats)

        assert len(feats) == len(sub_graphs), "Nr of features returned does not equal nr. of classification nodes!"

        return feats

    def training_step(self, batch, batch_idx):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sub_graphs, targets = batch
        out = self.forward(sub_graphs, mode='train')
        predictions = self.classifier(out)
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
        out = self.forward(sub_graphs, mode='val')
        predictions = self.classifier(out)

        val_accuracy = accuracy(predictions, targets)
        f1_macro = f1(predictions, targets, average='macro')
        f1_micro = f1(predictions, targets, average='micro')

        self.log_on_epoch('val_accuracy', val_accuracy)
        self.log_on_epoch('val_f1_macro', f1_macro)
        self.log_on_epoch('val_f1_micro', f1_micro)

        print(f"\nValidation accuracy: {val_accuracy}")
        print(f"Validation F1 macro: {f1_macro}")
        print(f"Validation F1 micro: {f1_micro}")

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default logs it per epoch (weighted average over batches)
        sub_graphs, targets = batch
        out = self.forward(sub_graphs, mode='test')
        predictions = self.classifier(out)

        self.log_on_epoch('test_accuracy', accuracy(predictions, targets))
        self.log_on_epoch('test_f1_macro', f1(predictions, targets, average='macro'))
        self.log_on_epoch('test_f1_micro', f1(predictions, targets, average='micro'))


def load_pretrained_encoder(checkpoint_path):
    """
    Load a pretrained encoder state dict and remove 'model.' from the keys in the state dict, so that solely
    the encoder can be loaded.

    Args:
        checkpoint_path (str) - Path to a checkpoint for the DocumentClassifier.
    Returns:
        encoder_state_dict (dict) - Containing all keys for weights of encoder.
    """
    checkpoint = torch.load(checkpoint_path)
    encoder_state_dict = {}
    for layer, param in checkpoint["state_dict"].items():
        if layer.startswith("model"):
            new_layer = layer[layer.index(".") + 1:]
            encoder_state_dict[new_layer] = param

    return encoder_state_dict


def get_classify_node_features(graphs, features):
    cl_n_indices, n_count = [], 0
    for graph in graphs:
        cl_n_indices.append(n_count + graph.center_idx)
        n_count += graph.num_nodes
    return features[cl_n_indices]


def get_subgraph_batch(graphs):
    batch = Batch.from_data_list(graphs)

    x = batch.x.float()
    if not x.is_sparse:
        x = x.to_sparse()

    return x, batch.edge_index
