import pytorch_lightning as pl
import torch
from torch import nn

from models.gat_encoder import GATEncoder
from models.train_utils import *


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

        self.model = GATEncoder(model_hparams['input_dim'], hidden_dim=model_hparams['cf_hid_dim'], num_heads=2)

        if checkpoint is not None:
            encoder = load_pretrained_encoder(checkpoint)
            self.model.load_state_dict(encoder)

        self.classifier = self.get_classifier(model_hparams['output_dim'])

        self.loss_module = nn.CrossEntropyLoss()

    def reset_classifier(self, num_classes):
        self.classifier = self.get_classifier(num_classes)

    def get_classifier(self, num_classes):
        cf_hidden_dim = self.hparams['model_hparams']['cf_hid_dim']
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

        lr = self.hparams.optimizer_hparams.lr
        lr_cl = self.hparams.optimizer_hparams.lr_cl
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
        self.log(metric, value, on_step=False, on_epoch=True, batch_size=self.hparams['batch_size'])

    def training_step(self, batch, batch_idx):

        sub_graphs, targets = batch
        out, _ = self.model(sub_graphs)

        # only predict for the center node
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)
        loss = self.loss_module(predictions, targets)

        self.log_on_epoch('train_accuracy', accuracy(predictions, targets))
        self.log_on_epoch('train_f1_macro', f1(predictions, targets, average='macro'))
        self.log_on_epoch('train_f1_micro', f1(predictions, targets, average='micro'))
        self.log('train_loss', loss, batch_size=self.hparams['batch_size'])

        # TODO: add scheduler
        # logging in optimizer step does not work, therefore here
        # self.log('lr_rate', self.lr_scheduler.get_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):

        sub_graphs, targets = batch
        out, _ = self.model(sub_graphs)

        # only predict for the center node
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)

        self.log_on_epoch('val_accuracy', accuracy(predictions, targets))
        self.log_on_epoch('val_f1_macro', f1(predictions, targets, average='macro'))
        self.log_on_epoch('val_f1_micro', f1(predictions, targets, average='micro'))

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default logs it per epoch (weighted average over batches)
        sub_graphs, targets = batch
        out, _ = self.model(sub_graphs)

        out = out[sub_graphs.ndata['classification_mask']]
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
