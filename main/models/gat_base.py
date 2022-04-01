import numpy as np
import torch.nn.functional as func
from torch import nn, optim

from models.GraphTrainer import GraphTrainer
from models.gat_encoder_sparse_pushkar import GatNet
from models.train_utils import *


class GatBase(GraphTrainer):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, label_names, batch_size):
        """
        Args:
            model_params - Hyperparameters for the whole model, as dictionary.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__(model_params["output_dim"])

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace + saves config in wandb
        self.save_hyperparameters()

        self.model = GatNet(model_params)

        # # TODO: move this to GatNet
        # if checkpoint is not None:
        #     encoder = load_pretrained_encoder(checkpoint)
        #     self.model.load_state_dict(encoder)

        # flipping the weights
        flipped_weights = torch.flip(model_params["class_weight"], dims=[0])

        # Loss function consistent with labels?
        # Verify that this is the binary cross entropy loss
        self.loss_module = nn.BCEWithLogitsLoss(weight=flipped_weights)

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer and enables training with different learning rates for encoder and classifier.
        Also initializes the learning rate scheduler.
        """

        lr = self.hparams.optimizer_hparams['lr']

        # weight_decay_enc = self.hparams.optimizer_hparams["weight_decay_enc"]
        weight_decay_enc = 5e-4

        params = list(self.named_parameters())

        def is_encoder(n):
            return n.startswith('model')

        grouped_parameters = [
            {
                'params': [p for n, p in params if is_encoder(n)],
                'lr': lr,
                'weight_decay': weight_decay_enc
            }
        ]

        optimizer = torch.optim.AdamW(grouped_parameters)

        self.lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.hparams.optimizer_hparams['warmup'],
                                                  max_iters=self.hparams.optimizer_hparams['max_iters'])

        return [optimizer], []

    def forward(self, batch, mode=None):

        support_graphs, query_graphs, support_targets, query_targets = batch

        # for test and val we only want to use query samples
        sub_graphs, targets = query_graphs, query_targets

        if mode == 'train':
            # collapse support and query set and train on whole
            sub_graphs = support_graphs + query_graphs
            targets = torch.cat([support_targets, query_targets])
        elif mode == 'val' or mode == 'test':
            # only validate on the query set to keep comparability with meta models
            sub_graphs = query_graphs
            targets = torch.cat([query_targets])

        # make a batch out of all sub graphs and push the batch through the model
        # [Data, Data, Data(x, y, ..)]
        x, edge_index, cl_mask = get_subgraph_batch(sub_graphs)

        logits = self.model(x, edge_index, mode)
        logits = logits[cl_mask]

        # make probabilities out of logits via sigmoid --> especially for the metrics; makes it more interpretable
        predictions = torch.sigmoid(logits).argmax(dim=-1)

        for mode_dict, _ in self.metrics.values():
            mode_dict[mode].update(predictions, targets)

        # logits are not yet put into a sigmoid layer, because the loss module does this combined
        return logits, targets

    def training_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logits, targets = self.forward(batch, mode='train')
        loss = self.loss_module(logits, func.one_hot(targets).float())

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train_loss", loss)

        # logging in optimizer step does not work, therefore here
        self.log('lr_rate', self.lr_scheduler.get_lr()[0])

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        logits, targets = self.forward(batch, mode='val')
        loss = self.loss_module(logits, func.one_hot(targets).float())

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"val_loss", loss)

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default, logs it per epoch (weighted average over batches)
        self.forward(batch, mode='test')


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


# noinspection PyProtectedMember
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler, combining warm-up with a cosine-shaped learning rate decay.
    """

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor()
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self):
        current_step = self.last_epoch
        lr_factor = 0.5 * (1 + np.cos(np.pi * current_step / self.max_num_iters))
        if current_step < self.warmup:
            lr_factor *= current_step * 1.0 / self.warmup
        return lr_factor
