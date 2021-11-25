import pytorch_lightning as pl
import sklearn
import torch
from torch import nn

from models.gat_encoder import GATEncoder


class DocumentClassifier(pl.LightningModule):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    def __init__(self, model_hparams, optimizer_hparams, checkpoint=None, h_search=False):
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
                # WHY??
                # nn.Linear(3 * cf_hidden_dim, cf_hidden_dim),
                nn.Linear(cf_hidden_dim, cf_hidden_dim),
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
                'lr': lr_enc  # ,
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
        out, node_mask = self.model(sub_graphs)

        # only predict for the center node
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)
        loss = self.loss_module(predictions, targets)

        self.log('train_accuracy', accuracy(predictions, targets), on_step=False, on_epoch=True)
        self.log('train_f1_macro', f1(predictions, targets, average='macro'), on_step=False, on_epoch=True)
        self.log('train_f1_micro', f1(predictions, targets, average='micro'), on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        # TODO: add scheduler
        # logging in optimizer step does not work, therefore here
        # self.log('lr_rate', self.lr_scheduler.get_lr()[0])
        return loss

    def validation_step(self, batch, batch_idx):

        sub_graphs, targets = batch
        out, node_mask = self.model(sub_graphs)

        # only predict for the center node
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)

        self.log('val_accuracy', accuracy(predictions, targets))
        self.log('val_f1_macro', f1(predictions, targets, average='macro'))
        self.log('val_f1_micro', f1(predictions, targets, average='micro'))

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default logs it per epoch (weighted average over batches)
        sub_graphs, targets = batch
        out, node_mask = self.model(sub_graphs)

        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)

        self.log('test_accuracy', accuracy(predictions, targets))
        self.log('test_f1_macro', f1(predictions, targets, average='macro'))
        self.log('test_f1_micro', f1(predictions, targets, average='micro'))


def accuracy(predictions, labels):
    # noinspection PyUnresolvedReferences
    return (labels == predictions.argmax(dim=-1)).float().mean().item()


def f1(predictions, targets, average='binary'):
    predictions_cpu = predictions.argmax(dim=-1).detach().cpu()
    targets_cpu = targets.detach().cpu()
    return sklearn.metrics.f1_score(targets_cpu, predictions_cpu, average=average).item()
