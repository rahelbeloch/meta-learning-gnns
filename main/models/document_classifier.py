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

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()

    @staticmethod
    def f1(predictions, labels):
        return sklearn.metrics.f1_score(labels, predictions)  # , average="samples"

    def training_step(self, batch, _):

        sub_graphs, targets = batch
        out, node_mask = self.model(sub_graphs)

        # only predict for the center node
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)
        loss = self.loss_module(predictions, targets)

        self.log('train_accuracy', self.accuracy(predictions, targets).item(), on_step=False, on_epoch=True)

        predictions_cpu = predictions.detach().cpu()
        targets_cpu = targets.detach().cpu()

        print(f"Pred shape: {str(predictions_cpu.shape)}")
        print(f"Pred type: {str(predictions_cpu.type)}")
        print(f"Pred content: {str(predictions_cpu[0])}")

        print(f"Targets shape: {str(targets_cpu.shape)}")
        print(f"Targets type: {str(targets_cpu.type)}")
        print(f"Targets content: {str(targets_cpu[0])}")

        f1 = self.f1(predictions_cpu, targets_cpu).item()
        self.log('train_f1', f1, on_step=False, on_epoch=True)
        self.log('train_loss', loss)

        # TODO: add scheduler
        # logging in optimizer step does not work, therefore here
        # self.log('lr_rate', self.lr_scheduler.get_lr()[0])
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

    def validation_step(self, subgraph_batch, _):
        # By default logs it per epoch (weighted average over batches)
        sub_graphs, labels = subgraph_batch

        out, node_mask = self.model(sub_graphs)
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)

        self.log('val_accuracy', self.accuracy(predictions, labels))

    def test_step(self, batch, _):
        # By default logs it per epoch (weighted average over batches)
        sub_graphs, labels = batch

        out, node_mask = self.model(sub_graphs)
        out = out[sub_graphs.ndata['classification_mask']]
        predictions = self.classifier(out)

        self.log('test_accuracy', self.accuracy(predictions, labels))
