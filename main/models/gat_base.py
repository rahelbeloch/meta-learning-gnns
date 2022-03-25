from torch import nn

from models.GraphTrainer import GraphTrainer
from models.gat_encoder_sparse_pushkar import GatNet
from models.train_utils import *


class GatBase(GraphTrainer):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, batch_size, label_names):
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

        new_targets = torch.zeros_like(logits)
        for i in range(new_targets.shape[0]):
            new_targets[i, targets[i]] = 1
        new_targets = new_targets.float()

        # print(f"\nLogits dtype {logits.dtype}")
        # print(f"Logits Shape {logits.shape}")
        # print(f"New Targets {new_targets}")
        # print(f"New Targets dtype {new_targets.dtype}")
        # print(f"New Targets Shape {new_targets.shape}")

        loss = self.loss_module(logits, new_targets)

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train_loss", loss)

        # TODO: add scheduler
        # logging in optimizer step does not work, therefore here
        # self.log('lr_rate', self.lr_scheduler.get_lr()[0])

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx):
        self.forward(batch, mode='val')

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
