import torch.nn.functional
import torch.nn.functional as func
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR

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

        # Deep copy of the model: one for train, one for val --> update validation model with weights from train model
        # validation fine-tuning should happen on a copy of the model NOT on the model which is trained
        # --> Training should not be affected by validation
        # self.validation_model = GatNet(model_params)

        # flipping the weights
        # pos_weight = 1 // model_params["class_weight"][0]
        pos_weight = torch.flip(model_params["class_weight"], dims=[0])
        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # self.automatic_optimization = False

    def configure_optimizers(self):

        train_optimizer, train_scheduler = self.get_optimizer()
        # val_optimizer, val_scheduler = self.get_optimizer(self.validation_model)
        # optimizers = [train_optimizer, val_optimizer]
        optimizers = [train_optimizer]

        schedulers = []
        if train_scheduler is not None:
            schedulers.append(train_scheduler)
        # if val_scheduler is not None:
        #     schedulers.append(val_scheduler)

        return optimizers, schedulers

    def get_optimizer(self, model=None):
        opt_params = self.hparams.optimizer_hparams

        model = self.model if model is None else model

        if opt_params['optimizer'] == 'Adam':
            optimizer = AdamW(model.parameters(), lr=opt_params['lr'],
                              weight_decay=opt_params['weight_decay'])
        elif opt_params['optimizer'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=opt_params['lr'], momentum=opt_params['momentum'],
                            weight_decay=opt_params['weight_decay'])
        else:
            raise ValueError("No optimizer name provided!")
        scheduler = None
        if opt_params['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=opt_params['lr_decay_epochs'],
                               gamma=opt_params['lr_decay_factor'])
        elif opt_params['scheduler'] == 'multi_step':
            scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 30, 40, 55],
                                    gamma=opt_params['lr_decay_factor'])
        return optimizer, scheduler

    def forward(self, sub_graphs, targets, mode=None):

        # make a batch out of all sub graphs and push the batch through the model
        x, edge_index, cl_mask = get_subgraph_batch(sub_graphs)

        logits = self.model(x, edge_index, mode)[cl_mask].squeeze()

        # make probabilities out of logits via sigmoid --> especially for the metrics; makes it more interpretable
        # predictions = (logits.sigmoid() > 0.5).long()
        # predictions = logits.sigmoid().argmax(dim=-1)
        predictions = torch.sigmoid(logits).argmax(dim=-1)

        for mode_dict, _ in self.metrics.values():
            # shapes should be: pred (batch_size), targets: (batch_size)
            mode_dict[mode].update(predictions, targets)

        # logits are not yet put into a sigmoid layer, because the loss module does this combined
        return logits

    def training_step(self, batch, batch_idx):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # train_opt, _ = self.optimizers()
        # train_opt.zero_grad()

        # collapse support and query set and train on whole
        support_graphs, query_graphs, support_targets, query_targets = batch
        sub_graphs = support_graphs + query_graphs
        targets = torch.cat([support_targets, query_targets])

        logits = self.forward(sub_graphs, targets, mode='train')

        # logits should be batch size x 1, not batch size x 2!
        # x 2 --> multiple label classification (only if labels are exclusive, can be only one and not multiple)

        loss = self.loss_module(logits, func.one_hot(targets).float())
        # loss = self.loss_module(logits, targets.float())

        # self.manual_backward(loss)
        # train_opt.step()

        # train_scheduler, _ = self.lr_schedulers()
        # train_scheduler.step()

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train/loss", loss)

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):

        support_graphs, query_graphs, support_targets, query_targets = batch

        if dataloader_idx == 0:
            pass

            # mode = 'val_support'
            #
            # # update the weights of the validation model with weights from trained model
            # self.validation_model.load_state_dict(self.model.state_dict())
            #
            # # Validation requires to finetune a model, hence we need to enable gradients
            # torch.set_grad_enabled(True)
            #
            # # Copy model for finetune on the support part and optimizer
            # self.validation_model.train()
            #
            # _, val_optimizer = self.optimizers()
            # val_optimizer.zero_grad()
            #
            # x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
            # logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()
            #
            # # TODO: log validation finetune metrics
            # # predictions = (logits.sigmoid() > 0.5).long()
            # predictions = torch.sigmoid(logits).argmax(dim=-1)
            #
            # for mode_dict, _ in self.metrics.values():
            #     # shapes should be: pred (batch_size), targets: (batch_size)
            #     mode_dict[mode].update(predictions, support_targets)
            #
            # loss = func.binary_cross_entropy_with_logits(logits, support_targets.float())
            #
            # self.log_on_epoch(f"val/support_loss", loss)
            #
            # # Calculate gradients and perform finetune update
            # self.manual_backward(loss)
            # # loss.backward()
            # val_optimizer.step()
            #
            # _, val_scheduler = self.lr_schedulers()
            # val_scheduler.step()
            #
            # # SGD does not keep any state --> Create an SGD optimizer again every time
            # # I enter the validation epoch; global or local should not be a difference
            # # different for ADAM --> Keeps running weight parameter, that changes
            # # per epoch, keeps momentum
            #
            # # Main constraint: Use same optimizer as in training, global ADAM validation
            #
            # torch.set_grad_enabled(False)

        elif dataloader_idx == 1:
            # Evaluate on meta test set
            mode = 'val_query'

            # with extra validation model
            # x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
            # logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()

            # predictions = (logits.sigmoid() > 0.5).long()
            #
            # for mode_dict, _ in self.metrics.values():
            #     # shapes should be: pred (batch_size), targets: (batch_size)
            #     mode_dict[mode].update(predictions, query_targets)

            # with only 1 model
            logits = self.forward(query_graphs, query_targets, mode)

            # loss = self.loss_module(logits, query_targets.float())
            loss = self.loss_module(logits, func.one_hot(query_targets).float())

            # only log this once in the end of an epoch (averaged over steps)
            self.log_on_epoch(f"val_query/loss", loss)

    def test_step(self, batch, batch_idx1, batch_idx2):
        # By default, logs it per epoch (weighted average over batches)

        # only validate on the query set to keep comparability with metamodels
        support_graphs, query_graphs, support_targets, query_targets = batch
        sub_graphs = query_graphs
        targets = query_targets

        self.forward(sub_graphs, targets, mode='test')

# def load_pretrained_encoder(checkpoint_path):
#     """
#     Load a pretrained encoder state dict and remove 'model.' from the keys in the state dict, so that solely
#     the encoder can be loaded.
#
#     Args:
#         checkpoint_path (str) - Path to a checkpoint for the DocumentClassifier.
#     Returns:
#         encoder_state_dict (dict) - Containing all keys for weights of encoder.
#     """
#     checkpoint = torch.load(checkpoint_path)
#     encoder_state_dict = {}
#     for layer, param in checkpoint["state_dict"].items():
#         if layer.startswith("model"):
#             new_layer = layer[layer.index(".") + 1:]
#             encoder_state_dict[new_layer] = param
#
#     return encoder_state_dict


# # noinspection PyProtectedMember
# class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
#     """
#     Learning rate scheduler, combining warm-up with a cosine-shaped learning rate decay.
#     """
#
#     def __init__(self, optimizer, warmup, max_iters):
#         self.warmup = warmup
#         self.max_num_iters = max_iters
#         super().__init__(optimizer)
#
#     def get_lr(self):
#         lr_factor = self.get_lr_factor()
#         return [base_lr * lr_factor for base_lr in self.base_lrs]
#
#     def get_lr_factor(self):
#         current_step = self.last_epoch
#         lr_factor = 0.5 * (1 + np.cos(np.pi * current_step / self.max_num_iters))
#         if current_step < self.warmup:
#             lr_factor *= current_step * 1.0 / self.warmup
#         return lr_factor
