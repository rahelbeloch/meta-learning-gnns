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
        super().__init__(model_params["output_dim"], validation_sets=['val_support', 'val_query'])

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace + saves config in wandb
        self.save_hyperparameters()

        self.model = GatNet(model_params)

        # Deep copy of the model: one for train, one for val --> update validation model with weights from train model
        # validation fine-tuning should happen on a copy of the model NOT on the model which is trained
        # --> Training should not be affected by validation
        self.validation_model = GatNet(model_params)

        # flipping the weights
        # train_class_weight = 1 // self.class_weights['train'][0]
        train_class_weight = torch.flip(model_params["class_weights"]['train'], dims=[0]).to(DEVICE)
        self.loss_module = nn.BCEWithLogitsLoss(pos_weight=train_class_weight)

        self.val_class_weight = torch.flip(model_params["class_weights"]['val'], dims=[0]).to(DEVICE)

        self.automatic_optimization = False

    def configure_optimizers(self):

        opt_params = self.hparams.optimizer_hparams
        train_optimizer, train_scheduler = self.get_optimizer(opt_params['lr'], opt_params['lr_decay_epochs'])
        val_optimizer, val_scheduler = self.get_optimizer(opt_params['lr_val'], opt_params['lr_decay_epochs_val'],
                                                          self.validation_model)
        optimizers = [train_optimizer, val_optimizer]

        schedulers = []
        if train_scheduler is not None:
            schedulers.append(train_scheduler)
        if val_scheduler is not None:
            schedulers.append(val_scheduler)

        return optimizers, schedulers

    def get_optimizer(self, lr, step_size, model=None):
        opt_params = self.hparams.optimizer_hparams

        model = self.model if model is None else model

        if opt_params['optimizer'] == 'Adam':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=opt_params['weight_decay'])
        elif opt_params['optimizer'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=lr, momentum=opt_params['momentum'],
                            weight_decay=opt_params['weight_decay'])
        else:
            raise ValueError("No optimizer name provided!")
        scheduler = None
        if opt_params['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=step_size, gamma=opt_params['lr_decay_factor'])
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
            mode_dict[mode].update(predictions, targets)

        # logits are not yet put into a sigmoid layer, because the loss module does this combined
        return logits

    def training_step(self, batch, batch_idx):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_opt, _ = self.optimizers()

        # collapse support and query set and train on whole
        support_graphs, query_graphs, support_targets, query_targets = batch
        sub_graphs = support_graphs + query_graphs
        targets = torch.cat([support_targets, query_targets])

        logits = self.forward(sub_graphs, targets, mode='train')

        # logits should be batch size x 1, not batch size x 2!
        # x 2 --> multiple label classification (only if labels are exclusive, can be only one and not multiple)

        loss = self.loss_module(logits, func.one_hot(targets).float())
        # loss = self.loss_module(logits, targets.float())

        train_opt.zero_grad()
        self.manual_backward(loss)
        train_opt.step()

        # TODO: Accumulate gradients?
        # self.manual_backward(loss)
        # n = 10
        # accumulate gradients of N batches
        # if (batch_idx + 1) % n == 0:
        #     train_opt.step()
        #     train_opt.zero_grad()

        # step every N epochs
        train_scheduler, _ = self.lr_schedulers()
        # print(f"Train SD, step size: {train_scheduler.step_size}")
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % train_scheduler.step_size == 0:
            print(f"Trainer epoch: {self.trainer.current_epoch + 1}")
            print("Reducing Train LR")
            print(f"LR before: {train_scheduler.get_last_lr()}")
            train_scheduler.step()
            print(f"LR after: {train_scheduler.get_last_lr()}")

        # only log this once in the end of an epoch (averaged over steps)
        self.log_on_epoch(f"train/loss", loss)

        # back propagate every step, but only log every epoch
        # sum the loss over steps and average at the end of one epoch and then log
        return dict(loss=loss)

    def validation_step(self, batch, batch_idx, dataloader_idx):

        # update the weights of the validation model with weights from trained model
        self.validation_model.load_state_dict(self.model.state_dict())

        support_graphs, query_graphs, support_targets, query_targets = batch

        if dataloader_idx == 0:
            # pass
            # print(f"Validation finetune: {batch_idx+1}")

            mode = 'val_support'

            # Validation requires to finetune a model, hence we need to enable gradients
            torch.set_grad_enabled(True)

            # Copy model for finetune on the support part and optimizer
            self.validation_model.train()

            _, val_optimizer = self.optimizers()

            x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
            logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()

            # predictions = (logits.sigmoid() > 0.5).long()
            predictions = torch.sigmoid(logits).argmax(dim=-1)

            for mode_dict, _ in self.metrics.values():
                mode_dict[mode].update(predictions, support_targets)

            # loss = func.binary_cross_entropy_with_logits(logits, support_targets.float())
            loss = func.binary_cross_entropy_with_logits(logits, func.one_hot(support_targets).float(),
                                                         pos_weight=self.val_class_weight)
            # loss = self.loss_module(logits, func.one_hot(support_targets).float())

            self.log_on_epoch(f"{mode}/loss", loss)

            # Calculate gradients and perform finetune update
            val_optimizer.zero_grad()
            self.manual_backward(loss)
            val_optimizer.step()

            # step every N epochs
            _, val_scheduler = self.lr_schedulers()
            # print(f"Val SD, step size: {val_scheduler.step_size}")
            # if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % val_scheduler.step_size == 0:
            #     print(f"Trainer epoch: {self.trainer.current_epoch + 1}")
            #     print("Reducing Val LR")
            val_scheduler.step()

            # SGD does not keep any state --> Create an SGD optimizer again every time
            # I enter the validation epoch; global or local should not be a difference
            # different for ADAM --> Keeps running weight parameter, that changes
            # per epoch, keeps momentum

            # Main constraint: Use same optimizer as in training, global ADAM validation

            torch.set_grad_enabled(False)

        elif dataloader_idx == 1:
            # print(f"Validation query test: {batch_idx + 1}")

            # Evaluate on meta test set
            mode = 'val_query'

            # with extra validation model
            x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
            logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()

            # predictions = (logits.sigmoid() > 0.5).long()
            predictions = torch.sigmoid(logits).argmax(dim=-1)

            for mode_dict, _ in self.metrics.values():
                mode_dict[mode].update(predictions, query_targets)

            # with only 1 model
            # logits = self.forward(query_graphs, query_targets, mode)

            # loss = self.loss_module(logits, func.one_hot(query_targets).float())
            loss = func.binary_cross_entropy_with_logits(logits, func.one_hot(query_targets).float(),
                                                         pos_weight=self.val_class_weight)
            # loss = self.loss_module(logits, func.one_hot(query_targets).float())

            # only log this once in the end of an epoch (averaged over steps)
            self.log_on_epoch(f"{mode}/loss", loss)

    def test_step(self, batch, batch_idx1, batch_idx2):
        """
        By default, logs it per epoch (weighted average over batches). Only validates on the query set of each batch to
        keep comparability with the meta learners.
        """

        _, sub_graphs, _, targets = batch
        self.forward(sub_graphs, targets, mode='test')
