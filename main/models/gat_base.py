import time

from torch.nn import BCEWithLogitsLoss

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer, get_or_none
from models.train_utils import *


# noinspection PyAbstractClass
class GatBase(GraphTrainer):
    """
    PyTorch Lightning module containing all model setup: Picking the correct encoder, initializing the classifier,
    and overwriting standard functions for training and optimization.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, other_params):
        """
        Args:
            model_params - Hyperparameters for the whole model, as dictionary.
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate,
            weight decay, etc.
        """
        super().__init__(validation_sets=['val_support', 'val_query'])

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace + saves config in wandb
        self.save_hyperparameters()

        self.model = GatNet(model_params)

        train_weight = get_or_none(other_params, 'train_loss_weight')
        print(f"Positive train weight: {train_weight}")
        self.train_loss = BCEWithLogitsLoss(pos_weight=train_weight)

        val_weight = get_or_none(other_params, 'val_loss_weight')
        print(f"Positive val weight: {val_weight}")
        self.validation_loss = BCEWithLogitsLoss(pos_weight=val_weight)

        # Deep copy of the model: one for train, one for val --> update validation model with weights from train model
        # validation fine-tuning should happen on a copy of the model NOT on the model which is trained
        # --> Training should not be affected by validation
        self.validation_model = GatNet(model_params)

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

    def forward(self, graphs, targets, mode=None):

        # batch all sub graphs and push the batch through the model
        x, edge_index, cl_mask = get_subgraph_batch(graphs)

        logits = self.model(x, edge_index, mode)[cl_mask].squeeze()

        # make probabilities out of logits via sigmoid --> especially for the metrics; makes it more interpretable
        if logits.ndim == 1:
            # binary classification
            predictions = (logits.sigmoid() > 0.5).float()
        else:
            # multiclass classification
            predictions = torch.softmax(logits, dim=1).argmax(dim=1)

        self.update_metrics(mode, predictions, targets)

        return logits

    def training_step(self, batch, batch_idx):

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_opt, _ = self.optimizers()

        # use support and query collapsed and train on whole
        sub_graphs, targets = batch

        logits = self.forward(sub_graphs, targets, mode='train')

        # logits should be batch size x 1, not batch size x 2!
        # x 2 --> multiple label classification (only if labels are exclusive, can be only one and not multiple)

        # Loss function is not weighted differently

        # 1. valdiation, not balanced --> loss weighting, see what and if it changes; (no loss weighting during training)
        # - keep using full validation set: 1 with balanced, 1 with unbalanced

        # BCE loss
        # BCE with logits loss
        # BCE with Sigmoid and 1 output of the model

        loss = self.train_loss(logits, targets.float())

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
        if self.trainer.current_epoch != 0 and (self.trainer.current_epoch + 1) % train_scheduler.step_size == 0:
            train_scheduler.step()

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
            mode = 'val_support'

            # Validation requires to finetune a model, hence we need to enable gradients
            torch.set_grad_enabled(True)

            # Copy model for finetune on the support part and optimizer
            self.validation_model.train()

            _, val_optimizer = self.optimizers()

            x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
            logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()

            support_predictions = (logits.sigmoid() > 0.5).float()
            self.update_metrics(mode, support_predictions, support_targets)

            loss = self.validation_loss(logits, support_targets.float())

            self.log_on_epoch(f"{mode}/loss", loss)

            # Calculate gradients and perform finetune update
            val_optimizer.zero_grad()
            self.manual_backward(loss)
            val_optimizer.step()

            # step every N epochs
            _, val_scheduler = self.lr_schedulers()
            if self.hparams.other_params['val_batches'] == (batch_idx + 1):
                val_scheduler.step()

            # SGD does not keep any state --> Create an SGD optimizer again every time
            # I enter the validation epoch; global or local should not be a difference
            # different for ADAM --> Keeps running weight parameter, that changes
            # per epoch, keeps momentum

            # Main constraint: Use same optimizer as in training, global ADAM validation

            torch.set_grad_enabled(False)

        elif dataloader_idx == 1:
            # Evaluate on meta test set
            mode = 'val_query'

            # testing on a query set that is oversampled should not be happening --> use original distribution
            # training is using a weighted loss --> validation set should use weighted loss as well
            # with extra validation model
            x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
            logits = self.validation_model(x, edge_index, mode)[cl_mask].squeeze()

            loss = self.validation_loss(logits, query_targets.float())

            query_predictions = (logits.sigmoid() > 0.5).float()
            self.update_metrics(mode, query_predictions, query_targets)

            # only log this once in the end of an epoch (averaged over steps)
            self.log_on_epoch(f"{mode}/loss", loss)

    def test_step(self, batch, batch_idx):
        """
        By default, logs it per epoch (weighted average over batches). Only validates on the query set of each batch to
        keep comparability with the meta learners.
        """

        _, sub_graphs, _, targets = batch
        self.forward(sub_graphs, targets, mode='test')

    def evaluation(self, n_classes, label_names, target_label):

        # Completely newly setting the output layer, erases all pretrained weights!
        self.model.reset_classifier_dimensions(n_classes)

        # reset the test metric with number of classes
        self.reset_test_metric(n_classes, label_names, target_label)


def evaluate(trainer, model, test_dataloader, target_label_name):
    """
    Tests a model on test and validation set.

    Args:
        trainer (pl.Trainer) - Lightning trainer to use.
        model (pl.LightningModule) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
    """

    print('\nTesting model on validation and test ..........\n')

    test_start = time.time()

    results = trainer.test(model, dataloaders=[test_dataloader], verbose=False)
    test_f1_target = results[0][f'test/f1_{target_label_name}']

    test_end = time.time()
    elapsed = test_end - test_start

    return test_f1_target, elapsed
