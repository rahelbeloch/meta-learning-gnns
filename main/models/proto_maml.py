from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn.functional as func
from torch import optim

from models.gat_base import get_classify_node_features, get_subgraph_batch
from models.gat_encoder_sparse_pushkar import SparseGATLayer
from models.proto_net import ProtoNet
from models.train_utils import evaluation_metrics
from samplers.batch_sampler import split_list


class ProtoMAML(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, input_dim, hidden_dim, feat_reduce_dim, opt_hparams, n_inner_updates, batch_size):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = SparseGATLayer(input_dim, hidden_dim, feat_reduce_dim)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.opt_hparams['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def adapt_few_shot(self, support_graphs, support_targets):

        x, edge_index = get_subgraph_batch(support_graphs)

        # Determine prototype initialization
        support_feats = self.model(x, edge_index).squeeze()
        support_feats = get_classify_node_features(support_graphs, support_feats)

        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)
        support_labels = self.get_labels(classes, support_targets)

        # Copy model for inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.opt_hparams['lr_inner'])
        local_optim.zero_grad()

        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1) ** 2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for _ in range(self.hparams.n_inner_updates):
            # Determine loss on the support set
            loss, _, _, _, _, _ = run_model(local_model, output_weight, output_bias, support_graphs, support_labels)

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()

            # Update output layer via SGD
            output_weight.data -= self.hparams.opt_hparams['lr_output'] * output_weight.grad
            output_bias.data -= self.hparams.opt_hparams['lr_output'] * output_bias.grad

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, classes

    @staticmethod
    def get_labels(classes, targets):
        # noinspection PyUnresolvedReferences
        return (classes[None, :] == targets[:, None]).long().argmax(dim=-1)

    def outer_loop(self, batch, mode="train"):
        accuracies = []
        f1_macros = []
        f1_micros = []
        losses = []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for task_batch in batch:

            graphs, targets = task_batch
            support_graphs, query_graphs = split_list(graphs)
            support_targets, query_targets = split_list(targets)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_graphs, support_targets)

            # Determine loss of query set
            query_labels = self.get_labels(classes, query_targets)
            loss, predictions, acc, f1, f1_macro, f1_micro = run_model(local_model, output_weight, output_bias,
                                                                       query_graphs, query_labels)

            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                for p_global, p_local in zip(self.model.parameters(), local_model.parameters()):
                    p_global.grad += p_local.grad  # First-order approx. -> add gradients of fine-tuned and base model

            accuracies.append(acc.mean().detach())
            f1_macros.append(f1_macro)
            f1_micros.append(f1_micro)
            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            # noinspection PyUnresolvedReferences
            opt.zero_grad()

        self.log(f"{mode}_loss", sum(losses) / len(losses))
        self.log(f"{mode}_acc", sum(accuracies) / len(accuracies))
        self.log_on_epoch(f"{mode}_f1_macro", sum(f1_macros) / len(f1_macros))
        self.log_on_epoch(f"{mode}_f1_micro", sum(f1_micros) / len(f1_micros))

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=self.hparams['batch_size'])

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to fine tune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, output_weight, output_bias, graphs, targets):
    """
    Execute a model with given output layer weights and inputs.
    """

    x, edge_index = get_subgraph_batch(graphs)
    feats = local_model(x, edge_index).squeeze()
    feats = get_classify_node_features(graphs, feats)

    predictions = func.linear(feats, output_weight, output_bias)

    loss = func.cross_entropy(predictions, targets)

    f1, f1_macro, f1_micro, acc = evaluation_metrics(predictions, targets)

    return loss, predictions, acc, f1, f1_macro, f1_micro
