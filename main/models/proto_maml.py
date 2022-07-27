import time
from collections import defaultdict
from copy import deepcopy
from statistics import mean, stdev

import torch.nn
import torch.nn.functional as func
from torch import optim
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torchmetrics import F1
from tqdm import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer, get_or_none
from models.proto_net import ProtoNet
from models.train_utils import *
from samplers.episode_sampler import split_list


# noinspection PyAbstractClass
class ProtoMAML(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, other_params):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__()
        self.save_hyperparameters()

        self.n_inner_updates = model_params['n_inner_updates']
        self.n_inner_updates_test = model_params['n_inner_updates_test']

        self.lr_inner = self.hparams.optimizer_hparams['lr_inner']
        self.k_shot_support = other_params['k_shot_support']

        train_weight = get_or_none(other_params, 'train_loss_weight')
        print(f"Positive train weight: {train_weight}")
        self.train_loss_module = BCEWithLogitsLoss(pos_weight=train_weight)

        val_weight = get_or_none(other_params, 'val_loss_weight')
        print(f"Positive val weight: {val_weight}")
        self.val_loss_module = BCEWithLogitsLoss(pos_weight=val_weight)

        self.model = GatNet(model_params)

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer_hparams

        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        optimizer, scheduler = self.get_optimizer(opt_params['lr'], opt_params['lr_decay_epochs'],
                                                  milestones=[140, 180])

        return [optimizer], [scheduler]

    def adapt_few_shot(self, x, edge_index, cl_mask, support_targets, mode, loss_module):

        # Determine prototype initialization
        support_feats = self.model(x, edge_index, mode).squeeze()[cl_mask]

        prototypes = ProtoNet.calculate_prototypes(support_feats, support_targets)

        # Copy model for inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.lr_inner)
        local_optim.zero_grad()

        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=1) ** 2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        updates = self.n_inner_updates if mode != 'test' else self.n_inner_updates_test

        losses = []

        # Optimize inner loop model on support set
        for _ in range(updates):
            # Determine loss on the support set
            loss, support_predictions = run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask,
                                                  support_targets, mode, torch.tensor(self.target_classes).squeeze(),
                                                  loss_module)

            self.update_metrics(mode, support_predictions, support_targets, set_name='support')

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()
            losses.append(loss.detach().item())

            # Update output layer via SGD
            output_weight.data -= self.hparams.optimizer_hparams['lr_output'] * output_weight.grad
            output_bias.data -= self.hparams.optimizer_hparams['lr_output'] * output_bias.grad

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias, torch.tensor(losses).mean().item()

    def outer_loop(self, batch, loss_module, mode):
        support_losses, query_losses = [], []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for graphs, targets in batch:

            # This should be done in the graph sampler: *2 because for both classes!!
            support_graphs, query_graphs = split_list(graphs, self.k_shot_support * 2)
            support_targets, query_targets = split_list(targets, self.k_shot_support * 2)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias, support_loss = self.adapt_few_shot(
                *get_subgraph_batch(support_graphs), support_targets, mode, loss_module)
            support_losses.append(support_loss)

            # Determine loss of query set
            query_loss, query_predictions = run_model(local_model, output_weight, output_bias,
                                                      *get_subgraph_batch(query_graphs), query_targets, mode,
                                                      torch.tensor(self.target_classes).squeeze(), loss_module)

            self.update_metrics(mode, query_predictions, query_targets, set_name='query')

            # Calculate gradients for query set loss
            if mode == "train":
                # initializes the grads in the outer model, as we used its support features for prototype computation
                query_loss.backward()

                for i, (p_global, p_local) in enumerate(zip(self.model.parameters(), local_model.parameters())):
                    if p_global.requires_grad is False:
                        continue
                    # First-order approx. -> add gradients of fine-tuned and base model
                    p_global.grad += p_local.grad

            query_losses.append(query_loss.detach().item())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            # noinspection PyUnresolvedReferences
            opt.zero_grad()

            train_scheduler = self.lr_schedulers()
            if self.trainer.current_epoch != 0 and (self.trainer.current_epoch + 1) % train_scheduler.step_size == 0:
                train_scheduler.step()

        self.log_on_epoch(f"{mode}/query_loss", torch.tensor(query_losses).mean())
        self.log_on_epoch(f"{mode}/support_loss", torch.tensor(support_losses).mean())

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, self.train_loss_module, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, self.val_loss_module, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, targets, mode, target_classes,
              loss_module=None):
    """
    Execute a model with given output layer weights and inputs.
    """

    logits = local_model(x, edge_index, mode)[cl_mask]

    # Expected
    # logits shape:         batch size x 64
    # output_weight shape:  2 x 64
    # output_bias shape:    2

    # out:                  80 x 2
    logits = func.linear(logits, output_weight, output_bias)

    logits_target = logits[:, target_classes]
    if logits_target.ndim == 1:
        logits_target = logits_target.unsqueeze(dim=1)

    if type(loss_module) != CrossEntropyLoss:
        targets = (targets.view(-1, 1) if not len(targets.shape) == 2 else targets).float()

    loss = loss_module(logits_target, targets) if loss_module is not None else None

    return loss, get_predictions(logits_target)


def test_protomaml(model, test_loader, label_names, loss_module, num_classes=1):
    mode = 'test'
    model = model.to(DEVICE)
    model.eval()

    test_start = time.time()

    # Iterate through the full dataset in two manners:
    # First, to select the k-shot batch. Second, to evaluate the model on all other batches.
    f1_fakes, f1_macros, f1_weights = defaultdict(list), [], []

    # test_data_outer = list(test_loader)[:5]
    # test_data_inner = test_data_outer

    # test_data_outer = tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing")
    # test_data_inner = test_loader

    for support_batch_idx, batch in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = batch

        # graphs are automatically put to device in adapt few shot
        support_targets = support_targets.to(DEVICE)

        # Finetune new model on support set
        local_model, output_weight, output_bias, _ = model.adapt_few_shot(*get_subgraph_batch(support_graphs),
                                                                          support_targets, mode, loss_module)

        f1_target = F1(num_classes=num_classes, average='none').to(DEVICE)
        f1_macro = F1(num_classes=num_classes, average='macro').to(DEVICE)
        f1_weighted = F1(num_classes=num_classes, average='weighted').to(DEVICE)

        with torch.no_grad():  # No gradients for query set needed
            local_model.eval()

            # Evaluate all examples in test dataset
            for query_batch_idx, test_batch in enumerate(test_loader):

                if support_batch_idx == query_batch_idx:
                    # Exclude support set elements
                    continue

                support_graphs, query_graphs, support_targets, query_targets = batch
                graphs = support_graphs + query_graphs
                targets = torch.cat([support_targets, query_targets]).to(DEVICE)

                _, pred = run_model(local_model, output_weight, output_bias, *get_subgraph_batch(graphs), targets, mode,
                                    torch.tensor(model.target_classes))

                f1_target.update(pred, targets)
                f1_macro.update(pred, targets)
                f1_weighted.update(pred, targets)

            # will be one-dim if binary, multi-dim if many target classes
            f1_fake = f1_target.compute()
            for i, label in enumerate(label_names):
                f1_fakes[label].append(f1_fake[i].item())
            f1_macros.append(f1_macro.compute().item())
            f1_weights.append(f1_weighted.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    for label in label_names:
        f1_fakes[label] = mean(f1_fakes[label])

    return (f1_fakes, 0.0), (mean(f1_macros), stdev(f1_macros)), (mean(f1_weights), stdev(f1_weights)), test_elapsed
