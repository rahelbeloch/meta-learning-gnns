import time
from copy import deepcopy
from statistics import mean, stdev

import torch.nn.functional as func
from torch import optim
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchmetrics import F1
from tqdm.auto import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.proto_net import ProtoNet
from models.train_utils import *
from samplers.batch_sampler import split_list


class ProtoMAML(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, label_names, batch_size):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__(validation_sets=['val'])
        self.save_hyperparameters()

        self.n_inner_updates = model_params['n_inner_updates']
        self.lr_output = self.hparams.optimizer_hparams['lr_output']

        # no class weighting for the meta learner
        self.loss_module = torch.nn.BCEWithLogitsLoss()

        self.model = GatNet(model_params)

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer_hparams

        lr = opt_params['lr']
        weight_decay = opt_params['weight_decay']

        if opt_params['optimizer'] == 'Adam':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_params['optimizer'] == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=lr, momentum=opt_params['momentum'], weight_decay=weight_decay)
        else:
            raise ValueError("No optimizer name provided!")

        scheduler = None
        if opt_params['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=opt_params['lr_decay_epochs'], gamma=opt_params['lr_decay_factor'])
        elif opt_params['scheduler'] == 'multi_step':
            scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 30, 40, 55],
                                    gamma=opt_params['lr_decay_factor'])
            # scheduler = MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)

        return [optimizer], [scheduler]

    def adapt_few_shot(self, support_graphs, support_targets, mode):

        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)

        # Determine prototype initialization
        support_feats = self.model(x, edge_index, mode)[cl_mask]

        prototypes = ProtoNet.calculate_prototypes(support_feats, support_targets)

        # Copy model for inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()  # local_model.classifier.out_features
        local_optim = optim.SGD(local_model.parameters(), lr=self.hparams.optimizer_hparams['lr_inner'])
        local_optim.zero_grad()

        # Create output layer weights with prototype-based initialization
        init_weight = 2 * prototypes
        init_bias = -torch.norm(prototypes, dim=0) ** 2
        output_weight = init_weight.detach().requires_grad_()
        output_bias = init_bias.detach().requires_grad_()

        # Optimize inner loop model on support set
        for _ in range(self.n_inner_updates):
            # Determine loss on the support set
            loss, logits = run_model(local_model, output_weight, output_bias, support_graphs, support_targets, mode,
                                     self.loss_module)

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()

            # Update output layer via SGD
            output_weight.data -= self.lr_output * output_weight.grad
            output_bias.data -= self.lr_output * output_bias.grad

            # Reset gradients
            local_optim.zero_grad()
            output_weight.grad.fill_(0)
            output_bias.grad.fill_(0)

        # Re-attach computation graph of prototypes
        output_weight = (output_weight - init_weight).detach() + init_weight
        output_bias = (output_bias - init_bias).detach() + init_bias

        return local_model, output_weight, output_bias

    def outer_loop(self, batch, mode):
        losses = []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for task_batch in batch:

            graphs, targets = task_batch
            support_graphs, query_graphs = split_list(graphs)
            support_targets, query_targets = split_list(targets)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias = self.adapt_few_shot(support_graphs, support_targets, mode)

            # Determine loss of query set
            loss, logits = run_model(local_model, output_weight, output_bias, query_graphs, query_targets, mode,
                                     self.loss_module)

            query_predictions = (logits.sigmoid() > 0.5).float()
            for mode_dict, _ in self.metrics.values():
                mode_dict[mode].update(query_predictions, query_targets)

            # Calculate gradients for query set loss
            if mode == "train":
                self.manual_backward(loss)
                # loss.backward()

                for i, (p_global, p_local) in enumerate(zip(self.model.parameters(), local_model.parameters())):
                    if p_global.requires_grad is False:
                        # Params with names attention_1.linear.weight, attention_2.linear.weight, ... do not require grad
                        # print(f"Param at position {i} does not require grad.")
                        continue
                    # First-order approx. -> add gradients of fine-tuned and base model
                    p_global.grad += p_local.grad

            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            # noinspection PyUnresolvedReferences
            opt.zero_grad()

        self.log_on_epoch(f"{mode}/loss", sum(losses) / len(losses))

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, output_weight, output_bias, graphs, targets, mode, loss_module=None):
    """
    Execute a model with given output layer weights and inputs.
    """

    x, edge_index, cl_mask = get_subgraph_batch(graphs)
    logits = local_model(x, edge_index, mode)[cl_mask]

    # gat base multi class
    # output_weight: 2 x 2, output_bias: 1 x 2, logits: 40 x 2
    # with proto dims 64: logits: 40 x 64,
    # gat base binary class
    # output_weight: 1 x 1, output_bias: 1 x 1, logits: 40 x 1

    # Expected
    # logits shape 20 x 64
    # output_weight shape 1 x 64
    # output_bias shape 1

    # Actual without transpose
    # logits shape 20 x 64
    # output_weight shape 64 x 1
    # output_bias shape 1

    logits = func.linear(logits, output_weight.T, output_bias)

    targets = targets.view(-1, 1) if not len(targets.shape) == 2 else targets
    loss = loss_module(logits, targets.float()) if loss_module is not None else None

    return loss, logits


def test_protomaml(model, test_loader, num_classes=1):
    mode = 'test'
    model = model.to(DEVICE)
    model.eval()

    # TODO: use inner loop updates of 200 --> should be higher than in training

    test_start = time.time()

    # Iterate through the full dataset in two manners:
    # First, to select the k-shot batch. Second, to evaluate the model on all other batches.

    f1_macros, f1_fakes, f1_reals = [], [], []

    for support_batch_idx, batch in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = batch

        # graphs are automatically put to device in adapt few shot
        support_targets = support_targets.to(DEVICE)

        # Finetune new model on support set
        local_model, output_weight, output_bias = model.adapt_few_shot(support_graphs, support_targets, mode)

        f1_target = F1(num_classes=num_classes, average='none').to(DEVICE)
        f1_macro = F1(num_classes=num_classes, average='macro').to(DEVICE)

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

                _, logits = run_model(local_model, output_weight, output_bias, graphs, targets, mode)

                pred = (logits.sigmoid() > 0.5).float()
                f1_target.update(pred, targets)
                f1_macro.update(pred, targets)

            f1_macros.append(f1_macro.compute().item())
            f1_fakes.append(f1_target.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fakes), stdev(f1_fakes)), (mean(f1_macros), stdev(f1_macros)), test_elapsed
