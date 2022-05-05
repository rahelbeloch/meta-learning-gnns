import time
from copy import deepcopy
from statistics import mean, stdev

import torch.nn.functional as func
from torch import optim
from torchmetrics import F1
from tqdm.auto import tqdm

from models.meta_learner import MetaLearner
from models.proto_net import ProtoNet
from models.train_utils import *
from samplers.batch_sampler import split_list


# noinspection PyAbstractClass
class ProtoMAML(MetaLearner):

    def adapt_few_shot(self, x, edge_index, cl_mask, support_targets, mode):

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
            loss, _ = run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, support_targets, mode,
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

            x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)
            query_x, query_edge_index, query_cl_mask = get_subgraph_batch(query_graphs)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias = self.adapt_few_shot(x_support, edge_index_support,
                                                                          cl_mask_support, support_targets, mode)

            # Determine loss of query set
            loss, query_predictions = run_model(local_model, output_weight, output_bias, query_x, query_edge_index,
                                                query_cl_mask, query_targets, mode, self.loss_module)

            self.update_metrics(mode, query_predictions, query_targets)

            # Calculate gradients for query set loss
            if mode == "train":
                self.manual_backward(loss)
                # loss.backward()

                for i, (p_global, p_local) in enumerate(zip(self.model.parameters(), local_model.parameters())):
                    if p_global.requires_grad is False:
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


def run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, targets, mode, loss_module=None):
    """
    Execute a model with given output layer weights and inputs.
    """

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

    return loss, (logits.sigmoid() > 0.5).float()


def test_protomaml(model, test_loader, num_classes=1):
    mode = 'test'
    model = model.to(DEVICE)
    model.eval()

    # TODO: use inner loop updates of 200 --> should be higher than in training

    test_start = time.time()

    # Iterate through the full dataset in two manners:
    # First, to select the k-shot batch. Second, to evaluate the model on all other batches.

    f1_fakes = []

    for support_batch_idx, batch in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = batch

        # graphs are automatically put to device in adapt few shot
        support_targets = support_targets.to(DEVICE)

        x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)

        # Finetune new model on support set
        local_model, output_weight, output_bias = model.adapt_few_shot(x_support, edge_index_support, cl_mask_support,
                                                                       support_targets, mode)

        f1_target = F1(num_classes=num_classes, average='none').to(DEVICE)

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

                x, edge_index, cl_mask = get_subgraph_batch(support_graphs)

                _, pred = run_model(local_model, output_weight, output_bias, x, edge_index, cl_mask, targets, mode)

                f1_target.update(pred, targets)

            f1_fakes.append(f1_target.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fakes), stdev(f1_fakes)), test_elapsed
