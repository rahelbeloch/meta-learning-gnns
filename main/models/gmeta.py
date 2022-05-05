import time
from copy import deepcopy
from statistics import mean, stdev

import torch
import torch.nn.functional as func
from torch import optim
from torch.optim import Adam
from torchmetrics import F1
from tqdm.auto import tqdm

from data_prep.data_utils import DEVICE
from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.proto_net import ProtoNet
from models.train_utils import get_subgraph_batch
from samplers.batch_sampler import split_list


class GMeta(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams):
        super(GMeta, self).__init__(validation_sets=['val'])
        self.save_hyperparameters()

        self.lr_inner = self.hparams.optimizer_hparams['lr_inner']

        self.n_inner_updates = model_params['n_inner_updates']
        self.n_inner_updates_test = model_params['n_inner_updates_test']

        pos_weight = 1 // model_params["class_weight"]['train'][1]
        print(f"Using positive weight: {pos_weight}")
        self.loss_module = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model = GatNet(model_params)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # self.forward(batch, mode="train")
        self.protomaml_forward(batch, mode="train")
        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        # self.finetune(batch, mode="val")
        self.protomaml_forward(batch, mode="val")
        torch.set_grad_enabled(False)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.hparams.optimizer_hparams['lr'])
        return [optimizer], []

    # def forward(self, batch, mode):
    #     losses_support = [0 for _ in range(self.n_inner_updates)]
    #     losses_query = [0 for _ in range(self.n_inner_updates + 1)]
    #     f1_fakes = [F1(num_classes=1, average='none', multiclass=False).to(DEVICE) for _ in
    #                 range(self.n_inner_updates + 1)]
    #
    #     # original state dict of the model for later update
    #     # original_state_dict = self.model.state_dict()
    #
    #     nan_query_loss = 0
    #
    #     for task_batch in batch:
    #
    #         graphs, targets = task_batch
    #         support_graphs, query_graphs = split_list(graphs)
    #         support_targets, query_targets = split_list(targets)
    #
    #         x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)
    #         x_query, edge_index_query, cl_mask_query = get_subgraph_batch(query_graphs)
    #
    #         ##############################
    #         # Initial loop for 0th step
    #         ##############################
    #
    #         # Step 3: put the support graphs through the model
    #         support_logits = self.model(x_support, edge_index_support, mode)[cl_mask_support]
    #
    #         # Step 4: compute support prototypes
    #         support_prototypes = ProtoNet.calculate_prototypes(support_logits, support_targets)
    #
    #         # Create a copy of the current model for the inner loop updates
    #         local_model = deepcopy(self.model)
    #         local_model.train()
    #         local_optim = optim.SGD(local_model.parameters(), lr=self.lr_inner)
    #         local_optim.zero_grad()
    #
    #         # Step 5: compute loss for k=0 / compute L_support
    #         logits = local_model(x_support, edge_index_support, mode)[cl_mask_support]
    #         loss_support, _ = proto_loss(logits, support_targets, support_prototypes)
    #         losses_support[0] += loss_support
    #
    #         # Query loss and f1 before first inner update
    #         with torch.no_grad():
    #             # Step 8: put the query graphs through the model
    #             query_logits = local_model(x_query, edge_index_query, mode)[cl_mask_query]
    #
    #             # Step 9: Use query logits and support prototypes to compute query loss for task i
    #             loss_query, predictions = proto_loss(query_logits, query_targets, support_prototypes)
    #
    #             if not True in torch.isnan(loss_query):
    #                 losses_query[0] += loss_query
    #                 f1_fakes[0].update(predictions, query_targets)
    #             else:
    #                 nan_query_loss += 1
    #
    #         # Step 6: Loss back propagates and update the GNN parameters
    #         # Calculate gradients and perform update
    #
    #         # self.manual_backward(loss_support)
    #         # loss_support.backward()
    #         # local_optim.step()
    #
    #         # grad = torch.autograd.grad(loss_support, self.model.parameters())
    #         # fast_weights = list(map(lambda p: p[1] - self.lr_inner * p[0], zip(grad, self.model.parameters())))
    #         # update_model_params(loss_support, self.model, self.lr_inner)
    #
    #         # loss and f1 after the first parameter update
    #         with torch.no_grad():
    #             # Step 8: put the query graphs through the model
    #             query_logits = local_model(x_query, edge_index_query, mode)[cl_mask_query]
    #
    #             # Step 9: Use query logits and support prototypes to compute query loss for task i
    #             loss_query, predictions = proto_loss(query_logits, query_targets, support_prototypes)
    #
    #             if not True in torch.isnan(loss_query):
    #                 losses_query[1] += loss_query
    #                 f1_fakes[1].update(predictions, query_targets)
    #             else:
    #                 nan_query_loss += 1
    #
    #         ##############################
    #         # Loop for 1st - end step
    #         ##############################
    #
    #         for n in range(1, self.n_inner_updates):
    #             # Step 3: put the support graphs through the model
    #             support_logits = local_model(x_support, edge_index_support, mode)[cl_mask_support]
    #
    #             # Step 4: compute support prototypes
    #             support_prototypes = ProtoNet.calculate_prototypes(support_logits, support_targets)
    #
    #             # Step 5: compute L_support
    #             loss_support, _ = proto_loss(support_logits, support_targets, support_prototypes)
    #             losses_support[n] += loss_support
    #
    #             # Step 6: Loss back propagates in order to update the GNN parameters
    #             # local_optim.zero_grad()
    #             # # self.manual_backward(loss_support)
    #             # loss_support.backward()
    #             # local_optim.step()
    #
    #             # 2. compute grad on theta_pi
    #             # grad = torch.autograd.grad(loss_support, fast_weights, retain_graph=True)
    #             # # 3. theta_pi = theta_pi - train_lr * grad
    #             # fast_weights = list(map(lambda p: p[1] - self.lr_inner * p[0], zip(grad, fast_weights)))
    #             # update_model_params(loss_support, self.model, self.lr_inner, retain_graph=True)
    #
    #             # Step 8: put the query graphs through the model
    #             query_logits = local_model(x_query, edge_index_query, mode)[cl_mask_query]
    #
    #             # Step 9: Use query logits and support prototypes to compute query loss for task i
    #             # loss_query will be overwritten, just keep the loss_query on last update step.
    #             loss_query, predictions = proto_loss(query_logits, query_targets, support_prototypes)
    #             losses_query[n + 1] += loss_query
    #             f1_fakes[n + 1].update(predictions, query_targets)
    #
    #     ##############################
    #     # End of all tasks
    #     ##############################
    #
    #     # sum over all losses on query set across all tasks
    #     task_num = len(batch)
    #     total_loss_query = losses_query[-1] / task_num
    #
    #     # self.model.load_state_dict(original_state_dict)
    #
    #     # optimize theta parameters
    #     outer_optimizer = self.optimizers()
    #     # noinspection PyUnresolvedReferences
    #     outer_optimizer.zero_grad()
    #
    #     self.manual_backward(total_loss_query, retain_graph=True)
    #     outer_optimizer.step()
    #
    #     f1_fake_total = torch.tensor([f1.compute() for f1 in f1_fakes]).sum() / task_num
    #     return f1_fake_total
    #
    # def finetune(self, batch, mode):
    #     f1_fakes = [F1(num_classes=1, average='none').to(DEVICE) for _ in range(self.n_inner_updates_test + 1)]
    #
    #     # fine-tuning on the copied model instead of self.model
    #     model = deepcopy(self.model)
    #
    #     graphs, targets = batch[0]
    #     support_graphs, query_graphs = split_list(graphs)
    #     support_targets, query_targets = split_list(targets)
    #
    #     x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)
    #     x_query, edge_index_query, cl_mask_query = get_subgraph_batch(query_graphs)
    #
    #     # put the support graphs through the model
    #     support_logits = model(x_support, edge_index_support, mode)[cl_mask_support]
    #
    #     # compute support prototypes
    #     support_prototypes = ProtoNet.calculate_prototypes(support_logits, support_targets)
    #
    #     # compute loss for k=0 / compute L_support
    #     loss_support, _ = proto_loss(support_logits, support_targets, support_prototypes)
    #
    #     # this is the loss and accuracy before first update
    #     with torch.no_grad():
    #         query_logits = model(x_query, edge_index_query, mode)[cl_mask_query]
    #         _, predictions = proto_loss(query_logits, query_targets, support_prototypes)
    #         f1_fakes[0].update(predictions, query_targets)
    #
    #     # update model parameters
    #     update_model_params(loss_support, model, self.lr_inner)
    #
    #     # this is the loss and accuracy after the first update
    #     with torch.no_grad():
    #         query_logits = model(x_query, edge_index_query, mode)[cl_mask_query]
    #         _, predictions = proto_loss(query_logits, query_targets, support_prototypes)
    #         f1_fakes[0].update(predictions, query_targets.float())
    #
    #     for k in range(1, self.n_inner_updates_test):
    #         support_logits = model(x_support, edge_index_support, mode)[cl_mask_support]
    #
    #         loss_support, _ = proto_loss(support_logits, support_targets, support_prototypes)
    #
    #         update_model_params(loss_support, model, self.lr_inner)
    #
    #         with torch.no_grad():
    #             query_logits = model(x_query, edge_index_query, mode)[cl_mask_query]
    #             _, predictions = proto_loss(query_logits, query_targets, support_prototypes)
    #             f1_fakes[0].update(predictions, query_targets.float())
    #
    #     del model
    #
    #     task_num = len(batch)
    #     f1_fake_total = torch.tensor([f1.compute() for f1 in f1_fakes]).sum() / task_num
    #     return f1_fake_total

    def protomaml_forward(self, batch, mode):
        query_losses = []
        f1 = F1(num_classes=1, average='none').to(DEVICE)

        # if mode == 'val':
        #     global_model = deepcopy(self.model)
        # elif mode == 'train':
        #     global_model = self.model
        # else:
        #     return

        self.model.zero_grad()
        nan_loss = []

        # Determine gradients for batch of tasks
        for idx, task_batch in enumerate(batch):
            graphs, targets = task_batch

            support_graphs, query_graphs = split_list(graphs)
            support_targets, query_targets = split_list(targets)

            # Perform inner loop adaptation
            local_model, support_prototypes = self.inner_loop_update(support_graphs, support_targets, mode)

            # after the last inner update, put query samples through the updated local model
            x_query, edge_index_query, cl_mask_query = get_subgraph_batch(query_graphs)

            query_logits = local_model(x_query, edge_index_query, mode)[cl_mask_query]
            last_query_loss, query_predictions = proto_loss(query_logits, query_targets, support_prototypes)

            # add the losses for each respective inner update step
            if True in torch.isnan(last_query_loss):
                nan_loss.append(idx)
            else:
                query_losses.append(last_query_loss)

            # Update F1 scores
            f1.update(query_predictions, query_targets)

            # TODO: calculate gradients for query set loss and add them to gradients of global model?

        ##############################
        # End of all tasks
        ##############################

        # take loss of last step and average over number of tasks
        total_loss_query = sum(query_losses) / len(query_losses)
        self.log_on_epoch(f"{mode}/loss", total_loss_query)

        if mode == "train":
            # optimize outer model parameters
            outer_optimizer = self.optimizers()
            # noinspection PyUnresolvedReferences
            outer_optimizer.zero_grad()

            # noinspection PyTypeChecker
            self.manual_backward(total_loss_query, retain_graph=True)
            outer_optimizer.step()

        f1_fake_total = f1.compute()

        return f1_fake_total

    def inner_loop_update(self, support_graphs, support_targets, mode):

        x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)

        # Determine prototype initialization
        support_logits = self.model(x_support, edge_index_support, mode)[cl_mask_support]
        support_prototypes = ProtoNet.calculate_prototypes(support_logits, support_targets)

        # Copy model for inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.lr_inner)
        local_optim.zero_grad()

        updates = self.n_inner_updates if mode != 'test' else self.n_inner_updates_test

        # Optimize inner loop model on support set
        for n in range(updates):
            # run the local model
            support_logits = local_model(x_support, edge_index_support, mode)[cl_mask_support]

            # compute the prototypical loss
            support_loss, _ = proto_loss(support_logits, support_targets, support_prototypes)

            # Calculate gradients and perform inner loop update
            # TODO: why do we have to retain the graph here?
            support_loss.backward(retain_graph=True)
            local_optim.step()
            local_optim.zero_grad()

        return local_model, support_prototypes


def proto_loss(logits, targets, prototypes):
    # expected shapes:
    #   query_logits: batch size x 64
    #   prototypes: 1 x 64
    #   dist: batch size x 1

    dist = torch.pow(prototypes[None, :] - logits[:, None], 2).sum(dim=2).squeeze()

    # original authors use cross-entropy loss as follows: L(p, y) = ∑ j yj log pj
    # where y indicates a true label’s one-hot encoding.
    # log_p_y = func.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    # G-Meta's version
    # log_p_y = func.log_softmax(-dist, dim=1)
    # loss_query = -log_p_y.mean()

    # Phillips version
    # predictions = func.log_softmax(-dist, dim=1)
    # loss_query = func.cross_entropy(predictions, query_targets)

    loss = func.binary_cross_entropy_with_logits(-dist, targets.float())
    return loss, (-dist.sigmoid() > 0.5).float()


def update_model_params(loss_support, model, lr_inner, retain_graph=None):
    # Compute gradient on theta_pi
    if retain_graph is None:
        grad = torch.autograd.grad(loss_support, model.parameters(), allow_unused=True)
    else:
        grad = torch.autograd.grad(loss_support, model.parameters(), retain_graph=retain_graph, allow_unused=True)

    # update model parameters
    # theta_pi = theta_pi - train_lr * grad

    # fast_weights = []
    # for i, (p_grad, p_model) in enumerate(zip(grad, self.model.parameters())):
    #     new_weight = p_model - self.n_inner_updates * p_grad
    #     fast_weights.append(new_weight)

    for i, (p_grad, p_model) in enumerate(zip(grad, model.parameters())):
        if p_model.requires_grad is False:
            continue
        p_model.data -= lr_inner * p_grad


def test_gmeta(model, test_loader, num_classes=1):
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
        local_model, support_prototypes = model.inner_loop_update(support_graphs, support_targets, mode)

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

                x, edge_index, cl_mask = get_subgraph_batch(graphs)

                logits = local_model(x, edge_index, mode)[cl_mask]
                _, predictions = proto_loss(logits, targets, support_prototypes)

                f1_target.update(predictions, targets)

            f1_fakes.append(f1_target.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fakes), stdev(f1_fakes)), test_elapsed
