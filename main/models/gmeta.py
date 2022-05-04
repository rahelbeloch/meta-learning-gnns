from copy import deepcopy

import numpy as np
import torch
import torch.functional as func
from torch.optim import Adam

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer
from models.proto_net import ProtoNet
from models.train_utils import get_subgraph_batch
from samplers.batch_sampler import split_list


class GMeta(GraphTrainer):
    def __init__(self, model_params):
        super(GMeta, self).__init__(validation_sets=['val'])

        self.inner_lr = self.hparams.optimizer_hparams['lr_inner']
        self.outer_lr = self.hparams.optimizer_hparams['lr_output']

        self.k_shot = model_params['k_shot']

        self.n_inner_updates = model_params['n_inner_updates']
        # self.n_inner_updates_test = model_params['n_inner_updates_test']

        self.model = GatNet(model_params)

        self.meta_optim = Adam(self.model.parameters(), lr=self.meta_lr)

    # def forward(self, x_support, y_support, x_query, y_query, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
    def forward(self, batch, mode):
        losses_support = [0 for _ in range(self.update_step)]
        losses_query = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]

        for task_batch in batch:

            graphs, targets = task_batch
            support_graphs, query_graphs = split_list(graphs)
            support_targets, query_targets = split_list(targets)

            x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)
            x_query, edge_index_query, cl_mask_query = get_subgraph_batch(query_graphs)

            support_features = None
            query_features = None

            # 1. run the i-th task and compute loss for k=0
            support_logits = self.model(x_support, edge_index_support, mode)[cl_mask_support]

            # loss, _, prototypes = proto_loss_spt(logits, y_support[i], self.k_spt)
            prototypes = ProtoNet.calculate_prototypes(support_logits, support_targets)
            logits, targets = ProtoNet.classify_features(prototypes, support_logits, support_targets)
            loss = self.loss_module(logits, targets.float())

            losses_support[0] += loss

            grad = torch.autograd.grad(loss, self.model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.n_inner_updates * p[0], zip(grad, self.model.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # # [setsz, nway]
                # logits_q, _ = self.net(x_query[i].to(DEVICE), c_qry[i].to(DEVICE), feat_qry, self.net.parameters())
                query_logits = self.model(x_query, edge_index_query, mode, query_features, self.model.parameters())[
                    cl_mask_query]
                loss_q, acc_q = proto_loss_qry(query_logits, query_targets, prototypes)
                losses_query[0] += loss_q
                corrects[0] = corrects[0] + acc_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                query_logits = self.model(x_query, edge_index_query, mode, query_features, fast_weights)[cl_mask_query]
                loss_q, acc_q = proto_loss_qry(query_logits, query_targets, prototypes)
                losses_query[1] += loss_q
                corrects[1] = corrects[1] + acc_q

            for k in range(1, self.n_inner_updates):
                # 1. run the i-th task and compute loss for k=1~K-1

                support_logits = self.model(x_support, edge_index_support, mode, support_features, fast_weights)[
                    cl_mask_support]
                # logits, _ = self.net(x_support[i].to(DEVICE), c_spt[i].to(DEVICE), feat_spt, fast_weights)
                loss, _, prototypes = proto_loss_spt(support_logits, support_targets, self.k_shot)
                losses_support[k] += loss

                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)

                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                query_logits = self.model(x_query, edge_index_query, mode, query_features, fast_weights)[cl_mask_query]
                # logits_q, _ = self.net(x_query[i].to(DEVICE), c_qry[i].to(DEVICE), feat_qry, fast_weights)

                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q, acc_q = proto_loss_qry(query_logits, query_targets, prototypes)
                losses_query[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        task_num = len(batch)
        total_loss_query = losses_query[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        total_loss_query.backward()
        self.meta_optim.step()

        accuracy = np.array(corrects) / task_num
        return accuracy

    # def finetune(self, x_support, y_support, x_query, y_query, c_spt, c_qry, n_spt, n_qry, g_spt, g_qry, feat):
    def finetune(self, batch, mode):
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # fine tunning on the copied model instead of self.model
        model = deepcopy(self.model)

        # x_support = x_support[0]
        # y_support = y_support[0]
        # x_query = x_query[0]
        # y_query = y_query[0]
        # c_spt = c_spt[0]
        # c_qry = c_qry[0]
        # n_spt = n_spt[0]
        # n_qry = n_qry[0]
        # g_spt = g_spt[0]
        # g_qry = g_qry[0]
        #
        # feat_spt = torch.Tensor(np.vstack(([feat[g_spt[j]][np.array(x)] for j, x in enumerate(n_spt)]))).to(DEVICE)
        # feat_qry = torch.Tensor(np.vstack(([feat[g_qry[j]][np.array(x)] for j, x in enumerate(n_qry)]))).to(DEVICE)
        support_features = None
        query_features = None

        graphs, targets = batch[0]
        support_graphs, query_graphs = split_list(graphs)
        support_targets, query_targets = split_list(targets)

        x_support, edge_index_support, cl_mask_support = get_subgraph_batch(support_graphs)
        x_query, edge_index_query, cl_mask_query = get_subgraph_batch(query_graphs)

        # 1. run the i-th task and compute loss for k=0
        # logits, _ = model(x_support.to(DEVICE), c_spt.to(DEVICE), feat_spt)
        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
        support_logits = self.model(x, edge_index, mode, support_features)[cl_mask]

        loss, _, prototypes = proto_loss_spt(support_logits, support_targets, self.k_shot)
        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, model.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # logits_q, _ = net(x_query.to(DEVICE), c_qry.to(DEVICE), feat_qry, net.parameters())
            query_logits = self.model(x_query, edge_index_query, mode, query_features, model.parameters())[
                cl_mask_query]

            loss_q, acc_q = proto_loss_qry(query_logits, query_targets, prototypes)
            corrects[0] = corrects[0] + acc_q

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # logits_q, _ = net(x_query.to(DEVICE), c_qry.to(DEVICE), feat_qry, fast_weights)
            query_logits = self.model(x_query, edge_index_query, mode, query_features, fast_weights)[cl_mask_query]

            loss_q, acc_q = proto_loss_qry(query_logits, query_targets, prototypes)
            corrects[1] = corrects[1] + acc_q

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            # logits, _ = net(x_support.to(DEVICE), c_spt.to(DEVICE), feat_spt, fast_weights)
            x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
            support_logits = self.model(x, edge_index, mode, support_features, fast_weights)[cl_mask]

            loss, _, prototypes = proto_loss_spt(support_logits, support_targets, self.k_shot)

            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=True)

            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.outer_lr * p[0], zip(grad, fast_weights)))

            # logits_q, _ = net(x_query.to(DEVICE), c_qry.to(DEVICE), feat_qry, fast_weights)
            query_logits = self.model(x_query, edge_index_query, mode, query_features, fast_weights)[cl_mask_query]

            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, acc_q = proto_loss_qry(query_logits, query_targets, prototypes)
            corrects[k + 1] = corrects[k + 1] + acc_q

        del model
        accuracies = np.array(corrects)
        return accuracies


def proto_loss_spt(features, targets, n_support):
    target_cpu = targets.to('cpu')
    input_cpu = features.to('cpu')

    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = 1
    n_query = n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[:n_support], classes))).view(-1)
    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = func.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val, prototypes


def proto_loss_qry(logits, targets, prototypes):
    target_cpu = targets.to('cpu')
    input_cpu = logits.to('cpu')

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    n_query = int(logits.shape[0] / n_classes)

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero(), classes))).view(-1)
    query_samples = input_cpu[query_idxs]

    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = func.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
