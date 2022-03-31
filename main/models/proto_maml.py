from copy import deepcopy

import torch.nn.functional as func
from torch import optim
## tqdm for loading bars
from tqdm.auto import tqdm

from models.GraphTrainer import GraphTrainer
from models.gat_encoder_sparse_pushkar import GatNet
from models.proto_net import ProtoNet, DEVICE
from models.train_utils import *
from samplers.batch_sampler import split_list


class ProtoMAML(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, opt_hparams, n_inner_updates, label_names):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            lr_output - Learning rate for the output layer in the inner loop
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__(n_classes=model_params['output_dim'])
        self.save_hyperparameters()

        self.model = GatNet(model_params)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.opt_hparams['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    def adapt_few_shot(self, support_graphs, support_targets, mode):

        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)

        # Determine prototype initialization
        support_feats = self.model(x, edge_index, mode).squeeze()
        support_feats = support_feats[cl_mask]

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
            loss, predictions = run_model(local_model, output_weight, output_bias, support_graphs, support_labels, mode)

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
        losses = []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for task_batch in batch:

            graphs, targets = task_batch
            support_graphs, query_graphs = split_list(graphs)
            support_targets, query_targets = split_list(targets)

            # Perform inner loop adaptation
            local_model, output_weight, output_bias, classes = self.adapt_few_shot(support_graphs, support_targets,
                                                                                   mode)

            # Determine loss of query set
            query_labels = self.get_labels(classes, query_targets)
            loss, predictions = run_model(local_model, output_weight, output_bias, query_graphs, query_labels, mode)

            pred = predictions.argmax(dim=-1)
            for mode_dict, _ in self.metrics.values():
                mode_dict[mode].update(pred, query_targets)

            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                # print("Model parameters.")
                count = 0
                for i, (p_global, p_local) in enumerate(zip(self.model.parameters(), local_model.parameters())):
                    if p_global.grad is None or p_local.grad is None:
                        # print(f"Grad none at position: {i}")
                        count += 1
                    else:
                        # First-order approx. -> add gradients of fine-tuned and base model
                        p_global.grad += p_local.grad

                # print(f"Grads None: {count}")

            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            # noinspection PyUnresolvedReferences
            opt.zero_grad()

        self.compute_and_log_metrics('train')
        # TODO: do we want to log on step or on epoch?
        self.log_on_epoch(f"{mode}_loss", sum(losses) / len(losses))

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to fine tune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, output_weight, output_bias, graphs, targets, mode):
    """
    Execute a model with given output layer weights and inputs.
    """

    x, edge_index, cl_mask = get_subgraph_batch(graphs)
    feats = local_model(x, edge_index, mode).squeeze()
    feats = feats[cl_mask]

    predictions = func.linear(feats, output_weight, output_bias)

    loss = func.cross_entropy(predictions, targets)

    return loss, predictions


def test_protomaml(model, dataset, k_shot=4):
    # pl.seed_everything(42)
    model = model.to(DEVICE)

    # We iterate through the full dataset in two manners. First, to select the k-shot batch.
    # Second, the evaluate the model on all other examples
    accuracies = []
    for (support_imgs, support_targets), support_indices in tqdm(zip(sample_dataloader, sampler),
                                                                 "Performing few-shot finetuning"):
        support_imgs = support_imgs.to(device)
        support_targets = support_targets.to(device)
        # Finetune new model on support set
        local_model, output_weight, output_bias, classes = model.adapt_few_shot(support_imgs, support_targets)
        with torch.no_grad():  # No gradients for query set needed
            local_model.eval()
            batch_acc = torch.zeros((0,), dtype=torch.float32, device=device)
            # Evaluate all examples in test dataset
            for query_imgs, query_targets in full_dataloader:
                query_imgs = query_imgs.to(device)
                query_targets = query_targets.to(device)
                query_labels = (classes[None, :] == query_targets[:, None]).long().argmax(dim=-1)
                _, _, acc = model.run_model(local_model, output_weight, output_bias, query_imgs, query_labels)
                batch_acc = torch.cat([batch_acc, acc.detach()], dim=0)
            # Exclude support set elements
            for s_idx in support_indices:
                batch_acc[s_idx] = 0
            batch_acc = batch_acc.sum().item() / (batch_acc.shape[0] - len(support_indices))
            accuracies.append(batch_acc)
    return mean(accuracies), stdev(accuracies)
