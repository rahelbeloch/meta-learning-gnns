import time
from copy import deepcopy
from statistics import mean, stdev

from torch import optim, nn
from torchmetrics import F1
from tqdm.auto import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer, get_or_none
from models.train_utils import *
from samplers.episode_sampler import split_list


# noinspection PyAbstractClass
class Maml(GraphTrainer):
    """
    First-Order MAML (FOML) which only uses first-order gradients.
    """

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, other_params):
        """
        Inputs
            lr - Learning rate of the outer loop Adam optimizer
            lr_inner - Learning rate of the inner loop SGD optimizer
            n_inner_updates - Number of inner loop updates to perform
        """
        super().__init__(validation_sets=['val'])
        self.save_hyperparameters()

        self.n_inner_updates = model_params['n_inner_updates']
        self.n_inner_updates_test = model_params['n_inner_updates_test']

        self.lr_inner = self.hparams.optimizer_hparams['lr_inner']
        self.k_shot_support = other_params['k_shot_support']

        # train_weight = get_loss_weight(model_params["class_weight"], 'train')
        self.train_loss_module = nn.BCEWithLogitsLoss(pos_weight=get_or_none(other_params, 'train_loss_weight'))

        # val_weight = get_loss_weight(model_params["class_weight"], 'val')
        self.val_loss_module = nn.BCEWithLogitsLoss(pos_weight=get_or_none(other_params, 'val_loss_weight'))

        self.model = GatNet(model_params)

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt_params = self.hparams.optimizer_hparams
        optimizer, scheduler = self.get_optimizer(opt_params['lr'], opt_params['lr_decay_epochs'],
                                                  milestones=[140, 180])
        return [optimizer], [scheduler]

    def adapt_few_shot(self, x, edge_index, cl_mask, support_targets, mode, loss_module):

        # Copy model for inner-loop model and optimizer
        local_model = deepcopy(self.model)
        local_model.train()
        local_optim = optim.SGD(local_model.parameters(), lr=self.lr_inner)
        local_optim.zero_grad()

        # Optimize inner loop model on support set
        for _ in range(self.n_inner_updates):
            # Determine loss on the support set
            loss, logits = run_model(local_model, x, edge_index, cl_mask, support_targets, mode, loss_module)

            # Calculate gradients and perform inner loop update
            loss.backward()
            local_optim.step()

        return local_model

    def outer_loop(self, batch, loss_module, mode):
        losses = []

        self.model.zero_grad()

        # Determine gradients for batch of tasks
        for graphs, targets in batch:

            support_graphs, query_graphs = split_list(graphs, self.k_shot_support)
            support_targets, query_targets = split_list(targets, self.k_shot_support)

            # Perform inner loop adaptation
            local_model = self.adapt_few_shot(*get_subgraph_batch(support_graphs), support_targets, mode, loss_module)

            # Determine loss of query set
            loss, query_predictions = run_model(local_model, *get_subgraph_batch(query_graphs), query_targets, mode,
                                                loss_module)

            self.update_metrics(mode, query_predictions, query_targets)

            # Calculate gradients for query set loss
            if mode == "train":
                loss.backward()

                for i, (p_global, p_local) in enumerate(zip(self.model.parameters(), local_model.parameters())):
                    if p_global.requires_grad is False:
                        continue

                    # First-order approx. -> add gradients of fine-tuned and base model
                    if p_global.grad is None:
                        p_global.grad = p_local.grad
                    else:
                        p_global.grad += p_local.grad

            losses.append(loss.detach())

        # Perform update of base model
        if mode == "train":
            opt = self.optimizers()
            opt.step()
            # noinspection PyUnresolvedReferences
            opt.zero_grad()

            train_scheduler = self.lr_schedulers()
            if self.trainer.current_epoch != 0 and (self.trainer.current_epoch + 1) % train_scheduler.step_size == 0:
                train_scheduler.step()

        self.log_on_epoch(f"{mode}/loss", sum(losses) / len(losses))

    def training_step(self, batch, batch_idx):
        self.outer_loop(batch, self.train_loss_module, mode="train")

        # Returning None means skipping the default training optimizer steps by PyTorch Lightning
        return None

    def validation_step(self, batch, batch_idx):
        # Validation requires to finetune a model, hence we need to enable gradients
        torch.set_grad_enabled(True)
        self.outer_loop(batch, self.val_loss_module, mode="val")
        torch.set_grad_enabled(False)


def run_model(local_model, x, edge_index, cl_mask, targets, mode, loss_module=None):
    """
    Execute a model with given output layer weights and inputs.
    """

    logits = local_model(x, edge_index, mode)[cl_mask]

    # Expected
    # logits shape batch size x 64
    # output_weight shape 1 x 64
    # output_bias shape 1

    targets = targets.view(-1, 1) if not len(targets.shape) == 2 else targets
    loss = loss_module(logits, targets.float()) if loss_module is not None else None

    return loss, (logits.sigmoid() > 0.5).float()


def test_maml(model, test_loader, num_classes=1):
    mode = 'test'
    model = model.to(DEVICE)
    model.eval()

    # TODO: use inner loop updates of 200 --> should be higher than in training

    test_start = time.time()

    # Iterate through the full dataset in two manners:
    # First, to select the k-shot batch. Second, to evaluate the model on all other batches.
    test_loss_weight = None
    loss_module = nn.BCEWithLogitsLoss(pos_weight=test_loss_weight)

    f1_fakes = []

    for support_batch_idx, batch in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = batch

        # graphs are automatically put to device in adapt few shot
        support_targets = support_targets.to(DEVICE)

        # Finetune new model on support set
        local_model = model.adapt_few_shot(*get_subgraph_batch(support_graphs), support_targets, mode, loss_module)

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

                _, pred = run_model(local_model, *get_subgraph_batch(graphs), targets, mode)

                f1_target.update(pred, targets)

            f1_fakes.append(f1_target.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    return (mean(f1_fakes), stdev(f1_fakes)), test_elapsed
