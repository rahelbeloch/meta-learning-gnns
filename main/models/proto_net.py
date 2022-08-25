import time
from collections import defaultdict
from statistics import mean, stdev

import torch.nn
import torchmetrics as tm
from torch import optim, nn
from tqdm.auto import tqdm

from models.gat_encoder_sparse_pushkar import GatNet
from models.graph_trainer import GraphTrainer, get_or_none
from models.train_utils import *


# noinspection PyAbstractClass
class ProtoNet(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, optimizer_hparams, other_params):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        train_weight = get_or_none(other_params, 'train_loss_weight')
        print(f"Positive train weight: {train_weight}")
        self.train_loss_module = nn.BCEWithLogitsLoss(pos_weight=train_weight)

        val_weight = get_or_none(other_params, 'val_loss_weight')
        print(f"Positive val weight: {val_weight}")
        self.val_loss_module = nn.BCEWithLogitsLoss(pos_weight=val_weight)

        self.model = GatNet(model_params)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.optimizer_hparams['lr'])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]

    @staticmethod
    def calculate_prototypes(features, targets):
        """
        Given a stack of features vectors and labels, return class prototypes.
        :param features:
        :param targets:
        :return:
        """

        # features shape: [N, proto_dim], targets shape: [N]

        classes, _ = torch.unique(targets).sort()  # Determine which classes we have

        prototypes = []
        for c in classes:
            # noinspection PyTypeChecker
            p = features[torch.where(targets == c)[0]].mean(dim=0)  # Average class feature vectors
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)

        return prototypes

    @staticmethod
    def classify_features(prototypes, feats, targets):
        """
        Classify new examples with prototypes and return classification error.
        """

        # Squared euclidean distance:   batch size x 2
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)

        # we only want to consider the distance for the target/fake class
        dist_fake = dist[:, 1]

        # predictions = func.log_softmax(-dist, dim=1)      # for CE loss
        # predictions = torch.sigmoid(-dist)  # for BCE loss

        # predictions = -dist  # for BCE with logits loss
        # for BCE with logits loss: don't use negative dist
        logits = dist_fake

        return logits.view(-1, 1), targets.view(-1, 1)

    def calculate_loss(self, batch, mode, loss_module):
        """
        Determine training loss for a given support and query set.
        """

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        support_graphs, query_graphs, support_targets, query_targets = batch

        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
        support_logits = self.model(x, edge_index, mode)[cl_mask]

        # support logits: 8 x 64
        # support targets: 8

        # TODO: not sure how to do this
        # self.update_metrics(mode, get_predictions(support_logits).float(), support_targets, set_name='support')

        assert support_logits.shape[0] == support_targets.shape[0], \
            "Nr of features returned does not equal nr. of classification nodes!"

        # support logits: episode size x 2, support targets: episode size x 1

        prototypes = ProtoNet.calculate_prototypes(support_logits, support_targets)

        x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
        query_logits = self.model(x, edge_index, mode)[cl_mask]

        assert query_logits.shape[0] == query_targets.shape[0], \
            "Nr of features returned does not equal nr. of classification nodes!"

        logits, targets = ProtoNet.classify_features(prototypes, query_logits, query_targets)
        # logits and targets: batch size x 1

        # targets have dimensions according to classes which are in the subgraph batch, i.e. if all sub graphs have the
        # same label, targets has 2nd dimension = 1

        loss = loss_module(logits, targets.float())

        if mode == 'train' or mode == 'val':
            self.log_on_epoch(f"{mode}/query_loss", loss)

        # make probabilities out of logits via sigmoid --> especially for the metrics; makes it more interpretable
        query_predictions = get_predictions(logits)

        self.update_metrics(mode, query_predictions, targets, set_name='query')

        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train", loss_module=self.train_loss_module)

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="val", loss_module=self.val_loss_module)


@torch.no_grad()
def test_proto_net(model, test_loader, label_names, k_shot=4, num_classes=1):
    """
    Use the trained ProtoNet & adapt to test classes. Pick k examples/sub graphs per class from which prototypes are
    determined. Test the metrics on all other sub graphs, i.e. use k sub graphs per class as support set and
    the rest of the dataset as a query set. Iterate through the dataset such that each sub graph has been once
    included in a support set.

     The average performance across all support sets tells how well ProtoNet is expected to perform
     when seeing only k examples per class.

    Inputs
        model - Pretrained ProtoNet model
        dataset - The dataset on which the test should be performed. Should be an instance of TorchGeomGraphDataset.
        data_feats - The encoded features of all articles in the dataset. If None, they will be newly calculated,
                    and returned for later usage.
        k_shot - Number of examples per class in the support set.
    """

    model = model.to(DEVICE)
    model.eval()

    # Iterate through the full dataset in two manners:
    #   First, to select the k-shot batch.
    #   Second, to evaluate the model on all the other examples

    test_start = time.time()

    f1_fakes, f1_macros, f1_weights = defaultdict(list), [], []

    for support_batch_idx, episode in tqdm(enumerate(test_loader), "Performing few-shot fine tuning in testing"):
        support_graphs, _, support_targets, _ = episode

        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
        support_feats = model.model(x, edge_index, 'test')
        support_feats = support_feats[cl_mask]
        support_targets = support_targets.to(DEVICE)

        assert support_feats.shape[0] == 2 * k_shot
        assert support_targets.shape[0] == 2 * k_shot
        assert len(set(support_targets.tolist())) == 2

        prototypes = model.calculate_prototypes(support_feats, support_targets)

        f1_target = tm.F1(num_classes=num_classes, average='none').to(DEVICE)
        f1_macro = tm.F1(num_classes=num_classes, average='macro').to(DEVICE)
        f1_weighted = tm.F1(num_classes=num_classes, average='weighted').to(DEVICE)

        with torch.no_grad():  # No gradients for query set needed

            # Evaluate all examples in test dataset
            for query_episode_idx, query_episode in enumerate(test_loader):

                if support_batch_idx == query_episode_idx:
                    # Exclude support set elements
                    continue

                support_graphs, query_graphs, support_targets, query_targets = query_episode
                graphs = support_graphs + query_graphs
                targets = torch.cat([support_targets, query_targets]).to(DEVICE)

                x, edge_index, cl_mask = get_subgraph_batch(graphs)
                feats = model.model(x, edge_index, 'test')
                feats = feats[cl_mask]

                logits, targets = model.classify_features(prototypes, feats, targets)
                predictions = get_predictions(logits)

                f1_target.update(predictions, targets)
                f1_macro.update(predictions, targets)
                f1_weighted.update(predictions, targets)

            f1_fake = f1_target.compute()
            for i, label in enumerate(label_names):
                f1_fakes[label].append(f1_fake[i].item())
            f1_macros.append(f1_macro.compute().item())
            f1_weights.append(f1_weighted.compute().item())

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f1_fakes)

    f1_fakes_std = defaultdict(float)
    for label in label_names:
        f1_fakes_std[label] = stdev(f1_fakes[label])
        f1_fakes[label] = mean(f1_fakes[label])

    return (f1_fakes, f1_fakes_std), (mean(f1_macros), stdev(f1_macros)), \
           (mean(f1_weights), stdev(f1_weights)), test_elapsed
