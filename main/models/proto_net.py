import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as func
from torch import optim

from models.gat_encoder import GATLayer
from models.train_utils import *


class ProtoNet(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, input_dim, cf_hidden_dim, lr, batch_size):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = GATLayer(c_in=input_dim, c_out=cf_hidden_dim, num_heads=4)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
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

        # Determine which classes we have
        classes, _ = torch.unique(targets).sort()
        prototypes = []
        for c in classes:
            # get all node features for this class and average them
            # noinspection PyTypeChecker
            p = torch.from_numpy(np.array(features)[torch.where(targets == c)[0]]).mean(dim=0)
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
        # Return the 'classes' tensor to know which prototype belongs to which class
        return prototypes, classes

    @staticmethod
    def classify_features(prototypes, classes, feats, targets):
        """
        Classify new examples with prototypes and return classification error.

        :param prototypes:
        :param classes:
        :param feats:
        :param targets:
        :return:
        """
        # Squared euclidean distance
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)
        predictions = func.log_softmax(-dist, dim=1)
        # noinspection PyUnresolvedReferences
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        return predictions, labels, accuracy(predictions, labels)

    def calculate_loss(self, batch, mode):
        """
        Determine training loss for a given support and query set.

        :param batch:
        :param mode:
        :return:
        """

        # sub_graphs, targets = batch
        # features = self.model(sub_graphs)  # Encode all sub graphs of support and query set
        # support_feats, query_feats, support_targets, query_targets = split_batch()

        support_graphs, query_graphs, support_targets, query_targets = batch

        support_feats = self.model(support_graphs)

        # TODO: remove; filter targets for subgraphs with more than 1 node
        actual_targets = []
        for i, g in enumerate(support_graphs):
            if g.num_nodes <= 1:
                continue
            actual_targets.append(support_targets[i])
        support_targets = torch.stack(actual_targets)

        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)

        query_feats = self.model(query_graphs)
        query_feats = query_feats[query_graphs.ndata['classification_mask']]
        predictions, targets, acc = ProtoNet.classify_features(prototypes, classes, query_feats, query_targets)

        meta_loss = func.cross_entropy(predictions, targets)

        if mode == 'train':
            self.log(f"{mode}_loss", meta_loss, batch_size=self.hparams['batch_size'])

        self.log_on_epoch(f"{mode}_accuracy", acc)
        self.log_on_epoch(f"{mode}_f1_macro", f1(predictions, targets, average='macro'))
        self.log_on_epoch(f"{mode}_f1_micro", f1(predictions, targets, average='micro'))

        return meta_loss

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True, batch_size=self.hparams['batch_size'])

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self.calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx1, batch_idx2):
        _ = self.calculate_loss(batch, mode="test")
