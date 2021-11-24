import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from models.gat_encoder import GATEncoder


class ProtoNet(pl.LightningModule):

    # noinspection PyUnusedLocal
    def __init__(self, input_dim, cf_hidden_dim, lr):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = GATEncoder(input_dim, hidden_dim=cf_hidden_dim, num_heads=4)

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
            p = features[torch.where(targets == c)[0]].mean(dim=0)
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
        predictions = F.log_softmax(-dist, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (predictions.argmax(dim=1) == labels).float().mean()
        return predictions, labels, acc

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
        print(f'Support features dim: {support_feats.shape}')
        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)

        query_feats = self.model(support_graphs)
        predictions, labels, acc = ProtoNet.classify_features(prototypes, classes, query_feats, query_targets)

        loss = F.cross_entropy(predictions, labels)

        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)

        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self.calculate_loss(batch, mode="val")
