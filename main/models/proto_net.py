import torch.nn.functional as func
from torch import optim

from models.GraphTrainer import GraphTrainer
from models.gat_encoder_sparse_pushkar import GatNet
from models.train_utils import *


class ProtoNet(GraphTrainer):

    # noinspection PyUnusedLocal
    def __init__(self, model_params, lr, batch_size, label_names):
        """
        Inputs
            proto_dim - Dimensionality of prototype feature space
            lr - Learning rate of Adam optimizer
        """
        super().__init__(model_params['output_dim'])
        self.save_hyperparameters()

        self.model = GatNet(model_params)
        # self.model = SparseGATLayer(in_features=model_params['input_dim'], out_features=model_params['hid_dim'],
        #                             feat_reduce_dim=model_params['feat_reduce_dim'])

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        support_graphs, query_graphs, support_targets, query_targets = batch

        x, edge_index, cl_mask = get_subgraph_batch(support_graphs)
        support_feats = self.model(x, edge_index, cl_mask, mode)

        assert support_feats.shape[0] == support_targets.shape[0], \
            "Nr of features returned does not equal nr. of classification nodes!"

        prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)

        x, edge_index, cl_mask = get_subgraph_batch(query_graphs)
        query_feats = self.model(x, edge_index, cl_mask, mode)

        assert query_feats.shape[0] == query_targets.shape[0], \
            "Nr of features returned does not equal nr. of classification nodes!"

        predictions, targets, acc = ProtoNet.classify_features(prototypes, classes, query_feats, query_targets)

        meta_loss = func.cross_entropy(predictions, targets)

        if mode == 'train':
            self.log(f"{mode}_loss", meta_loss)

        self.f1_target[mode].update(predictions, targets)
        self.f1_macro[mode].update(predictions, targets)
        self.log_on_epoch(f"{mode}_accuracy", accuracy(predictions, targets))

        return meta_loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx1, batch_idx2):
        self.calculate_loss(batch, mode="test")
