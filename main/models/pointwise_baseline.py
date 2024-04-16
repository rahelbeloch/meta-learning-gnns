import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseMLP(nn.Module):
    """
    Literally just an MLP, but built to imitate the behaviour of our GATs.
    Should be equally expressive.

    """

    def __init__(self, model_params):
        super(PointwiseMLP, self).__init__()

        self.in_dim = model_params["input_dim"]
        self.hid_dim = model_params["hid_dim"]
        self.fc_dim = model_params["fc_dim"]
        self.output_dim = model_params["output_dim"]
        self.n_heads = model_params["n_heads"]

        self.mask_p = model_params["node_mask_p"]
        self.dropout = model_params["dropout"]
        self.attn_dropout = model_params["attn_dropout"]

        self.feature_extractor = nn.Sequential(
            nn.Linear(
                in_features=self.in_dim,
                out_features=self.n_heads * self.hid_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(
                in_features=self.n_heads * self.hid_dim,
                out_features=self.n_heads * self.hid_dim,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(
                in_features=self.n_heads * self.hid_dim,
                out_features=self.fc_dim,
            ),
            nn.ReLU(),
        )

        self.classifier = self.get_classifier(self.output_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def reset_classifier(self, num_classes: int = None):
        # Adapting the classifier dimensions
        if num_classes is None:
            num_classes = self.output_dim

        self.classifier = self.get_classifier(num_classes).to(self.device)

    def get_classifier(self, output_dim):
        return nn.Linear(self.fc_dim, output_dim, bias=True)

    def node_mask(self, x):
        if self.training:
            node_mask = torch.rand((x.shape[0], 1)).type_as(x)
            x = node_mask.ge(self.mask_p) * x

        return x

    def extract_features(self, x, edge_index):
        if torch.cuda.is_available():
            assert x.is_cuda

        x = self.node_mask(x)

        # Attention on input
        x = self.feature_extractor(x)

        return x

    def forward(self, x, edge_index, mode=None):
        #! Deprecated argument: mode
        # Mode left here for legacy purposes, no longer serves a purpose
        # All dropout are now registered modules (i.e. model.eval())

        x = self.extract_features(x, edge_index)

        # Classification head
        logits = self.classifier(x)

        return logits
