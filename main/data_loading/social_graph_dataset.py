import torch
from torch.utils.data import Dataset

from data_prep.post_processing import SocialGraph


class IterableSocialGraph(Dataset, SocialGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph
