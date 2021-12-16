import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import GraphSAINTSampler


class FewShotSubgraphSampler(Sampler):

    def __init__(self, data, n_way=2, k_shot=5, include_query=False, shuffle=True, shuffle_once=False):
        """
        Support sets should contain n_way * k_shot examples. So, e.g. 2 * 5 = 10 sub graphs.
        Query set is of same size ...

        Inputs:
            data - Class containing the graph data.
            n_way - Number of classes to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
            include_query - If True, returns batch of size n_way*k_shot*2, which
                            can be split into support and query set. Simplifies
                            the implementation of sampling the same classes but
                            distinct examples for support and query set.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training).
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation).
        """
        super().__init__(None)
        self.data_targets = data.targets
        self.mask = data.mask  # the mask for the idx to be used for this split
        self.n_way = n_way
        self.k_shot = k_shot if not include_query else k_shot * 2
        self.shuffle = shuffle
        self.include_query = include_query
        self.batch_size = self.n_way * self.k_shot  # Number of overall samples per batch

        # Organize examples by class
        self.classes = torch.unique(self.data_targets[self.mask]).tolist()
        self.num_classes = len(self.classes)

        # Number of K-shot batches that each class can provide
        self.batches_per_class = {}
        self.indices_per_class = {}

        for c in self.classes:
            self.indices_per_class[c] = torch.where((self.data_targets == c) & self.mask)[0]
            # number of examples we have per class // number of shots -> amount of batches we can create from this class
            self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot

        # Create a list of classes from which we select the N classes per batch
        self.num_batches = sum(self.batches_per_class.values()) // self.n_way  # total batches we can create
        self.target_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]

        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in
                         range(self.batches_per_class[c])]
            self.target_list = np.array(self.target_list)[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c in self.classes:
            perm = torch.randperm(self.indices_per_class[c].shape[0])
            self.indices_per_class[c] = self.indices_per_class[c][perm]

        # Shuffle the target list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        random.shuffle(self.target_list)

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(int)
        for it in range(self.num_batches):
            class_batch = self.target_list[it * self.n_way:(it + 1) * self.n_way]  # Select N classes for the batch
            index_batch = []
            for c in class_batch:  # For each class, select the next K examples and add them to the batch
                index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c] + self.k_shot])
                start_index[c] += self.k_shot
            if self.include_query:  # If we return support+query set, sort them so that they are easy to split
                index_batch = index_batch[::2] + index_batch[1::2]
            yield index_batch

    def __len__(self):
        return self.num_batches

    @staticmethod
    def get_collate_fn(model):
        collate_fn = None

        if model == 'gat':
            def collate_fn(batch_samples):
                """
                Receives a batch of samples (sub graphs and labels) node IDs for which sub graphs need to be generated
                on the flight.
                :param batch_samples: List of pairs where each pair is: (graph, label)
                """
                graphs, labels = list(map(list, zip(*batch_samples)))

                # for DGL
                # return dgl.batch(graphs), torch.LongTensor(labels)

                return graphs, torch.LongTensor(labels)
        elif model == 'prototypical':
            def collate_fn(batch_samples):
                """
                Receives a batch of samples (sub graphs and labels) node IDs for which sub graphs need to be generated
                on the flight.
                :param batch_samples: List of pairs where each pair is: (graph, label)
                """
                graphs, labels = list(map(list, zip(*batch_samples)))

                support_sub_graphs, query_sub_graphs = split_list(graphs)
                support_labels, query_labels = split_list(labels)

                # for DGL
                # return dgl.batch(support_sub_graphs), dgl.batch(query_sub_graphs), torch.LongTensor(
                #     support_labels), torch.LongTensor(query_labels)

                return support_sub_graphs, query_sub_graphs, torch.LongTensor(support_labels), torch.LongTensor(
                    query_labels)

        return collate_fn


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


class KHopSampler(GraphSAINTSampler):

    def __init__(self, data, batch_size: int, k_hops: int = 1, save_dir: Optional[str] = None,
                 log: bool = True, **kwargs):
        super().__init__(data, batch_size, k_hops, save_dir=save_dir, log=log, **kwargs)

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.k_hops}_{self.sample_coverage}.pt'

    # def __sample_nodes__(self, batch_size):
    #     # instead of calculating shortest distance, we find the following ways to get sub graphs are quicker
    #     h_hop_neighbors = self.get_hop_neighbors(node_id)
    #
    #     sub_graph = dgl.node_subgraph(self.graph.graph, h_hop_neighbors, store_ids=True)
    #
    #     # create mask for the classification; which node we want to classify
    #     classification_mask = torch.zeros(sub_graph.num_nodes())
    #     # noinspection PyTypeChecker
    #     classification_mask[torch.where(sub_graph.ndata[dgl.NID] == node_id)[0]] = 1
    #
    #     # sub_graph.ndata['classification_mask'] = classification_mask.bool()  # mask tensors must be bool
    #     # start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
    #     # node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
    #
    #     return node_idx.view(-1)
