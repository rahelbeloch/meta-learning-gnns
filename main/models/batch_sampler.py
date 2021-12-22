import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import GraphSAINTSampler
from torch_geometric.utils import k_hop_subgraph


class FewShotSampler(Sampler):

    def __init__(self, targets, mask, n_way=2, k_shot=5, include_query=False, shuffle=True, shuffle_once=False):
        """
        Support sets should contain n_way * k_shot examples. So, e.g. 2 * 5 = 10 sub graphs.
        Query set is of same size ...

        Inputs:
            targets - Tensor containing all targets of the graph.
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
        self.data_targets = targets
        self.mask = mask  # the mask for the idx to be used for this split
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


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


class KHopSampler(GraphSAINTSampler):

    def __init__(self, graph, model, batch_sampler, n_way: int, k_shots: int, k_hops: int = 1,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        super().__init__(graph.data, batch_size=n_way * k_shots, save_dir=save_dir, log=log,
                         batch_sampler=batch_sampler, **kwargs)

        self.k_hops = k_hops
        self.edge_index = graph.edge_index

        self.model_type = model

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.k_hops}_{self.sample_coverage}.pt'

    @property
    def b_size(self):
        """
        Can not use the fitting property name batch_size, because of the code structure of GraphSAINTSampler, therefore
        using this property name. If the batch_sampler includes query and support set in one batch/episode, the
        batch/episode size is 2 * k_shots * n_classes.
        """
        return self.__batch_size__ if not self.batch_sampler.include_query else self.__batch_size__ * 2

    def __getitem__(self, idx):
        node_idx = self.__sample_nodes__(idx).unique()
        adj, _ = self.adj.saint_subgraph(node_idx)
        # noinspection PyTypeChecker
        return node_idx, adj, torch.where(node_idx == idx)[0].item()

    def __sample_nodes__(self, node_id):
        node_idx, edge_index, node_mapping_idx, edge_mask = k_hop_subgraph(node_id.unsqueeze(dim=0),
                                                                           self.k_hops,
                                                                           self.edge_index,
                                                                           relabel_nodes=True,
                                                                           flow="target_to_source")

        return node_idx

    # noinspection PyPropertyAccess
    def __collate__(self, data_list):
        """
        Overrides the GraphSAINTSampler collate function because that one only accepts one single data point.
        Iterates over every single data point in the list, creates a Data object and normalizes it.
        :param data_list: List of node indices and adjacency matrices for (sub) graphs.
        :return:
        """
        # The Saint Graph Sampler expects 1 data point (and we have more) and normalizes them. This method overrides
        # the graph saint normalizing method and normalizes every single point in our list.
        # Returns the subgraphs in TorchGerom Batches

        data_list_collated = []

        # TODO
        for node_idx, adj, center_node in data_list:
            # data_collated = super().__collate__(data)

            data = self.data.__class__()
            data.num_nodes = node_idx.size(0)
            row, col, edge_idx = adj.coo()
            data.edge_index = torch.stack([row, col], dim=0)

            for key, item in self.data:
                if key in ['edge_index', 'num_nodes']:
                    continue
                if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                    data[key] = item[node_idx]
                elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
                    data[key] = item[edge_idx]
                else:
                    data[key] = item

            # TODO: normalization
            #     if self.sample_coverage > 0:
            #         data.node_norm = self.node_norm[node_idx]
            #         data.edge_norm = self.edge_norm[edge_idx]

            data.x = self.data.x[node_idx]
            data.y = self.data.y[node_idx]
            data.mask = self.batch_sampler.mask[node_idx]
            data.center_idx = center_node

            data_list_collated.append((data, data.y[data.center_idx].item()))

        if self.model_type == 'gat':
            sup_graphs, labels = list(map(list, zip(*data_list_collated)))
            return sup_graphs, torch.LongTensor(labels)
        elif self.model_type == 'prototypical':
            sup_graphs, labels = list(map(list, zip(*data_list_collated)))

            supp_sub_graphs, query_sub_graphs = split_list(sup_graphs)
            supp_labels, query_labels = split_list(labels)
            return supp_sub_graphs, query_sub_graphs, torch.LongTensor(supp_labels), torch.LongTensor(query_labels)

    def __len__(self):
        return len(self.batch_sampler)

    # @staticmethod
    # def get_collate_fn(model):
    #     collate_fn = None
    #
    #     if model == 'gat':
    #         def collate_fn(batch_samples):
    #             """
    #             Receives a batch of samples (sub graphs and labels) node IDs for which sub graphs need to be generated
    #             on the flight.
    #             :param batch_samples: List of pairs where each pair is: (graph, label)
    #             """
    #             sup_graphs, labels = list(map(list, zip(*batch_samples)))
    #
    #             return sup_graphs, torch.LongTensor(labels)
    #     elif model == 'prototypical':
    #         def collate_fn(batch_samples):
    #             """
    #             Receives a batch of samples (sub graphs and labels) node IDs for which sub graphs need to be generated
    #             on the flight.
    #             :param batch_samples: List of pairs where each pair is: (graph, label)
    #             """
    #             graphs, labels = list(map(list, zip(*batch_samples)))
    #
    #             support_sub_graphs, query_sub_graphs = split_list(graphs)
    #             support_labels, query_labels = split_list(labels)
    #
    #             # for DGL
    #             # return dgl.batch(support_sub_graphs), dgl.batch(query_sub_graphs), torch.LongTensor(
    #             #     support_labels), torch.LongTensor(query_labels)
    #
    #             return support_sub_graphs, query_sub_graphs, torch.LongTensor(support_labels), torch.LongTensor(
    #                 query_labels)
    #
    #     return collate_fn

# class KHopSamplerAndFewShotSampler(GraphSAINTSampler):
#
#     def __init__(self, data, mask, n_way=2, k_shot=5, include_query=False, shuffle=True, shuffle_once=False,
#                  k_hops: int = 1, save_dir: Optional[str] = None, log: bool = True, **kwargs):
#         """
#         Support sets should contain n_way * k_shot examples. So, e.g. 2 * 5 = 10 sub graphs.
#         Query set is of same size ...
#
#         Inputs:
#             data - Class containing the graph data.
#             n_way - Number of classes to sample per batch.
#             k_shot - Number of examples to sample per class in the batch.
#             include_query - If True, returns batch of size n_way*k_shot*2, which
#                             can be split into support and query set. Simplifies
#                             the implementation of sampling the same classes but
#                             distinct examples for support and query set.
#             shuffle - If True, examples and classes are newly shuffled in each
#                       iteration (for training).
#             shuffle_once - If True, examples and classes are shuffled once in
#                            the beginning, but kept constant across iterations
#                            (for validation).
#         """
#         super().__init__(data, batch_size=n_way * k_shot, save_dir=save_dir, log=log, **kwargs)
#
#         # self.data_targets = data.targets
#         self.data_targets = data.y
#         self.mask = mask  # the mask for the idx to be used for this split
#         self.k_hops = k_hops
#
#         self.n_way = n_way
#         self.k_shot = k_shot if not include_query else k_shot * 2
#         self.shuffle = shuffle
#         self.include_query = include_query
#         # self.batch_size = self.n_way * self.k_shot  # Number of overall samples per batch
#
#         # Organize examples by class
#         self.classes = torch.unique(self.data_targets[self.mask]).tolist()
#         self.num_classes = len(self.classes)
#
#         # Number of K-shot batches that each class can provide
#         self.batches_per_class = {}
#         self.indices_per_class = {}
#
#         for c in self.classes:
#             self.indices_per_class[c] = torch.where((self.data_targets == c) & self.mask)[0]
#             # number of examples we have per class // number of shots -> amount of batches we can create from this class
#             self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot
#
#         # Create a list of classes from which we select the N classes per batch
#         self.num_batches = sum(self.batches_per_class.values()) // self.n_way  # total batches we can create
#         self.target_list = [c for c in self.classes for _ in range(self.batches_per_class[c])]
#
#         if shuffle_once or self.shuffle:
#             self.shuffle_data()
#         else:
#             # For testing, we iterate over classes instead of shuffling them
#             sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in
#                          range(self.batches_per_class[c])]
#             self.target_list = np.array(self.target_list)[np.argsort(sort_idxs)].tolist()
#
#     def shuffle_data(self):
#         # Shuffle the examples per class
#         for c in self.classes:
#             perm = torch.randperm(self.indices_per_class[c].shape[0])
#             self.indices_per_class[c] = self.indices_per_class[c][perm]
#
#         # Shuffle the target list from which we sample. Note that this way of shuffling
#         # does not prevent to choose the same class twice in a batch. However, for
#         # training and validation, this is not a problem.
#         random.shuffle(self.target_list)
#
#     def __iter__(self):
#         # Shuffle data
#         if self.shuffle:
#             self.shuffle_data()
#
#         # Sample few-shot batches
#         start_index = defaultdict(int)
#         for it in range(self.num_batches):
#             class_batch = self.target_list[it * self.n_way:(it + 1) * self.n_way]  # Select N classes for the batch
#             index_batch = []
#             for c in class_batch:  # For each class, select the next K examples and add them to the batch
#                 index_batch.extend(self.indices_per_class[c][start_index[c]:start_index[c] + self.k_shot])
#                 start_index[c] += self.k_shot
#             if self.include_query:  # If we return support+query set, sort them so that they are easy to split
#                 index_batch = index_batch[::2] + index_batch[1::2]
#             yield index_batch
#
#     def __len__(self):
#         return self.num_batches
#
#     @property
#     def __filename__(self):
#         return f'{self.__class__.__name__.lower()}_{self.k_hops}_{self.sample_coverage}.pt'
#
#     def __getitem__(self, idx):
#         node_idx = self.__sample_nodes__(idx).unique()
#         adj, _ = self.adj.saint_subgraph(node_idx)
#         return node_idx, adj
#
#     def __sample_nodes__(self, node_id):
#         node_idx, edge_index, node_mapping_idx, edge_mask = k_hop_subgraph(node_id.unsqueeze(dim=0),
#                                                                            self.k_hops,
#                                                                            self.edge_index,
#                                                                            relabel_nodes=True,
#                                                                            flow="target_to_source")
#
#         # return TorchGeomSubGraph(node_id.item(), node_idx, node_feats, edge_index, node_mapping_idx, edge_mask)
#
#         # pre implementation
#         # start = torch.randint(0, self.N, (batch_size,), dtype=torch.long)
#         # node_idx = self.adj.random_walk(start.flatten(), self.walk_length)
#
#         return node_idx
#
#     def __collate__(self, data_list):
#         # The Saint Graph Sampler expects 1 data point (and we have more) and normalizes them. So for now no
#         # normalization
#
#         data_list_collated = []
#         for node_idx, adj in data_list:
#             # data_collated = super().__collate__(data)
#
#             data = self.data.__class__()
#             data.num_nodes = node_idx.size(0)
#             row, col, edge_idx = adj.coo()
#             data.edge_index = torch.stack([row, col], dim=0)
#
#             for key, item in self.data:
#                 if key in ['edge_index', 'num_nodes']:
#                     continue
#                 if isinstance(item, torch.Tensor) and item.size(0) == self.N:
#                     data[key] = item[node_idx]
#                 elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
#                     data[key] = item[edge_idx]
#                 else:
#                     data[key] = item
#
#             if self.sample_coverage > 0:
#                 data.node_norm = self.node_norm[node_idx]
#                 data.edge_norm = self.edge_norm[edge_idx]
#
#             data_list_collated.append(data)
#         return data_list
#
#     @staticmethod
#     def get_collate_fn(model):
#         collate_fn = None
#
#         if model == 'gat':
#             def collate_fn(batch_samples):
#                 """
#                 Receives a batch of samples (sub graphs and labels) node IDs for which sub graphs need to be generated
#                 on the flight.
#                 :param batch_samples: List of pairs where each pair is: (graph, label)
#                 """
#                 graphs, labels = list(map(list, zip(*batch_samples)))
#
#                 # for DGL
#                 # return dgl.batch(graphs), torch.LongTensor(labels)
#
#                 return graphs, torch.LongTensor(labels)
#         elif model == 'prototypical':
#             def collate_fn(batch_samples):
#                 """
#                 Receives a batch of samples (sub graphs and labels) node IDs for which sub graphs need to be generated
#                 on the flight.
#                 :param batch_samples: List of pairs where each pair is: (graph, label)
#                 """
#                 graphs, labels = list(map(list, zip(*batch_samples)))
#
#                 support_sub_graphs, query_sub_graphs = split_list(graphs)
#                 support_labels, query_labels = split_list(labels)
#
#                 return support_sub_graphs, query_sub_graphs, torch.LongTensor(support_labels), torch.LongTensor(
#                     query_labels)
#
#         return collate_fn
