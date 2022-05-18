from typing import Optional

import torch
from torch_geometric.loader import GraphSAINTSampler
from torch_geometric.utils import k_hop_subgraph

from samplers.episode_sampler import split_list
from train_config import META_MODELS


class KHopSampler(GraphSAINTSampler):

    def __init__(self, graph, model, batch_sampler, n_way: int, k_shots: int, k_hops: int = 1, mode=None,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        super().__init__(graph.data, batch_size=n_way * k_shots, save_dir=save_dir, log=log,
                         batch_sampler=batch_sampler, **kwargs)

        self.k_hops = k_hops
        self.edge_index = graph.edge_index
        self.adjacency_matrix = graph.adj
        self.model_type = model
        self.mode = mode

        self.mask = graph.mask(f'{mode}_mask')

        if mode == 'train':
            val_nodes = torch.where(graph.mask('val_mask'))[0]
            test_nodes = torch.where(graph.mask('test_mask'))[0]
            self.test_val_nodes = set(torch.cat((val_nodes, test_nodes), dim=0).tolist())
        else:
            self.test_val_nodes = None

    @property
    def __filename__(self):
        return f'{self.__class__.__name__.lower()}_{self.k_hops}_{self.sample_coverage}.pt'

    @property
    def b_sampler(self):
        return self.batch_sampler

    @property
    def b_size(self):
        """
        Can not use the fitting property name batch_size, because of the code structure of GraphSAINTSampler, therefore
        using this property name. If the batch_sampler includes query and support set in one batch/episode, the
        batch/episode size is 2 * k_shots * n_classes.
        """
        return self.b_sampler.b_size

    def __getitem__(self, idx):
        node_indices = self.__sample_nodes__(idx).unique()
        adj, _ = self.adj.saint_subgraph(node_indices)
        # noinspection PyTypeChecker
        return node_indices, adj, (idx.item(), torch.where(node_indices == idx)[0].item())

    def __sample_nodes__(self, node_id):
        node_indices, edge_index, node_mapping_idx, edge_mask = k_hop_subgraph(node_id.unsqueeze(dim=0),
                                                                               self.k_hops,
                                                                               self.edge_index,
                                                                               relabel_nodes=True,
                                                                               flow="target_to_source")

        if self.mode == 'train':
            # check if any of test/val nodes are samples as side node; if True --> remove them!
            subgraph_nodes = set(node_indices.tolist())
            intersect = self.test_val_nodes.intersection(subgraph_nodes)
            if len(intersect) is not 0:
                new_subgraph_nodes = subgraph_nodes - intersect
                assert node_id.item() in new_subgraph_nodes, "Center node ID was removed!"
                node_indices = torch.LongTensor(list(new_subgraph_nodes))

        return node_indices

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
        # Returns the sub graphs in TorchGeom Batches

        data_list_collated = []

        for node_indices, adj, center_indices in data_list:
            data, target = self.as_data_target(adj, center_indices, node_indices)
            data_list_collated.append((data, target))

        sup_graphs, labels = list(map(list, zip(*data_list_collated)))

        if self.model_type in ['prototypical', 'gat'] or (self.model_type in META_MODELS and self.mode == 'test'):

            supp_sub_graphs, query_sub_graphs = split_list(sup_graphs)
            supp_labels, query_labels = split_list(labels)
            return supp_sub_graphs, query_sub_graphs, torch.LongTensor(supp_labels), torch.LongTensor(query_labels)

        elif self.model_type in META_MODELS:

            # converts list of all given samples (e.g. (local batch size * task batch size) x 3) into a list of
            # batches (task batch size x 3 x local batch size). Makes it easier to process in the PL Maml.
            targets = torch.tensor(labels)

            local_b_size = self.batch_sampler.local_batch_size
            task_b_size = self.batch_sampler.task_batch_size

            # all samples should be divisible by 2 (support/query), the local batch size and the size of one task
            assert len(sup_graphs) == targets.shape[0] == (local_b_size * task_b_size)

            targets = targets.chunk(task_b_size, dim=0)

            # no torch chunking for list of graphs
            graphs = [sup_graphs[i:i + local_b_size] for i in range(0, len(sup_graphs), local_b_size)]

            # should have size: 16 (self.task_batch_size)
            assert len(targets) == len(graphs) == task_b_size

            # should have size: 30 (self.local_batch_size)
            assert len(graphs[0]) == len(targets[0]) == local_b_size

            return list(zip(graphs, targets))

    def as_data_target(self, adj, center_indices, node_indices):
        # data_collated = super().__collate__(data)

        row, col, edge_idx = adj.coo()
        data = self.data.__class__(self.data.x[node_indices], torch.stack([row, col], dim=0), None, None)

        # TODO: normalization
        #     if self.sample_coverage > 0:
        #         data.node_norm = self.node_norm[node_idx]
        #         data.edge_norm = self.edge_norm[edge_idx]

        data.orig_center_idx, data.new_center_idx = center_indices

        # VERY IMPORTANT: batch sampler works with indices based on the mask --> have to get the masked y here first!
        target = self.data.y[self.mask][data.orig_center_idx].item()

        for s in self.b_sampler.sets:
            # TODO: fix this, must be the target class!!
            if data.orig_center_idx in self.b_sampler.indices_per_class[s][target]:
                data.set_type = s
                break

        return data, target

    def __len__(self):
        return len(self.batch_sampler)


class KHopSamplerSimple(GraphSAINTSampler):

    def __init__(self, graph, k_hops: int = 1, save_dir: Optional[str] = None, log: bool = True, **kwargs):
        super().__init__(graph.data, batch_size=1, save_dir=save_dir, log=log, batch_sampler=None, **kwargs)

        self.k_hops = k_hops
        self.edge_index = graph.edge_index

    def __getitem__(self, idx):
        node_indices = self.__sample_nodes__(idx).unique()
        adj, _ = self.adj.saint_subgraph(node_indices)
        # noinspection PyTypeChecker
        data, target = self.as_data_target(adj, (idx.item(), torch.where(node_indices == idx)[0].item()), node_indices)
        return data, target

    def __sample_nodes__(self, node_id):
        node_indices, edge_index, node_mapping_idx, edge_mask = k_hop_subgraph(node_id.unsqueeze(dim=0),
                                                                               self.k_hops,
                                                                               self.edge_index,
                                                                               relabel_nodes=True,
                                                                               flow="target_to_source")

        return node_indices

    def as_data_target(self, adj, center_indices, node_indices):
        # data_collated = super().__collate__(data)

        row, col, edge_idx = adj.coo()
        data = self.data.__class__(self.data.x[node_indices], torch.stack([row, col], dim=0), None, None)

        # TODO: normalization
        #     if self.sample_coverage > 0:
        #         data.node_norm = self.node_norm[node_idx]
        #         data.edge_norm = self.edge_norm[edge_idx]

        # data.mask = self.batch_sampler.mask[node_idx]
        data.orig_center_idx, data.new_center_idx = center_indices
        target = self.data.y[data.orig_center_idx].item()
        return data, target
