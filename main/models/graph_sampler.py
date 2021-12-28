from typing import Optional

import torch
from torch_geometric.loader import GraphSAINTSampler
from torch_geometric.utils import k_hop_subgraph

from models.batch_sampler import split_list


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

            data.x = self.data.x[node_idx].float().to_sparse()
            data.y = self.data.y[node_idx].to_sparse()
            data.mask = self.batch_sampler.mask[node_idx].to_sparse()
            data.center_idx = center_node
            data.edge_index = data.edge_index.to_sparse()
            data.edge_attr = data.edge_attr.to_sparse()

            data_list_collated.append((data, data.y[data.center_idx].item()))

        if self.model_type == 'gat':
            sup_graphs, labels = list(map(list, zip(*data_list_collated)))
            return sup_graphs, torch.LongTensor(labels)
        elif self.model_type == 'prototypical':
            sup_graphs, labels = list(map(list, zip(*data_list_collated)))

            supp_sub_graphs, query_sub_graphs = split_list(sup_graphs)
            supp_labels, query_labels = split_list(labels)
            return supp_sub_graphs, query_sub_graphs, torch.LongTensor(supp_labels), torch.LongTensor(query_labels)
        elif self.model_type == 'gmeta':

            # converts list of all given samples (e.g. (local batch size * task batch size) x 3) into a list of
            # batches (task batch size x 3 x local batch size). Makes it easier to process in the PL Maml.
            sup_graphs, labels = list(map(list, zip(*data_list_collated)))

            targets = torch.tensor(labels)

            local_b_size = self.batch_sampler.local_batch_size
            task_b_size = self.batch_sampler.task_batch_size

            assert len(sup_graphs) == targets.shape[0] == (local_b_size * task_b_size)

            targets = targets.chunk(task_b_size, dim=0)

            # no torch chunking for list of graphs
            graphs = [sup_graphs[i:i + local_b_size] for i in range(0, len(sup_graphs), local_b_size)]

            # should have size: 16 (self.task_batch_size)
            assert len(targets) == len(graphs) == task_b_size

            # should have size: 30 (self.local_batch_size)
            assert len(graphs[0]) == len(targets[0]) == local_b_size

            return list(zip(graphs, targets))

    def __len__(self):
        return len(self.batch_sampler)
