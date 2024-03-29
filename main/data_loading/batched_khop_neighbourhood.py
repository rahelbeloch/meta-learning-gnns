import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch_geometric 
from torch_geometric.utils import degree, to_torch_coo_tensor

from data_prep.post_processing import SocialGraph


class BatchedKHopNeighbourhoodBase(SocialGraph, Dataset):
    def __init__(
        self,
        args: dict,
        structure_mode: str,
        cur_fold: int,
        split: str,
        k_hop: int = 2,
        batch_size: int = 32,
        node_weights_dist: str = "inv_node_degree",
        **superkwargs,
    ):
        super().__init__(
            args=args,
            structure_mode=structure_mode,
            cur_fold=cur_fold,
            split=split,
            **superkwargs,
        )

        self.k_hop = k_hop
        if node_weights_dist in {
            "uniform",
            "inv_node_degree",
            "inv_neigh_degree",
            "inv_neigh_user_degree",
        }:
            self.node_weights_dist = node_weights_dist
        else:
            raise ValueError(
                f"Node weight distribution `{node_weights_dist}` not recognized."
            )

        if batch_size is not None and batch_size > 0:
            self.batch_size = batch_size
        else:
            raise ValueError("Batch size must be a positive integer.")

    @property
    def neighbourhood_dir(self):
        return self.data_structure_path(self.__str__().lower())

    def _generate_node_weights(self):
        self.log("\nComputing node weights...")

        if self.node_weights_dist == "inv_node_degree":
            # Node gets sampled according proportional to its degree
            self.node_weights = 1 / degree(self.graph.edge_index[0])

        elif self.node_weights_dist == "inv_neigh_degree":
            # A node gets sampled proportional to the average degree of its neighbours
            self.node_weights = torch.ones(self.graph.num_nodes)

            sparse_adj = to_torch_coo_tensor(self.graph.edge_index)

            self.node_weights = torch.matmul(sparse_adj, self.node_weights)
            self.node_weights = torch.matmul(sparse_adj, self.node_weights)

            self.node_weights = 1 / self.node_weights

        elif self.node_weights_dist == "inv_neigh_user_degree":
            labelled_nodes = torch.where(self.graph.mask)[0]

            num_nodes = self.graph.num_nodes
            num_labelled_nodes = labelled_nodes.shape[0]
            num_docs = len(self.valid_docs)

            # Use a message passing scheme
            # First get the adjacency
            sparse_adj = to_torch_coo_tensor(
                self.graph.edge_index,
                size=(num_nodes, num_nodes),
            )

            # Create the initial messages
            # A 1 on each labelled document node
            # Column is message initial location
            # Row is message propagated to a neighbour
            messages = torch.sparse_coo_tensor(
                torch.stack([labelled_nodes, labelled_nodes]),
                torch.ones((num_labelled_nodes,)),
                (num_nodes, num_nodes),
            )

            # Left multiplied with the adjecency matrix
            # The message gets passed to all its neighbours
            messages = torch.matmul(sparse_adj, messages)

            # Left multiplied with the adjecency matrix
            # The message gets passed to all its neighbours' neighbours
            messages = torch.matmul(sparse_adj, messages)

            # Now get the users which received the message from the original labelled node
            # Check which nodes are users
            user_mask = messages.indices()[0] > num_docs

            senders = messages.indices()[1, user_mask]

            # Force every value to be a 1
            user_received_messages = messages.values()[user_mask].bool().float()

            # Add the user received messages (as booleans) to the original senders
            # And presto! The node weights of the labelled nodes are the number of users
            # that reside in its 2-hop neighbourhood
            self.node_weights = torch.zeros((num_nodes,))

            self.node_weights.index_add_(
                dim=0,
                index=senders,
                source=user_received_messages,
            )

            # self.node_weights = (self.node_weights.max() + 1) - self.node_weights
            self.node_weights = 1 / self.node_weights

        elif self.node_weights_dist == "uniform":
            # All nodes are equally likely to get samples
            self.node_weights = torch.ones((self.graph.idx.shape[0],))

    def build_graph(self, *args, **kwargs):
        super().build_graph(*args, **kwargs)

        self._generate_node_weights()

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        subgraph_idx = self.batches[index]

        batch_info = torch.load(
            self.neighbourhood_dir / f"batch_{subgraph_idx}.pt", weights_only=True
        )

        return batch_info

    def __repr__(self):
        raise NotImplementedError()

    def __str__(self):
        return self.__repr__()
