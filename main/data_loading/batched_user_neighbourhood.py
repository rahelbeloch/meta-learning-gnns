import os
import random
import time
import typing
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import starmap
import multiprocessing as mp
from functools import partial

import numpy as np
import pymetis
import torch
from torch.nn.utils.rnn import pad_sequence
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.convert import to_scipy_sparse_matrix

from data_loading.batched_khop_neighbourhood import BatchedKHopNeighbourhoodBase
from utils.logging import calc_elapsed_time
from utils.rng import stochastic_method


class BatchedKHopUserNeighbourhood(BatchedKHopNeighbourhoodBase):
    def __init__(
        self,
        args: dict,
        structure_mode: str,
        cur_fold: int,
        split: str,
        min_k_hop: int = 2,
        max_k_hop: int = 5,
        labels_per_graph: int = 1,
        batch_size: int = 32,
        max_samples_per_partition: int = -1,
        max_nodes_per_subgraph: int = 2048,
        walk_length: int = 3,
        node_weights_dist: str = "inv_node_degree",
        label_dist: typing.Optional[str] = None,
        prefix: typing.Optional[str] = None,
        **superkwargs,
    ):
        super().__init__(
            args,
            structure_mode,
            cur_fold,
            split,
            2,
            batch_size,
            node_weights_dist,
            **superkwargs,
        )

        # Graph statistics
        self.max_samples_per_partition = max_samples_per_partition
        self.labels_per_graph = labels_per_graph

        self.min_k_hop = min_k_hop
        self.max_k_hop = max_k_hop
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.walk_length = walk_length

        # Compression stats
        self.central_nodes = []
        self.needed_k_hops = []

        self.pre_num_nodes = []
        self.pre_num_edges = []
        self.pre_num_labels = []

        self.post_num_nodes = []
        self.post_num_edges = []
        self.post_num_labels = []

        self.prefix = prefix

        self.node_occurences = Counter()

        if label_dist is None or label_dist == "node":
            self.label_dist = None
        elif label_dist == "frequency":
            self.label_dist = "frequency"

    def __repr__(self):
        _repr = f"BatchedKHopUserNeighbourhood(mode={self.structure_mode}, split={self.split}, max_k_hop={self.max_k_hop}, batch_size={self.batch_size}, budget={self.max_nodes_per_subgraph}"

        if self.prefix is not None:
            return _repr + f", version={self.prefix})"
        else:
            return _repr + ")"

    @stochastic_method
    def partition_into_batches(self):
        start_time = time.time()
        self.print_step("Partitioning graph into batches")

        # Generate the edge_list from the edge_index
        edge_list = to_scipy_sparse_matrix(
            self.graph.edge_index, num_nodes=self.graph.num_nodes
        )
        edge_list = edge_list.tolil(copy=True)
        edge_list = edge_list.rows

        # Use METIS to partition into buckets
        # A batch consists of samples from all buckets
        # Maximizes the coverage per batch
        self.log("Using METIS to partition graph")
        n_cuts, membership = pymetis.part_graph(
            self.batch_size,
            adjacency=edge_list,
            options=pymetis.Options(
                seed=self.seed,
            ),
        )

        self.log(f"Cut size: {n_cuts}, {n_cuts/self.graph.num_edges*100:.2f}%")

        # Distribute the samples in buckets to batches
        num_docs = sum(1 for split in self.graph.splits if split != "user")
        num_users = sum(1 for split in self.graph.splits if split == "user")

        # Distribute the partitions into their buckets
        user_to_partition = defaultdict(list)
        for i, partition in enumerate(membership[num_docs:]):
            user_to_partition[partition] += [num_docs + i]

        rng = random.Random(self.seed)
        # Sort and shuffle the clusters
        # Subsample if necessary
        self.clusters = list(
            map(
                lambda x: rng.sample(x[1], len(x[1]))[: self.max_samples_per_partition],
                sorted(user_to_partition.items(), key=lambda x: x[0]),
            )
        )

        self.log("\nFinished partitioning graph.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    @stochastic_method
    def generate_subgraph(self, central_id, partition_id):
        def check_subgraph(subset, edge_index):
            subset = subset.detach()
            edge_index = edge_index.detach()

            num_nodes = subset.shape[0]

            # Generate the subgraph indices relative to this subgraph
            batch_idx = torch.arange(0, num_nodes)

            # Check the number of potential label nodes sampled
            label_info = {l: [] for l in self.labels}

            # Find where the labels are and store
            sufficient_labels = True
            for l in self.labels:
                label_mask = self.graph.y[subset] == l

                # Get the location of the labels in terms of the batched subgraph
                label_idx = batch_idx[label_mask]

                if label_idx.shape[0] < self.labels_per_graph:
                    sufficient_labels = False
                    break
                else:
                    # Get the corresponding label weights
                    label_weights = self.node_weights[subset][label_idx]
                    label_weights = label_weights / label_weights.sum()

                    label_info[l] = [label_idx, label_weights]

            if sufficient_labels:
                return subset, edge_index, num_nodes, label_info
            else:
                return None

        subgraph_info = dict()

        # Subgraph generation ==========================================================
        # We sample a center node and sample a k-hop subgraph around it
        # Once we have enough label nodes we return the graph properties needed
        # Can happen that the subgraph radius becomes unreasonably large
        # Then we just give up on it
        subgraph_node_id = torch.tensor([central_id])

        for k_hop in range(self.min_k_hop, self.max_k_hop + 1):
            # Sample the k-hop subgraph
            subset, edge_index, _, _ = k_hop_subgraph(
                node_idx=subgraph_node_id,
                num_hops=k_hop,
                edge_index=self.graph.edge_index,
                relabel_nodes=True,
            )

            result = check_subgraph(subset, edge_index)

            if result is not None:
                subset, edge_index, num_nodes, label_info = result
                break

        if k_hop == self.max_k_hop and result is None:
            return None

        # Subgraph subsampling =========================================================
        # The sampled graph has enough label support but its too large to fit into
        # memory when batched
        # Here we stochastically subsample the graph using random walks from the
        # found label nodes.

        subgraph_info["central_nodes"] = central_id
        subgraph_info["needed_k_hops"] = k_hop

        subgraph_info["pre_num_nodes"] = subset.shape[0]
        subgraph_info["pre_num_edges"] = edge_index.shape[1]
        subgraph_info["pre_num_labels"] = sum(
            map(lambda x: len(x[0]), label_info.values())
        )

        # In case we have zero-budget, i.e. text classification
        # Only has self-edges
        if self.max_nodes_per_subgraph == 0:
            subset_ = []
            label_info_ = dict()
            num_nodes = 0
            for l in sorted(list(self.labels.keys())):
                label_locs, label_weights = label_info[l]

                num_l_labels = label_locs.squeeze().shape[0]
                label_info_[l] = [
                    torch.arange(num_nodes, num_nodes + num_l_labels),
                    label_weights.squeeze(),
                ]

                subset_ += [label_locs.squeeze()]

                num_nodes += num_l_labels

            subset = subset[torch.cat(subset_)]

            edge_index = torch.stack(
                torch.arange(0, num_nodes).repeat(2).split(num_nodes)
            )

            label_info = label_info_

        # Subsampling with non-zero budget
        elif subset.shape[0] > self.max_nodes_per_subgraph:
            max_nodes_per_subgraph_per_label = round(
                self.max_nodes_per_subgraph / len(self.labels)
            )

            nodes = set()
            cur_nodes_size = 0
            for l in self.labels:
                label_locs, label_weights = label_info[l]

                # Get the starting locations
                # Needs lots of them
                # Sample them based on the node distribution
                path_starts = torch.multinomial(
                    label_weights,
                    num_samples=max(
                        label_weights.shape[0], 5 * max_nodes_per_subgraph_per_label
                    ),
                    replacement=True,
                    generator=self.rng,
                )

                # Convert the labels to subgraph_node_ids
                path_starts = label_locs[path_starts]

                # Generate random walks from those start location of some length
                adj = SparseTensor(
                    row=edge_index[0],
                    col=edge_index[1],
                    value=torch.arange(edge_index.shape[1], device=edge_index.device),
                    sparse_sizes=(subset.shape[0], subset.shape[0]),
                )

                # Literally no way to pass a seed here...
                sampled_paths = adj.random_walk(
                    path_starts.flatten(), walk_length=self.walk_length
                )

                # Adding paths to sampled nodes until budget is met
                for path in sampled_paths:
                    path = subset[path]
                    unique_nodes = set(path.tolist())
                    nodes.update(unique_nodes)

                    if len(nodes) - cur_nodes_size >= max_nodes_per_subgraph_per_label:
                        break

                cur_nodes_size = len(nodes)

            subset = torch.tensor(sorted(list(nodes)))
            edge_index = self.graph.subgraph(subset).edge_index

            result = check_subgraph(subset, edge_index)
            if result is not None:
                subset, edge_index, num_nodes, label_info = result

            else:
                print("Subsampled graph has too few labels remaining.")
                return None

        subgraph_info["post_num_nodes"] = subset.shape[0]
        subgraph_info["post_num_edges"] = edge_index.shape[1]
        subgraph_info["post_num_labels"] = sum(
            map(lambda x: len(x[0]), label_info.values())
        )

        # Subgraph saving ==============================================================
        # Save the generated subgraph to disk
        torch.save(
            obj={
                "partition_id": partition_id,
                "central_nodes": subgraph_node_id,
                "graph_idx": subset,
                "edge_index": edge_index,
                "num_nodes": num_nodes,
                "num_edges": edge_index.shape[1],
                "label_info": label_info,
            },
            f=self.neighbourhood_dir / f"subgraph_{central_id}.pt",
        )

        return (central_id, partition_id, subgraph_info)

    @stochastic_method
    def generate_batches(self, num_workers: int = 0):
        start_time = time.time()
        self.print_step("Generating k-hop neigbourhoods")
        os.makedirs(self.neighbourhood_dir, exist_ok=True)
        self.log(f"Find files in: {self.neighbourhood_dir}")

        # Get the node ids partitioned into clusters
        clusters = deepcopy(self.clusters)

        # Flatten the clusters into a queue
        flattened_clusters = [
            (node_id, cluster_id)
            for cluster_id, cluster in enumerate(clusters)
            for node_id in cluster
        ]

        self.log("\n+=== Processing graphs ===+")

        start = time.time()

        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        if num_workers > 0:
            self.log(f"Using {num_workers} workers")
            # Build an mp worker pool
            # Controls the number of active workers
            with mp.Pool(processes=num_workers) as pool:
                subgraphs = pool.starmap(
                    func=self.generate_subgraph,
                    iterable=flattened_clusters,
                )

        else:
            self.log("Running on main process")
            # Otherwise just do everything over the main process
            subgraphs = starmap(
                self.generate_subgraph,
                flattened_clusters,
            )
            subgraphs = list(subgraphs)

        del self.rng

        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # Filter out all the unsuccesfull subgraphs
        clusters_succes = defaultdict(list)

        for result in subgraphs:
            if result is not None:
                node_id, partition_id, subgraph_info = result

                clusters_succes[partition_id].append(node_id)

                self.pre_num_nodes.append(subgraph_info["pre_num_nodes"])
                self.post_num_nodes.append(subgraph_info["post_num_nodes"])
                self.pre_num_edges.append(subgraph_info["pre_num_edges"])
                self.post_num_edges.append(subgraph_info["post_num_edges"])
                self.pre_num_labels.append(subgraph_info["pre_num_labels"])
                self.post_num_labels.append(subgraph_info["post_num_labels"])
                self.needed_k_hops.append(subgraph_info["needed_k_hops"])

        self.log("\nSubgraph Stats:")
        self.pre_num_nodes = np.array(self.pre_num_nodes)
        self.post_num_nodes = np.array(self.post_num_nodes)
        self.log(
            f"{'Nodes':<7} | Pre: {np.mean(self.pre_num_nodes)/1e+3:5.2f}K, Post: {np.mean(self.post_num_nodes)/1e+3:5.2f}K, Reduction: {1 - np.mean(self.post_num_nodes / self.pre_num_nodes):.2f}"
        )

        self.pre_num_edges = np.array(self.pre_num_edges)
        self.post_num_edges = np.array(self.post_num_edges)
        self.log(
            f"{'Edges':<7} | Pre: {np.mean(self.pre_num_edges)/1e+6:5.2f}M, Post: {np.mean(self.post_num_edges)/1e+6:5.2f}M, Reduction: {1 - np.mean(self.post_num_edges / self.pre_num_edges):.2f}"
        )

        self.pre_num_labels = np.array(self.pre_num_labels)
        self.post_num_labels = np.array(self.post_num_labels)
        self.log(
            f"{'Labels':<7} | Pre: {np.mean(self.pre_num_labels):6.0f}, Post: {np.mean(self.post_num_labels):6.0f}, Reduction: {1 - np.mean(self.post_num_labels / self.pre_num_labels):.2f}"
        )

        pre_density = (
            2 * self.pre_num_edges / (self.pre_num_nodes * (self.pre_num_nodes - 1))
        )
        post_density = (
            2 * self.post_num_edges / (self.post_num_nodes * (self.post_num_nodes - 1))
        )
        self.log(
            f"{'Density':<7} | Pre: {np.mean(pre_density):6.0e}, Post: {np.mean(post_density):6.0e}, {'Growth: ':<11}{np.mean(post_density / pre_density):.2f}"
        )

        counts, _ = np.histogram(
            self.needed_k_hops, bins=self.max_k_hop - self.min_k_hop - 1
        )
        Z = counts.sum()

        self.log("\nHops Needed:")
        for hops, count in zip(range(self.min_k_hop, self.max_k_hop + 1), counts):
            self.log(f"{hops} | {count:>5} {count / Z * 100:.2f}%")

        self.log(f"\n+=== Assigning subgraphs to batches ===+")
        # Rebuild the clusters lists
        # Keeping only those subgraphs that were succesful

        # Resort the clusters
        clusters_succes = list(
            map(lambda x: x[1], sorted(clusters_succes.items(), key=lambda x: x[0]))
        )

        # Fuse the batches together
        # Ideally, each batch gets one sample from each partition
        batches = defaultdict(list)
        for node_ids in clusters_succes:
            for batch_id, node_id in enumerate(node_ids):
                batches[batch_id].append(node_id)

        batches = list(map(lambda x: x[1], sorted(batches.items(), key=lambda x: x[0])))

        self.log(
            f"\nFound {sum(map(len, batches))} samples over {len(batches)} batches."
        )
        self.log(
            f"Found {sum(map(lambda x: len(x) < self.batch_size, batches))} batches smaller than batchsize"
        )
        self.log("Folding smaller batches into larger ones.")

        # Batches at the end are going to be smaller than batch size
        # Just take those and put them into another batch that needs more samples
        ptr_l = 0
        # Iterate over the batches from left to right
        while ptr_l < len(batches) - 1:
            batch_l = batches[ptr_l]

            # If the batch is large enough go to next batch
            if len(batch_l) == self.batch_size:
                ptr_l += 1
                continue

            # If batch needs more samples, take some from a smaller batch
            elif len(batch_l) < self.batch_size:
                # Figure out how much of the right batch can be put in the left batch
                n_needed = self.batch_size - len(batch_l)
                n_available = len(batches[-1])
                n_given = min(n_needed, n_available)

                # Move that amount
                batches[ptr_l] = batch_l + batches[-1][:n_given]
                batches[-1] = batches[-1][n_given:]

                # Get rid of the right batch if empty
                if len(batches[-1]) == 0:
                    del batches[-1]

            else:
                raise ValueError("Batch found larger than batch size...")

        self.log(
            f"\nFound {sum(map(len, batches))} samples over {len(batches)} batches."
        )
        self.log(
            f"Found {sum(map(lambda x: len(x) < self.batch_size, batches))} batches smaller than batchsize"
        )

        self.log("\n+=== Aggregating subgraphs into batches ===+")
        start = time.time()

        self.batches = list()
        self.batch_information = defaultdict(list)

        for batch_n, batch in enumerate(batches):
            batched_subgraphs = {
                "batch_n": batch_n,
                "partition_id": [],
                "batch_ptr": [],
                "central_nodes": [],
                "graph_idx": [],
                "edge_index": [],
                "num_nodes": 0,
                "num_edges": 0,
            }
            for l in self.labels:
                batched_subgraphs.update(
                    {f"label_{l}_locs": [], f"label_{l}_probs": []}
                )

            batch_ptr = torch.tensor(0)

            for central_id in batch:
                subgraph = torch.load(
                    self.neighbourhood_dir / f"subgraph_{central_id}.pt",
                    weights_only=True,
                )

                cur_batch_ptr = batch_ptr.detach().clone()

                self.node_occurences.update(subgraph["graph_idx"].tolist())

                # Populate the batch with subgraph information
                batched_subgraphs["partition_id"] += [subgraph["partition_id"]]
                batched_subgraphs["batch_ptr"] += [cur_batch_ptr]
                batched_subgraphs["central_nodes"] += [subgraph["central_nodes"]]
                batched_subgraphs["graph_idx"] += [subgraph["graph_idx"]]
                batched_subgraphs["edge_index"] += [
                    subgraph["edge_index"] + cur_batch_ptr
                ]
                batched_subgraphs["num_nodes"] += subgraph["num_nodes"]
                batched_subgraphs["num_edges"] += subgraph["num_edges"]

                for l in self.labels:
                    locs, probs = subgraph["label_info"][l]

                    # Also update the label locations with the batch ptr
                    locs = locs + cur_batch_ptr

                    batched_subgraphs[f"label_{l}_locs"].append(locs)
                    batched_subgraphs[f"label_{l}_probs"].append(probs)

                batch_ptr += subgraph["num_nodes"]

            # Aggregate the subgraphs together into a single disjoint graph
            for k, v in batched_subgraphs.items():
                if "label" not in k:
                    if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                        if len(v[0].shape) > 0:
                            batched_subgraphs[k] = torch.cat(v, dim=-1)
                        else:
                            batched_subgraphs[k] = torch.stack(v, dim=0)

                    elif (
                        isinstance(v, int)
                        or isinstance(v, float)
                        or isinstance(v, bool)
                    ):
                        batched_subgraphs[k] = torch.tensor(v)

                else:
                    batched_subgraphs[k] = pad_sequence(
                        v,
                        batch_first=True,
                        padding_value=-1 if "locs" in k else 0.0,
                    )

            # Save the batched subgraph
            torch.save(
                batched_subgraphs,
                self.neighbourhood_dir / f"batch_{batch_n}.pt",
            )

            self.batches.append(batch_n)

            # Delete the subgraph from the saved dir
            for central_id in batch:
                (self.neighbourhood_dir / f"subgraph_{central_id}.pt").unlink()

            self.batch_information["batch_size"] += [
                batched_subgraphs["central_nodes"].shape[0]
            ]
            self.batch_information["num_nodes"] += [
                batched_subgraphs["num_nodes"].item()
            ]
            self.batch_information["num_edges"] += [
                batched_subgraphs["num_edges"].item()
            ]

            if (
                batch_n == 0
                or batch_n % max(1, len(batches) // 10) == 0
                or batch_n == len(batches) - 1
            ):
                self.log(
                    f"{batch_n:04} | Batch size: {self.batch_information['batch_size'][-1]} Num. nodes: {self.batch_information['num_nodes'][-1] / 1e+3:.2f}K Num. edges: {self.batch_information['num_edges'][-1] / 1e+6:.2f}M"
                )

        self.batch_information = dict(self.batch_information)

        if self.label_dist is None:
            self._label_dist = self.node_weights

        elif self.label_dist == "frequency":
            self._label_dist = torch.zeros((self.graph.num_nodes,))

            idx, counts = list(map(list, zip(*self.node_occurences.items())))

            counts = torch.tensor(counts, dtype=torch.float)

            self._label_dist[idx] = counts
            self._label_dist = 1 / self._label_dist

        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

        assert len(self.batches) > 0, "Batches is empty."

        self.log("\nFinished generating batched subgraphs.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    @stochastic_method
    def collate_fn(self, batch):
        if len(batch) != 1:
            raise ValueError("Batch size should be 1")
        batch = batch[0]

        graph = Data(
            x=self.graph.x[batch["graph_idx"]],
            edge_index=batch["edge_index"],
            num_nodes=batch["num_nodes"],
            num_edges=batch["num_edges"],
            y=torch.full((batch["num_nodes"],), fill_value=self.label_mask),
            mask=torch.full((batch["num_nodes"],), fill_value=False),
        )

        for l in self.labels:
            if self.label_dist is None:
                chosen_locs = torch.multinomial(
                    batch[f"label_{l}_probs"],
                    num_samples=self.labels_per_graph,
                    replacement=False,
                )

            elif self.label_dist == "frequency":
                label_graph_idx = batch["graph_idx"][batch[f"label_{l}_locs"]]
                label_weights = self._label_dist[label_graph_idx]

                # Could be that certain nodes do not appear in distribution?
                label_weights.masked_fill_(batch[f"label_{l}_probs"] == 0.0, torch.nan)
                label_weights = torch.nan_to_num(
                    label_weights,
                    nan=0.0,
                    neginf=0.0,
                    posinf=0.0,
                )
                label_weights /= torch.nansum(label_weights, dim=-1).unsqueeze(-1)

                assert (
                    ((label_weights != 0.0).sum(dim=-1) >= self.labels_per_graph)
                    .all()
                    .item()
                ), f"Not enough labels to sample anymore..., {batch['batch_n']}, {l}"

                try:
                    chosen_locs = torch.multinomial(
                        label_weights,
                        num_samples=self.labels_per_graph,
                        replacement=False,
                    )

                except:
                    print("Batch id:", batch["batch_n"])
                    print("Label:", l)
                    print("Shape:", label_weights.shape)
                    print("Sum:", label_weights.sum())
                    print("Mean:", label_weights.float().mean())
                    print("Min:", label_weights.min())
                    print("Max:", label_weights.max())
                    print("Nan per row:", torch.isnan(label_weights).sum(dim=-1))
                    print("Non-zeros per row:", (label_weights != 0.0).sum(dim=-1))
                    print("Zeros per row:", (label_weights == 0.0).sum(dim=-1))
                    print(label_weights)
                    raise ValueError("Can't sample")

            chosen_labels = torch.gather(
                batch[f"label_{l}_locs"], dim=-1, index=chosen_locs
            ).reshape(-1)

            graph.y.index_fill_(
                dim=-1,
                index=chosen_labels,
                value=l,
            )

            graph.mask.index_fill_(
                dim=-1,
                index=chosen_labels,
                value=True,
            )

        return graph
