import os
import time
import typing
from collections import defaultdict
from itertools import zip_longest
import multiprocessing as mp

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from data_loading.batched_khop_neighbourhood import BatchedKHopNeighbourhoodBase
from utils.logging import calc_elapsed_time


class BatchedKHopDocumentNeighbourhood(BatchedKHopNeighbourhoodBase):
    def __init__(
        self,
        args: dict,
        structure_mode: str,
        cur_fold: int,
        split: str,
        k_hop: int = 2,
        batch_size: int = 32,
        node_weights_dist: str = "inv_node_degree",
        prefix: str = None,
        _doc_limit: int = -1,
        **superkwargs,
    ):
        super().__init__(
            args=args,
            structure_mode=structure_mode,
            cur_fold=cur_fold,
            split=split,
            k_hop=k_hop,
            batch_size=batch_size,
            node_weights_dist=node_weights_dist,
            **superkwargs,
        )

        self._doc_limit = _doc_limit

        self.prefix = prefix

    def __repr__(self):
        _repr = f"BatchedKHopDocumentNeighbourhood(mode={self.structure_mode}, split={self.split}, k_hop={self.k_hop}, batch_size={self.batch_size}"

        if self.prefix is not None:
            return _repr + f", version={self.prefix})"
        else:
            return _repr + ")"

    def partition_into_batches(self):
        # Nothing happens here at the moment
        start_time = time.time()
        self.print_step("Partitioning graph into batches")

        print("Document graphs need no partitioning.")

        self.log("\nFinished partitioning graph.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def generate_subgraph(self, central_id):
        doc_node_id = torch.tensor([central_id])

        graph_node_id = torch.searchsorted(
            self.graph.idx,
            doc_node_id,
        )

        # Sample the k-hop subgraph
        subset, edge_index, mapping, _ = k_hop_subgraph(
            node_idx=graph_node_id,
            num_hops=self.k_hop,
            edge_index=self.graph.edge_index,
            relabel_nodes=True,
        )

        subset = subset.detach().clone()
        edge_index = edge_index.detach().clone()

        num_nodes = subset.shape[0]

        torch.save(
            obj={
                "central_nodes": doc_node_id,
                "label_locs": mapping,
                "graph_idx": subset,
                "edge_index": edge_index,
                "num_nodes": num_nodes,
                "num_edges": edge_index.shape[1],
            },
            f=self.neighbourhood_dir / f"eval_subgraph_{central_id}.pt",
        )

        return central_id

    def generate_batches(
        self, num_workers: int = 0, batches: typing.Iterable[int] = None
    ):
        start_time = time.time()
        self.print_step("Generating k-hop neigbourhoods")
        os.makedirs(self.neighbourhood_dir, exist_ok=True)
        print("Find files in:", self.neighbourhood_dir)

        if batches is None:
            # Interleaves the labels
            # Just cause we can
            label2nodeid = {l: [] for l in self.labels.keys()}
            for split, doc_node_id in zip(self.graph.splits, self.graph.node_ids):
                if split == self.split:
                    label = self.nodeid2label[doc_node_id].item()
                    label2nodeid[label] += [doc_node_id]

            split_docs = [
                x
                for interwoven_list in zip_longest(*label2nodeid.values())
                for x in interwoven_list
                if x is not None
            ]

            split_docs = split_docs[: self._doc_limit]

        else:
            split_docs = [doc_node_id for batch in batches for doc_node_id in batch]

        print("\n+=== Processing graphs ===+")
        start = time.time()

        if num_workers > 0:
            # Build an mp worker pool
            # Controls the number of active workers
            print(f"Using {num_workers} workers")
            with mp.Pool(processes=num_workers) as pool:
                subgraphs = pool.map(
                    func=self.generate_subgraph,
                    iterable=split_docs,
                )

        else:
            # Otherwise just do everything over the main process
            print("Running on main process")
            subgraphs = map(
                self.generate_subgraph,
                split_docs,
            )
            subgraphs = list(subgraphs)

        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        print(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

        print("\n+=== Assigning subgraphs to batches ===+")
        if batches is None:
            batch_n = 0
            batches = defaultdict(list)
            for node_id in subgraphs:
                if len(batches[batch_n]) < self.batch_size:
                    batches[batch_n] += [node_id]
                else:
                    batch_n += 1
                    batches[batch_n] += [node_id]

            batches = [batch for batch in batches.values() if len(batch) > 0]

        else:
            print("\nUsing provided `batches`.")

        print(f"\nFound {sum(map(len, batches))} samples over {len(batches)} batches.")
        print(
            f"Found {sum(map(lambda x: len(x) < self.batch_size, batches))} batches smaller than batchsize"
        )

        print("\n+=== Aggregating subgraphs into batches ===+")
        self.batches = list()
        self.batch_information = defaultdict(list)

        start = time.time()

        for batch_n, batch in enumerate(batches):
            batched_subgraphs = {
                "batch_ptr": [],
                "central_nodes": [],
                "label_locs": [],
                "graph_idx": [],
                "edge_index": [],
                "num_nodes": 0,
                "num_edges": 0,
            }

            batch_ptr = torch.tensor(0)

            for central_id in batch:
                subgraph = torch.load(
                    self.neighbourhood_dir / f"eval_subgraph_{central_id}.pt",
                    weights_only=True,
                )

                cur_batch_ptr = batch_ptr.detach().clone()

                # Populate the batch with subgraph information
                batched_subgraphs["batch_ptr"] += [cur_batch_ptr]
                batched_subgraphs["central_nodes"] += [subgraph["central_nodes"]]
                batched_subgraphs["label_locs"] += [
                    subgraph["label_locs"] + cur_batch_ptr
                ]
                batched_subgraphs["graph_idx"] += [subgraph["graph_idx"]]
                batched_subgraphs["edge_index"] += [
                    subgraph["edge_index"] + cur_batch_ptr
                ]
                batched_subgraphs["num_nodes"] += subgraph["num_nodes"]
                batched_subgraphs["num_edges"] += subgraph["num_edges"]

                batch_ptr += subgraph["num_nodes"]

            # Aggregate the subgraphs together into a single disjoint graph
            for k, v in batched_subgraphs.items():
                if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    if len(v[0].shape) > 0:
                        batched_subgraphs[k] = torch.cat(v, dim=-1)
                    else:
                        batched_subgraphs[k] = torch.stack(v, dim=0)

                elif isinstance(v, int) or isinstance(v, float) or isinstance(v, bool):
                    batched_subgraphs[k] = torch.tensor(v)

            # Save the batched subgraph
            torch.save(
                batched_subgraphs,
                self.neighbourhood_dir / f"batch_{batch_n}.pt",
            )

            self.batches.append(batch_n)

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
                print(
                    f"{batch_n:04} | Batch size: {self.batch_information['batch_size'][-1]} Num. nodes: {self.batch_information['num_nodes'][-1] / 1e+3:.2f}K Num. edges: {self.batch_information['num_edges'][-1] / 1e+6:.2f}M"
                )

        self.batch_information = dict(self.batch_information)

        # Delete the subgraphs from the saved dir
        # These have all been aggregated into batches
        for batch_n, batch in enumerate(batches):
            for central_id in batch:
                subgraph_fp = self.neighbourhood_dir / f"eval_subgraph_{central_id}.pt"

                if subgraph_fp.exists():
                    subgraph_fp.unlink()

        end = time.time()
        hours, minutes, seconds = calc_elapsed_time(start, end)
        print(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

        assert len(self.batches) > 0, "Batches is empty."

        self.log("\nFinished generating batched subgraphs.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def collate_fn(self, batch):
        if len(batch) != 1:
            raise ValueError("Batchsize should be 1.")
        batch = batch[0]

        graph = Data(
            x=self.graph.x[batch["graph_idx"]],
            edge_index=batch["edge_index"],
            num_nodes=batch["num_nodes"],
            num_edges=batch["num_edges"],
            y=torch.full((batch["num_nodes"],), fill_value=self.label_mask),
            mask=torch.full((batch["num_nodes"],), fill_value=False),
        )

        graph.y[batch["label_locs"]] = self.graph.y[
            batch["graph_idx"][batch["label_locs"]]
        ]
        graph.mask[batch["label_locs"]] = True

        return graph
