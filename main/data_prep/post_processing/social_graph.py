import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_scipy_sparse_matrix,
    from_scipy_sparse_matrix,
    coalesce,
)
import scipy.sparse as sp

from data_prep.post_processing import PostProcessing
from utils.logging import calc_elapsed_time


class SocialGraph(PostProcessing):
    splits = ["train", "val", "test"]

    def __init__(
        self,
        args,
        structure_mode,
        cur_fold,
        split,
        keep_cc="largest",
        version=None,
        processed_or_structured="structured",
    ):
        super().__init__(
            args["data"],
            cur_fold=cur_fold,
            version=version,
            processed_or_structured=processed_or_structured,
        )

        self.label_mask = args["data"]["label_mask"]

        self.structure_mode = structure_mode
        self.split = split
        self.keep_cc = keep_cc

    def prep(self):
        start_time = time.time()

        self.print_step("Gathering needed graph info")

        self.doc2nodeid = self.load_file("doc2nodeid")
        self.user2nodeid = self.load_file("user2nodeid")

        self.node_ids = list(self.doc2nodeid.values()) + list(self.user2nodeid.values())

        compressed_doc_features = self.load_file("compressed_dataset")
        compressed_doc_features.set_format(type="torch", columns=["x", "y"])

        self.split2nodeids = dict()
        self.nodeid2storageid = dict()
        self.nodeid2split = dict()
        self.nodeid2label = dict()
        for split in self.splits:
            split_dataset = compressed_doc_features[split]

            split_doc_nodes = sorted(split_dataset["node_id"])
            self.split2nodeids[split] = split_doc_nodes

            for storage_id, (node_id, label) in enumerate(
                zip(split_dataset["node_id"], split_dataset["y"])
            ):
                self.nodeid2split[node_id] = split
                self.nodeid2storageid[node_id] = storage_id
                self.nodeid2label[node_id] = label

        self.log("\nFinished loading needed files.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def build_graph(self):
        start_time = time.time()
        self.print_step("Building PyG Graph")

        compressed_doc_features = self.load_file("compressed_dataset")
        compressed_doc_features.set_format(type="torch", columns=["x", "y"])

        adj_matrix = self.load_file("adj_matrix")
        num_nodes = adj_matrix.shape[0]

        edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
        edge_index = coalesce(edge_index)

        node_ids = []
        features = []
        labels = []
        mask = []
        splits = []

        for node_id in sorted(self.nodeid2split.keys()):
            split = self.nodeid2split[node_id]
            storage_id = self.nodeid2storageid[node_id]

            row = compressed_doc_features[split][storage_id]

            node_ids.append(node_id)

            features.append(row["x"])

            if split == self.split:
                mask.append(torch.tensor(True, dtype=torch.bool))
                labels.append(row["y"])
            else:
                mask.append(torch.tensor(False, dtype=torch.bool))
                labels.append(torch.tensor(self.label_mask, dtype=torch.long))

            splits.append(split)

        num_docs = len(labels)

        num_users = len(self.user2nodeid)
        for node_id in sorted(list(self.user2nodeid.values())):
            node_ids.append(node_id)

            features.append(torch.zeros_like(features[0]))

            mask.append(torch.tensor(False, dtype=torch.bool))
            labels.append(torch.tensor(self.label_mask, dtype=torch.long))

            splits.append("user")

        features = torch.stack(features, dim=0)
        labels = torch.stack(labels, dim=0)
        mask = torch.stack(mask, dim=0)

        self.graph = Data(
            edge_index=edge_index,
            x=features,
            y=labels,
            mask=mask,
            idx=torch.arange(0, num_nodes),
            node_ids=node_ids,
            splits=splits,
        )

        self.log("Found:")
        self.log(f"  - {num_nodes} nodes")
        self.log(f"  - {num_docs} docs")
        self.log(f"  - {torch.sum(mask).item()} labelled docs")
        self.log(f"  - {num_users} users")
        self.log(f"  - {edge_index.shape[1]} edges")

        self.log("\nFinished building graph.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def split_graph(self):
        start_time = time.time()
        self.print_step("Splitting graph into train/val/test partitions")

        # Decide which nodes we keep based on the mode and split used ==========
        if self.structure_mode == "transductive":
            doc_nodes_to_keep = [
                graph_id
                for graph_id, split in zip(self.graph.idx, self.graph.splits)
                if split in set(self.splits)
            ]

        elif self.structure_mode == "inductive":
            doc_nodes_to_keep = [
                graph_id
                for graph_id, split in zip(self.graph.idx, self.graph.splits)
                if split == self.split
            ]

        elif self.structure_mode == "augmented":
            doc_nodes_to_keep = [
                graph_id
                for graph_id, split in zip(self.graph.idx, self.graph.splits)
                if split in {"train", self.split}
            ]

        doc_nodes_to_keep = torch.stack(doc_nodes_to_keep, dim=0)

        # Now for users
        user_nodes_to_keep = [
            graph_id
            for graph_id, split in zip(self.graph.idx, self.graph.splits)
            if split == "user"
        ]
        user_nodes_to_keep = torch.stack(user_nodes_to_keep)

        nodes_to_keep = torch.cat([doc_nodes_to_keep, user_nodes_to_keep])

        split_graph = self.graph.subgraph(nodes_to_keep)

        self.log(f"\nKept {split_graph.num_nodes}/{self.graph.num_nodes} nodes.")
        self.log(
            f"Removed {self.graph.num_nodes - split_graph.num_nodes} nodes with different split."
        )

        # Check the connected components ===============================================
        # Remove some CCs if too many
        # Most are just isolated users now
        num_docs = doc_nodes_to_keep.shape[0]

        adj = to_scipy_sparse_matrix(
            split_graph.edge_index, num_nodes=split_graph.num_nodes
        )

        num_components, component = sp.csgraph.connected_components(
            adj,
            connection="weak",
        )
        self.log(f"\nSplit generates {num_components} connected components.")

        if num_components == 1:
            self.log("Keeping all connected components.")
            subset = np.ones_like(nodes_to_keep).astype(bool)

        elif self.keep_cc == "all_docs":
            self.log("Keeping all connected components containing a document node.")
            components_to_keep = np.unique(component[:num_docs])
            subset = np.in1d(component, components_to_keep)

        elif self.keep_cc == "largest":
            self.log("Keeping only the largest connected component.")
            _, count = np.unique(component, return_counts=True)
            subset = np.in1d(component, count.argsort()[-1:])

        else:
            raise ValueError(f"CC pruning methods {self.keep_cc} not recognized.")

        prev_keep_nodes = nodes_to_keep.shape[0]
        nodes_to_keep = nodes_to_keep[subset]
        cur_keep_nodes = nodes_to_keep.shape[0]
        self.log(f"\nKept {cur_keep_nodes}/{prev_keep_nodes} nodes.")
        self.log(f"Removed {prev_keep_nodes - cur_keep_nodes} nodes in wrong CC.")

        split_graph = self.graph.subgraph(nodes_to_keep)

        nodes_to_keep_set = set(nodes_to_keep.tolist())
        split_graph.splits = [
            split for i, split in enumerate(self.graph.splits) if i in nodes_to_keep_set
        ]
        split_graph.node_ids = [
            split
            for i, split in enumerate(self.graph.node_ids)
            if i in nodes_to_keep_set
        ]

        assert len(split_graph.splits) == split_graph.x.shape[0], "Splits not subset"
        assert (
            len(split_graph.node_ids) == split_graph.x.shape[0]
        ), "Node_ids not subset"

        prev_label_counts = [0 for _ in range(len(self.labels))]

        for doc_node_id in self.split2nodeids[self.split]:
            prev_label_counts[self.nodeid2label[doc_node_id].item()] += 1

        cur_label_counts = torch.bincount(split_graph.y[split_graph.mask]).tolist()

        self.log("\nUpdated label count:")
        for label, clss in sorted(self.labels.items(), key=lambda x: x[0]):
            self.log(
                f"  - {clss:<8}: {cur_label_counts[label]}/{prev_label_counts[label]}"
                + f"[{(cur_label_counts[label] / prev_label_counts[label]) * 100:.2f}%]"
            )

        prev_num_users = sum(1 for split in self.graph.splits if split == "user")
        cur_num_users = sum(1 for split in split_graph.splits if split == "user")
        self.log(
            f"\nUsers kept {cur_num_users}/{prev_num_users} [{(cur_num_users / prev_num_users) * 100:.2f}%]"
        )

        self.graph = split_graph

        self.log("\nFinished splitting graph.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def __repr__(self):
        return f"SocialGraph(mode={self.structure_mode}, split={self.split}, keep_cc={self.keep_cc})"

    def __str__(self):
        return self.__repr__()
