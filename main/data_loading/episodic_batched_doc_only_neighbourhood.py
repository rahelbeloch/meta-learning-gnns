import typing
from copy import deepcopy
import time
from math import ceil

import torch
from torch.utils.data import IterableDataset

from data_prep.post_processing import SocialGraph
from data_loading.batched_doc_neighbourhood import BatchedKHopDocumentNeighbourhood
from utils.logging import calc_elapsed_time
from utils.rng import stochastic_method


class EpisodicKHopDocsOnlySocialGraph(SocialGraph, IterableDataset):
    """
    An adapted variant of `EpisodicKHopNeighbourhoodSocialGraph` that does not use
    user-base sampling to generate support graphs. Instead, these are document based
    graphs, sampled IID from the underlying graph
    """

    def __init__(
        self,
        args,
        structure_mode,
        cur_fold,
        split,
        k: int,
        shots: typing.List[int],
        prop_query: int,
        doc_k_hop: int,
        node_weights_dist: str,
        max_samples_per_partition: int,
        prefix: typing.Optional[str] = None,
        _doc_limit: int = -1,
        **super_kwargs,
    ):
        super().__init__(
            args=args,
            structure_mode=structure_mode,
            cur_fold=cur_fold,
            split=split,
            **super_kwargs,
        )

        self.k = k
        self.all_k = shots
        self.prop_query = prop_query

        self._args = args

        self.node_weights_dist = node_weights_dist
        self.max_samples_per_partition = max_samples_per_partition

        # Args and kwargs for `BatchedKHopDocumentNeighbourhood`
        # Used for sampling support set
        self.doc_k_hop = doc_k_hop
        self._doc_limit = _doc_limit

        self.prefix = prefix

    def __repr__(self):
        _repr = f"EpisodicNonGraph(mode={self.structure_mode}, split={self.split}, k_shot={self.k}, prop_query={self.prop_query}"  # noqa: E501

        if self.prefix is not None:
            return _repr + f", version={self.prefix})"
        else:
            return _repr + ")"

    def split_graph(self):
        super().split_graph()

        start_time = time.time()
        self.print_step("Splitting labelled nodes into support and query subsets")

        # Figure out which nodes belong to which label
        rng = torch.random.manual_seed(self.seed)

        split_nodes = [
            self.graph.idx[self.graph.y == l] for l in sorted(list(self.labels.keys()))
        ]

        if self.split != "train":
            self.log(
                "For non train splits, no need to generate disjoint support/query sets."
            )
            self.support_nodes = []
            self.query_nodes = []
            for l in sorted(list(self.labels.keys())):
                label_nodes = split_nodes[l][
                    torch.randperm(split_nodes[l].shape[0], generator=rng)
                ]

                self.query_nodes.append([])
                self.support_nodes.append(label_nodes)

            self.support_graph = deepcopy(self.graph.detach())

            return

        # Count em up
        class_counts = torch.tensor(list(map(lambda x: x.shape[0], split_nodes)))
        class_prob = class_counts / class_counts.sum()

        self.log(
            f"Current class counts: {class_counts}, {[float(f'{pp*100:.2f}') for pp in class_prob]} %"  # noqa: E501
        )

        # Find split size ======================================================
        # Try and find an approximate subset size for the query set
        approx_query_size = (self.prop_query * class_counts).int()
        self.log(f"Approx query class count: {approx_query_size}")

        # Now get the actual number
        # Should be divisible by all shots
        self.log("\nFinding class sizes for query set")
        class_counts_divisible = [
            self.get_evenly_divisible_nr(count, self.all_k)
            for count in approx_query_size.tolist()
        ]

        for c, (approx_query_c, usable_query_c) in enumerate(
            zip(approx_query_size, class_counts_divisible)
        ):
            self.log(f"{c} | {approx_query_c} -> {usable_query_c}")

        # We want to keep the max class count bounded in all episodes
        max_to_remain_below_k_times_min_class = min(class_counts_divisible) * min(
            self.all_k
        )

        class_counts_divisible = torch.tensor(class_counts_divisible)
        if torch.any(class_counts_divisible > max_to_remain_below_k_times_min_class):
            self.log(
                f"\nClipping max class count to {max_to_remain_below_k_times_min_class} samples"  # noqa: E501
            )

            class_counts_divisible = torch.clip(
                class_counts_divisible,
                min=0,
                max=max_to_remain_below_k_times_min_class,
            )

        else:
            self.log("\nOther classes within reasonable bound of found min class size.")

        # Find out how much we deviate from the natural class distribution =====
        class_counts_divisible = class_counts_divisible.int()
        class_prob_divisible = class_counts_divisible / class_counts_divisible.sum()

        self.log(
            f"\nUpdated class counts: {class_counts_divisible.tolist()}, {[float(f'{pp.item()*100:.2f}') for pp in class_prob_divisible]} %"  # noqa: E501
        )

        self.log("Divergence from actual class distribution")
        self.log(
            f"RMSE:{torch.sqrt(torch.mean(torch.pow((class_prob - class_prob_divisible)*100, 2))).item():>5.2f}%"  # noqa: E501
        )
        self.log(
            f"KLD:{torch.sum(class_prob_divisible * (torch.log(class_prob_divisible) - torch.log(class_prob))):>10.2e}"  # noqa: E501
        )

        max_episodes = torch.min(class_counts_divisible)
        self.log(f"\n#Episodes per epoch @ k=min({self.all_k}): {max_episodes}")
        self.log(f"#Episodes per epoch @ k={self.k}: {(max_episodes / self._g).int()}")

        self.class_counts_divisible = class_counts_divisible
        self.max_episodes = max_episodes.item()

        # Split nodes into support and query subsets ===========================
        self.support_nodes = []
        self.query_nodes = []
        for l, class_count in enumerate(self.class_counts_divisible.tolist()):
            label_nodes = split_nodes[l][
                torch.randperm(split_nodes[l].shape[0], generator=rng)
            ]

            self.query_nodes.append(label_nodes[:class_count])
            self.support_nodes.append(label_nodes[class_count:])

        # Copy the graph into a separate support and query version =============
        if self.structure_mode == "transductive":
            # Assuming graph has already been split, nothing to do here

            self.support_graph = deepcopy(self.graph.detach())
            self.query_graph = deepcopy(self.graph.detach())

        elif self.structure_mode == "augmented" or self.structure_mode == "inductive":
            user_nodes = torch.tensor(
                [
                    graph_id
                    for graph_id, split in zip(self.graph.idx, self.graph.splits)
                    if split == "user"
                ]
            )

            # Split the support graph
            support_graph_nodes_to_keep = torch.sort(
                torch.cat(
                    [
                        torch.cat(self.support_nodes),
                        user_nodes,
                    ]
                )
            ).values

            support_local_graph_idx_to_keep = torch.searchsorted(
                self.graph.idx,
                support_graph_nodes_to_keep,
            )

            self.support_graph = deepcopy(self.graph.detach()).subgraph(
                support_local_graph_idx_to_keep
            )

            nodes_to_keep_set = set(support_local_graph_idx_to_keep.tolist())
            self.support_graph.splits = [
                split
                for i, split in enumerate(self.support_graph.splits)
                if i in nodes_to_keep_set
            ]
            self.support_graph.node_ids = [
                split
                for i, split in enumerate(self.support_graph.node_ids)
                if i in nodes_to_keep_set
            ]

            # Split the query graph
            query_graph_nodes_to_keep = torch.cat(
                [
                    torch.sort(torch.cat(self.query_nodes)).values,
                    user_nodes,
                ]
            )

            query_local_graph_idx_to_keep = torch.searchsorted(
                self.graph.idx,
                query_graph_nodes_to_keep,
            )

            self.query_graph = deepcopy(self.graph.detach()).subgraph(
                query_local_graph_idx_to_keep
            )

            nodes_to_keep_set = set(query_local_graph_idx_to_keep.tolist())
            self.query_graph.splits = [
                split
                for i, split in enumerate(self.query_graph.splits)
                if i in nodes_to_keep_set
            ]
            self.query_graph.node_ids = [
                split
                for i, split in enumerate(self.query_graph.node_ids)
                if i in nodes_to_keep_set
            ]

        # Label masking ========================================================
        # Mask the labels in the support graph
        # Should not include labels belonging to support subset
        self.support_graph.y = torch.full_like(
            self.support_graph.idx,
            fill_value=self.label_mask,
        )
        self.support_graph.mask = torch.zeros_like(self.support_graph.idx)

        num_docs = sum(map(len, split_nodes))
        num_docs_kept = 0
        for l, node_ids in enumerate(self.support_nodes):
            local_mask = torch.isin(
                self.support_graph.idx, node_ids, assume_unique=True
            )

            num_docs_kept += local_mask.sum().item()

            self.support_graph.y.masked_fill_(local_mask, value=l)

            self.support_graph.mask.masked_fill_(local_mask, value=1.0)

        self.log(
            f"\nLabels on {num_docs_kept}/{num_docs} doc nodes for support subset."
        )

        # Mask the labels in the query graph
        # Should not include labels belonging to support subset
        num_docs_kept = 0
        self.query_graph.mask = torch.zeros_like(self.query_graph.idx)
        self.query_graph.y = torch.full_like(
            self.query_graph.idx,
            fill_value=self.label_mask,
        )

        for l, node_ids in enumerate(self.query_nodes):
            local_mask = torch.isin(self.query_graph.idx, node_ids, assume_unique=True)

            num_docs_kept += local_mask.sum().item()

            self.query_graph.y.masked_fill_(local_mask, value=l)

            self.query_graph.mask.masked_fill_(local_mask, value=1.0)
        self.log(f"Labels on {num_docs_kept}/{num_docs} doc nodes for query subset.")

        # Determine episodes and dynamic query shot ============================
        # Find the number of episodes for current k
        # Remainder defined to be 0, but still need to cast to int
        cur_episodes = int(self.max_episodes / self._g)

        # Make some buckets with 0 per class
        self.query_episode_samples = [[] for _ in range(cur_episodes)]

        # Divvy up the query samples over the buckets
        for c in self.labels.keys():
            idx = torch.randperm(self.max_episodes, generator=rng).tolist()

            iteration = 0
            samples_to_distribute = self.query_nodes[c].tolist()

            while len(samples_to_distribute) > 0:
                bucket = int(iteration % cur_episodes)

                sample = samples_to_distribute.pop(-1)

                self.query_episode_samples[idx[bucket]] += [sample]

                iteration += 1

        self.log("\nFinished generating support and query subgraphs.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    @property
    def _g(self):
        # The growth factor of current k to minimum k
        return self.k / min(self.all_k)

    @staticmethod
    def get_evenly_divisible_nr(nr, divisors, max_checks=100):
        current_nr = nr

        times_checked = 0
        while True:
            if times_checked >= max_checks:
                break

            times_checked += 1
            divisible = True
            for d in divisors:
                if current_nr % d != 0:
                    divisible = False
                    break

            if divisible:
                return current_nr
            else:
                current_nr -= 1

    @stochastic_method
    def generate_batches(self, num_workers: int = 0):
        self.log("\n>>> Generating support batches <<<\n")

        self.support_graph_dataset = BatchedKHopDocumentNeighbourhood(
            args=self._args,
            structure_mode=self.structure_mode,
            cur_fold=self.cur_fold,
            split=self.split,
            k_hop=self.doc_k_hop,
            batch_size=2 * self.k,
            node_weights_dist=self.node_weights_dist,
            prefix="meta_train_support",
            _doc_limit=self._doc_limit,
            version=self.version,
        )

        self.support_graph_dataset.graph = self.support_graph
        self.support_graph_dataset.valid_docs = set(
            torch.cat(self.support_nodes).tolist()
        )
        self.support_graph_dataset.prep()

        self.support_graph_dataset._generate_node_weights()

        # Determine which samples go in the support dataset ====================
        # Static k though, which helps
        def sample_episode_document_nodes(label_nodes, label_weights):
            sampled_document_nodes = []
            for i in range(self.num_classes):
                # Check how many nodes we can sample
                num_non_zero = label_weights[i].sum()

                if num_non_zero < self.k:
                    # Reset their weights if not enough samples remain
                    # for a full episode
                    label_weights[i] = torch.ones_like(label_nodes[i])
                    num_non_zero = label_weights[i].shape[0]

                # Sample uniformly from the remaining nodes
                sampled_idx = torch.multinomial(
                    label_weights[i] / num_non_zero,
                    num_samples=self.k,
                )

                # Add the nodes to a list
                sampled_document_nodes.append(label_nodes[i][sampled_idx])

                # Set the probability of those nodes being sampled again to 0
                label_weights[i][sampled_idx] = 0

            episode_doc_nodes = torch.cat(sampled_document_nodes)

            return episode_doc_nodes, label_weights

        support_label_nodes = [
            torch.where(self.support_graph.y == label)[0]
            for label in range(self.num_classes)
        ]
        support_label_weights = [
            torch.ones_like(l_nodes) for l_nodes in support_label_nodes
        ]

        total_support_episodes = ceil(max(map(len, support_label_nodes)) / self.k)

        self.support_episode_samples = []
        for i in range(total_support_episodes):
            if (
                self.max_samples_per_partition != -1
                and i == self.max_samples_per_partition
            ):
                break

            (
                support_episode_nodes,
                support_label_weights,
            ) = sample_episode_document_nodes(
                support_label_nodes,
                support_label_weights,
            )

            support_episode_node_idx = self.support_graph_dataset.graph.idx[
                support_episode_nodes
            ]

            self.support_episode_samples.append(support_episode_node_idx)

        self.support_graph_dataset.generate_batches(
            num_workers=num_workers,
            batches=self.support_episode_samples[: self._doc_limit],
        )

        # Only need a query dataset if training
        # During testing, we evaluate on the entire graph at once
        if self.split == "train":
            self.log("\n>>> Generating query batches <<<\n")

            self.query_graph_dataset = BatchedKHopDocumentNeighbourhood(
                args=self._args,
                structure_mode=self.structure_mode,
                cur_fold=self.cur_fold,
                split=self.split,
                k_hop=self.doc_k_hop,
                batch_size=1,
                node_weights_dist=self.node_weights_dist,
                prefix="meta_train_query",
                _doc_limit=self._doc_limit,
                version=self.version,
            )

            self.query_graph_dataset.graph = self.query_graph
            self.query_graph_dataset.valid_docs = set(
                torch.cat(self.query_nodes).tolist()
            )
            self.query_graph_dataset._generate_node_weights()

            self.query_graph_dataset.generate_batches(
                num_workers=0, batches=self.query_episode_samples[: self._doc_limit]
            )

    def partition_into_batches(self, *args, **kwargs):
        pass

    def __len__(self):
        if self.split == "train":
            return self.max_episodes
        else:
            return len(self.support_graph_dataset)

    @stochastic_method
    def __iter__(self):
        if self.split == "train":
            n_support_batches = len(self.support_graph_dataset)

            support_idx = torch.randperm(n_support_batches).tolist()

            n_query_batches = len(self.query_graph_dataset)

            query_idx = torch.randperm(n_query_batches).tolist()

            for i in range(self.max_episodes):
                support_graph = self.support_graph_dataset[support_idx[i]]

                query_graph = self.query_graph_dataset[query_idx[i]]

                yield support_graph, query_graph

        else:
            # If the current split is not train, the query set is just the entire graph
            # No need to perform subgraph sampling

            n_support_batches = len(self.support_graph_dataset)

            support_idx = torch.randperm(n_support_batches).tolist()

            for i in range(n_support_batches):
                support_graph = self.support_graph_dataset[support_idx[i]]

                query_graph = None

                yield support_graph, query_graph

    @stochastic_method
    def collate_fn_train(self, batch):
        support_graph, query_graph = batch[0]

        support_graph = self.support_graph_dataset.collate_fn([support_graph])
        query_graph = self.query_graph_dataset.collate_fn([query_graph])

        return (
            support_graph,
            query_graph,
        )

    def collate_fn_eval(self, batch):
        support_batch, _ = batch[0]

        graph_idx = support_batch["graph_idx"]

        support_graph = self.support_graph_dataset.collate_fn([support_batch])

        # Figure out which labels are being povided in the support set
        labelled_support_nodes_graph_idx = torch.isin(
            self.graph.idx, graph_idx[support_graph.mask]
        )

        query_graph = deepcopy(self.graph)

        # Mask out the labelled support set labels
        # This avoids biasing the results to seen examples
        query_graph.y.masked_fill_(
            labelled_support_nodes_graph_idx,
            self.label_mask,
        )

        query_graph.mask.masked_fill_(
            labelled_support_nodes_graph_idx,
            False,
        )

        return (
            support_graph,
            query_graph,
        )
