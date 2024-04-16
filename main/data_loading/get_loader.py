from torch.utils.data import DataLoader
import torch_geometric

from data_prep.post_processing import SocialGraph
from data_loading.social_graph_dataset import IterableSocialGraph
from data_loading.batched_user_neighbourhood import BatchedKHopUserNeighbourhood
from data_loading.episodic_batched_khop_neighbourdhood import (
    EpisodicKHopNeighbourhoodSocialGraph,
)
from data_loading.episodic_batched_doc_only_neighbourhood import (
    EpisodicKHopDocsOnlySocialGraph,
)


def get_dataset(args, split: str, load: bool = False):
    fold = args["fold"]
    assert fold in range(max(1, args["data"]["num_splits"]))

    if args["structure"]["structure"] == "full":
        if load:
            cls_init = IterableSocialGraph.load
        else:
            cls_init = SocialGraph

        graph_dataset = cls_init(
            args,
            # Social graph kwargs
            structure_mode=args["structure"]["structure_mode"],
            cur_fold=fold,
            split=split,
            keep_cc=args["structure"]["keep_cc"],
            version=args["version"],
        )

    elif args["structure"]["structure"] == "khop":
        if split == "train":
            if load:
                cls_init = BatchedKHopUserNeighbourhood.load
            else:
                cls_init = BatchedKHopUserNeighbourhood

            graph_dataset = cls_init(
                args,
                # Social graph kwargs
                structure_mode=args["structure"]["structure_mode"],
                cur_fold=fold,
                split=split,
                keep_cc=args["structure"]["keep_cc"],
                version=args["version"],
                # Batched khop kwargs
                min_k_hop=args["structure"]["min_k_hop"],
                max_k_hop=args["structure"]["max_k_hop"],
                labels_per_graph=args["structure"]["labels_per_graph"],
                node_weights_dist=args["structure"]["node_weights_dist"],
                label_dist=args["structure"]["label_dist"],
                max_nodes_per_subgraph=args["structure"]["max_nodes_per_subgraph"],
                walk_length=args["structure"]["walk_length"],
                batch_size=args["structure"]["batch_size"],
                max_samples_per_partition=args["structure"][
                    "max_samples_per_partition"
                ],
            )

        else:
            if load:
                cls_init = IterableSocialGraph.load
            else:
                cls_init = SocialGraph

            graph_dataset = cls_init(
                args,
                # Social graph kwargs
                structure_mode=args["structure"]["structure_mode"],
                cur_fold=fold,
                split=split,
                keep_cc=args["structure"]["keep_cc"],
                version=args["version"],
            )

    elif args["structure"]["structure"] == "episodic_khop":
        if load:
            cls_init = EpisodicKHopNeighbourhoodSocialGraph.load
        else:
            cls_init = EpisodicKHopNeighbourhoodSocialGraph

        graph_dataset = cls_init(
            args,
            # Social graph kwargs
            structure_mode=args["structure"]["structure_mode"],
            cur_fold=fold,
            split=split,
            keep_cc=args["structure"]["keep_cc"],
            version=args["version"],
            # Batched khop kwargs
            doc_k_hop=args["structure"]["doc_k_hop"],
            min_k_hop=args["structure"]["min_k_hop"],
            max_k_hop=args["structure"]["max_k_hop"],
            node_weights_dist=args["structure"]["node_weights_dist"],
            label_dist=args["structure"]["label_dist"],
            max_nodes_per_subgraph=args["structure"]["max_nodes_per_subgraph"],
            walk_length=args["structure"]["walk_length"],
            max_samples_per_partition=(
                args["structure"]["max_samples_per_partition"]
                if split == "train"
                else args["structure"]["max_samples_per_eval_partition"]
            ),
            # Episodic kwargs
            k=args["k"],
            shots=args["shots"],
            prop_query=args["structure"]["prop_query"] if split == "train" else 0.0,
        )

    elif args["structure"]["structure"] == "episodic_doc_only_khop":
        if load:
            cls_init = EpisodicKHopDocsOnlySocialGraph.load
        else:
            cls_init = EpisodicKHopDocsOnlySocialGraph

        graph_dataset = cls_init(
            args,
            # Social graph kwargs
            structure_mode=args["structure"]["structure_mode"],
            cur_fold=fold,
            split=split,
            keep_cc=args["structure"]["keep_cc"],
            version=args["version"],
            max_samples_per_partition=(
                args["structure"]["max_samples_per_partition"]
                if split == "train"
                else args["structure"]["max_samples_per_eval_partition"]
            ),
            # Batched khop kwargs
            doc_k_hop=args["structure"]["doc_k_hop"],
            node_weights_dist=args["structure"]["node_weights_dist"],  # Episodic kwargs
            k=args["k"],
            shots=args["shots"],
            prop_query=args["structure"]["prop_query"] if split == "train" else 0.0,
        )

    else:
        raise ValueError(
            "`structure.structure` must be one of [`full`, `khop`, `episodic_khop`, `episodic_doc_only_khop`]"
        )

    graph_dataset.change_data_dir(args["data"])

    return graph_dataset


def get_dataloader(args, split: str, **dataloader_kwargs):
    fold = args["fold"]
    assert fold in range(max(1, args["data"]["num_splits"]))

    dataset = get_dataset(args, split, load=True)

    if args["structure"]["structure"] == "full":
        loader = torch_geometric.loader.DataLoader(
            dataset, batch_size=1, **dataloader_kwargs
        )

    elif args["structure"]["structure"] == "khop":
        if split == "train":
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                collate_fn=dataset.collate_fn,
                **dataloader_kwargs,
            )

        else:
            loader = torch_geometric.loader.DataLoader(
                dataset, batch_size=1, **dataloader_kwargs
            )

    elif args["structure"]["structure"] in {"episodic_khop", "episodic_doc_only_khop"}:
        if split == "train":
            loader = DataLoader(
                dataset,
                batch_size=1,
                collate_fn=dataset.collate_fn_train,
                **dataloader_kwargs,
            )

        else:
            loader = DataLoader(
                dataset,
                batch_size=1,
                collate_fn=dataset.collate_fn_eval,
                **dataloader_kwargs,
            )

    else:
        raise ValueError(
            "`structure.structure` must be one of [`full`, `khop`, `episodic_khop`, `episodic_doc_only_khop`]"
        )

    return loader
