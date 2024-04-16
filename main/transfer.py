import shutil
from pathlib import Path
import re
import os
import json
import pickle
import csv
from copy import deepcopy

import hydra
from omegaconf import OmegaConf
import datasets
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

from data_prep.graph_io import GraphIO
from data_prep.content_processing import ContentProcessor
from data_prep.post_processing import PostProcessing
from data_prep.post_processing.feature_extraction import FeatureExtractorDataset
from data_loading.get_loader import get_dataset, get_dataloader
from models import GatNonEpisodic, GatMAML, GatPrototypical
from utils.logging import get_results_dir
from utils.graph_functions import avg_pool_doc_neighbours


def print_step(string):
    print("\n" + "=" * 100)
    print(f"{string.upper():^100s}")
    print("=" * 100 + "\n")


def find_checkpoint_file_path(args: dict, checkpoint: str):
    checkpoint_dir = Path(args["log_path"], args["checkpoint_dir"], checkpoint)

    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint dir does not exist:\n\t{checkpoint_dir}")

    checkpoints_found = dict()
    for f in (checkpoint_dir / "checkpoints").glob("*.ckpt"):
        step = re.search(r"step\=([0-9]+)", str(f)).group(1)
        checkpoints_found[step] = f

    if len(checkpoints_found) == 0:
        raise ValueError(f"No checkpoint file found:\n\t{checkpoint_dir}")
    elif len(checkpoints_found) == 1:
        step = list(checkpoints_found.keys())[0]
    elif len(checkpoints_found) > 1:
        steps = list(checkpoints_found.keys())
        if args["checkpoint_strategy"] == "earliest":
            step = min(steps)
        elif args["checkpoint_strategy"] == "latest":
            step = max(steps)

    checkpoint_fp = checkpoints_found[step]

    return checkpoint_fp


def init_trainer_and_logger(args):
    if args.get("trainer", None) is not None:
        trainer_kwargs = args["trainer"]
    else:
        trainer_kwargs = {}

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        move_metrics_to_cpu=True,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
        inference_mode=False,
        **trainer_kwargs,
    )

    return trainer


def dump_results(args, prefix, results_dict=None, hparams=None, preds=None, gt=None):
    results_dir = get_results_dir(
        results_dir=args["results_path"],
        data_args=args["data"],
        structure_args=args["structure"],
        checkpoint=(
            args["checkpoint"]
            if args["checkpoint_name"] is None
            else args["checkpoint_name"]
        ),
        version=args["version"],
        fold=args["fold"],
    )
    os.makedirs(results_dir, exist_ok=True)

    if results_dict is not None:
        results_name = f"{prefix}"

        if args["learning_algorithm"]["meta_learner"] != "non_episodic":
            results_name += "_eval[episodic]"
            results_name += f"_k[{args['learning_algorithm']['k']}]"
            results_name += (
                f"_nupdates[{args['learning_algorithm']['n_inner_updates']}]"
            )
            results_name += f"_innerlr[{args['learning_algorithm']['lr_inner']}]"
            results_name += f"_headlr[{args['learning_algorithm']['head_lr_inner']}]"
            results_name += (
                f"_classweights[{args['learning_algorithm']['class_weights']}]"
            )
            results_name += f"_budget[{args['structure']['max_nodes_per_subgraph']}]"

        else:
            results_name += "_eval[standard]"

    else:
        results_name = "hparams"

    results_fp = results_dir / results_name

    print(f"Saving {prefix} results to:\n\t{results_fp}")
    if results_fp.exists():
        print(">>WARNING<< RESULTS FILEPATH ALREADY EXISTS. OVERWRITING >>WARNING<<")

    if results_dict is not None:
        with open(str(results_fp) + ".json", "w") as fp:
            json.dump(results_dict, fp)

    elif hparams is not None:
        with open(str(results_fp) + ".pickle", "wb") as fp:
            pickle.dump(hparams, fp)

    elif preds is not None and gt is not None:
        torch.save(
            preds.detach().cpu(),
            f=str(results_fp) + "preds.pt",
        )

        torch.save(
            gt.detach().cpu(),
            f=str(results_fp) + "gt.pt",
        )

        csv_output = torch.stack([preds.detach().cpu(), gt.detach().cpu()], dim=1)
        with open(str(results_fp) + "_out.csv", "w") as f:
            wr = csv.writer(f)
            wr.writerows(csv_output.tolist())

    else:
        raise ValueError("Some value must be provided")


@hydra.main(version_base=None, config_path="./config", config_name="transfer")
def transfer(args):
    print("*" * 100)
    if args["print_config"]:
        print(OmegaConf.to_yaml(args, resolve=True))
    else:
        "Config loaded but not printing."
    print("*" * 100)

    os.makedirs(args["log_path"], exist_ok=True)
    os.makedirs(args["results_path"], exist_ok=True)

    args = OmegaConf.to_container(args, resolve=True)

    # ==========================================================================
    # Device
    # ==========================================================================
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        cur_device = torch.cuda.current_device()
        print("Current device:", cur_device, torch.cuda.get_device_name(cur_device))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args["seed"])

    # ==========================================================================
    # Find the right checkpoint
    # ==========================================================================
    if args.get("checkpoint_address_file", None) is not None:
        with open(args["checkpoint_address_file"], "r") as f:
            checkpoint = f.readlines()[-1]
            args["checkpoint"] = checkpoint
    elif args.get("checkpoint", None) is not None:
        checkpoint = args["checkpoint"]
    else:
        raise ValueError(
            "Either the `checkpoint` argument must be specified, or `checkpoint_address_file` must be."
        )

    ckpt_fp = find_checkpoint_file_path(
        args,
        checkpoint,
    )

    print(f"Loading from:\n\t{ckpt_fp}")

    checkpoint_obj = torch.load(
        ckpt_fp,
        map_location=torch.device("cpu"),
    )

    origin_training_data_params = checkpoint_obj["hyper_parameters"][
        "training_data_params"
    ]

    del checkpoint_obj

    if origin_training_data_params["dataset"] == args["data"]["dataset"]:
        raise ValueError(
            "Training data of the checkpoint matches provided transfer dataset. Transfer to same dataset makes no sense."
        )

    if not args["reset_checkpoint"]:
        version = f"transfer_{checkpoint}"
    else:
        version = f"reset_{checkpoint}_{args['checkpoint_reset_seed']}"

    if args["avg_pool_doc_neighbours"]:
        version += "_avg_pool_user_init"

    # The feature extraction now comes from the checkpoint
    args_data_updated = deepcopy(args["data"])

    args_data_updated["feature_type"] = origin_training_data_params["feature_type"]
    args_data_updated["compression"] = origin_training_data_params["compression"]
    args_data_updated["vocab_size"] = origin_training_data_params["vocab_size"]
    args_data_updated["compressed_size"] = origin_training_data_params[
        "compressed_size"
    ]
    args_data_updated["use_joint_vocab"] = origin_training_data_params[
        "use_joint_vocab"
    ]

    # ==========================================================================
    # Data Transfer
    # ==========================================================================
    if not args["skip_data_transfer"]:
        # This is just for dev purposes
        # origin_training_data_params["fold"] = 0

        origin_training_data_params["processed_data_dir"] = args["data"][
            "processed_data_dir"
        ]
        origin_training_data_params["tsv_dir"] = args["data"]["tsv_dir"]
        origin_training_data_params["complete_dir"] = args["data"]["complete_dir"]
        origin_training_data_params["processed_dir"] = args["data"]["processed_dir"]
        origin_training_data_params["structure_dir"] = args["data"]["structure_dir"]

        fold = origin_training_data_params["fold"]

        # ==========================================================================
        # (Re)-Tokenization
        # ==========================================================================
        # Get the tokenizer applied to the content dataset
        origin_content_processor = ContentProcessor(
            origin_training_data_params,
            enforce_raw=False,
            version=args["orig_version"],
        )

        tokenizer = origin_content_processor.load_file("tokenizer")

        # Get the data belonging to the transfer dataset
        # Already has text tokenized, but we'll overwrite that
        transfer_content_processor = ContentProcessor(
            args["data"]
            | {
                "num_splits": 5
                if args["data"]["num_splits"] == 0
                else args["data"]["num_splits"]
            },
            enforce_raw=False,
        )

        # Re-tokenize the transfer data, but now using the origin data's tokenizer
        transfer_data = transfer_content_processor.load_file("doc_dataset")
        transfer_data.map(lambda x: tokenizer(x["raw_text"]))

        # ==============================================================================
        # Feature extraction
        # ==============================================================================
        # Load necessary model and data ================================================
        # First 'build' the doc dataset using transfer docs now
        # Should process all docs from all folds/splits at once
        transfer_doc_dataset = FeatureExtractorDataset(
            args=args["data"],
            cur_fold=0,
            split="train",
            version=args["version"],
        )
        idx = [
            element
            for sublist in transfer_doc_dataset.fold_idx.values()
            for element in sublist
        ]

        transfer_doc_dataset.data = transfer_data.select(idx)
        transfer_doc_dataset.idx = [
            i for i in range(transfer_doc_dataset.data.num_rows)
        ]

        fold_idx_to_dataset_idx_mapping = {
            id: transfer_doc_dataset.idx[i] for i, id in enumerate(idx)
        }

        doc2nodeid = transfer_doc_dataset.load_file("doc2nodeid")
        transfer_doc_dataset.node_ids = torch.tensor(
            [doc2nodeid[doc_id] for doc_id in transfer_doc_dataset.data["doc_id"]]
        )

        # Then load in the origin dataset feature extractor
        origin_post_processor = PostProcessing(
            args=origin_training_data_params,
            cur_fold=fold,
            version=args["orig_version"],
            processed_or_structured="processed",
        )

        feature_extractor = origin_post_processor.load_file("feature_extractor")
        feature_extractor.to(device)
        feature_extractor.eval()

        # Compression ==================================================================
        loader = DataLoader(
            transfer_doc_dataset,
            batch_size=args["feature_extraction"]["batch_size"],
            collate_fn=transfer_doc_dataset.collate_fn,
            shuffle=False,
        )

        prog_bar_updates = max(len(loader) // args["feature_extraction"]["prog_bar"], 1)

        node_ids_storage = []
        embeddings_storage = []
        targets_storage = []

        prev_i = 0
        pbar = tqdm(
            loader,
            desc="Compressing all transfer docs with origin feature extractor",
            total=len(loader),
        )
        for i, (node_ids, x, y) in enumerate(pbar):
            x = {k: v.to(device) for k, v in x.items()}

            with torch.inference_mode():
                compressed_x = feature_extractor.compress(x)

            node_ids_storage.append(node_ids)
            embeddings_storage.append(compressed_x.cpu().detach())
            targets_storage.append(y)

            if i == 0 or i % prog_bar_updates == 0 or i == len(loader) - 1:
                pbar.update(i - prev_i)
                prev_i = i

        node_ids_storage = torch.cat(node_ids_storage, dim=0)
        embeddings_storage = torch.cat(embeddings_storage, dim=0)
        targets_storage = torch.cat(targets_storage, dim=0)

        # ==========================================================================
        # FINAL DATA CONVERSION & STRUCTURING
        # ==========================================================================
        # COMPLETE =================================================================
        # Populate with mock split indices
        new_split_idx = []
        old_split_idx = transfer_doc_dataset.load_file(file_type="split_idx")
        for fold_idx in old_split_idx:
            if args["data"]["num_splits"] > 0:
                new_test_idx = fold_idx["train"] + fold_idx["val"]
                new_val_idx = fold_idx["test"]
            else:
                new_test_idx = fold_idx["train"] + fold_idx["val"] + fold_idx["test"]
                new_val_idx = []

            new_split_idx.append(
                {
                    "train": [],
                    "val": new_val_idx,
                    "test": new_test_idx,
                }
            )

        graph_io = GraphIO(
            args_data_updated | {"num_splits": args["data"]["num_splits"]},
            version=version,
            enforce_raw=False,
        )

        graph_io.save_file(file_type="split_idx", obj=new_split_idx)

        # Transfer some of the needed files from complete of the transfer dataset
        # to the new adjusted transfer dataset
        shutil.copy(
            transfer_doc_dataset._get_file_name("adj_matrix"),
            graph_io.data_complete_path(),
        )
        shutil.copy(
            transfer_doc_dataset._get_file_name("doc2nodeid"),
            graph_io.data_complete_path(),
        )
        shutil.copy(
            transfer_doc_dataset._get_file_name("user2nodeid"),
            graph_io.data_complete_path(),
        )

        for fold in range(max(1, args["data"]["num_splits"])):
            # PROCESSED ============================================================
            # Populate with mock compressed datasets

            val_data_idx = list(
                map(
                    lambda x: fold_idx_to_dataset_idx_mapping[x],
                    new_split_idx[fold]["val"],
                )
            )
            test_data_idx = list(
                map(
                    lambda x: fold_idx_to_dataset_idx_mapping[x],
                    new_split_idx[fold]["test"],
                )
            )

            split_datasets = datasets.DatasetDict(
                {
                    "train": datasets.Dataset.from_dict(
                        {
                            "node_id": [],
                            "x": [],
                            "y": [],
                        },
                        split="train",
                    ),
                    "val": datasets.Dataset.from_dict(
                        {
                            "node_id": node_ids_storage[val_data_idx]
                            if args["data"]["num_splits"] > 0
                            else [],
                            "x": embeddings_storage[val_data_idx]
                            if args["data"]["num_splits"] > 0
                            else [],
                            "y": targets_storage[val_data_idx]
                            if args["data"]["num_splits"] > 0
                            else [],
                        },
                        split="val",
                    ),
                    "test": datasets.Dataset.from_dict(
                        {
                            "node_id": node_ids_storage[test_data_idx],
                            "x": embeddings_storage[test_data_idx],
                            "y": targets_storage[test_data_idx],
                        },
                        split="test",
                    ),
                }
            )

            post_processor = PostProcessing(
                args_data_updated | {"num_splits": args["data"]["num_splits"]},
                version=version,
                cur_fold=fold,
                processed_or_structured="processed",
            )

            post_processor.save_file(file_type="compressed_dataset", obj=split_datasets)

            # STRUCTURED ===========================================================
            # Structure the datasets
            # Per fold:
            # One large one for testing
            # One small one for validation

            # Process the validation split
            if args["data"]["num_splits"] > 0:
                graph_dataset = get_dataset(
                    {
                        "k": args["k"],
                        "shots": args["shots"],
                        "data": args_data_updated
                        | {
                            "num_splits": args["data"]["num_splits"],
                        },
                        "structure": args["structure"],
                        "fold": fold,
                        "version": version,
                    },
                    "val",
                    load=False,
                )

                graph_dataset.prep()
                graph_dataset.build_graph()
                graph_dataset.split_graph()

                if args["avg_pool_doc_neighbours"]:
                    if hasattr(graph_dataset, "graph"):
                        graph_dataset.graph = avg_pool_doc_neighbours(
                            graph_dataset.graph
                        )

                    if hasattr(graph_dataset, "support_graph"):
                        graph_dataset.support_graph = avg_pool_doc_neighbours(
                            graph_dataset.support_graph
                        )

                    if hasattr(graph_dataset, "query_graph"):
                        graph_dataset.query_graph = avg_pool_doc_neighbours(
                            graph_dataset.query_graph
                        )

                if callable(getattr(graph_dataset, "generate_batches", None)):
                    graph_dataset.partition_into_batches()
                    graph_dataset.generate_batches(
                        num_workers=args["structure"]["num_workers"]
                    )

                graph_dataset.save()

                del graph_dataset

            # Process the test split
            graph_dataset = get_dataset(
                {
                    "k": args["k"],
                    "shots": args["shots"],
                    "data": args_data_updated
                    | {
                        "num_splits": args["data"]["num_splits"],
                    },
                    "structure": args["structure"],
                    "fold": fold,
                    "version": version,
                },
                "train",
                load=False,
            )

            graph_dataset.prep()
            graph_dataset.build_graph()
            graph_dataset.split_graph()

            if args["avg_pool_doc_neighbours"]:
                if hasattr(graph_dataset, "graph"):
                    graph_dataset.graph = avg_pool_doc_neighbours(graph_dataset.graph)

                if hasattr(graph_dataset, "support_graph"):
                    graph_dataset.support_graph = avg_pool_doc_neighbours(
                        graph_dataset.support_graph
                    )

                if hasattr(graph_dataset, "query_graph"):
                    graph_dataset.query_graph = avg_pool_doc_neighbours(
                        graph_dataset.query_graph
                    )

            if callable(getattr(graph_dataset, "generate_batches", None)):
                graph_dataset.partition_into_batches()
                graph_dataset.generate_batches(
                    num_workers=args["structure"]["num_workers"]
                )

            graph_dataset.save()

            del graph_dataset

    else:
        print("Skipping transfer data.")

    args["data"] = args_data_updated

    # ==========================================================================
    # Model Evaluation
    # ==========================================================================
    if args["use_val"] or args["use_test"]:
        # ======================================================================
        # Model creation
        # ======================================================================
        if args["learning_algorithm"]["meta_learner"] == "non_episodic":
            print("\n>>> NON-EPISODIC <<<\n")

            model = GatNonEpisodic.load_from_checkpoint(
                ckpt_fp,
                map_location=device,
            )

        elif args["learning_algorithm"]["meta_learner"] == "maml":
            print("\n>>> MAML <<<\n")

            model = GatMAML.load_from_checkpoint(
                ckpt_fp,
                map_location=device,
            )

        elif (
            args["learning_algorithm"]["meta_learner"] == "proto"
            or args["learning_algorithm"]["meta_learner"] == "protomaml"
        ):
            print("\n>>> PROTOTYPICAL <<<\n")

            model = GatPrototypical.load_from_checkpoint(
                ckpt_fp,
                map_location=device,
            )

        else:
            raise NotImplementedError("Have yet to implement the other models.")

        # If resetting the checkpoint weights
        if args["reset_checkpoint"]:
            print("\nModel weights being reset.")
            print(f"Using seed: {args['checkpoint_reset_seed']}")
            global_rng_state = torch.random.get_rng_state()
            torch.manual_seed(args["checkpoint_reset_seed"])

            reset_parameters = set()

            for layer in model.children():
                for n, ll in layer.named_modules():
                    if hasattr(ll, "reset_parameters"):
                        ll.reset_parameters()

                        for nn, _ in ll.named_parameters():
                            reset_parameters.add(".".join([n, nn]))

            for n, p in model.model.named_parameters():
                if n not in reset_parameters:
                    if "bias" in n:
                        torch.nn.init.constant_(p, 0.0)

                    else:
                        print(f"Found parameter that I don't know how to reset: {n}")

            torch.random.set_rng_state(global_rng_state)

        # If the classes are the same, keep the classifier head attached
        # Otherwise reset it
        if (
            args["learning_algorithm"]["reset_classifier"]
            or model.training_data_params["num_classes"] != args["data"]["num_classes"]
        ):
            # Alter the meta learning hyperparameters
            print("\nClassification head *will* be reset.")

            model.model.reset_classifier(args["data"]["num_classes"])
            model.n_classes = args["data"]["num_classes"]

        else:
            print("\nClassification head will *not* be reset.")

        # Handling the other learning algorithm parameters
        if args["learning_algorithm"]["class_weights"] is not None:
            model.register_class_weights(args["learning_algorithm"]["class_weights"])
            model.register_class_weights(
                args["learning_algorithm"]["class_weights"], prefix="eval"
            )

        else:
            model.register_class_weights(args["data"]["class_weights"])
            model.register_class_weights(args["data"]["class_weights"], prefix="eval")
        args["learning_algorithm"]["class_weights"] = model.eval_class_weights

        if args["learning_algorithm"]["lr_inner"] is not None:
            model.eval_lr_inner = args["learning_algorithm"]["lr_inner"]
        else:
            model.eval_lr_inner = model.lr_inner
        print(f"Inner LR: {model.eval_lr_inner:.2e}")
        args["learning_algorithm"]["lr_inner"] = model.eval_lr_inner

        if args["learning_algorithm"]["head_lr_inner"] is not None:
            model.eval_head_lr_inner = args["learning_algorithm"]["head_lr_inner"]
        else:
            model.eval_head_lr_inner = model.head_lr_inner
        print(f"CLF Head Inner LR: {model.eval_head_lr_inner:.2e}")
        args["learning_algorithm"]["head_lr_inner"] = model.eval_head_lr_inner

        if args["learning_algorithm"]["n_inner_updates"] is not None:
            model.eval_n_inner_updates = args["learning_algorithm"]["n_inner_updates"]
        else:
            model.eval_n_inner_updates = model.n_inner_updates
        print(f"Inner updates: {model.eval_n_inner_updates:d}")
        args["learning_algorithm"]["n_inner_updates"] = model.eval_n_inner_updates

        # ==========================================================================
        # EVALUATION
        # ==========================================================================
        trainer = init_trainer_and_logger(args)

        for fold in range(max(1, args["data"]["num_splits"])):
            if args["use_val"]:
                print_step(f"evaluation on validation set of fold {fold}")
                model.val_prefix = "val"
                model.test_prefix = "val"

                val_loader = get_dataloader(
                    args=args | {"fold": fold, "version": version},
                    split="val",
                    **args["data_loading"],
                )

                val_results = trainer.validate(model, val_loader, verbose=False)[0]
                dump_results(
                    args | {"fold": fold, "version": version},
                    prefix=model.test_prefix,
                    results_dict=val_results,
                )

            if args["use_test"]:
                print_step("evaluation on test set")
                model.val_prefix = "test"
                model.test_prefix = "test"

                test_loader = get_dataloader(
                    args=args | {"fold": fold, "version": version},
                    split="test",
                    **args["data_loading"],
                )

                test_results = trainer.validate(model, test_loader, verbose=False)[0]
                dump_results(
                    args | {"fold": fold, "version": version},
                    prefix=model.test_prefix,
                    results_dict=test_results,
                )

    else:
        print("Skipping transfer evaluation")


if __name__ == "__main__":
    transfer()
