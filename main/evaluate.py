import os
from pathlib import Path
import re
import json
import csv
import pickle

import hydra
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from data_loading.get_loader import get_dataloader
from models import GatNonEpisodic, GatMAML, GatPrototypical
from utils.logging import get_results_dir

if torch.cuda.is_available():
    torch.cuda.empty_cache()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"


def print_step(string):
    print("\n" + "=" * 100)
    print(f"{string.upper():^100s}")
    print("=" * 100 + "\n")


def find_checkpoint_file_path(args: dict, checkpoint: str):
    checkpoint_dir = Path(args["log_path"], args["checkpoint_dir"], checkpoint)

    if not checkpoint_dir.exists():
        raise ValueError("Checkpoint dir does not exist:\n\t", checkpoint_dir)

    checkpoints_found = dict()
    for f in (checkpoint_dir / "checkpoints").glob("*.ckpt"):
        step = re.search(r"step\=([0-9]+)", str(f)).group(1)
        checkpoints_found[step] = f

    if len(checkpoints_found) == 0:
        raise ValueError("No checkpoint file found:\n\t", checkpoint_dir)
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


def get_summary_line(args, results_dict, prefix):
    if f"{prefix}/loss" in results_dict:
        loss = results_dict[f"{prefix}/loss"]
    elif f"{prefix}/query_mean_loss" in results_dict:
        loss = results_dict[f"{prefix}/query_mean_loss"]

    summary = f"Loss: {loss:.4f} F1: ["
    summary += ", ".join(
        [
            f"{results_dict[f'{prefix}/f1_{l}']*100:.2f}"
            for l in range(args["data"]["num_classes"])
        ]
    )
    summary += "]%"
    summary += f" F1 Macro: {results_dict[f'{prefix}/f1_macro']*100:.2f}"
    summary += f" MCC: {results_dict[f'{prefix}/mcc']:.4f}"

    return summary


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


@hydra.main(version_base=None, config_path="./config", config_name="evaluate")
def evaluate(args):
    if args["print_config"]:
        print(OmegaConf.to_yaml(args, resolve=True))
    else:
        "Config loaded but not printing."

    args = OmegaConf.to_container(args, resolve=True)

    os.makedirs(args["log_path"], exist_ok=True)
    os.makedirs(args["results_path"], exist_ok=True)

    # ==========================================================================
    # Device
    # ==========================================================================
    print_step("device information")

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        cur_device = torch.cuda.current_device()
        print("Current device:", cur_device, torch.cuda.get_device_name(cur_device))

    # ==========================================================================
    # Experiment definiton
    # ==========================================================================
    print_step("experiment & global args")
    seed = args["seed"]
    print(f"Running with seed: {args['seed']}")

    # reproducible results
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ==========================================================================
    # Data loaders construction
    # ==========================================================================
    print_step("data loader construction")

    train_loader = get_dataloader(args=args, split="train", **args["data_loading"])
    val_loader = get_dataloader(args=args, split="val", **args["data_loading"])
    test_loader = get_dataloader(args=args, split="test", **args["data_loading"])

    # ==========================================================================
    # Model construction
    # ==========================================================================
    print_step("model construction")

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

    if args["learning_algorithm"]["meta_learner"] is None:
        raise ValueError("No meta learner specified.")

    elif args["learning_algorithm"]["meta_learner"] == "non_episodic":
        print("\n>>> NON-EPISODIC <<<\n")

        model = GatNonEpisodic.load_from_checkpoint(
            ckpt_fp,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    elif args["learning_algorithm"]["meta_learner"] == "maml":
        print("\n>>> MAML <<<\n")

        model = GatMAML.load_from_checkpoint(
            ckpt_fp,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    elif (
        args["learning_algorithm"]["meta_learner"] == "proto"
        or args["learning_algorithm"]["meta_learner"] == "protomaml"
    ):
        print("\n>>> PROTOTYPICAL <<<\n")

        model = GatPrototypical.load_from_checkpoint(
            ckpt_fp,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    else:
        raise NotImplementedError(
            f"Have not implemented {args['learning_algorithm']['meta_learner']}."
        )

    # If the dataset is the same, keep the classifier head attached
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

    # ==========================================================================
    # EVALUATION LOOPS
    # ==========================================================================

    trainer = init_trainer_and_logger(args)

    summaries = {}
    if args["use_train"]:
        print_step("evaluation on train set")
        model.val_prefix = "train"
        model.test_prefix = "train"

        train_results = trainer.validate(model, train_loader, verbose=False)[0]
        dump_results(
            args,
            prefix=model.test_prefix,
            results_dict=train_results,
        )

        train_summary = get_summary_line(args, train_results, model.test_prefix)
        print("Train:", train_summary)
        summaries["train"] = train_summary

    if args["use_val"]:
        print_step("evaluation on validation set")
        model.val_prefix = "val"
        model.test_prefix = "val"

        if args["learning_algorithm"]["meta_learner"] == "non_episodic":
            val_results = trainer.test(model, val_loader, verbose=False)[0]
            dump_results(
                args,
                prefix=model.test_prefix,
                results_dict=val_results,
                preds=model.test_epoch_preds,
                gt=model.test_epoch_gt,
            )

        else:
            val_results = trainer.validate(model, val_loader, verbose=False)[0]
            dump_results(
                args,
                prefix=model.test_prefix,
                results_dict=val_results,
            )

        val_summary = get_summary_line(args, val_results, model.test_prefix)
        print("Val:", val_summary)
        summaries["val"] = val_summary

    if args["use_test"]:
        print_step("evaluation on test set")
        model.val_prefix = "test"
        model.test_prefix = "test"

        if args["learning_algorithm"]["meta_learner"] == "non_episodic":
            test_results = trainer.test(model, test_loader, verbose=False)[0]
            dump_results(
                args,
                prefix=model.test_prefix,
                results_dict=test_results,
                preds=model.test_epoch_preds,
                gt=model.test_epoch_gt,
            )

        else:
            test_results = trainer.validate(model, test_loader, verbose=False)[0]
            dump_results(
                args,
                prefix=model.test_prefix,
                results_dict=test_results,
            )

        test_summary = get_summary_line(args, test_results, model.test_prefix)
        print("Test:", test_summary)
        summaries["test"] = test_summary

    dump_results(args, "summary", results_dict=summaries)

    dump_results(
        args,
        prefix="",
        hparams=model.hparams | {"optimizer": model.opt_params},
    )


if __name__ == "__main__":
    evaluate()
