import os
from copy import deepcopy
from pathlib import Path
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from pytorch_lightning.loggers import WandbLogger
from lightning_lite.utilities.seed import seed_everything

from data_prep.post_processing import PostProcessing
from data_loading import get_dataloader
from models import GatNonEpisodic, GatMAML, GatPrototypical

if torch.cuda.is_available():
    torch.cuda.empty_cache()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["WANDB_SILENT"] = "true"


def print_step(string):
    print("\n" + "=" * 100)
    print(f"{string.upper():^100s}")
    print("=" * 100 + "\n")


def init_wandb(args):
    wandb_config = {
        "seed": args["seed"],
        "k": args["k"],
        "structure_mode": args["structure_mode"],
        "model_architecture": args["model_architecture"],
        "meta_learner": args["learning_algorithm"]["meta_learner"],
    }
    wandb_config["data"] = args["data"]
    wandb_config["structure"] = args["structure"]
    wandb_config["data_loading"] = args["data_loading"]
    wandb_config["model"] = args["model"]
    wandb_config["learning_algorithm"] = args["learning_algorithm"]
    wandb_config["evaluation"] = args["evaluation"]
    wandb_config["optimizer"] = args["optimizer"]
    wandb_config["early_stopping"] = args["callbacks"]["early_stopping"]
    wandb_config["trainer"] = args["trainer"]

    logger = WandbLogger(
        project=args["logger"]["project"],
        log_model=False,
        save_dir=args["log_path"],
        offline=args["logger"]["mode"] == "offline",
        config=wandb_config,
        **args["logger"]["kwargs"],
    )
    logger.experiment.name = logger.experiment.id

    return logger


def init_trainer(args, logger):
    """
    Initializes a Lightning Trainer for respective parameters as given in the function header. Creates a proper
    folder name for the respective model files, initializes logging and early stopping.
    """

    early_stopping_cb = callbacks.EarlyStopping(
        patience=args["callbacks"]["early_stopping"]["patience"],
        monitor=args["callbacks"]["early_stopping"]["metric"],
        mode=args["callbacks"]["early_stopping"]["mode"],
        verbose=False,
        check_on_train_epoch_end=False,
    )

    checkpoint_cb = callbacks.ModelCheckpoint(
        save_weights_only=True,
        monitor=args["callbacks"]["early_stopping"]["metric"],
        mode=args["callbacks"]["early_stopping"]["mode"],
        **args["callbacks"]["checkpoint"],
    )

    lr_monitor_cb = callbacks.LearningRateMonitor(logging_interval="step")

    progress_bar_cb = callbacks.TQDMProgressBar(
        refresh_rate=args["callbacks"]["progress_bar"]["refresh_rate"]
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        move_metrics_to_cpu=True,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[
            early_stopping_cb,
            checkpoint_cb,
            lr_monitor_cb,
            progress_bar_cb,
        ],
        inference_mode=False,
        **args["trainer"],
    )

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


@hydra.main(version_base=None, config_path="./config", config_name="train")
def train(args: DictConfig):
    print("*" * 100)
    if args["print_config"]:
        print(OmegaConf.to_yaml(args, resolve=True))
    else:
        "Config loaded but not printing."
    print("*" * 100)

    os.makedirs(args["log_path"], exist_ok=True)

    args = OmegaConf.to_container(args, resolve=True)

    # ==========================================================================
    # Device
    # ==========================================================================
    print_step("device information")

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        cur_device = torch.cuda.current_device()
        print("Current device:", cur_device, torch.cuda.get_device_name(cur_device))
    else:
        warnings.warn(">>> CUDA NOT AVAILABLE <<<")

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

    # Figure out the dimensionality of the compressed features
    compressed_size = (
        PostProcessing(
            args["data"],
            cur_fold=args["fold"],
            version=args["version"],
            processed_or_structured="processed",
        )
        .load_file("feature_extractor")
        .compressed_size
    )

    print("\nInferring input feature dimensionality from feature extractor.")
    print(compressed_size)
    args["model"]["input_dim"] = compressed_size

    # ==========================================================================
    # Model construction
    # ==========================================================================
    print_step("model construction")

    # Need to set the model output as the number of classes
    # Also, for prototypical networks, check rule

    if args["learning_algorithm"]["meta_learner"] is None:
        raise ValueError("No meta learner specified.")

    elif args["learning_algorithm"]["meta_learner"] == "non_episodic":
        print("\n>>> NON-EPISODIC <<<\n")

        model = GatNonEpisodic(
            training_data_params=args["data"],
            training_structure_params=args["structure"],
            n_classes=args["data"]["num_classes"],
            optimizer_hparams=args["optimizer"],
            evaluation_params=args["evaluation"],
            ignore_index=args["data"]["label_mask"],
            model_params=args["model"],
            learning_hparams=args["learning_algorithm"],
            model_architecture=args["model_architecture"],
        )

    elif args["learning_algorithm"]["meta_learner"] == "maml":
        if args["structure"]["structure"] not in {
            "episodic_khop",
            "episodic_doc_only_khop",
        }:
            raise ValueError("If using MAML, graph structure cannot be `full`.")

        print("\n>>> MAML <<<\n")

        model = GatMAML(
            training_data_params=args["data"],
            training_structure_params=args["structure"],
            n_classes=args["data"]["num_classes"],
            optimizer_hparams=args["optimizer"],
            evaluation_params=args["evaluation"],
            ignore_index=args["data"]["label_mask"],
            model_params=args["model"],
            learning_hparams=args["learning_algorithm"],
            model_architecture=args["model_architecture"],
        )

    elif (
        args["learning_algorithm"]["meta_learner"] == "proto"
        or args["learning_algorithm"]["meta_learner"] == "protomaml"
    ):
        if args["structure"]["structure"] not in {
            "episodic_khop",
            "episodic_doc_only_khop",
        }:
            raise ValueError(
                "If using Prototypical networks, graph structure cannot be `full`."
            )

        print("\n>>> PROTOTYPICAL <<<\n")

        model = GatPrototypical(
            training_data_params=args["data"],
            training_structure_params=args["structure"],
            n_classes=args["data"]["num_classes"],
            optimizer_hparams=args["optimizer"],
            evaluation_params=args["evaluation"],
            ignore_index=args["data"]["label_mask"],
            model_params=args["model"],
            learning_hparams=args["learning_algorithm"],
            model_architecture=args["model_architecture"],
        )

    else:
        raise NotImplementedError(
            f"Have not implemented {args['learning_algorithm']['meta_learner']}."
        )

    print_step("model summary")
    print(pl.utilities.model_summary.ModelSummary(model, max_depth=3))

    # ==========================================================================
    # Trainer construction
    # ==========================================================================
    print_step("trainer construction")

    logger = init_wandb(args)

    experiment_dir = deepcopy(logger.experiment.name)
    print(f"Experiment dir: {experiment_dir}")

    trainer = init_trainer(args, logger)

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================
    print_step("starting training loop")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model path: {model_path}")

    # ==========================================================================
    # BEST VALIDATION LOOP
    # ==========================================================================
    print_step("starting validation with best checkpoint")
    if Path(model_path).exists() and Path(model_path).is_file():
        model = model.load_from_checkpoint(model_path)
    else:
        print(">>WARNING<<\nNo checkpoint found. Using current weights.")

    model.val_prefix = "val_best"
    model.test_prefix = "val_best"

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        move_metrics_to_cpu=True,
        logger=logger,
        inference_mode=False,
        **args["trainer"],
    )

    trainer.test(model, val_loader)

    logger.experiment.finish(0, quiet=True)

    if args.get("checkpoint_address_file", None) is not None:
        with open(args["checkpoint_address_file"], "a") as f:
            f.write(f"{experiment_dir}")

    print(f"\n\n\nRun dir for bash:\n{experiment_dir}")


if __name__ == "__main__":
    train()
