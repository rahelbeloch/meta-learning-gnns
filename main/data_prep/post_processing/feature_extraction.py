import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from lightning_lite import seed_everything
import datasets
from transformers import logging
from tqdm import tqdm

from data_prep.post_processing import PostProcessing
from models import FeatureExtractor
from utils.io import save_json_file
from utils.logging import get_results_dir
from utils.metrics import (
    compute_clf_metrics,
    compute_aupr_metrics,
    summarize_clf_metrics,
)
from utils.rng import stochastic_method

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HYDRA_FULL_ERROR"] = "1"

logging.set_verbosity(40)


@stochastic_method
def train_feature_extractor_and_compress(args, version):
    print("=" * 100)
    print(f"\tTraining feature extractor for {args['data']['dataset']} dataset.")
    print("=" * 100)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cur_device = torch.cuda.current_device()
        print("Current device:", cur_device, torch.cuda.get_device_name(cur_device))
    else:
        cur_device = torch.device("cpu")
        print("Current device:", "cpu")

    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []
    for fold in range(max(1, args["data"]["num_splits"])):
        print("\n" + "-" * 100)
        print(f"\tFold: {fold}")
        print("-" * 100)

        (train_dataset, _, _), (
            train_metrics,
            val_metrics,
            test_metrics,
        ) = train_feature_extractor(
            args,
            fold=fold,
            device=cur_device,
            version=version,
        )

        all_train_metrics += [train_metrics]
        all_val_metrics += [val_metrics]
        all_test_metrics += [test_metrics]

    results_dir = get_results_dir(
        results_dir=args["results_path"],
        data_args=args["data"],
        structure_args="text_baseline",
        version=version,
        fold="summary",
    )
    os.makedirs(results_dir, exist_ok=True)

    train_dataset.log("=" * 100)
    train_dataset.log(f"\tEvaluation {args['data']['dataset']} dataset.")
    train_dataset.log("=" * 100)

    train_dataset.log("\n" + "+" * 50)
    train_dataset.log("\tSummary statistics for split: train")
    train_dataset.log("+" * 50)
    train_metrics_summarized, train_summary_string = summarize_clf_metrics(
        all_train_metrics
    )
    save_json_file(train_metrics_summarized, results_dir / "train.json")
    train_dataset.log(train_summary_string)

    train_dataset.log("\n" + "+" * 50)
    train_dataset.log("\tSummary statistics for split: val")
    train_dataset.log("+" * 50)
    val_metrics_summarized, val_summary_string = summarize_clf_metrics(all_val_metrics)
    save_json_file(val_metrics_summarized, results_dir / "val.json")
    train_dataset.log(val_summary_string)

    train_dataset.log("\n" + "+" * 50)
    train_dataset.log("\tSummary statistics for split: test")
    train_dataset.log("+" * 50)
    test_metrics_summarized, test_summary_string = summarize_clf_metrics(
        all_test_metrics
    )
    save_json_file(test_metrics_summarized, results_dir / "test.json")
    train_dataset.log(test_summary_string)


@stochastic_method
def train_feature_extractor(
    args, fold: int = 0, device: torch.DeviceObjType = torch.device("cpu"), version=None
):
    def run_validation_epoch(args, model, loader, device):
        prog_bar_updates = max(len(loader) // args["feature_extraction"]["prog_bar"], 1)

        weights = torch.tensor(args["data"]["class_weights"]).float().to(device)

        model.eval()

        losses = []
        targets = []
        preds = []
        pbar = tqdm(loader, desc="Eval epoch", total=len(loader))
        prev_i = 0

        with torch.inference_mode():
            for i, batch in enumerate(pbar):
                _, x, y = batch

                x = {k: v.to(device) for k, v in x.items()}

                logits = model(x)

                loss = F.cross_entropy(
                    logits,
                    y.to(device),
                    weight=weights,
                )

                losses.append(loss.cpu().detach())
                targets.append(y.cpu().detach())
                preds.append(logits.cpu().detach().argmax(dim=-1))

                if i == 0 or i % prog_bar_updates == 0 or i == len(loader) - 1:
                    pbar.update(i - prev_i)
                    prev_i = i

        losses = torch.stack(losses)
        targets = torch.cat(targets)
        preds = torch.cat(preds)

        epoch_metrics = compute_clf_metrics(
            preds=preds, gt=targets, num_classes=args["data"]["num_classes"]
        )
        epoch_metrics["loss"] = torch.mean(losses)

        return epoch_metrics

    def run_epoch(args, model, optimizer, loader, device):
        prog_bar_updates = max(len(loader) // args["feature_extraction"]["prog_bar"], 1)

        weights = torch.tensor(args["data"]["class_weights"]).float().to(device)

        model.train()

        train_losses = []
        pbar = tqdm(loader, desc="Train epoch", total=len(loader))
        prev_i = 0

        for i, batch in enumerate(pbar):
            _, x, y = batch

            x = {k: v.to(device) for k, v in x.items()}

            logits = model(x)

            loss = F.cross_entropy(logits, y.to(device), weight=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.cpu().detach())

            if i == 0 or i % prog_bar_updates == 0 or i == len(loader) - 1:
                pbar.update(i - prev_i)
                prev_i = i

        train_losses = torch.stack(train_losses)

        return train_losses

    train_dataset = FeatureExtractorDataset(
        args["data"],
        split="train",
        cur_fold=fold,
        version=version,
    )

    val_dataset = FeatureExtractorDataset(
        args["data"],
        split="val",
        cur_fold=fold,
        version=version,
    )

    if args["data"]["num_splits"] > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args["feature_extraction"]["batch_size"],
            collate_fn=train_dataset.collate_fn,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args["feature_extraction"]["batch_size"],
            collate_fn=val_dataset.collate_fn,
            shuffle=False,
        )

    test_dataset = FeatureExtractorDataset(
        args["data"],
        split="test",
        cur_fold=fold,
        version=version,
    )

    feature_extractor = FeatureExtractor(
        feature_type=args["data"]["feature_type"],
        compression=args["data"]["compression"],
        vocab_size=args["data"]["vocab_size"],
        compressed_size=args["data"]["compressed_size"],
        out_dim=args["data"]["num_classes"],
        p_dropout=args["feature_extraction"]["p_dropout"],
        p_mask_token=args["feature_extraction"]["p_mask_token"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        feature_extractor.parameters(),
        lr=args["feature_extraction"]["lr"],
        weight_decay=args["feature_extraction"]["weight_decay"],
    )

    optimize_on = args["feature_extraction"]["optimize_on"]
    best_metric = None
    if optimize_on == "loss":
        best_metric = float("inf")
    else:
        best_metric = -float("inf")

    patience = args["feature_extraction"]["patience"]

    seed_everything(args["data"]["seed"])

    if args["data"]["num_splits"] == 0 or args["feature_extraction"]["patience"] == 0:
        state_dict = {
            "state_dict": feature_extractor.get_state_dict(),
            "hparams": feature_extractor.hparams,
            "metrics": None,
            "epoch": None,
        }

        train_dataset.save_file(file_type="feature_extractor", obj=state_dict)

        patience = 0

    for epoch in range(args["feature_extraction"]["n_epochs"]):
        save_flag = False
        if patience == 0:
            print("Early stopping.")
            break

        train_losses = run_epoch(
            args,
            feature_extractor,
            optimizer,
            loader=train_loader,
            device=device,
        )
        epoch_metrics = run_validation_epoch(
            args, feature_extractor, val_loader, device=device
        )

        print(
            f"Epoch {epoch} | Train Loss: {torch.mean(train_losses).item():.2e} Eval Loss: {epoch_metrics['loss'].item():.2e} F1 Macro: {epoch_metrics['f1_macro'].item():.2f} MCC: {epoch_metrics['mcc'].item():.2f}"
        )

        if optimize_on == "loss" and epoch_metrics["loss"].item() < best_metric:
            save_flag = True

        elif optimize_on != "loss" and epoch_metrics[optimize_on].item() > best_metric:
            save_flag = True

        else:
            save_flag = False

        if save_flag:
            print(f">>> New best {optimize_on} <<<")
            best_metric = epoch_metrics[optimize_on].item()
            patience = args["feature_extraction"]["patience"]

            state_dict = {
                "state_dict": feature_extractor.get_state_dict(),
                "hparams": feature_extractor.hparams,
                "metrics": epoch_metrics,
                "epoch": epoch,
            }

            train_dataset.save_file(file_type="feature_extractor", obj=state_dict)

        else:
            patience -= 1

    feature_extractor = train_dataset.load_file(file_type="feature_extractor")
    feature_extractor.to(device)
    feature_extractor.eval()

    print("Compressing...")
    weights = torch.tensor(args["data"]["class_weights"]).float().to(device)

    split_metrics = dict()
    split_datasets = dict()

    if args["data"]["num_splits"] > 0:
        datasets_zip = zip(
            [train_dataset, val_dataset, test_dataset], ["train", "val", "test"]
        )
    else:
        datasets_zip = zip([test_dataset], ["test"])

        split_metrics["train"] = dict()
        split_metrics["val"] = dict()

    for dataset, split in datasets_zip:
        losses = []
        targets = []
        preds = []
        probs = []

        split_node_ids = []
        split_embeddings = []
        split_targets = []

        loader = DataLoader(
            dataset,
            batch_size=args["feature_extraction"]["batch_size"],
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )

        prog_bar_updates = max(len(loader) // args["feature_extraction"]["prog_bar"], 1)

        prev_i = 0
        pbar = tqdm(loader, desc=f"Compressing {split} split", total=len(loader))
        for i, (node_ids, x, y) in enumerate(pbar):
            x = {k: v.to(device) for k, v in x.items()}

            with torch.inference_mode():
                compressed_x = feature_extractor.compress(x)

            split_node_ids.append(node_ids)
            split_embeddings.append(compressed_x)
            split_targets.append(y)

            logits = feature_extractor.clf(compressed_x)

            loss = F.cross_entropy(
                logits,
                y.to(device),
                weight=weights,
            )

            compressed_x = compressed_x.cpu().detach()

            losses.append(loss.cpu().detach())
            targets.append(y.cpu().detach())
            preds.append(logits.cpu().detach().argmax(dim=-1))
            probs.append(F.softmax(logits.cpu().detach(), dim=-1))

            if i == 0 or i % prog_bar_updates == 0 or i == len(loader) - 1:
                pbar.update(i - prev_i)
                prev_i = i

        split_node_ids = torch.cat(split_node_ids, dim=0)
        split_embeddings = torch.cat(split_embeddings, dim=0)
        split_targets = torch.cat(split_targets, dim=0)

        split_dataset = datasets.Dataset.from_dict(
            {
                "node_id": split_node_ids,
                "x": split_embeddings,
                "y": split_targets,
            },
            split=split,
        )

        split_datasets[split] = split_dataset

        losses = torch.stack(losses)
        targets = torch.cat(targets)
        preds = torch.cat(preds)
        probs = torch.cat(probs)

        epoch_metrics = compute_clf_metrics(
            preds=preds, gt=targets, num_classes=args["data"]["num_classes"]
        )
        epoch_metrics |= compute_aupr_metrics(
            probs=probs, gt=targets, num_classes=args["data"]["num_classes"]
        )
        epoch_metrics["loss"] = torch.mean(losses)

        split_metrics[split] = epoch_metrics

    split_datasets = datasets.DatasetDict(split_datasets)

    dataset.save_file(file_type="compressed_dataset", obj=split_datasets)

    results_dir = get_results_dir(
        results_dir=args["results_path"],
        data_args=args["data"],
        structure_args="text_baseline",
        version=version,
        fold=fold,
    )
    os.makedirs(results_dir, exist_ok=True)

    save_json_file(
        {k: v.item() for k, v in split_metrics["train"].items()},
        results_dir / "train.json",
    )
    save_json_file(
        {k: v.item() for k, v in split_metrics["val"].items()}, results_dir / "val.json"
    )
    save_json_file(
        {k: v.item() for k, v in split_metrics["test"].items()},
        results_dir / "test.json",
    )

    return (train_dataset, val_dataset, test_dataset), (
        split_metrics["train"],
        split_metrics["val"],
        split_metrics["test"],
    )


class FeatureExtractorDataset(PostProcessing, Dataset):
    def __init__(self, args, cur_fold: int, split: str, **super_kwargs):
        super().__init__(
            args, cur_fold, processed_or_structured="processed", **super_kwargs
        )

        self.split = split

        doc_dataset = self.load_file("doc_dataset")
        self.data = doc_dataset.select(self.fold_idx[self.split])

        doc2nodeid = self.load_file("doc2nodeid")
        self.node_ids = torch.tensor(
            [doc2nodeid[doc_id] for doc_id in self.data["doc_id"]]
        )

        self.idx = [i for i in range(self.data.num_rows)]

        if args["feature_type"] == "lm-embeddings":
            self.feature_type = "lm-embeddings"
            tokenizer = self.load_file("tokenizer")
            self.pad_token_id = tokenizer.tokenizer.pad_token_id

    def collate_fn(self, batch):
        node_ids = self.node_ids[batch]
        batch = self.data[batch]
        y = torch.tensor(batch["y"])

        if self.feature_type == "lm-embeddings":
            model_inputs = dict()
            for k, v in batch.items():
                v = list(map(torch.tensor, v))

                if k == "input_ids":
                    model_inputs[k] = pad_sequence(
                        v, batch_first=True, padding_value=self.pad_token_id
                    )

                elif k == "attention_mask" or k == "special_tokens_mask":
                    model_inputs[k] = pad_sequence(v, batch_first=True, padding_value=0)

            return node_ids, model_inputs, y

        elif self.feature_type == "one-hot":
            batch_size = batch["input_ids"].shape[0]
            model_inputs = torch.zeros((batch_size, self.vocab_size))
            for row, present_tokens in enumerate(batch["input_ids"]):
                model_inputs[row][present_tokens] = 1.0

            return node_ids, {"model_input": model_inputs}, y

    def __getitem__(self, index):
        return self.idx[index]

    def __len__(self):
        return len(self.data)
