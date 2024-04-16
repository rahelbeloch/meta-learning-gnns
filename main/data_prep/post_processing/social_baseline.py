import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from data_prep.post_processing import PostProcessing
from utils.io import save_json_file
from utils.logging import get_results_dir
from utils.metrics import (
    compute_clf_metrics,
    compute_aupr_metrics,
    summarize_clf_metrics,
)


def train_social_baseline(args, version):
    all_train_metrics = []
    all_val_metrics = []
    all_test_metrics = []
    for fold in range(args["data"]["num_splits"]):
        social_baseline = SocialBaseline(
            args=args["data"],
            cur_fold=fold,
            version=version,
        )

        (train_docs, val_docs, test_docs), (
            train_labels,
            val_labels,
            test_labels,
        ) = social_baseline.get_features_and_labels()

        social_baseline.train(train_docs)

        split_metrics = dict()

        results_dir = get_results_dir(
            results_dir=args["results_path"],
            data_args=args["data"],
            structure_args="social_baseline",
            version=version,
            fold=fold,
        )
        os.makedirs(results_dir, exist_ok=True)

        for split, split_features, split_labels in zip(
            ["train", "val", "test"],
            [train_docs, val_docs, test_docs],
            [train_labels, val_labels, test_labels],
        ):
            probs = social_baseline.predict_proba(split_features, split_labels)
            split_labels = torch.tensor(split_labels)

            split_metrics[split] = {
                "loss": F.cross_entropy(probs, split_labels),
                **compute_clf_metrics(
                    torch.argmax(probs, dim=1),
                    split_labels,
                    num_classes=social_baseline.num_labels,
                ),
                **compute_aupr_metrics(
                    probs,
                    split_labels,
                    num_classes=social_baseline.num_labels,
                ),
            }

            save_json_file(
                {k: v.item() for k, v in split_metrics[split].items()},
                results_dir / f"{split}.json",
            )

        all_train_metrics.append(split_metrics["train"])
        all_val_metrics.append(split_metrics["val"])
        all_test_metrics.append(split_metrics["test"])

    results_dir = get_results_dir(
        results_dir=args["results_path"],
        data_args=args["data"],
        structure_args="social_baseline",
        version=version,
        fold="summary",
    )
    os.makedirs(results_dir, exist_ok=True)

    social_baseline.log("=" * 100)
    social_baseline.log(
        f"\tEvaluation social baseline on {args['data']['dataset']} dataset."
    )
    social_baseline.log("=" * 100)

    social_baseline.log("\n" + "+" * 50)
    social_baseline.log("\tSummary statistics for split: train")
    social_baseline.log("+" * 50)
    train_metrics_summarized, train_summary_string = summarize_clf_metrics(
        all_train_metrics
    )
    save_json_file(train_metrics_summarized, results_dir / "train.json")
    social_baseline.log(train_summary_string)

    social_baseline.log("\n" + "+" * 50)
    social_baseline.log("\tSummary statistics for split: val")
    social_baseline.log("+" * 50)
    val_metrics_summarized, val_summary_string = summarize_clf_metrics(all_val_metrics)
    save_json_file(val_metrics_summarized, results_dir / "val.json")
    social_baseline.log(val_summary_string)

    social_baseline.log("\n" + "+" * 50)
    social_baseline.log("\tSummary statistics for split: test")
    social_baseline.log("+" * 50)
    test_metrics_summarized, test_summary_string = summarize_clf_metrics(
        all_test_metrics
    )
    save_json_file(test_metrics_summarized, results_dir / "test.json")
    social_baseline.log(test_summary_string)


class SocialBaseline(PostProcessing):
    def __init__(self, args, cur_fold, **super_kwargs):
        super().__init__(
            args, cur_fold, processed_or_structured="processed", **super_kwargs
        )

        self.num_labels = len(self.labels)

    def get_features_and_labels(self):
        doc_dataset = self.load_file("doc_dataset")

        train_docs = doc_dataset.select(self.fold_idx["train"])
        train_labels = train_docs["y"]
        train_docs = train_docs["doc_id"]

        val_docs = doc_dataset.select(self.fold_idx["val"])
        val_labels = val_docs["y"]
        val_docs = val_docs["doc_id"]

        test_docs = doc_dataset.select(self.fold_idx["test"])
        test_labels = test_docs["y"]
        test_docs = test_docs["doc_id"]

        return (
            train_docs,
            val_docs,
            test_docs,
        ), (
            train_labels,
            val_labels,
            test_labels,
        )

    def train(self, train_docs):
        invalid_docs = self.load_file("invalid_docs")
        self.invalid_users = self.load_file("invalid_users")
        self.doc2users = self.load_file("doc2users")
        doc2label = self.load_file("doc2labels")

        self.prior = [0 for _ in range(self.num_labels)]

        self.user_prop = defaultdict(lambda: [0 for _ in range(self.num_labels)])
        for doc_id in train_docs:
            if doc_id in invalid_docs:
                print("Found an invalid doc!")

            incident_users = self.doc2users.get(doc_id, set()) - self.invalid_users
            if len(incident_users) == 0:
                continue

            label = doc2label[doc_id]
            for user_id in incident_users:
                self.user_prop[user_id][label] += 1

            self.prior[label] += 1

        self.user_prop = dict(self.user_prop)
        for user_id, prop in self.user_prop.items():
            prop = torch.tensor(prop)
            self.user_prop[user_id] = prop / prop.sum()

        self.prior = torch.tensor(self.prior)
        self.prior = self.prior / self.prior.sum()

    def predict_proba(self, docs, labels):
        probs = []
        for doc_id in docs:
            incident_users = self.doc2users.get(doc_id, set()) - self.invalid_users

            if len(incident_users) == 0:
                n = 1
                prob = self.prior
            else:
                n = 0
                prob = torch.zeros((self.num_labels,))
                for user_id in incident_users:
                    if user_id in self.user_prop:
                        prob += self.user_prop[user_id]
                        n += 1
                # In case none of the incident users for this document have been seen before
                # Fal back onto prior
                if n == 0:
                    n = 1
                    prob = self.prior

            probs.append(prob / n)

        probs = torch.stack(probs)

        return probs
