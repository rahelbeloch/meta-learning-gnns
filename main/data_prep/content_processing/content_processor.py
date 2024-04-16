import abc
import time
from collections import Counter, defaultdict

import numpy as np
import datasets
from datasets import Dataset

from data_prep.graph_io import GraphIO
from data_prep.tokenizers import OneHotTokenizer, LMTokenizer
from utils.logging import calc_elapsed_time
from utils.io import load_json_file


class ContentProcessor(GraphIO):
    def __init__(self, args, **super_kwargs):
        super().__init__(
            args,
            **super_kwargs,
        )

        self.summary["Num isolated docs"] = 0
        self.summary["Num unlabelled docs"] = 0

    @abc.abstractmethod
    def load_content(self, invalid_docs):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_doc_interactions(self, invalid_users):
        raise NotImplementedError()

    def prep(self):
        start_time = time.time()

        self.print_step("Loading needed files")

        self.log("Loading content...\n")
        invalid_docs = set()
        invalid_docs, doc2content, doc2labels = self.load_content(invalid_docs)

        self.save_file(file_type="doc2content", obj=doc2content)
        self.save_file(file_type="doc2labels", obj=doc2labels)
        self.save_file(file_type="invalid_docs", obj=invalid_docs)

        self.log("Aggregating doc-user relations...")
        invalid_users = set()
        invalid_users, doc2user, user2doc = self.load_doc_interactions(invalid_users)

        self.save_file(file_type="doc2users", obj=doc2user)
        self.save_file(file_type="user2docs", obj=user2doc)
        self.save_file(file_type="invalid_users", obj=invalid_users)

        self.log("\nFinished loading needed files.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def sanitize_documents(self):
        """
        Initial cleaning of documents:
            1. Imports the documents
            2. Sanitizes them
            3. Trains or loads a tokenizer
            4. Tokenizes the documents
            5. Runs an initial check if documents are valid
        """
        start_time = time.time()

        self.print_step("Sanitizing Documents")

        invalid_docs = self.load_file(file_type="invalid_docs")
        doc2content = self.load_file(file_type="doc2content")
        doc2labels = self.load_file(file_type="doc2labels")

        self.log("Generating dataset object")
        doc_dataset = []
        for doc_id in doc2content.keys():
            content_line = {"doc_id": doc_id}
            content_line["raw_text"] = doc2content[doc_id]
            content_line["y"] = doc2labels[doc_id]

            doc_dataset += [content_line]

        doc_dataset = Dataset.from_list(doc_dataset)
        doc_dataset.set_format(type="numpy", columns=["y"])

        self.log("\nSanitizing...")
        if self.feature_type == "one-hot":
            self.log("Running with a OneHot Tokenizer")
            tokenizer = OneHotTokenizer(self.vocab_size, self.use_joint_vocab)
            self.log("Building vocab...")
            if self.use_joint_vocab:
                joint_vocab_fp = (
                    self.data_tsv_dir / f"joint_vocab_{self.vocab_size}.json"
                )

                if joint_vocab_fp.exists():
                    self.log("Found pre-defined joint vocab...")
                    tokenizer.vocab = load_json_file(joint_vocab_fp)

                else:
                    raise ValueError(
                        f"Specified `{self.use_joint_vocab}` but file {joint_vocab_fp} not found."
                    )

            else:
                tokenizer.build_vocab(list(doc2content.values()))

            self.log("Tokenizing documents...")
            doc_dataset = doc_dataset.map(lambda row: tokenizer(row["raw_text"]))

            self.log("Filtering all OOV docs...")
            all_oov_idx = doc_dataset.filter(
                lambda row: row["length"] <= row["oov_count"]
            )["doc_id"]
            invalid_docs.update(all_oov_idx)
            num_all_oov = len(all_oov_idx)

            self.log("Filtering too short docs...")
            too_short_idx = doc_dataset.filter(
                lambda row: row["length"] < self.min_len
            )["doc_id"]
            invalid_docs.update(too_short_idx)
            num_too_short = len(too_short_idx)

        elif self.feature_type == "lm-embeddings":
            self.log(f"Running with a LMEmbeddings Tokenizer ({self.compression})")
            tokenizer = LMTokenizer(
                self.compression,
                truncation=True,
                return_length=True,
                return_special_tokens_mask=True,
            )

            self.log("Tokenizing documents...")
            doc_dataset = doc_dataset.map(lambda row: tokenizer(row["raw_text"]))

            self.log("Filtering all OOV docs...")
            all_oov_idx = doc_dataset.filter(
                lambda row: row["length"] == row["special_tokens_mask"].sum()
            )["doc_id"]
            invalid_docs.update(all_oov_idx)
            num_all_oov = len(all_oov_idx)

            self.log("Filtering too short docs...")
            too_short_idx = doc_dataset.filter(
                lambda row: (row["length"] - row["special_tokens_mask"].sum())
                < self.min_len
            )["doc_id"]
            invalid_docs.update(too_short_idx)
            num_too_short = len(too_short_idx)

        self.log("Applying filters...")
        doc_dataset = doc_dataset.filter(lambda row: row["doc_id"] not in invalid_docs)

        doc_lengths = doc_dataset["length"]

        self.log("\n+=== Stats ===+")
        self.log("Doc lengths:")
        doc_length_stats_summary = f"Mean: {np.mean(doc_lengths):.2f}"
        doc_length_stats_summary += f" Std. Dev.: {np.std(doc_lengths):.2f}"
        doc_length_stats_summary += (
            f" Quantiles: ["
            + ", ".join(
                map(
                    lambda x: f"{int(x):d}",
                    np.quantile(doc_lengths, [0, 0.25, 0.50, 0.75, 1]),
                )
            )
            + "]"
        )
        self.summary["Doc lengths"] = doc_length_stats_summary

        self.log(doc_length_stats_summary)

        self.log(
            f"\nTotal docs not found (pre tokenization): {self.summary['File not found']}"
        )
        self.log(
            f"Total docs without content (pre tokenization): {self.summary['Empty file']}"
        )
        self.log(f"Total docs without content (post tokenization): {num_all_oov}")
        self.summary["All OOV docs"] = num_all_oov

        self.log(f"Total docs too short content (post tokenization): {num_too_short}")
        self.summary["Too short docs"] = num_too_short

        self.log("\nLabel counts (post filtering):")
        counts = np.bincount(doc_dataset["y"])
        percentage = counts / counts.sum()

        self.summary["Label counts"] = dict()
        self.summary["Label counts summary"] = ""
        for i in range(counts.shape[0]):
            self.summary["Label counts"][self.labels[i]] = counts[i]
            self.summary[
                "Label counts summary"
            ] += f"Portion of label '{self.labels[i]}': {counts[i]} ({percentage[i]*100:.2f}%)\n"

        self.log(self.summary["Label counts summary"])

        self.save_file(file_type="tokenizer", obj=tokenizer)

        self.save_file(file_type="doc2labels", obj=doc2labels)

        self.save_file(file_type="invalid_docs", obj=invalid_docs)

        self.save_file(file_type="summary")

        self.log("Finished sanitization & tokenization.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

        return doc_dataset

    def filter_users(self):
        start_time = time.time()

        self.print_step("Filtering Users")

        invalid_docs = self.load_file("invalid_docs")
        doc2labels = self.load_file("doc2labels")

        doc2users = self.load_file(file_type="doc2users")
        user2docs = self.load_file(file_type="user2docs")

        self.summary["num interacting users (pre filter)"] = len(user2docs)
        self.log(f"Num interacting users (pre filter): {len(user2docs)}")

        self.log(f"\n+== Pre User Filtering ==+")
        self.summary["num interacting users (post invalid docs filter)"] = len(
            user2docs
        )
        self.log(f"Num interacting users (post invalid docs filter): {len(user2docs)}")

        num_interactions = {
            user_id: len(doc_ids) for user_id, doc_ids in user2docs.items()
        }

        self.log("Num interactions:")
        num_interactions_stats_summary = (
            f"Mean: {np.mean(list(num_interactions.values())):.2f}"
        )
        num_interactions_stats_summary += (
            f" Std. Dev.: {np.std(list(num_interactions.values())):.2f}"
        )
        num_interactions_stats_summary += (
            f" Quantiles: ["
            + ", ".join(
                map(
                    lambda x: f"{int(x):d}",
                    np.quantile(
                        list(num_interactions.values()), [0, 0.25, 0.50, 0.75, 1]
                    ),
                )
            )
            + "]"
        )
        num_interactions_stats_summary += (
            f" E[log(x)]={np.mean(np.log(list(num_interactions.values()))):.2f}"
        )
        num_interactions_stats_summary += f" exp(E[log(x)])={np.exp(np.mean(np.log(list(num_interactions.values())))):.2f}"
        self.summary[
            "num_interactions_stats (pre user filter)"
        ] = num_interactions_stats_summary
        self.log(self.summary["num_interactions_stats (pre user filter)"])

        self.log(f"\n+== User Doc Threshold ==+")
        user2doc_label = {
            user_id: Counter(map(lambda x: doc2labels.get(x, None), doc_ids)).items()
            for user_id, doc_ids in user2docs.items()
        }

        invalid_users = self.load_file("invalid_users")
        num_docs_without_label = 0
        num_users_share_too_large_prop = 0
        for user_id, doc_class_counts in user2doc_label.items():
            for label, count in doc_class_counts:
                if label is None:
                    num_docs_without_label += count
                    continue

                prop = count / self.summary["Label counts"][self.labels[label]]
                if prop > self.user_doc_threshold:
                    invalid_users.add(user_id)
                    num_users_share_too_large_prop += 1

                    for doc_id in user2docs[user_id]:
                        if doc_id not in doc2users:
                            invalid_docs.add(doc_id)
                            continue

                        doc2users[doc_id].discard(user_id)

                    del user2docs[user_id], num_interactions[user_id]
                    break

        self.summary["num_users_share_too_large_prop"] = num_users_share_too_large_prop
        self.log(
            f"Number of users sharing too much of one class: {num_users_share_too_large_prop}"
        )
        self.log(f"Number of documents without label: {num_docs_without_label}")

        # self.log(f"\n+== User Truncation ==+")
        # user_interactions_sorted = [
        #    k for k, v in sorted(num_interactions.items(), key=lambda item: item[1], reverse=True)
        #    ]

        # top_users_excluded = int((self.top_users_excluded / 100) * len(user_interactions_sorted))
        # self.summary["top_users_excluded"] = top_users_excluded
        # self.log(f"Removing top users: {top_users_excluded}")
        # invalid_users.update(user_interactions_sorted[:top_users_excluded])

        # for user_id in user_interactions_sorted[:top_users_excluded]:
        #    for doc_id in user2docs[user_id]:
        #        doc2users[doc_id].remove(user_id)
        #    del user2docs[user_id], num_interactions[user_id]

        self.log(f"\n+== Post User Filtering ==+")
        self.summary["num_invalid_users"] = len(invalid_users)

        num_interactions = {
            user_id: len(doc_ids) for user_id, doc_ids in user2docs.items()
        }

        self.log(f"Users removed: {len(invalid_users)}")

        self.log("Num interations:")
        num_interactions_stats_summary = (
            f"Mean: {np.mean(list(num_interactions.values())):.2f}"
        )
        num_interactions_stats_summary += (
            f" Std. Dev.: {np.std(list(num_interactions.values())):.2f}"
        )
        num_interactions_stats_summary += (
            f" Quantiles: ["
            + ", ".join(
                map(
                    lambda x: f"{int(x):d}",
                    np.quantile(
                        list(num_interactions.values()), [0, 0.25, 0.50, 0.75, 1]
                    ),
                )
            )
            + "]"
        )
        num_interactions_stats_summary += (
            f" E[log(x)]={np.mean(np.log(list(num_interactions.values()))):.2f}"
        )
        num_interactions_stats_summary += f" exp(E[log(x)])={np.exp(np.mean(np.log(list(num_interactions.values())))):.2f}"
        self.summary[
            "num_interactions_stats (post user filter)"
        ] = num_interactions_stats_summary
        self.log(self.summary["num_interactions_stats (post user filter)"])

        num_docs_without_label = 0
        class_users = {l: set() for l in self.labels.keys()}
        for doc_id, user_ids in doc2users.items():
            try:
                label = doc2labels[doc_id]
            except KeyError:
                num_docs_without_label += 1

            class_users[label].update(user_ids)

        self.summary["distinct_user_per_class"] = dict()
        self.log("Number of distinct users interacting with class:")
        for k, v in sorted(class_users.items(), key=lambda item: item[0]):
            label = self.labels[k]
            count = len(v)
            self.summary["distinct_user_per_class"][label] = count
            self.log(f"\t'{label}': {count}")

        self.summary["isolated_docs_per_class"] = defaultdict(int)
        self.log("Number of isolated docs per class:")
        for doc_id, user_ids in doc2users.items():
            if len(user_ids) == 0:
                try:
                    label = self.labels[doc2labels[doc_id]]
                except KeyError:
                    continue
                self.summary["isolated_docs_per_class"][label] += 1

        self.summary["isolated_docs_per_class"] = dict(
            self.summary["isolated_docs_per_class"]
        )
        for l, count in self.summary["Label counts"].items():
            n_isolated = self.summary["isolated_docs_per_class"].get(l, 0)
            n_total = self.summary["Label counts"].get(l, 0)

            self.log(
                f"\t'{l}': {n_isolated}/{n_total} [{n_isolated/n_total * 100:.2f}%]"
            )

        self.save_file(file_type="doc2users", obj=doc2users)
        self.save_file(file_type="user2docs", obj=user2docs)

        self.save_file(file_type="invalid_docs", obj=invalid_docs)
        self.save_file(file_type="invalid_users", obj=invalid_users)

        self.save_file(file_type="summary")

        self.log("\nFinished filtering users.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def apply_filters(self, doc_dataset):
        start_time = time.time()

        self.print_step("Applying filters")

        self.log("Applying filters to components...")
        doc2content = self.load_file(file_type="doc2content")
        doc2labels = self.load_file(file_type="doc2labels")
        doc2users = self.load_file(file_type="doc2users")
        user2docs = self.load_file(file_type="user2docs")
        invalid_docs = self.load_file(file_type="invalid_docs")
        invalid_users = self.load_file(file_type="invalid_users")

        user2docs_invalid = set()
        for user_id, docs in user2docs.items():
            user2docs[user_id] = docs - invalid_docs

            if len(user2docs[user_id]) == 0:
                user2docs_invalid.add(user_id)

        for doc_id, users in doc2users.items():
            doc2users[doc_id] = users - invalid_users

        for doc_id, users in doc2users.items():
            if len(doc2users[doc_id]) == 0 and self.filter_out_isolated_docs:
                invalid_docs.add(doc_id)
                self.summary["Num isolated docs"] += 1

        if doc_id not in doc2labels or doc2labels[doc_id] is None:
            invalid_docs.add(doc_id)
            self.summary["Num unlabelled docs"] += 1

        for doc_id in invalid_docs:
            if doc_id in doc2content.items():
                del doc2content[doc_id]
            if doc_id in doc2labels.items():
                del doc2labels[doc_id]
            if doc_id in doc2users.items():
                del doc2users[doc_id]

        for user_id in set.union(invalid_users, user2docs_invalid):
            if user_id in user2docs:
                del user2docs[user_id]

        self.save_file(file_type="doc2content", obj=doc2content)
        self.save_file(file_type="doc2labels", obj=doc2labels)

        self.save_file(file_type="doc2users", obj=doc2users)
        self.save_file(file_type="user2docs", obj=user2docs)

        self.save_file(file_type="invalid_docs", obj=invalid_docs)
        self.save_file(file_type="invalid_users", obj=invalid_users)

        del doc2content, doc2labels, doc2users, user2docs, invalid_users

        self.log("+== Stats ==+")
        self.log(f"Num isolated docs: {self.summary['Num isolated docs']}")
        self.log(f"Num unlabelled docs: {self.summary['Num unlabelled docs']}")

        self.log("\nApplying filters to dataset object...")
        pre_filter_num_rows = doc_dataset.num_rows
        doc_dataset = doc_dataset.filter(lambda row: row["doc_id"] not in invalid_docs)
        post_filter_num_rows = doc_dataset.num_rows

        self.log(
            f"Removed {pre_filter_num_rows-post_filter_num_rows}/{pre_filter_num_rows} [{(1-post_filter_num_rows/pre_filter_num_rows)*100:.2f}%] rows from `doc_dataset`"
        )

        self.save_file(file_type="doc_dataset", obj=doc_dataset)

        self.save_file(file_type="summary")

        self.log("\nLabel counts (post filtering):")
        counts = np.bincount(doc_dataset["y"])
        percentage = counts / counts.sum()

        self.summary["Label counts"] = dict()
        self.summary["Label counts summary"] = ""
        self.summary["Label counts"] = dict()
        self.summary["Label counts summary"] = ""
        for i in range(counts.shape[0]):
            self.summary["Label counts"][self.labels[i]] = counts[i]
            self.summary[
                "Label counts summary"
            ] += f"Portion of label '{self.labels[i]}': {counts[i]} ({percentage[i]*100:.2f}%)\n"

        self.log(self.summary["Label counts summary"])

        self.log("\nFinished applying filters.")
        end_time = time.time()
        hours, minutes, seconds = calc_elapsed_time(start_time, end_time)
        self.log(f"Time taken: {hours:02d}:{minutes:02d}:{seconds:02d}")
