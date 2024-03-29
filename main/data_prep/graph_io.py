import os
import shutil
import sys
import pickle
from pathlib import Path

import datasets
from datasets import Dataset
from scipy.sparse import save_npz, load_npz

from data_prep import (
    SUMMARY_FILE,
    LOG_FILE,
    FEATURE_TYPES,
    DOC_2_CONTENT_FILE,
    DOC_2_LABELS_FILE,
    DOC_2_USERS_FILE,
    USER_2_DOCS_FILE,
    INVALID_DOCS_FILE,
    INVALID_USERS_FILE,
    SPLIT_ID_FILE,
    USER_2_USERS_FILE,
    NODE_ID_2_TYPE_FILE,
    DOC_2_NODE_ID_FILE,
    USER_2_NODE_ID_FILE,
    EDGE_LIST_FILE,
    ADJACENCY_MATRIX_FILE,
    EDGE_TYPE_FILE,
)
from data_prep.tokenizers import OneHotTokenizer, LMTokenizer
from utils.io import save_json_file, load_json_file, create_dir


class GraphIO:
    def __init__(
        self,
        args,
        version=None,
        overwrite: bool = False,
        enforce_raw=True,
        delay_making_subdirs: bool = False,
    ):
        self.dataset = args["dataset"]
        self.version = version
        self.overwrite = overwrite

        self._delay_making_subdirs = delay_making_subdirs

        # Ensure the directory structure is in place ===========================
        self.raw_data_dir = Path(args["raw_data_dir"])
        if enforce_raw and not (self.raw_data_dir / self.dataset).exists():
            raise ValueError(
                f"Wanting to preprocess data for dataset '{self.dataset}', but raw data in path"
                f" with raw data '{self.raw_data_dir / self.dataset}' does not exist!"
            )

        self.data_dir = args["processed_data_dir"]
        data_path = Path(self.data_dir)

        self.data_tsv_dir = create_dir(
            data_path / args["tsv_dir"] / self.dataset
        ).parent
        self.data_complete_dir = create_dir(
            data_path / args["complete_dir"] / self.dataset
        ).parent
        self.data_processed_dir = create_dir(
            data_path / args["processed_dir"] / self.dataset
        ).parent
        self.data_structure_dir = create_dir(
            data_path / args["structure_dir"] / self.dataset
        ).parent

        # Name of the specific dataset version =================================
        self.seed = args["seed"]
        data_dir_name = f"seed[{self.seed}]"

        self.num_splits = args["num_splits"]
        data_dir_name += f"_splits[{self.num_splits}]"

        # Filters
        self.filters_str = ""

        self.min_len = args["min_len"]
        data_dir_name += f"_minlen[{self.min_len}]"
        self.filters_str += f"minlen[{self.min_len}]"

        self.filter_out_isolated_docs = args["filter_out_isolated_docs"]
        data_dir_name += f"_filterisolated[{self.filter_out_isolated_docs}]"
        self.filters_str += f"_filterisolated[{self.filter_out_isolated_docs}]"

        self.top_users = args["top_users"]
        data_dir_name += f"_topk[{self.top_users}]"
        self.filters_str += f"_topk[{self.top_users}]"

        self.top_users_excluded = args["top_users_excluded"]
        data_dir_name += f"_topexcl[{self.top_users_excluded}]"
        self.filters_str += f"_topexcl[{self.top_users_excluded}]"

        self.user_doc_threshold = args["user_doc_threshold"]
        data_dir_name += f"_userdoc[{int(self.user_doc_threshold * 100)}]"
        self.filters_str += f"_userdoc[{int(self.user_doc_threshold * 100)}]"

        self.feature_type = args["feature_type"]
        if self.feature_type not in FEATURE_TYPES:
            raise ValueError(
                f"Trying to create features of type {self.feature_type} which is not supported!"
            )
        data_dir_name += f"_featuretype[{self.feature_type}]"

        if args["feature_type"] != "one-hot":
            self.vocab_origin = "external"
        else:
            self.use_joint_vocab = args["use_joint_vocab"]
            if self.use_joint_vocab:
                self.vocab_origin = "joint"
            else:
                self.vocab_origin = args["dataset"]

        self.compression = args["compression"]
        self.vocab_size = args["vocab_size"]
        self.compressed_size = args["compressed_size"]
        data_dir_name += f"_vocab[{self.vocab_origin}][{self.compression.replace('/', '_')}][{self.vocab_size}x{self.compressed_size}]"

        self.user_compression_pre_or_post = args["pre_or_post_compression"]
        self.user2doc_aggregator = args["user2doc_aggregator"]
        data_dir_name += f"_userfeatures[{self.user_compression_pre_or_post}][{self.user2doc_aggregator}]"

        if self.version is not None:
            data_dir_name += f"_version[{str(version)}]"

        self.data_dir_name = Path(data_dir_name)

        # Actually create the dataset directory structure ======================
        if not self._delay_making_subdirs:
            create_dir(self.data_tsv_path())
            create_dir(self.data_complete_path())
            create_dir(self.data_processed_path())
            create_dir(self.data_structure_path())

        # Some additional, but important arguments =============================
        self.num_classes = args["num_classes"]
        self.class_weights = args["class_weights"]
        self.labels = args["labels"]

        # The summary stores all preprocessing actions =========================
        self.summary = self.load_file("summary")

    def change_data_dir(self, args, verbose: bool = True):
        if verbose:
            print("Changing data directories")
            print(f"Raw data: {self.raw_data_dir} -> {args['raw_data_dir']}")
            print(f"Processed data: {self.data_dir} -> {args['processed_data_dir']}")

        self.raw_data_dir = Path(args["raw_data_dir"])

        self.data_dir = args["processed_data_dir"]
        data_path = Path(self.data_dir)

        self.data_tsv_dir = create_dir(
            data_path / args["tsv_dir"] / self.dataset
        ).parent
        self.data_complete_dir = create_dir(
            data_path / args["complete_dir"] / self.dataset
        ).parent
        self.data_processed_dir = create_dir(
            data_path / args["processed_dir"] / self.dataset
        ).parent
        self.data_structure_dir = create_dir(
            data_path / args["structure_dir"] / self.dataset
        ).parent

    def data_raw_path(self, *parts):
        return self.raw_data_dir.joinpath(*parts)

    def data_tsv_path(self, *parts):
        return self.data_tsv_dir.joinpath(self.dataset, self.data_dir_name, *parts)

    def data_complete_path(self, *parts):
        return self.data_complete_dir.joinpath(self.dataset, self.data_dir_name, *parts)

    def data_processed_path(self, *parts):
        return self.data_processed_dir.joinpath(
            self.dataset, self.data_dir_name, *parts
        )

    def data_structure_path(self, *parts):
        return self.data_structure_dir.joinpath(
            self.dataset, self.data_dir_name, *parts
        )

    def reset(self):
        for dir_path in [
            self.data_tsv_path(),
            self.data_complete_path(),
            self.data_processed_path(),
        ]:
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

    def get_engagement_files(self):
        return self.data_tsv_path("engagements").glob("*")

    def log(self, log_string):
        sys.stdout.write(log_string + "\n")
        if not self._delay_making_subdirs:
            with open(self.data_complete_path(LOG_FILE), "a") as f:
                f.write(log_string + "\n")

    def print_step(self, step_title):
        self.log(
            f'\n{"-" * 100}\n \t\t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}'
        )

    def _get_file_name(self, file_type: str) -> str:
        if file_type == "summary":
            return self.data_complete_path(SUMMARY_FILE)

        elif file_type == "doc_dataset":
            return self.data_tsv_path()

        elif file_type == "tokenizer":
            tokenizer_file = f"origin[{self.vocab_origin}]"
            tokenizer_file += f"_featuretype[{self.feature_type}]"
            tokenizer_file += f"_vocabsize[{self.vocab_size}]"
            tokenizer_file += ".pickle"

            return self.data_processed_dir.joinpath(
                self.dataset, self.data_dir_name, tokenizer_file
            )

        elif file_type == "doc2content":
            return self.data_complete_path(DOC_2_CONTENT_FILE)

        elif file_type == "doc2labels":
            return self.data_complete_path(DOC_2_LABELS_FILE)

        elif file_type == "doc2users":
            return self.data_complete_path(DOC_2_USERS_FILE)

        elif file_type == "user2docs":
            return self.data_complete_path(USER_2_DOCS_FILE)

        elif file_type == "user2users":
            return self.data_complete_path(USER_2_USERS_FILE)

        elif file_type == "nodeid2type":
            return self.data_complete_path(NODE_ID_2_TYPE_FILE)

        elif file_type == "doc2nodeid":
            return self.data_complete_path(DOC_2_NODE_ID_FILE)
        elif file_type == "user2nodeid":
            return self.data_complete_path(USER_2_NODE_ID_FILE)

        elif file_type == "invalid_docs":
            return self.data_complete_path(INVALID_DOCS_FILE)

        elif file_type == "invalid_users":
            return self.data_complete_path(INVALID_USERS_FILE)

        elif file_type == "split_idx":
            return self.data_complete_path(SPLIT_ID_FILE)

        elif file_type == "edge_list":
            return self.data_complete_path(EDGE_LIST_FILE)

        elif file_type == "adj_matrix":
            return self.data_complete_path(ADJACENCY_MATRIX_FILE)

        elif file_type == "edge_type":
            return self.data_complete_path(EDGE_TYPE_FILE)

        else:
            raise NotImplementedError(f"No file name for file type: `{file_type}`")

    def save_file(self, file_type, obj=None):
        if self._delay_making_subdirs:
            return

        file_path = self._get_file_name(file_type)

        if file_type == "summary":
            with open(file_path, "wb") as f:
                pickle.dump(self.summary, f)

        elif file_type == "doc_dataset":
            obj.save_to_disk(file_path)

        elif file_type == "tokenizer":
            if self.feature_type == "one-hot":
                save_json_file(obj.vocab, self.data_complete_path("vocab.json"))

            obj.save(file_path)

        elif file_type in {"doc2content", "doc2labels", "doc2nodeid"}:
            save_json_file(obj, file_path)

        elif file_type in {
            "doc2users",
            "user2docs",
            "user2users",
            "nodeid2type",
            "user2nodeid",
            "split_idx",
            "edge_list",
        }:
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)

        elif file_type in {"invalid_docs", "invalid_users"}:
            save_json_file(list(obj), file_path)

        elif file_type in {"adj_matrix", "edge_type"}:
            save_npz(file_path, obj)

        else:
            raise NotImplementedError(f"Cannot save this file type: `{file_type}`")

    def load_file(self, file_type):
        file_path = self._get_file_name(file_type)

        if file_type == "summary":
            if file_path.exists():
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)
            else:
                obj = dict()

        elif file_type == "doc_dataset":
            obj = Dataset.load_from_disk(file_path)

        elif file_type == "tokenizer":
            if self.feature_type == "one-hot":
                obj = OneHotTokenizer.load(file_path)
            elif self.feature_type == "lm-embeddings":
                obj = LMTokenizer.load(file_path)

        elif file_type in {"doc2content", "doc2labels", "doc2nodeid"}:
            obj = load_json_file(file_path)

        elif file_type in {
            "doc2users",
            "user2docs",
            "user2users",
            "nodeid2type",
            "user2nodeid",
            "split_idx",
            "edge_list",
        }:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)

        elif file_type in {"invalid_docs", "invalid_users"}:
            obj = load_json_file(file_path)
            obj = set(obj)

        elif file_type in {"adj_matrix", "edge_type"}:
            obj = load_npz(file_path)

        else:
            raise NotImplementedError(f"Cannot load this file type: `{file_type}`")

        return obj
