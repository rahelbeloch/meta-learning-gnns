import os
import sys
import pickle

import torch
from datasets import load_from_disk

from data_prep import LOG_FILE
from data_prep.graph_io import GraphIO
from models import FeatureExtractor


class PostProcessing(GraphIO):
    def __init__(
        self, args, cur_fold: int, processed_or_structured, version, **super_kwargs
    ):
        self.cur_fold = cur_fold

        super().__init__(args, version=version, enforce_raw=False, **super_kwargs)

        self.fold_idx = self.load_file("split_idx")[self.cur_fold]

        os.makedirs(self.data_processed_path(), exist_ok=True)
        os.makedirs(self.data_structure_path(), exist_ok=True)

        if processed_or_structured in {"processed", "structured"}:
            self.processed_or_structured = processed_or_structured
        else:
            raise ValueError(
                "`processed_or_structured` must be one of {'processed', 'structured'}"
            )

    def data_processed_path(self, *parts):
        return super().data_processed_path(str(self.cur_fold), *parts)

    def data_structure_path(self, *parts):
        return super().data_structure_path(str(self.cur_fold), *parts)

    def save_file(self, file_type, obj=None):
        if file_type == "feature_extractor":
            torch.save(obj, self.data_processed_path("feature_extractor.pt"))

        elif file_type == "compressed_dataset":
            obj.save_to_disk(self.data_processed_path())

        else:
            super().save_file(file_type, obj)

    def load_file(self, file_type):
        if file_type == "feature_extractor":
            state_dict = torch.load(
                self.data_processed_path("feature_extractor.pt"),
                map_location="cpu",
            )

            obj = FeatureExtractor(**state_dict["hparams"])
            obj.load_state_dict(state_dict["state_dict"], strict=False)

        elif file_type == "compressed_dataset":
            obj = load_from_disk(self.data_processed_path())

        else:
            obj = super().load_file(file_type)

        return obj

    @property
    def _file_name(self):
        return self.__str__().lower() + ".pickle"

    def save(self):
        if self.processed_or_structured == "processed":
            save_path = self.data_processed_path(self._file_name)
        elif self.processed_or_structured == "structured":
            save_path = self.data_structure_path(self._file_name)

        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)

        if instance.processed_or_structured == "processed":
            save_path = instance.data_processed_path(instance._file_name)
        elif instance.processed_or_structured == "structured":
            save_path = instance.data_structure_path(instance._file_name)

        if save_path.exists():
            instance.log(f"Loading from:\n\t{save_path}")

            with open(save_path, "rb") as f:
                state_dict = pickle.load(f)

            for k, v in state_dict.items():
                instance.__setattr__(k, v)

            return instance
        else:
            raise ValueError(f"No file found at:\n\t{save_path}")

    def log(self, log_string):
        if self.processed_or_structured == "processed":
            log_dir = self.data_processed_path
        elif self.processed_or_structured == "structured":
            log_dir = self.data_structure_path

        sys.stdout.write(log_string + "\n")
        with open(log_dir(self.__str__().lower() + "_" + LOG_FILE), "a") as f:
            f.write(log_string + "\n")
