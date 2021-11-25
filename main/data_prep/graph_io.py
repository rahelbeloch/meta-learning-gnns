import abc
import datetime

import nltk
from importlib_resources import files

nltk.download('punkt')

from data_prep.data_preprocess_utils import *

from data_prep.config import *


class GraphIO:

    def __init__(self, dataset, data_dir, tsv_dir=TSV_DIR, complete_dir=COMPLETE_DIR):
        self.dataset = dataset

        data_path = files(data_dir)
        print(f"data path {data_path}, exists: {(data_path / '').exists()}")
        raw_path = data_path / RAW_DIR
        if not (raw_path / dataset).exists():
            raise ValueError(f"Wanting to preprocess data for dataset '{dataset}', but raw data in path"
                             f" with raw data '{raw_path / dataset}' does not exist!")

        self.data_raw_dir = raw_path
        self.data_tsv_dir = self.create_dir(data_path / tsv_dir)
        self.data_complete_dir = self.create_dir(data_path / complete_dir)

        self.valid_docs = None

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

    @staticmethod
    def create_dir(dir_name):
        if not dir_name.exists():
            dir_name.mkdir()
        return dir_name

    def data_raw_path(self, *parts):
        return os.path.join(self.data_raw_dir, *parts)

    def data_tsv_path(self, *parts):
        return os.path.join(self.data_tsv_dir, self.dataset, *parts)

    def data_complete_path(self, *parts):
        return os.path.join(self.data_complete_dir, self.dataset, *parts)

    @staticmethod
    def np_converter(obj):
        """
        A converter which can be used when dumping JSON strings to files, as JSON can not work with numpy data types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError
