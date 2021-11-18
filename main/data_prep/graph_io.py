import datetime

import nltk

nltk.download('punkt')

from data_prep.data_preprocess_utils import *

from data_prep.config import *

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']


class GraphIO:

    def __init__(self, dataset, raw_dir=RAW_DIR, tsv_dir=TSV_DIR, complete_dir=COMPLETE_DIR):
        self.dataset = dataset

        full_raw_path = os.path.join(raw_dir, dataset)
        if not os.path.exists(full_raw_path):
            raise ValueError(f"Wanting to preprocess data for dataset '{dataset}', but raw data in path"
                             f" with raw data '{full_raw_path}' does not exist!")

        self.data_raw_dir = raw_dir
        self.data_tsv_dir = self.create_dir(tsv_dir)
        self.data_complete_dir = self.create_dir(complete_dir)

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

    @staticmethod
    def create_dir(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
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
