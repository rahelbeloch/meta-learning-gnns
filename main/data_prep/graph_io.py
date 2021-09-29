import os

from data_prep.config import RAW_DIR, TSV_DIR


class GraphIO:

    def __init__(self, dataset, raw_dir=RAW_DIR, tsv_dir=TSV_DIR, complete_dir=None):
        self.dataset = dataset

        self.data_raw_dir = self.create_dir(raw_dir)
        self.data_tsv_dir = self.create_dir(tsv_dir)
        self.data_complete_dir = self.create_dir(complete_dir)

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

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

    def print_header(self, header_title):
        print(f'\n\n{"-" * 50}\n{header_title}\n{"-" * 50}')

    @staticmethod
    def print_step_1(step_title):
        print(f'\n{"=" * 50}\n \t\t{step_title}\n{"=" * 50}')

    @staticmethod
    def calc_elapsed_time(start, end):
        hours, rem = divmod(end - start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)
