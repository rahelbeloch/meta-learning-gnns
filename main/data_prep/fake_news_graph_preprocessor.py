import argparse
import os
from collections import defaultdict

import pandas as pd

from data_prep.config import *
from data_prep.graph_io import GraphPreprocessor


class FakeNewsGraphPreprocessor(GraphPreprocessor):

    def __init__(self, config):
        super().__init__(config)

        self.aggregate_user_contexts()
        self.create_doc_user_splits()
        self.create_doc_id_dicts()
        self.filter_user_contexts()
        self.create_adjacency_matrix()

        # self.create_feature_matrix()
        # self.create_labels()
        # self.create_split_masks()

        # self.create_dgl_graph()

    def get_doc_key(self, name, name_type='dir'):
        if name_type == 'dir':
            return name.split('gossipcop-')[1].split('/')[-1]
        elif name_type == 'file':
            return name.split('.')[0]
        else:
            raise ValueError("Name type to get ID from is neither file, nor dir!")

    def aggregate_user_contexts(self):
        self.print_step("Aggregating follower/ing relations")

        dest_dir = self.data_complete_path("engagements")
        if not os.path.exists(dest_dir):
            print(f"Creating destination dir:  {dest_dir}\n")
            os.makedirs(dest_dir)

        docs_users = defaultdict(set)
        count = 0
        for user_context in ['tweets', 'retweets']:
            print("\nIterating over : ", user_context)

            src_dir = self.data_raw_path(self.dataset, user_context)
            if not os.path.exists(src_dir):
                raise ValueError(f'Source directory {src_dir} does not exist!')

            for root, dirs, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    user_ids = pd.read_csv(os.path.join(root, file))['user_id']
                    user_ids = list(set([s for s in user_ids if isinstance(s, int)]))

                    doc = file.split('.')[0]
                    docs_users[doc].update(user_ids)
                    if count == 1:
                        print(doc, docs_users[doc])
                    if count % 2000 == 0:
                        print(f"{count} done")

        self.save_user_docs(count, dest_dir, docs_users)

    def create_doc_user_splits(self):
        self.create_user_splits(self.data_raw_path(self.dataset, 'complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_raw_dir', type=str, default=RAW_DIR,
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=COMPLETE_DIR,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=TSV_DIR,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='gossipcop',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_k', type=int, default=50, help='Number of top users.')

    parser.add_argument('--exclude_frequent', type=bool, default=False, help='TODO')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeNewsGraphPreprocessor(args.__dict__)
