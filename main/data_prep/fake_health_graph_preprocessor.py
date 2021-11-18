import argparse
import os
from collections import defaultdict

import numpy as np
from scipy.sparse import load_npz

from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file, save_json_file
from data_prep.graph_preprocessor import GraphPreprocessor


class FakeHealthGraphPreprocessor(GraphPreprocessor):
    """
    Does all the preprocessing work to later be able to quickly load graph datasets from existing files.
    This includes creating and storing (in json files) the following components:
        - Engagement information (following/follower)
        - Document and user splits
        - Doc2id and id2doc dictionaries
        - Followers and following list of only the users who exist in the dataset
        - Adjacency matrix
        - Feature matrix
        - Labels
        - Split masks
    """

    @staticmethod
    def get_doc_key(name, name_type):
        return name.split('.')[0]

    def __init__(self, config):
        super().__init__(config)

        # self.aggregate_user_contexts()
        # if self.only_valid_users:
        #     self.filter_valid_users()
        # self.create_user_splits()
        # self.create_doc_id_dicts()
        # self.filter_contexts('ids')
        # self.create_adj_matrix()
        # self.create_feature_matrix()
        # self.create_labels()
        # self.create_split_masks()

    def aggregate_user_contexts(self):
        self.print_step("Aggregating follower/ing relations")

        src_dir = self.data_raw_path("engagements", self.dataset)
        if not os.path.exists(src_dir):
            raise ValueError(f'Source directory {src_dir} does not exist!')

        docs_users = defaultdict(set)
        count = 0
        for root, _, files in os.walk(src_dir):
            if root.endswith("replies"):
                continue
            for count, file in enumerate(files):
                if file.startswith('.'):
                    continue

                src_file = load_json_file(os.path.join(root, file))
                doc_name = root.split('/')[-2]
                docs_users[doc_name].update(src_file['user']['id'])
                if count % 10000 == 0:
                    print(f"{count} done")

        print(f"\nTotal tweets/re-tweets in the data set = {count}")
        self.save_user_doc_engagements(docs_users)

    def create_labels(self):
        """
        Create labels for each node of the graph
        """
        self.print_step("Creating labels")

        self.maybe_load_id_mappings()

        doc_labels = load_json_file(self.data_raw_path('reviews', self.dataset + '.json'))

        if self.n_total is None:
            adj_matrix = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k))
            self.n_total = adj_matrix.shape[0]
            del adj_matrix

        self.load_doc_splits()
        split_docs = self.train_docs + self.val_docs

        print("\nCreating doc2labels dictionary...")
        doc2labels = {}

        for count, doc in enumerate(doc_labels):
            if str(doc['news_id']) in split_docs:
                label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
                doc2labels[str(doc['news_id'])] = label

        # print(len(doc2labels.keys()))
        # print(len(doc2id.keys()) - len(doc_splits['test_docs']))
        assert len(doc2labels.keys()) == len(self.doc2id.keys()) - len(self.test_docs)
        print(f"\nLen of doc2labels = {len(doc2labels)}")

        self.save_labels(doc2labels)

    def save_labels(self, doc2labels):

        doc2labels_file = self.data_complete_path(DOC_2_LABELS_FILE_NAME)
        print(f"Saving doc2labels for {self.dataset} at: {doc2labels_file}")
        save_json_file(doc2labels, doc2labels_file)

        labels_list = np.zeros(self.n_total, dtype=int)
        for key, value in doc2labels.items():
            labels_list[self.doc2id[str(key)]] = value

        # Sanity Checks
        # print(sum(labels_list))
        # print(len(labels_list))
        # print(sum(labels_list[2402:]))
        # print(sum(labels_list[:2402]))

        labels_file = self.data_complete_path(TRAIN_LABELS_FILE_NAME)
        print(f"\nLabels list construction done! Saving in : {labels_file}")
        save_json_file({'labels_list': list(labels_list)}, labels_file, converter=self.np_converter)

        # Create the all_labels file
        all_labels = np.zeros(self.n_total, dtype=int)
        all_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
        for doc in doc2labels.keys():
            all_labels[self.doc2id[str(doc)]] = doc2labels[str(doc)]

        print("\nSum of labels this test set = ", int(sum(all_labels)))
        print("Len of labels = ", len(all_labels))

        print(f"\nall_labels list construction done! Saving in : {all_labels_file}")
        save_json_file({'all_labels': list(all_labels)}, all_labels_file, converter=self.np_converter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_raw_dir', type=str, default=RAW_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=COMPLETE_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=TSV_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='HealthStory',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_k', type=int, default=50, help='Number of top users.')

    parser.add_argument('--user_doc_threshold', type=float, default=0.3, help='Threshold defining how many articles '
                                                                              'of any class users may max have shared '
                                                                              'to be included in the graph.')

    parser.add_argument('--valid_users', type=bool, default=True, help='Flag if only top K and users not sharing '
                                                                       'more than X% of any class should be used.')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeHealthGraphPreprocessor(args.__dict__)
