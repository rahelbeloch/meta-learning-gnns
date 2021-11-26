import argparse

import numpy as np

from data_prep.config import *
from data_prep.data_preprocess_utils import save_json_file, load_json_file
from data_prep.fake_news_tsv_processor import LABELS
from data_prep.graph_preprocessor import GraphPreprocessor


class FakeNewsGraphPreprocessor(GraphPreprocessor):

    def __init__(self, config):
        super().__init__(config)

        self.load_doc_splits()

        #if self.only_valid_users:
        #    self.filter_valid_users()
        #self.create_user_splits(max_users=100)
        #self.create_doc_id_dicts()
        #self.filter_contexts()
        #self.create_adj_matrix()
        self.create_feature_matrix()
        self.create_labels()
        self.create_split_masks()

    def labels(self):
        return LABELS

    @staticmethod
    def get_doc_key(name, name_type='dir'):
        if name_type == 'dir':
            return name.split('gossipcop-')[1].split('/')[-1]
        elif name_type == 'file':
            return name.split('.')[0]
        elif name_type == 'filepath':
            return name.split('/')[-1].split('.')[0]
        else:
            raise ValueError("Name type to get ID from is neither file, nor dir!")

    def create_labels(self):

        self.print_step('Creating labels')

        self.maybe_load_id_mappings()

        print("Loading doc2labels dictionary...")
        doc2labels = load_json_file(self.data_complete_path(DOC_2_LABELS_FILE_NAME))

        train_docs = self.train_docs + self.val_docs
        labels_list = np.zeros(len(train_docs), dtype=int)

        for doc_key, label in doc2labels.items():
            if doc_key not in train_docs:
                continue
            labels_list[self.doc2id[doc_key]] = label

        assert len(labels_list) == len(self.doc2id.keys()) - len(self.test_docs)
        print(f"\nLen of (train) labels = {len(labels_list)}")

        labels_file = self.data_complete_path(TRAIN_LABELS_FILE_NAME)
        print(f"\nLabels list construction done! Saving in : {labels_file}")
        save_json_file({'labels_list': list(labels_list)}, labels_file, converter=self.np_converter)

        # Create the all_labels file
        all_labels = np.zeros(self.n_total, dtype=int)
        all_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
        for doc_key in doc2labels.keys():
            if doc_key not in self.doc2id:
                continue
            all_labels[self.doc2id[doc_key]] = doc2labels[doc_key]

        print("\nSum of all labels = ", int(sum(all_labels)))
        print("Len of all labels = ", len(all_labels))

        print(f"\nAll labels list construction done! Saving in : {all_labels_file}")
        save_json_file({'all_labels': list(all_labels)}, all_labels_file, converter=self.np_converter)


if __name__ == '__main__':
    complete_dir = COMPLETE_small_DIR
    tsv_dir = TSV_small_DIR

    # complete_dir = COMPLETE_DIR
    # tsv_dir = TSV_DIR

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=complete_dir,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=tsv_dir,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='gossipcop',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_k', type=int, default=30, help='Number (in K) of top users.')

    parser.add_argument('--user_doc_threshold', type=float, default=0.3, help='Threshold defining how many articles '
                                                                              'of any class users may max have shared '
                                                                              'to be included in the graph.')

    parser.add_argument('--valid_users', type=bool, default=True, help='Flag if only top K and users not sharing '
                                                                       'more than X% of any class should be used.')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeNewsGraphPreprocessor(args.__dict__)
