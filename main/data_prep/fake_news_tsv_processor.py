import argparse
import json
from collections import defaultdict

import nltk

from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file, sanitize_text
from data_prep.data_preprocessor import DataPreprocessor
from data_prep.graph_io import FEATURE_TYPES

LABELS = {0: 'fake', 1: 'real'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for gossipcop.
    """

    def __init__(self, config, data_dir, tsv_dir, comp_dir):
        super().__init__(config, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=comp_dir)

    @property
    def labels(self):
        return LABELS

    def filter_valid_users(self):
        """
        From the user engagements folder, loads all document files (containing user IDs who interacted with
        the respective document), counts how many document each user interacted with and filter out users:
            - identify users that shared at least X% of the articles of any class
            - exclude top 3% sharing users (these are bots that have shared almost everything)
            - pick from the remaining the top K active users
        """

        self.print_step("Applying restrictions on users")

        print(f"Filtering users who in any class shared articles more than : {self.user_doc_threshold * 100}%")

        user_stats = defaultdict(lambda: {'fake': 0, 'real': 0})

        # no_label = []
        n_docs = 0

        for file_path in self.get_engagement_files():

            # only restrict users interacting with this document ID if we actually use this doc in our splits
            doc_key = file_path.stem

            is_fake = self.is_label(doc_key, 'fake')
            is_real = self.is_label(doc_key, 'real')

            if not is_fake and not is_real:
                # no_label.append(doc_key)
                # raise ValueError(f"Found document which does not have a label: {doc_key}")
                continue

            n_docs += 1

            label = 'fake' if is_fake else 'real'

            for u in load_json_file(file_path)['users']:
                user_stats[u][label] += 1

        # print(f"Total docs without label : {len(no_label)}")

        super().filter_users(user_stats, n_docs)

    def is_label(self, doc_key, label):
        return (self.data_raw_path(self.dataset, label) / doc_key / 'news content.json').exists()

    def filter_documents(self, min_len):
        """
        Filters documents based on if they were shared by any user that is contained in 'valid_users'.
        For all documents that:
            - have content
            - are shared at least by 1 (valid) user
        extracts the document contents and saves then as TSV file.

        :param min_len: Minimum required length for articles.
        :return:
        """

        self.print_step("Filtering documents and creating doc2label file")

        self.maybe_load_non_interaction_docs()
        self.maybe_load_valid_users()

        dest_dir = self.data_tsv_path('engagements')

        doc2labels = {}
        contents = []

        no_content, no_valid_users, no_interactions = [], [], []
        invalid_min_length = []

        for label in LABELS.keys():
            # load all files from this label folder
            for folder_name in self.data_raw_path(self.dataset, LABELS[label]).glob('*'):
                doc_name = folder_name.stem

                file_contents = folder_name / 'news content.json'
                if not file_contents.exists():
                    no_content.append(doc_name)
                    continue

                if doc_name in self.non_interaction_docs:
                    no_interactions.append(doc_name)
                    continue

                # get the engagement file
                document_user_file = dest_dir.joinpath(f'{str(doc_name)}.json')
                users = load_json_file(document_user_file)['users']

                # check if the engagement file contains at least 1 user that is in valid users
                if len(set(users).intersection(self.valid_users)) == 0:
                    no_valid_users.append(doc_name)
                    continue

                with open(file_contents, 'r') as f:
                    doc = json.load(f)
                    text = sanitize_text(doc['text'])
                    tokens = set(nltk.word_tokenize(text))
                    if len(tokens) >= min_len:
                        contents.append([doc_name, "_".join(tokens), label])
                        doc2labels[doc_name] = label
                    else:
                        invalid_min_length.append(doc_name)

        print(f"Total docs without content : {len(no_content)}")
        print(f"Total docs without interactions (before valid users) : {len(no_interactions)}")
        print(f"Total docs without interactions (after valid users) : {len(no_valid_users)}")
        print(f"Total docs invalid and removed (length < {min_len}) = {len(set(invalid_min_length))}")

        assert len(doc2labels) == len(contents), "doc2labels is not of same length as contents list!"

        self.store_doc2label(doc2labels)
        self.store_doc_contents(contents)


if __name__ == '__main__':
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_train_nodes = int(COMPLETE_small_DIR.split('-')[1])
    # max_nr_users = 2000

    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    num_train_nodes = None
    max_users = None

    min_len = 25

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=complete_dir,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=tsv_dir,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='gossipcop', help='The name of the dataset we want to process.')

    parser.add_argument('--top_users', type=int, default=30, help='Number (in K) of top users.')

    parser.add_argument('--top_users_excluded', type=int, default=1,
                        help='Percentage (in %) of top sharing users that are excluded (the bot users).')

    parser.add_argument('--user_doc_threshold', type=float, default=0.3, help='Threshold defining how many articles '
                                                                              'of any class users may max have shared '
                                                                              'to be included in the graph.')

    parser.add_argument('--valid_users', type=bool, default=True, help='Flag if only top K and users not sharing '
                                                                       'more than X% of any class should be used.')

    parser.add_argument('--feature_type', type=str, default='one-hot', help='The type of features to use.',
                        choices=FEATURE_TYPES)

    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of the vocabulary used (if one-hot).')

    parser.add_argument('--train-size', dest='train_size', type=float, default=0.875, help='Size of train split.')

    parser.add_argument('--val-size', dest='val_size', type=float, default=0.125, help='Size of validation split.')

    parser.add_argument('--test-size', dest='test_size', type=float, default=0.0, help='Size of train split.')

    args, unparsed = parser.parse_known_args()
    args = args.__dict__

    preprocessor = TSVPreprocessor(args, args['data_dir'], args['data_tsv_dir'], args['data_complete_dir'])

    preprocessor.aggregate_user_contexts()

    if args['valid_users']:
        preprocessor.filter_valid_users()

    preprocessor.filter_documents(min_len=min_len)

    data = preprocessor.preprocess_documents(num_train_nodes=num_train_nodes)

    preprocessor.create_document_splits(data)

    preprocessor.create_user_splits(max_users)
