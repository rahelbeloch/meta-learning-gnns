import argparse
import csv
from collections import defaultdict

import nltk

from data_prep.config import *
from data_prep.data_preprocess_utils import sanitize_text
from data_prep.data_preprocessor import DataPreprocessor
from data_prep.graph_io import FEATURE_TYPES

# Report metrics: Macro, racism, sexism; None is majority class
LABELS = {0: 'racism', 1: 'sexism', 2: 'none'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for twitter hate speech.
    """

    def __init__(self, config, data_dir, tsv_dir, comp_dir, content_file):
        super().__init__(config, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=comp_dir)

        self.content_file = content_file

    @property
    def labels(self):
        return LABELS

    def filter_valid_users(self):
        """
        From the authors.txt file, loads all lines (containing a combination of tweet ID and user ID),
        counts how many document each user interacted with and identifies users that shared at least X% of
        the articles of any class. Also picks the top K active users.
        """

        self.print_step("Applying restrictions on users")

        print(f"Filtering users who in any class shared articles more than : {self.user_doc_threshold * 100}%")

        doc2labels = {}
        data_file = self.data_raw_path(self.dataset, self.content_file)
        with open(data_file, encoding='utf-8') as content_data:
            reader = csv.DictReader(content_data)

            # skip the header
            next(reader, None)

            for row in reader:
                doc2labels[row['id']] = int(row['annotation'])

        user_stats = defaultdict(lambda: {'racism': 0, 'sexism': 0, 'none': 0})
        used_docs = 0

        authors_file = self.data_raw_path(self.dataset, 'authors.txt')
        for count, author_entry in enumerate(open(authors_file, 'r').read().split('\n')):

            if len(author_entry) == 0:
                continue

            doc_key, user_key = author_entry.split()

            # only restrict users interacting with this document ID if we actually use this doc in our splits
            if doc_key not in doc2labels:
                continue

            used_docs += 1
            user_stats[user_key][self.labels[doc2labels[doc_key]]] += 1

        super().filter_users(user_stats, used_docs)

    def filter_documents(self, min_len):
        """
        Filters documents based on if they were shared by any user that is contained in 'valid_users'.
        For all documents that:
            - have content
            - are shared at least by 1 (valid) user
        extracts the document contents and saves then as TSV file. The .tsv file contains the fields:
        ID, article title, article content and the label.

        :param min_len: Minimum required length for articles.
        """

        self.print_step("Filtering documents and creating doc2label file")

        self.maybe_load_valid_users()

        doc2labels = {}
        contents = []

        no_content, no_valid_users, no_interactions = [], [], []
        invalid_min_length = []

        data_file = self.data_raw_path(self.dataset, self.content_file)
        with open(data_file, encoding='utf-8') as content_data:
            reader = csv.DictReader(content_data)

            # skip the header
            next(reader, None)

            for row in reader:
                tweet_id = row['id']
                tweet_content = row['tweet']
                annotation = int(row['annotation'])

                if len(tweet_content) == 0:
                    no_content.append(tweet_id)
                    continue

                # TODO: do this
                # # check if the engagement file contains at least 1 user that is in valid users
                # if len(set(users).intersection(self.valid_users)) == 0:
                #     no_valid_users.append(doc_name)
                #     continue

                text = sanitize_text(tweet_content)
                tokens = set(nltk.word_tokenize(text))
                if len(tokens) >= min_len:
                    contents.append([tweet_id, "_".join(tokens), annotation])
                    doc2labels[tweet_id] = annotation
                else:
                    invalid_min_length.append(tweet_id)

        print(f"Total docs without content : {len(no_content)}")
        print(f"Total docs without interactions (before valid users) : {len(no_interactions)}")
        print(f"Total docs without interactions (after valid users) : {len(no_valid_users)}")
        print(f"Total docs invalid and removed (length < {min_len}) = {len(set(invalid_min_length))}")

        assert len(doc2labels) == len(contents), "doc2labels is not of same length as contents list!"

        self.store_doc2label(doc2labels)
        self.store_doc_contents(contents)

    def create_user_splits(self, max_users=None):

        self.print_step("Creating user splits")

        self.maybe_load_valid_users()
        self.load_doc_splits()

        print("\nCollecting users for splits file..")

        train_users, val_users, test_users = set(), set(), set()

        authors_file = self.data_raw_path(self.dataset, 'authors.txt')
        for count, author_entry in enumerate(open(authors_file, 'r').read().split('\n')):

            if max_users is not None and (len(train_users) + len(test_users) + len(val_users)) >= max_users:
                break

            if len(author_entry) == 0:
                continue

            doc_key, author_id = author_entry.split()
            if not self.valid_user(author_id):
                continue

            if doc_key in self.train_docs:
                train_users.add(author_id)
            if doc_key in self.val_docs:
                val_users.add(author_id)
            if doc_key in self.test_docs:
                test_users.add(author_id)

        super().store_user_splits(train_users, test_users, val_users)

    def print_label_distribution(self, labels, split):
        racism, sexism, none = self.get_label_distribution(labels)

        denom = racism + sexism + none
        racism_avg = racism / denom
        sexism_avg = sexism / denom
        none_avg = none / denom

        print(f"\nRacism labels in {split} split  = {racism_avg * 100:.2f}% ({racism} samples)")
        print(f"Sexism labels in {split} split  = {sexism_avg * 100:.2f}%  ({sexism} samples)")
        print(f"None labels in {split} split  = {none_avg * 100:.2f}%  ({none} samples)")

    @staticmethod
    def get_label_distribution(labels):
        return (labels == 0).sum(), (labels == 1).sum(), (labels == 2).sum()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=COMPLETE_DIR,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=TSV_DIR,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='twitterHateSpeech',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_users', type=int, default=30, help='Number (in K) of top users.')

    parser.add_argument('--top_users_excluded', type=int, default=0,
                        help='Percentage (in %) of top sharing users that are excluded (the bot users).')

    parser.add_argument('--user_doc_threshold', type=float, default=0.3, help='Threshold defining how many articles '
                                                                              'of any class users may max have shared '
                                                                              'to be included in the graph.')

    parser.add_argument('--valid_users', type=bool, default=True, help='Flag if only top K and users not sharing '
                                                                       'more than X% of any class should be used.')

    parser.add_argument('--feature_type', type=str, default='one-hot', help='The type of features to use.',
                        choices=FEATURE_TYPES)

    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of the vocabulary used (if one-hot).')

    parser.add_argument('--train_size', type=float, default=0.0, help='Size of train split.')

    parser.add_argument('--val_size', type=float, default=0.25, help='Size of validation split.')

    parser.add_argument('--test_size', type=float, default=0.75, help='Size of train split.')

    args, unparsed = parser.parse_known_args()
    args = args.__dict__

    preprocessor = TSVPreprocessor(args, args['data_dir'], args['data_tsv_dir'], args['data_complete_dir'],
                                   'twitter_data_waseem_hovy.csv')

    if args['valid_users']:
        preprocessor.filter_valid_users()

    preprocessor.filter_documents(min_len=6)

    data = preprocessor.preprocess_documents()

    preprocessor.create_document_splits(data)

    preprocessor.create_user_splits()
