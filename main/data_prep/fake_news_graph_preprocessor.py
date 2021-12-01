import argparse
import copy
from collections import defaultdict
from json import JSONDecodeError

from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file
from data_prep.fake_news_tsv_processor import LABELS
from data_prep.graph_io import FEATURE_TYPES
from data_prep.graph_preprocessor import GraphPreprocessor, USER_CONTEXTS_FILTERED


class FakeNewsGraphPreprocessor(GraphPreprocessor):

    def __init__(self, config, max_users=None):
        super().__init__(config)

        self.load_doc_splits()

        if self.only_valid_users:
            self.filter_valid_users()
        self.create_user_splits(max_users)
        self.create_doc_id_dicts()
        self.filter_contexts()
        self.create_feature_matrix()
        self.create_adj_matrix()
        self.create_labels()
        self.create_split_masks()

    @property
    def labels(self):
        return LABELS

    def filter_valid_users(self):
        """
        From the user engagements folder, loads all document files (containing user IDs who interacted with
        the respective document), counts how many document each user interacted with and identifies users
        that shared at least X% of the articles of any class. Also picks the top K active users.
        """

        self.print_step("Applying restrictions on users")

        print(f"Filtering users who in any class shared articles more than : {self.user_doc_threshold * 100}%")

        doc2labels = load_json_file(self.data_complete_path(DOC_2_LABELS_FILE_NAME))
        user_stats = defaultdict(lambda: {'fake': 0, 'real': 0})

        used_docs = 0
        for count, file_path in enumerate(self.data_tsv_path('engagements').glob('*')):

            # only restrict users interacting with this document ID if we actually use this doc in our splits
            doc_key = file_path.stem
            if not self.doc_used(doc_key):
                continue

            used_docs += 1

            try:
                src_file = load_json_file(file_path)
            except UnicodeDecodeError:
                # TODO: fix this error / keep track of files for which this happens
                print(f"Exception for doc {doc_key}")
                continue
            except JSONDecodeError:
                # TODO: fix this error / keep track of files for which this happens
                print(f"Exception for doc {doc_key}")
                continue

            users = src_file['users']

            for u in users:
                if doc_key in doc2labels:
                    user_stats[u][self.labels[doc2labels[doc_key]]] += 1

        super().filter_users(user_stats, used_docs)

    def create_user_splits(self, max_users):
        """
        Walks through all users that interacted with documents and, divides them on train/val/test splits.
        """

        self.print_step("Creating user splits")

        self.maybe_load_valid_users()

        print("\nCollecting users for splits file..")

        train_users, val_users, test_users = set(), set(), set()

        # walk through user-doc engagement files created before
        files = list(self.data_tsv_path('engagements').glob('*'))
        print_iter = int(len(files) / 20)

        for count, file_path in enumerate(files):
            if max_users is not None and (len(train_users) + len(test_users) + len(val_users)) >= max_users:
                break

            try:
                src_file = load_json_file(file_path)
            except UnicodeDecodeError:
                # TODO: fix this error / keep track of files for which this happens
                print("Exception")
                continue

            if count % print_iter == 0:
                print("{} / {} done..".format(count + 1, len(files)))

            users = src_file['users']

            users_filtered = [u for u in users if self.valid_user(u)]

            doc_key = file_path.stem
            if doc_key in self.train_docs:
                train_users.update(users_filtered)
            if doc_key in self.val_docs:
                val_users.update(users_filtered)
            if doc_key in self.test_docs:
                test_users.update(users_filtered)

        super().store_user_splits(train_users, test_users, val_users)

    def docs_to_adj(self, adj_matrix, edge_type):

        adj_matrix = copy.deepcopy(adj_matrix)
        edge_type = copy.deepcopy(edge_type)

        edge_list = []
        not_used = 0

        for count, file_path in enumerate(self.data_tsv_path('engagements').glob('*')):
            doc_key = file_path.stem
            if doc_key == '':
                continue
            src_file = load_json_file(file_path)
            users = map(str, src_file['users'])
            for user in users:
                if doc_key in self.test_docs:
                    # no connections between users and test documents!
                    continue

                if doc_key in self.doc2id and user in self.user2id:
                    doc_id = self.doc2id[doc_key]
                    user_id = self.user2id[user]

                    # for DGL graph creation; edges are reversed later
                    edge_list.append((doc_id, user_id))
                    # edge_list.append((user_id, doc_id))

                    adj_matrix[doc_id, user_id] = 1
                    adj_matrix[user_id, doc_id] = 1
                    edge_type[doc_id, user_id] = 2
                    edge_type[user_id, doc_id] = 2
                else:
                    not_used += 1

        return adj_matrix, edge_type, edge_list, not_used

    def users_to_adj(self, adj_matrix, edge_type, edge_list):

        adj_matrix = copy.deepcopy(adj_matrix)
        edge_type = copy.deepcopy(edge_type)
        edge_list = copy.deepcopy(edge_list)

        not_found = 0

        for user_context in USER_CONTEXTS_FILTERED:
            print(f"\n    - from {user_context} folder...")
            for count, file_path in enumerate(self.data_raw_path(user_context).glob('*')):
                src_file = load_json_file(file_path)
                user_id = str(int(src_file['user_id']))
                if user_id not in self.user2id:
                    continue
                followers = src_file['followers'] if user_context == 'user_followers_filtered' else \
                    src_file['following']
                for follower in list(map(str, followers)):
                    if follower not in self.user2id:
                        not_found += 1
                        continue

                    user_id1, user_id2 = self.user2id[user_id], self.user2id[follower]

                    # for DGL graph creation; edges are reversed later
                    edge_list.append((user_id2, user_id1))
                    # edge_list.append((user_id1, user_id2))

                    adj_matrix[user_id1, user_id2] = 1
                    adj_matrix[user_id2, user_id1] = 1
                    edge_type[user_id1, user_id2] = 3
                    edge_type[user_id2, user_id1] = 3

        return adj_matrix, edge_type, edge_list, not_found

    def get_feature_id_mapping(self, feature_ids):
        feature_id_mapping = defaultdict(list, {k: [] for k in self.user2id.values()})
        for count, file_path in enumerate(self.data_tsv_path('engagements').rglob('*.json')):
            doc_users = load_json_file(file_path)
            doc_key = file_path.stem

            # Each user of this doc has its features as the features of the doc
            if doc_key not in self.doc2id or doc_key not in feature_ids:
                continue

            for user in doc_users['users']:
                user_key = str(user)
                if user_key not in self.user2id:
                    continue

                feature_id_mapping[self.user2id[user_key]].append(feature_ids[doc_key])

        return feature_id_mapping


if __name__ == '__main__':
    # complete_dir = COMPLETE_small_DIR
    # tsv_dir = TSV_small_DIR
    # max_nr_users = 2000

    complete_dir = COMPLETE_DIR
    tsv_dir = TSV_DIR
    max_nr_users = None

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

    parser.add_argument('--feature_type', type=str, default='one-hot', help='The type of features to use.',
                        choices=FEATURE_TYPES)

    parser.add_argument('--max_vocab', type=int, default=10000, help='Size of the vocabulary used (if one-hot).')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeNewsGraphPreprocessor(args.__dict__, max_nr_users)
