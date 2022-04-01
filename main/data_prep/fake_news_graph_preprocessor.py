import argparse
import copy
from collections import defaultdict

from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file
from data_prep.fake_news_tsv_processor import LABELS
from data_prep.graph_io import FEATURE_TYPES
from data_prep.graph_preprocessor import GraphPreprocessor, USER_CONTEXTS_FILTERED


class FakeNewsGraphPreprocessor(GraphPreprocessor):

    def __init__(self, config):
        super().__init__(config)

        self.load_doc_splits()

        self.create_doc_id_dicts()
        self.create_follower_following_relationships()
        self.create_feature_matrix()
        self.create_adj_matrix()
        self.create_labels()
        self.create_split_masks()

    @property
    def labels(self):
        return LABELS

    def docs_to_adj(self, adj_matrix, edge_type):

        adj_matrix = copy.deepcopy(adj_matrix)
        edge_type = copy.deepcopy(edge_type)

        edge_list = []
        not_used = 0

        for file_path in self.get_engagement_files():
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

                    edge_list.append((doc_id, user_id))
                    edge_list.append((user_id, doc_id))

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

                    edge_list.append((user_id2, user_id1))
                    edge_list.append((user_id1, user_id2))

                    adj_matrix[user_id1, user_id2] = 1
                    adj_matrix[user_id2, user_id1] = 1
                    edge_type[user_id1, user_id2] = 3
                    edge_type[user_id2, user_id1] = 3

        return adj_matrix, edge_type, edge_list, not_found

    def get_feature_id_mapping(self, feature_ids):
        feature_id_mapping = defaultdict(list, {k: [] for k in self.user2id.values()})
        for count, file_path in enumerate(self.get_engagement_files()):
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

    complete_dir = COMPLETE_DIR
    tsv_dir = TSV_DIR

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=complete_dir,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=tsv_dir,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='gossipcop',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_users', type=int, default=30, help='Number (in K) of top users.')

    parser.add_argument('--top_users_excluded', type=int, default=1,
                        help='Percentage (in %) of top sharing users that are excluded (the bot users).')

    parser.add_argument('--valid_users', type=bool, default=True, help='Flag if only top K and users not sharing '
                                                                       'more than X% of any class should be used.')

    parser.add_argument('--feature_type', type=str, default='one-hot', help='The type of features to use.',
                        choices=FEATURE_TYPES)

    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of the vocabulary used (if one-hot).')

    parser.add_argument('--train-size', dest='train_size', type=float, default=0.7, help='Size of train split.')

    parser.add_argument('--val-size', dest='val_size', type=float, default=0.1, help='Size of validation split.')

    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2, help='Size of train split.')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeNewsGraphPreprocessor(args.__dict__)
