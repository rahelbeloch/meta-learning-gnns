import argparse
import copy
from collections import defaultdict

from data_prep.config import *
from data_prep.graph_io import FEATURE_TYPES
from data_prep.graph_preprocessor import GraphPreprocessor
from data_prep.twitter_tsv_processor import LABELS


class TwitterGraphPreprocessor(GraphPreprocessor):

    def __init__(self, config):
        super().__init__(config)

        self.load_doc_splits()

        self.create_doc_id_dicts()
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

        edge_list, not_used = [], 0

        authors_file = self.data_raw_path(self.dataset, 'authors.txt')
        for count, author_entry in enumerate(open(authors_file, 'r').read().split('\n')):

            if len(author_entry) == 0:
                continue

            doc_key, user = author_entry.split()

            if doc_key not in self.doc2id or user not in self.user2id:
                not_used += 1
                continue

            if doc_key in self.test_docs:
                # no connections between users and test documents!
                continue

            doc_id, user_id = self.doc2id[doc_key], self.user2id[user]

            # for DGL graph creation; edges are reversed later
            edge_list.append((doc_id, user_id))
            # edge_list.append((user_id, doc_id))

            adj_matrix[doc_id, user_id] = 1
            adj_matrix[user_id, doc_id] = 1
            edge_type[doc_id, user_id] = 2
            edge_type[user_id, doc_id] = 2

        return adj_matrix, edge_type, edge_list, not_used

    def users_to_adj(self, adj_matrix, edge_type, edge_list):

        adj_matrix = copy.deepcopy(adj_matrix)
        edge_type = copy.deepcopy(edge_type)
        edge_list = copy.deepcopy(edge_list)

        not_found = 0

        authors_file = self.data_raw_path(self.dataset, 'authors.edgelist')
        for count, author_entry in enumerate(open(authors_file, 'r').read().split('\n')):
            if len(author_entry) == 0:
                continue

            user1, user2 = author_entry.split()

            if user1 not in self.user2id or user2 not in self.user2id:
                not_found += 1
                continue

            user_id1, user_id2 = self.user2id[user1], self.user2id[user2]

            # for DGL graph creation; edges are reversed later
            edge_list.append((user_id2, user_id1))
            # edge_list.append((user_id1, user_id2))

            adj_matrix[user_id1, user_id2] = 1
            adj_matrix[user_id2, user_id1] = 1
            edge_type[user_id1, user_id2] = 3
            edge_type[user_id2, user_id1] = 3

        return adj_matrix, edge_type, edge_list, not_found

    def get_feature_id_mapping(self, feature_ids):
        feature_id_mapping = defaultdict(lambda: [])
        authors_file = self.data_raw_path(self.dataset, 'authors.txt')
        for count, author_entry in enumerate(open(authors_file, 'r').read().split('\n')):
            if len(author_entry) == 0:
                continue

            doc_key, user_key = author_entry.split()
            if doc_key not in self.doc2id or user_key not in self.user2id or doc_key not in feature_ids:
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

    parser.add_argument('--data_set', type=str, default='twitterHateSpeech',
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

    parser.add_argument('--train_size', type=float, default=0.0, help='Size of train split.')

    parser.add_argument('--val_size', type=float, default=0.25, help='Size of validation split.')

    parser.add_argument('--test_size', type=float, default=0.75, help='Size of train split.')

    args, unparsed = parser.parse_known_args()

    preprocessor = TwitterGraphPreprocessor(args.__dict__)
