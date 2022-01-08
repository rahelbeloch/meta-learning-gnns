import abc
import copy
import csv
import time
from collections import OrderedDict

import torch
from scipy.sparse import lil_matrix, save_npz

from data_prep.config import *
from data_prep.data_preprocess_utils import *
from data_prep.graph_io import GraphIO

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']


class GraphPreprocessor(GraphIO):

    def __init__(self, config):
        super().__init__(config['data_set'], config['feature_type'], config['max_vocab'], config['data_dir'],
                         config['data_tsv_dir'], config['data_complete_dir'])

        self.only_valid_users = config['valid_users']
        self.top_k = config['top_k']
        self.user_doc_threshold = config['user_doc_threshold']

        self.train_size = config['train_size']
        self.val_size = config['val_size']
        self.test_size = config['test_size']

        # temporary attributes for data which has been loaded and will be reused
        self.doc2id, self.user2id = None, None
        self.train_docs, self.test_docs, self.val_docs, self.n_nodes = None, None, None, None
        self.valid_users = None

    def doc_used(self, doc_key):
        return doc_key in self.train_docs or doc_key in self.val_docs or doc_key in self.test_docs

    def load_doc_splits(self):
        file_name = DOC_SPLITS_FILE_NAME % (
            self.feature_type, self.max_vocab, self.train_size, self.val_size, self.test_size)
        doc_splits = load_json_file(self.data_tsv_path(file_name))
        self.train_docs = doc_splits['train_docs'] if 'train_docs' in doc_splits else []
        self.val_docs = doc_splits['val_docs'] if 'val_docs' in doc_splits else []
        self.test_docs = doc_splits['test_docs'] if 'test_docs' in doc_splits else []

    def maybe_load_valid_users(self):
        if self.valid_users is None:
            self.valid_users = self.load_if_exists(self.data_complete_path(VALID_USERS % self.top_k))

    def maybe_load_id_mappings(self):
        if self.user2id is None:
            self.user2id = self.load_if_exists(self.data_complete_path(self.get_file_name(USER_2_ID_FILE_NAME)))
        if self.doc2id is None:
            self.doc2id = self.load_if_exists(self.data_complete_path(self.get_file_name(DOC_2_ID_FILE_NAME)))
        self.n_nodes = len(self.user2id) + len(self.doc2id)

    def is_restricted(self, statistics):
        for label, percentage in statistics.items():
            if percentage >= self.user_doc_threshold:
                return True
        return False

    def filter_users(self, user_stats, used_docs):
        """
        Counts how many document each user interacted with and identifies users that shared at least X% of
        the articles of any class. Also picks the top K active users.
        """

        # based on user stats, exclude some
        n_docs = len(self.train_docs) + len(self.test_docs) + len(self.val_docs)

        assert n_docs == used_docs, "Total nr of documents used does not equal the number of docs for which " \
                                    "we restrict users!"

        user_stats_avg = copy.deepcopy(user_stats)
        for user, stat in user_stats_avg.items():
            for label in self.labels.values():
                stat[label] = stat[label] / n_docs

        # filter for 30% in either one of the classes
        restricted_users = []
        for user_id, stat in user_stats_avg.items():
            if self.is_restricted(stat):
                restricted_users.append(user_id)

        restricted_users_file = self.data_complete_path(RESTRICTED_USERS % self.user_doc_threshold)
        save_json_file(restricted_users, restricted_users_file)

        print(f'Nr. of restricted users : {len(restricted_users)}')
        print(f"Restricted users stored in : {restricted_users_file}")

        print(f"\nCollecting top K users as valid users : {self.top_k * 1000}")

        # dict with user_ids and total shared/interacted docs
        users_shared_sorted = dict(sorted(user_stats.items(), key=lambda it: sum(it[1].values()), reverse=True))

        # remove users that we have already restricted before
        users_total_shared = OrderedDict()
        for key, value in users_shared_sorted.items():
            if key not in restricted_users:
                users_total_shared[key] = value

        # select the top k
        valid_users = list(users_total_shared.keys())[:self.top_k * 1000]

        valid_users_file = self.data_complete_path(VALID_USERS % self.top_k)
        save_json_file(valid_users, valid_users_file)
        print(f"Valid/top k users stored in : {valid_users_file}\n")

        self.valid_users = valid_users

    def store_user_splits(self, train_users, test_users, val_users):
        if self.only_valid_users:
            all_users = set.union(*[train_users, val_users, test_users])
            print(f'All users: {len(all_users)}')
            assert len(all_users) <= self.top_k * 1000, \
                f"Total nr of users for all splits is greater than top K {self.top_k}!"

        file_name = USER_SPLITS_FILE_NAME % \
                    (self.feature_type, self.max_vocab, self.train_size, self.val_size, self.test_size)
        user_splits_file = self.data_complete_path(file_name)
        print("User splits stored in : ", user_splits_file)
        temp_dict = {'train_users': list(train_users), 'val_users': list(val_users), 'test_users': list(test_users)}
        save_json_file(temp_dict, user_splits_file)

    def valid_user(self, user):
        return not self.only_valid_users or user in self.valid_users

    def create_doc_id_dicts(self):
        """
        Creates and saves doc2id and node2id dictionaries based on the document and user splits created by
        `create_doc_user_splits` function.
        """

        self.print_step("Creating doc2id and user2id dicts")

        n_train = len(self.train_docs)
        print("Train docs = ", n_train)
        doc2id_train, node_type_train = self.doc_node_info(self.train_docs)

        n_val = len(self.val_docs)
        print("Val docs = ", n_val)
        doc2id_val, node_type_val = self.doc_node_info(self.val_docs, offset=n_train)

        n_test = len(self.test_docs)
        print("Test docs = ", n_test)
        doc2id_test, node_type_test = self.doc_node_info(self.test_docs, offset=n_train + n_val)

        doc2id = {**doc2id_train, **doc2id_val, **doc2id_test}

        # only for Python 3.9+
        # self.doc2id = doc2id_train | doc2id_val

        assert len(set(doc2id.values())) == len(doc2id), "Doc2ID contains duplicate IDs!!"
        assert len(doc2id) == (n_val + n_train + n_test), "Doc2id does not contain all documents!"
        print("New doc2id len including test docs = ", len(doc2id))
        doc2id_file = self.data_complete_path(self.get_file_name(DOC_2_ID_FILE_NAME))
        print("Saving doc2id_train in : ", doc2id_file)
        save_json_file(doc2id, doc2id_file)
        self.doc2id = doc2id

        file_name = USER_SPLITS_FILE_NAME % \
                    (self.feature_type, self.max_vocab, self.train_size, self.val_size, self.test_size)
        splits = load_json_file(self.data_complete_path(file_name))
        train_users, val_users, test_users = splits['train_users'], splits['val_users'], splits['test_users']
        all_users = list(set(train_users + val_users + test_users))

        print('\nTrain users = ', len(train_users))
        print("Test users = ", len(test_users))
        print("Val users = ", len(val_users))
        print("All users = ", len(all_users))

        common_users = len(set(train_users + val_users).intersection(set(test_users)))
        print("\nUsers common between train/val and test = ", common_users)

        node_type_user = []
        user2id = {}

        for count, user in enumerate(all_users):
            user2id[str(user)] = count + len(self.doc2id)
            node_type_user.append(2)

        assert len(set(user2id.values())) == len(user2id), "User2ID contains duplicate IDs!!"

        print("\nUser2id size = ", len(user2id))
        user2id_train_file = self.data_complete_path(self.get_file_name(USER_2_ID_FILE_NAME))
        print("Saving user2id_train in : ", user2id_train_file)
        save_json_file(user2id, user2id_train_file)
        self.user2id = user2id

        # TODO: do we need node type and node2id?
        # # node type should contain train and val docs and train, val and test users
        # node_type = node_type_train + node_type_val + node_type_user
        # assert len(node_type) == n_val + n_train + len(all_users)
        #
        # print(f"\nNode type size = {len(node_type)}")
        # node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME % self.top_k)
        # node_type = np.array(node_type)
        # print(f"Saving node type in : {node_type_file}")
        # np.save(node_type_file, node_type, allow_pickle=True)

        # node2id = self.doc2id.copy()
        # node2id.update(self.user2id)
        # assert len(node2id) == len(self.user2id) + len(self.doc2id), \
        #     "Length of node2id is not the sum of doc2id and user2id length!"
        #
        # print("\nNode2id size = ", len(node2id))
        # node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
        # print("Saving node2id_lr in : ", node2id_file)
        # save_json_file(node2id, node2id_file)

        print("\nDone ! All files written.")

    @staticmethod
    def doc_node_info(docs, offset=None):
        doc2id, node_type = {}, []

        for doc_count, doc_name in enumerate(docs):
            doc2id[doc_name] = doc_count if offset is None else doc_count + offset
            node_type.append(1)

        assert len(docs) == len(doc2id) == len(node_type), \
            "doc2id and node type for train to not match number of train documents!"

        return doc2id, node_type

    def filter_contexts(self):
        self.print_step("Creating filtered follower-following")

        all_users = load_json_file(self.data_complete_path(self.get_file_name(USER_2_ID_FILE_NAME)))
        print("Total users in this dataset = ", len(all_users))

        print_iter = int(20000 / 10)
        unknown_user_ids = 0

        for user_context in USER_CONTEXTS:
            print(f"\n    - from {user_context}  folder...")

            user_context_filtered_dir = self.data_raw_path(self.dataset, f'{user_context}_filtered')
            self.create_dir(user_context_filtered_dir)

            user_context_src_dir = self.data_raw_path(self.dataset, user_context)
            for count, file_path in enumerate(user_context_src_dir.glob('*')):

                user_id = file_path.stem
                if user_id not in all_users:
                    unknown_user_ids += 1
                    continue

                if count == 0:
                    print("Writing filtered lists in : ", user_context_filtered_dir)
                    print("Printing every: ", print_iter)

                dest_file_path = (user_context_filtered_dir / (user_id + '.json'))

                # skip if we have already a file for this user
                if dest_file_path.is_file():
                    continue

                follower_key = 'followers' if user_context == 'user_followers' else 'following'
                followers = load_json_file(file_path)[follower_key]

                followers = [f for f in followers if f in all_users]  # only use follower if contained in all_users
                followers = list(map(int, followers))

                temp = set()
                for follower in followers:
                    temp.update([follower])

                follower_json = {'user_id': user_id, follower_key: list(temp)}
                save_json_file(follower_json, dest_file_path)

                if count % print_iter == 0:
                    print(f"{count + 1} done..")

    def create_adj_matrix(self):

        self.print_step(f"Processing {self.dataset} dataset for adj_matrix")

        self.maybe_load_id_mappings()
        self.load_doc_splits()

        # this assumes users and documents are unique in all data sets!
        n_users, n_docs = len(self.user2id), len(self.doc2id)
        print("\nNumber of unique users = ", n_users)
        print("Number of docs = ", n_users)
        print("\nLength user2id = ", n_users)
        print("Length doc2id = ", n_docs)

        # Creating and filling the adjacency matrix (doc-user edges); includes test docs!
        self.n_nodes = n_docs + n_users
        adj_matrix = lil_matrix((self.n_nodes, self.n_nodes))
        edge_type = lil_matrix((self.n_nodes, self.n_nodes))

        # Creating self-loops for each node (diagonals are 1's)
        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, i] = 1
            edge_type[i, i] = 1

        print(f"\nSize of adjacency matrix = {adj_matrix.shape} \nPrinting every  {int(n_docs / 10)} docs")
        start = time.time()

        print("\nPreparing entries for doc-user pairs...")

        adj_matrix, edge_type, edge_list, not_used = self.docs_to_adj(adj_matrix, edge_type)

        # add self connections
        for n in range(adj_matrix.shape[0]):
            adj_matrix[n, n] = 1
            edge_list.append((n, n))

        # get node indices which are bots (nodes with have > X number of neighboring nodes)

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not used users and docs = {not_used}")
        print(f"Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Non-zero entries edge_type = {edge_type.getnnz()}")

        start = time.time()
        key_errors, not_found, overlaps = 0, 0, 0
        print("\nPreparing entries for user-user pairs...")
        print(f"Printing every {int(n_users / 10)}  users done.")

        adj_matrix, edge_type, edge_list, not_used = self.users_to_adj(adj_matrix, edge_type, edge_list)

        edge_list_file = self.data_complete_path(self.get_file_name(EDGE_LIST_FILE_NAME))
        print("\nSaving edge list in :", edge_list_file)
        save_json_file(edge_list, edge_list_file)

        # TODO: because of filtered out users we now here have a graph which still has around 282 nodes that
        # are not test nodes but do not have connections to any users --> filter them out
        # single_self_connection = np.argwhere(adj_matrix.sum(axis=0) > 1)[:, 1]

        # there may only be len(test_docs) number of nodes with no connections to training nodes!

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not found user_ids = {not_found}")
        print(f"Total Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Total Non-zero entries edge_type = {edge_type.getnnz()}")

        # SAVING everything

        adj_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k)
        print(f"\nMatrix construction done! Saving in  {adj_file}")
        save_npz(adj_file, adj_matrix.tocsr())

        edge_type_file = self.data_complete_path(self.get_file_name(EDGE_TYPE_FILE_NAME))
        print(f"\nEdge type construction done! Saving in  {edge_type_file}")
        save_npz(edge_type_file, edge_type.tocsr())

        # Not needed
        # rows, cols = adj_matrix.nonzero()
        # edge_index = np.vstack((np.array(rows), np.array(cols)))
        # print("Edge index shape = ", edge_index.shape)
        # edge_index_file = self.data_complete_path(self.get_file_name(EDGE_INDEX_FILE_NAME))
        # print("saving edge_index format in :  ", edge_index_file)
        # np.save(edge_index_file, edge_index, allow_pickle=True)

        edge_type = edge_type[edge_type.nonzero()].toarray().squeeze(0)
        print("edge_type shape = ", edge_type.shape)
        edge_list_type = self.data_complete_path(self.get_file_name(EDGE_LIST_FILE_NAME))
        print("saving edge_type list format in :  ", edge_list_type)
        np.save(edge_list_type, edge_list_type, allow_pickle=True)

    def create_feature_matrix(self):

        self.maybe_load_id_mappings()

        self.print_step("Creating feature matrix")

        # load all texts for test, train and val documents
        folder_name = DOC_SPLITS_FOLDER_NAME % \
                      (self.feature_type, self.max_vocab, self.train_size, self.val_size, self.test_size)
        split_path = self.data_tsv_path(folder_name)

        all_texts = {}
        for split in ['test', 'train', 'val']:
            try:
                reader = csv.DictReader(open(split_path / f'{split}.tsv', encoding='utf-8'), delimiter='\t')
            except FileNotFoundError:
                print(f"No split file found for split: {split}.")
                continue
            for row in reader:
                all_texts[row['id']] = row['text'].split(' ')

        assert len(all_texts) == len(self.doc2id), "Nr of texts from doc splits does not equal to doc2id!"

        print(f"\nNr of docs = {len(self.doc2id)}")
        print(f"Nr of users = {len(self.user2id)}")

        vocabulary, feature_size = self.get_vocab_token2idx(all_texts)

        print("\nCreating features for docs nodes...")
        start = time.time()

        features_docs = []
        feature_ids = {}
        feature_idx = 0
        for doc_key, tokens in all_texts.items():

            if self.feature_type == 'one-hot':
                indices = [vocabulary[token] for token in tokens if token in vocabulary]
                doc_feat = torch.zeros(feature_size)
                doc_feat[indices] = 1
            elif 'glove' in self.feature_type:
                idx_vectors = torch.stack([vocabulary[token] for token in tokens])
                # noinspection PyUnboundLocalVariable
                doc_feat = idx_vectors.mean(dim=0) if 'average' in self.feature_type else idx_vectors.sum(dim=0)
            else:
                raise ValueError(f"Trying to create features of type {self.feature_type} which is not unknown!")

            features_docs.append(doc_feat)
            feature_ids[doc_key] = feature_idx
            feature_idx += 1

        doc_features = torch.stack(features_docs)

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs")

        print("\nCreating features for users nodes...")
        start = time.time()

        feature_id_mapping = self.get_feature_id_mapping(feature_ids)
        feature_id_mapping = dict(sorted(feature_id_mapping.items(), key=lambda it: it[0]))

        features_users = []
        for user_id, doc_ids in feature_id_mapping.items():
            features_users.append(doc_features[doc_ids].sum(axis=0))
        user_features = torch.stack(features_users)

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        print(f"\nCreating feature matrix and storing doc and user features...")
        start = time.time()
        total = doc_features.shape[0] + user_features.shape[0]

        feature_matrix = lil_matrix((total, feature_size))
        print(f"Size of feature matrix = {feature_matrix.shape}")

        feature_matrix[:len(doc_features)] = doc_features
        feature_matrix[len(doc_features):] = user_features

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        if self.feature_type == 'one-hot':
            # turning into 1-hot (as user features were aggregated from documents)
            feature_matrix = feature_matrix >= 1
            feature_matrix = feature_matrix.astype(int)

        filename = self.data_complete_path(self.get_file_name(FEAT_MATRIX_FILE_NAME))
        print(f"\nMatrix construction done! Saving in: {filename}")
        save_npz(filename, feature_matrix.tocsr())

    def get_file_name(self, file):
        return file % (self.top_k, self.feature_type, self.max_vocab, self.train_size, self.val_size, self.test_size)

    @abc.abstractmethod
    def docs_to_adj(self, adj_matrix, edge_type):
        """
        Creates connections for docs and users in the adj_matrix, adds the edge type for every node.
        :param adj_matrix: Adjacency matrix
        :param edge_type: Edge type matrix
        :return: Copy of adj_matrix and edge_type with filled values, a list which describes the edges.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def users_to_adj(self, adj_matrix, edge_type, edge_list):
        raise NotImplementedError

    @abc.abstractmethod
    def get_feature_id_mapping(self, feature_ids):
        raise NotImplementedError

    def create_labels(self):

        self.print_step('Creating labels')

        self.maybe_load_id_mappings()

        print("Loading doc2labels dictionary...")
        doc2labels = load_json_file(self.data_complete_path(DOC_2_LABELS_FILE_NAME))

        train_docs = self.train_docs + self.val_docs
        train_labels = np.zeros(len(train_docs), dtype=int)

        # must be length of all nodes, but we only fill labels for train and val
        all_labels = np.zeros(self.n_nodes, dtype=int)

        for doc_key in train_docs:
            if doc_key not in self.doc2id:
                continue
            if doc_key not in doc2labels:
                raise ValueError(f'Can not retrieve label for document with key: {doc_key}')
            label = doc2labels[doc_key]
            doc_id = self.doc2id[doc_key]
            train_labels[doc_id] = label
            all_labels[doc_id] = label

        for doc_key in self.test_docs:
            if doc_key not in self.doc2id:
                continue
            if doc_key not in doc2labels:
                raise ValueError(f'Can not retrieve label for document with key: {doc_key}')
            all_labels[self.doc2id[doc_key]] = doc2labels[doc_key]

        assert len(train_labels) == len(self.doc2id.keys()) - len(self.test_docs)
        print(f"\nLen of (train) labels = {len(train_labels)}")

        # labels_file = self.data_complete_path(self.get_file_name(TRAIN_LABELS_FILE_NAME))
        # print(f"\nLabels list construction done! Saving in : {labels_file}")
        # save_json_file({'labels_list': list(train_labels)}, labels_file, converter=self.np_converter)

        print("\nSum of all labels = ", int(sum(all_labels)))
        print("Len of all labels = ", len(all_labels))

        all_labels_file = self.data_complete_path(self.get_file_name(ALL_LABELS_FILE_NAME))
        print(f"\nAll labels list construction done! Saving in : {all_labels_file}")
        save_json_file({'all_labels': list(all_labels)}, all_labels_file, converter=self.np_converter)

    def create_split_masks(self):
        """
        Creates split masks over all the document nodes for train/test/val set.
        """

        self.print_step("Creating split masks")

        self.maybe_load_id_mappings()

        train_mask, val_mask, test_mask = np.zeros(self.n_nodes), np.zeros(self.n_nodes), np.zeros(self.n_nodes)

        for doc, doc_id in self.doc2id.items():
            doc_n = str(doc)
            if doc_n in self.train_docs:
                train_mask[doc_id] = 1
            elif doc_n in self.val_docs:
                val_mask[doc_id] = 1
            elif doc_n in self.test_docs:
                test_mask[doc_id] = 1

        print("train_mask sum = ", int(sum(train_mask)))
        print("val_mask sum = ", int(sum(val_mask)))
        print("test_mask sum = ", int(sum(test_mask)))

        mask_dict = {'train_mask': list(train_mask), 'val_mask': list(val_mask), 'test_mask': list(test_mask)}
        split_mask_file = self.data_complete_path(self.get_file_name(SPLIT_MASK_FILE_NAME))
        print("\nWriting split mask file in : ", split_mask_file)
        save_json_file(mask_dict, split_mask_file)
