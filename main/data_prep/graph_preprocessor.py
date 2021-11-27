import abc
import copy
import csv
import time
from collections import defaultdict, OrderedDict
from json import JSONDecodeError

import nltk
import torch
from scipy.sparse import lil_matrix, save_npz
from torchtext.vocab import GloVe

from data_prep.config import *
from data_prep.data_preprocess_utils import *
from data_prep.graph_io import GraphIO

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']
FEATURE_TYPES = ['one-hot', 'glove-sum', 'glove-average']


class GraphPreprocessor(GraphIO):

    def __init__(self, config):
        super().__init__(config['data_set'], config['data_dir'], config['data_tsv_dir'],
                         config['data_complete_dir'])
        self.only_valid_users = config['valid_users']
        self.top_k = config['top_k']
        self.user_doc_threshold = config['user_doc_threshold']

        # temporary attributes for data which has been loaded and will be reused
        self.doc2id, self.user2id = None, None
        self.train_docs, self.test_docs, self.val_docs, self.n_total = None, None, None, None
        self.valid_users = None

        self.vocab = None

    def doc_used(self, doc_id):
        return doc_id in self.train_docs or doc_id in self.test_docs or doc_id in self.val_docs

    def load_doc_splits(self):
        doc_splits = load_json_file(self.data_tsv_path(DOC_SPLITS_FILE_NAME))
        self.train_docs = doc_splits['train_docs']
        self.test_docs = doc_splits['test_docs']
        self.val_docs = doc_splits['val_docs']

    def maybe_load_valid_users(self):
        if self.valid_users is None:
            self.valid_users = self.load_if_exists(self.data_complete_path(VALID_USERS % self.top_k))

    def maybe_load_id_mappings(self):
        if self.user2id is None:
            self.user2id = self.load_if_exists(self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k))
        if self.doc2id is None:
            self.doc2id = self.load_if_exists(self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k))
        self.n_total = len(self.user2id) + len(self.doc2id)

    @staticmethod
    @abc.abstractmethod
    def get_doc_key(name, name_type):
        raise NotImplementedError

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

        restriction_docs = 0
        for count, file_path in enumerate(self.data_tsv_path('engagements').glob('*')):

            # only restrict users interacting with this document ID if we actually use this doc in our splits
            doc_key = file_path.stem
            if not self.doc_used(doc_key):
                continue

            restriction_docs += 1

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
                    user_stats[u][self.labels()[doc2labels[doc_key]]] += 1

        # based on user stats, exclude some
        n_docs = len(self.train_docs) + len(self.test_docs) + len(self.val_docs)

        assert n_docs == restriction_docs, "Total nr of documents used does not equal the number of docs for " \
                                           "which we restrict users!"

        user_stats_avg = copy.deepcopy(user_stats)
        for user, stat in user_stats_avg.items():
            for label in self.labels().values():
                stat[label] = stat[label] / n_docs

        # filter for 30% in either one of the classes
        restricted_users = []
        for user_id, stat in user_stats_avg.items():
            restricted = stat['fake'] >= self.user_doc_threshold or stat['real'] >= self.user_doc_threshold
            if restricted:
                # print(f'User with ID {user_id} shared {stat["fake"]} of fake and {stat["real"]} of real.')
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

            # TODO: fix the runtime here, this is super slow
            users_filtered = [u for u in users if self.valid_user(u)]

            doc_key = file_path.stem
            if doc_key in self.train_docs:
                train_users.update(users_filtered)
            if doc_key in self.val_docs:
                val_users.update(users_filtered)
            if doc_key in self.test_docs:
                test_users.update(users_filtered)

        if self.only_valid_users:
            all_users = set.union(*[train_users, val_users, test_users])
            print(f'All users: {len(all_users)}')
            assert len(all_users) <= self.top_k * 1000, \
                f"Total nr of users for all splits is greater than top K {self.top_k}!"

        user_splits_file = self.data_complete_path(USER_SPLITS_FILE_NAME)
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
        save_json_file(doc2id, self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k))
        self.doc2id = doc2id

        splits = load_json_file(self.data_complete_path(USER_SPLITS_FILE_NAME))
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
        user2id_train_file = self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k)
        print("Saving user2id_train in : ", user2id_train_file)
        save_json_file(user2id, user2id_train_file)
        self.user2id = user2id

        # node type should contain train and val docs and train, val and test users
        node_type = node_type_train + node_type_val + node_type_user
        assert len(node_type) == n_val + n_train + len(all_users)

        print(f"\nNode type size = {len(node_type)}")
        node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME % self.top_k)
        node_type = np.array(node_type)
        print(f"Saving node type in : {node_type_file}")
        np.save(node_type_file, node_type, allow_pickle=True)

        # print("\nAdding test docs..")
        # n_test = len(self.test_docs)
        # print("Test docs = ", n_test)
        # orig_doc2id_len = len(self.doc2id)

        # for test_count, doc in enumerate(self.test_docs):
        #     self.doc2id[doc] = test_count + len(self.user2id) + orig_doc2id_len

        node2id = self.doc2id.copy()
        node2id.update(self.user2id)
        assert len(node2id) == len(self.user2id) + len(self.doc2id), \
            "Length of node2id is not the sum of doc2id and user2id length!"

        print("\nNode2id size = ", len(node2id))
        node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
        print("Saving node2id_lr in : ", node2id_file)
        save_json_file(node2id, node2id_file)

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

    def filter_contexts(self, follower_key=None):
        self.print_step("Creating filtered follower-following")

        all_users = load_json_file(self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k))
        print("Total users in this dataset = ", len(all_users))

        unknown_user_ids = 0
        for user_context in USER_CONTEXTS:
            print(f"\n    - from {user_context}  folder...")

            dest_dir = self.data_raw_path(self.dataset, f'{user_context}_filtered')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            user_context_src_dir = self.data_raw_path(self.dataset, user_context)
            for root, dirs, files in os.walk(user_context_src_dir):
                for count, file in enumerate(files):

                    user_id = file.split(".")[0]
                    if user_id not in all_users:
                        unknown_user_ids += 1
                        continue

                    print_iter = int(len(files) / 10)
                    if count == 0:
                        print("Total user files = ", len(files))
                        print("Writing filtered lists in : ", dest_dir)
                        print("Printing every: ", print_iter)

                    dest_file_path = os.path.join(dest_dir, str(user_id) + '.json')

                    # skip if we have already a file for this user
                    if os.path.isfile(dest_file_path):
                        continue

                    src_file = load_json_file(os.path.join(root, file))

                    follower_dest_key = 'followers' if user_context == 'user_followers' else 'following'

                    # will be different for FakeHealth and FakeNews
                    follower_src_key = follower_key if follower_key is not None else follower_dest_key

                    # only use follower if it is contained in all_users
                    followers = [f for f in src_file[follower_src_key] if f in all_users]
                    followers = list(map(int, followers))

                    temp = set()
                    for follower in followers:
                        temp.update([follower])

                    save_json_file({'user_id': user_id, follower_dest_key: list(temp)}, dest_file_path)

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
        self.n_total = n_docs + n_users
        adj_matrix = lil_matrix((self.n_total, self.n_total))
        edge_type = lil_matrix((self.n_total, self.n_total))

        # Creating self-loops for each node (diagonals are 1's)
        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, i] = 1
            edge_type[i, i] = 1

        print(f"\nSize of adjacency matrix = {adj_matrix.shape} \nPrinting every  {int(n_docs / 10)} docs")
        start = time.time()

        print("\nPreparing entries for doc-user pairs...")
        edge_list = []
        not_found = 0
        no_connections_added_doc_ids = set()
        for count, file_path in enumerate(self.data_tsv_path('engagements').glob('*')):
            doc_key = file_path.stem
            if doc_key == '':
                continue
            src_file = load_json_file(file_path)
            users = map(str, src_file['users'])
            for user in users:
                if doc_key in self.test_docs:
                    # no connections between users and test documents!
                    no_connections_added_doc_ids.add(self.doc2id[doc_key])
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
                    not_found += 1

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not Found users = {not_found}")
        print(f"Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Non-zero entries edge_type = {edge_type.getnnz()}")

        start = time.time()
        key_errors, not_found, overlaps = 0, 0, 0
        print("\nPreparing entries for user-user pairs...")
        print(f"Printing every {int(n_users / 10)}  users done.")

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
                        continue

                    user_id1 = self.user2id[user_id]
                    user_id2 = self.user2id[follower]

                    # for DGL graph creation; edges are reversed later
                    edge_list.append((user_id2, user_id1))
                    # edge_list.append((user_id1, user_id2))

                    adj_matrix[user_id1, user_id2] = 1
                    adj_matrix[user_id2, user_id1] = 1
                    edge_type[user_id1, user_id2] = 3
                    edge_type[user_id2, user_id1] = 3

                else:
                    not_found += 1

        edge_list_file = self.data_complete_path(EDGE_LIST_FILE_NAME % self.top_k)
        print("\nSaving edge list in :", edge_list_file)
        save_json_file(edge_list, edge_list_file)

        # TODO: because of filtered out users we now here have a grad which still has around 282 nodes that
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

        edge_type_file = self.data_complete_path(EDGE_TYPE_FILE_NAME % self.top_k)
        print(f"\nEdge type construction done! Saving in  {edge_type_file}")
        save_npz(edge_type_file, edge_type.tocsr())

        rows, cols = adj_matrix.nonzero()
        edge_index = np.vstack((np.array(rows), np.array(cols)))
        print("Edge index shape = ", edge_index.shape)
        edge_index_file = self.data_complete_path(EDGE_INDEX_FILE_NAME % self.top_k)
        print("saving edge_index format in :  ", edge_index_file)
        np.save(edge_index_file, edge_index, allow_pickle=True)

        edge_type = edge_type[edge_type.nonzero()].toarray().squeeze(0)
        print("edge_type shape = ", edge_type.shape)
        edge_type_file = self.data_complete_path(EDGE_TYPE_FILE_NAME % self.top_k)
        print("saving edge_type list format in :  ", edge_type_file)
        np.save(edge_type_file, edge_index, allow_pickle=True)

    @staticmethod
    def build_vocab(text_tokens, max_count=-1, min_count=2, max_vocab=15000):

        # creating word frequency dict
        word_freq = {}
        for text_token in text_tokens:
            for token in set(text_token):
                word_freq[token] = word_freq.get(token, 0) + 1
        word_freq = [(f, w) for (w, f) in word_freq.items()]
        word_freq.sort(reverse=True)

        # collect token counts
        token_counts = []
        for (count, token) in word_freq:
            if max_count != -1 and count > max_count:
                continue
            if count < min_count:
                continue
            token_counts.append((count, token))

        token_counts.sort(reverse=True)
        if max_vocab != -1:
            token_counts = token_counts[:max_vocab]
        # NIV: not in vocab token, i.e., out of vocab
        token_counts.append((0, 'NIV'))

        vocab = {}
        for (i, (count, token)) in enumerate(token_counts):
            vocab[token] = i + 1

        return vocab

    def create_feature_matrix(self, feature_type='one-hot', max_vocab=10000):

        if feature_type not in FEATURE_TYPES:
            raise ValueError(f"Trying to create features of type {feature_type} which is not supported!")

        self.maybe_load_id_mappings()

        self.print_step("Creating feature matrix")

        # load all texts for test, train and val documents
        split_path = self.data_tsv_path('splits')

        all_texts = []
        for split in ['test', 'train', 'val']:
            reader = csv.DictReader(open(split_path / f'{split}.tsv', encoding='utf-8'), delimiter='\t')
            for row in reader:
                tokens = set(nltk.word_tokenize(row['text']))
                all_texts.append(list(tokens))

        assert len(all_texts) == len(self.doc2id), "Nr of texts from doc splits does not equal to doc2id!"

        print(f"\nNr of docs = {len(self.doc2id)}")
        print(f"Nr of users = {len(self.user2id)}")

        if feature_type == 'one-hot':
            # create own vocabulary from all data
            token2idx = self.build_vocab(all_texts)
        elif 'glove' in feature_type:
            glove = GloVe(name='twitter.27B', dim=200, max_vectors=max_vocab)
            token2idx = glove.stoi
        else:
            raise ValueError(f"Trying to create features of type {feature_type} which is not unknown!")

        print("\nCreating features for docs nodes...")
        start = time.time()

        features_docs = []
        for tokens in all_texts:
            indices = torch.tensor([token2idx[token] for token in tokens if token in token2idx])

            # Use only 10k most common tokens
            indices = indices[indices < max_vocab]

            if len(indices) == 0:
                continue

            if feature_type == 'one-hot':
                doc_feat = torch.zeros(max_vocab)
                doc_feat[indices] = 1
            elif 'glove' in feature_type:
                # noinspection PyUnboundLocalVariable
                vectors = glove.vectors[indices]
                doc_feat = vectors.mean(dim=0) if 'average' in feature_type else vectors.sum(dim=0)
            else:
                raise ValueError(f"Trying to create features of type {feature_type} which is not unknown!")

            features_docs.append(doc_feat)
        doc_features = torch.stack(features_docs)  # .to(self._device)

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs")

        print("\nCreating features for users nodes...")
        start = time.time()

        feature_id_mapping = defaultdict(lambda: [])
        for count, file_path in enumerate(self.data_tsv_path('engagements').rglob('*.json')):
            doc_users = load_json_file(file_path)
            doc_key = file_path.stem

            # Each user of this doc has its features as the features of the doc
            if doc_key not in self.doc2id:
                continue

            for user in doc_users['users']:
                user_key = str(user)
                if user_key not in self.user2id:
                    continue
                feature_id_mapping[self.user2id[user_key]].append(self.doc2id[str(doc_key)])

        feature_id_mapping = dict(sorted(feature_id_mapping.items(), key=lambda it: it[0]))
        features_users = []
        for user_id, doc_ids in feature_id_mapping.items():
            features_users.append(doc_features[doc_ids].sum(axis=0))
        user_features = torch.stack(features_users)  # .to(self._device)

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        print(f"\nCreating feature matrix and storing doc and user features...")
        start = time.time()
        feature_matrix = lil_matrix((self.n_total, max_vocab))
        print(f"Size of feature matrix = {feature_matrix.shape}")

        feature_matrix[:len(doc_features)] = doc_features
        feature_matrix[len(doc_features):] = user_features

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        # TODO: do we need this?
        feature_matrix = feature_matrix >= 1
        feature_matrix = feature_matrix.astype(int)

        # Sanity Checks
        # feat_matrix_sum = np.array(feat_matrix.sum(axis=1)).squeeze(1)

        filename = self.data_complete_path(FEAT_MATRIX_FILE_NAME % self.top_k)
        print(f"\nMatrix construction done! Saving in: {filename}")
        save_npz(filename, feature_matrix.tocsr())

    @abc.abstractmethod
    def create_labels(self):
        raise NotImplementedError

    def create_split_masks(self):
        """
        Creates split masks over all the document nodes for train/test/val set.
        """

        self.print_step("Creating split masks")

        self.maybe_load_id_mappings()

        train_mask, val_mask, test_mask = np.zeros(self.n_total), np.zeros(self.n_total), np.zeros(self.n_total)

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
        split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
        print("\nWriting split mask file in : ", split_mask_file)
        save_json_file(mask_dict, split_mask_file)
