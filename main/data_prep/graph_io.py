import abc
import csv
import datetime
import json
import os
import re
import time

import nltk
from scipy.sparse import load_npz

nltk.download('stopwords')

from nltk.corpus import stopwords

import numpy as np
from scipy.sparse import lil_matrix, save_npz
from sklearn.model_selection import StratifiedShuffleSplit

from data_prep.config import *

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']


class GraphIO:

    def __init__(self, dataset, raw_dir=RAW_DIR, tsv_dir=TSV_DIR, complete_dir=COMPLETE_DIR):
        self.dataset = dataset

        self.data_raw_dir = self.create_dir(raw_dir)
        self.data_tsv_dir = self.create_dir(tsv_dir)
        self.data_complete_dir = self.create_dir(complete_dir)

        self.train_docs, self.test_docs, self.val_docs = None, None, None

    @staticmethod
    def create_dir(dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        return dir_name

    def data_raw_path(self, *parts):
        return os.path.join(self.data_raw_dir, *parts)

    def data_tsv_path(self, *parts):
        return os.path.join(self.data_tsv_dir, self.dataset, *parts)

    def data_complete_path(self, *parts):
        return os.path.join(self.data_complete_dir, self.dataset, *parts)

    def maybe_load_doc_splits(self):
        doc_splits = load_json_file(self.data_tsv_path(DOC_SPLITS_FILE_NAME))
        self.train_docs = doc_splits['train_docs']
        self.test_docs = doc_splits['test_docs']
        self.val_docs = doc_splits['val_docs']


class GraphPreprocessor(GraphIO):

    def __init__(self, config):
        super().__init__(config['data_set'], config['data_raw_dir'], config['data_tsv_dir'],
                         config['data_complete_dir'])
        self.exclude_frequent_users = config['exclude_frequent']
        self.top_k = config['top_k']

        # temporary attributes for data which has been loaded and will be reused
        self.doc2id, self.user2id = None, None
        self.n_total = None

    def maybe_load_id_mappings(self):
        if self.user2id is None:
            user2id_file = self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k)
            if os.path.exists(user2id_file):
                self.user2id = json.load(open(user2id_file, 'r'))
        if self.doc2id is None:
            doc2id_file = self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k)
            if os.path.exists(doc2id_file):
                self.doc2id = json.load(open(doc2id_file, 'r'))

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

    # @staticmethod
    # def print_header(header_title):
    #     print(f'\n\n{"-" * 50}\n{header_title}\n{"-" * 50}')

    @staticmethod
    def calc_elapsed_time(start, end):
        hours, rem = divmod(end - start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)

    @staticmethod
    def save_user_docs(count, dest_dir, docs_users):
        print(f"\nTotal tweets/re-tweets in the data set = {count}")
        print(f"\nWriting all the info in the dir: {dest_dir}")

        for doc, user_list in docs_users.items():
            document_user_file = os.path.join(dest_dir, f'{str(doc)}.json')
            with open(document_user_file, 'w+') as j:
                temp_dict = {'users': list(user_list)}
                json.dump(temp_dict, j)

        print("\nDONE..!!")

    @staticmethod
    @abc.abstractmethod
    def get_doc_key(name, name_type):
        raise NotImplementedError

    @abc.abstractmethod
    def aggregate_user_contexts(self):
        """
        Aggregates only user IDs from different folders of tweets/retweets to a single place. Creates a folder which
        has files named after document Ids. Each file contains all the users that interacted with it.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def create_doc_user_splits(self):
        raise NotImplementedError

    def create_user_splits(self, src_dir):

        self.print_step("Creating doc and user splits")

        self.maybe_load_doc_splits()

        print("\nCreating users in splits file..")

        train_users, val_users, test_users = set(), set(), set()
        for root, dirs, files in os.walk(src_dir):
            for count, file in enumerate(files):
                doc_key = self.get_doc_key(file, name_type='file')
                src_file_path = os.path.join(root, file)

                try:
                    src_file = json.load(open(src_file_path, 'r'))
                except UnicodeDecodeError:
                    print("Exception")
                    continue

                users = src_file['users']
                users = [int(s) for s in users if isinstance(s, int)]
                if str(doc_key) in self.train_docs:
                    train_users.update(users)
                if str(doc_key) in self.val_docs:
                    val_users.update(users)
                if str(doc_key) in self.test_docs:
                    test_users.update(users)

        user_splits_file = self.data_complete_path(USER_SPLITS_FILE_NAME)
        print("User splits stored in : ", user_splits_file)

        temp_dict = {'train_users': list(train_users), 'val_users': list(val_users), 'test_users': list(test_users)}
        with open(user_splits_file, 'w+') as j:
            json.dump(temp_dict, j)

    # @abc.abstractmethod
    # def create_doc_id_dicts(self):
    #     raise NotImplementedError

    def create_doc_id_dicts(self):
        """
        Creates and saves doc2id and node2id dictionaries based on the document and user splits created by
        `create_doc_user_splits` function. Also puts constraints on the users which are used
        (e.g. only top k active users).
        """

        self.print_step("Creating dicts")
        print("\nCreating doc2id and user2id dicts....\n")

        user_splits_file = self.data_complete_path(USER_SPLITS_FILE_NAME)
        usr_splits = json.load(open(user_splits_file, 'r'))
        train_users, val_users, test_users = usr_splits['train_users'], usr_splits['val_users'], usr_splits[
            'test_users']

        self.maybe_load_doc_splits()

        doc2id = {}
        node_type = []

        for train_count, doc in enumerate(self.train_docs):
            doc2id[str(doc)] = train_count
            node_type.append(1)

        print("Train docs = ", len(self.train_docs))
        print("doc2id train = ", len(doc2id))
        print("Node type = ", len(node_type))

        # assert len(self.train_docs) == len(node_type), "Length of train docs is not the same as length of node type!"
        # assert len(self.train_docs) == len(self.doc2id), "Length of train docs is not the same as length of doc2ids!"

        for val_count, doc in enumerate(self.val_docs):
            doc2id[str(doc)] = val_count + len(self.train_docs)
            node_type.append(1)

        print("\nVal docs = ", len(self.val_docs))
        print("doc2id train = ", len(doc2id))
        print("Node type = ", len(node_type))

        assert len(self.train_docs) + len(self.val_docs) == len(node_type), \
            "Sum of train docs and val docs length is not the same as length of node type!"

        doc2id_file = self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k)
        print("Saving doc2id dict in :", doc2id_file)
        with open(doc2id_file, 'w+') as j:
            json.dump(doc2id, j)

        # print('\nTrain users = ', len(train_users))
        # print("Test users = ", len(test_users))
        # print("Val users = ", len(val_users))
        # print("All users = ", len(all_users))

        if self.exclude_frequent_users:
            print("\nRestricting users ... ")

            # Exclude most frequent users
            restricted_users_file = self.data_tsv_path(RESTRICTED_USERS)
            valid_users_file = self.data_tsv_path(VALID_USERS % self.top_k)

            restricted_users, valid_users = [], []
            try:
                restricted_users = json.load(open(restricted_users_file, 'r'))['restricted_users']
                valid_users = json.load(open(valid_users_file, 'r'))['valid_users']
            except FileNotFoundError:
                print("\nDid not find file for restricted and valid users although they should be excluded!\n")

            train_users = [u for u in train_users if str(u) not in restricted_users and str(u) in valid_users]
            val_users = [u for u in val_users if str(u) not in restricted_users and str(u) in valid_users]
            test_users = [u for u in test_users if str(u) not in restricted_users and str(u) in valid_users]
        else:
            train_users = [u for u in train_users]
            val_users = [u for u in val_users]
            test_users = [u for u in test_users]

        # train_users = list(set(train_users)-set(done_users['done_users'])-set(restricted_users['restricted_users']))

        all_users = list(set(train_users + val_users + test_users))

        print('\nTrain users = ', len(train_users))
        print("Test users = ", len(test_users))
        print("Val users = ", len(val_users))
        print("All users = ", len(all_users))

        a = set(train_users + val_users)
        b = set(test_users)
        print("\nUsers common between train/val and test = ", len(a.intersection(b)))

        self.maybe_load_id_mappings()

        user2id_train = {}
        for count, user in enumerate(all_users):
            user2id_train[str(user)] = count + len(self.doc2id)
            node_type.append(2)

        user2id_train_file = self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k)
        print("\nuser2id size = ", len(user2id_train))
        print("Saving user2id_train in : ", user2id_train_file)
        with open(user2id_train_file, 'w+') as j:
            json.dump(user2id_train, j)

        node2id = self.doc2id.copy()
        node2id.update(user2id_train)

        assert len(node2id) == len(user2id_train) + len(self.doc2id), \
            "Length of node2id is not the sum of doc2id and user2id length!"

        print("\nnode2id size = ", len(node2id))
        node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
        print("Saving node2id_lr_train in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        # node type already contains train and val docs and train, val and test users
        assert len(node_type) == len(self.val_docs) + len(self.train_docs) + len(all_users)

        print("\nNode type size = ", len(node_type))
        node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME % self.top_k)
        node_type = np.array(node_type)
        print("Saving node_type in :", node_type_file)
        np.save(node_type_file, node_type, allow_pickle=True)

        print("\nAdding test docs..")
        orig_doc2id_len = len(self.doc2id)
        for test_count, doc in enumerate(self.test_docs):
            self.doc2id[str(doc)] = test_count + len(user2id_train) + orig_doc2id_len
        print("Test docs = ", len(self.test_docs))
        print("doc2id = ", len(self.doc2id))
        with open(self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k), 'w+') as j:
            json.dump(self.doc2id, j)

        node2id = self.doc2id.copy()
        node2id.update(user2id_train)
        print("node2id size = ", len(node2id))
        node2id_file = self.data_complete_path('node2id_lr_top{self.top_k}.json')
        print("Saving node2id_lr in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        print("\nDone ! All files written..")

    def filter_contexts(self, follower_key):
        """
        Reduces the follower/ing data to users that are among the top k users.
        """

        self.print_step("Creating filtered follower-following")

        with open(self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k), 'r') as j:
            all_users = json.load(j)

        # print_iter = int(len(all_users) / 10)
        print("Total users in this dataset = ", len(all_users))

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

                    src_file_path = os.path.join(root, file)
                    src_file = json.load(open(src_file_path, 'r'))

                    follower_dest_key = 'followers' if user_context == 'user_followers' else 'following'

                    # will be different for FakeHealth and FakeNews
                    follower_src_key = follower_key if follower_key is not None else follower_dest_key

                    # only use follower if it is contained in all_users
                    followers = [f for f in src_file[follower_src_key] if f in all_users]
                    followers = list(map(int, followers))

                    temp = set()
                    for follower in followers:
                        temp.update([follower])
                    temp_dict = {'user_id': user_id, follower_dest_key: list(temp)}
                    with open(dest_file_path, 'w+') as v:
                        json.dump(temp_dict, v)

                    if count % print_iter == 0:
                        # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                        print(f"{count + 1} done..")

    def create_adj_matrix(self, src_dir):

        self.print_step(f"Processing {self.dataset} dataset for adj_matrix")

        self.maybe_load_id_mappings()
        self.maybe_load_doc_splits()

        # this assumes users and documents are unique in all data sets!
        num_users, num_docs = len(self.user2id), len(self.doc2id) - len(self.test_docs)
        print("\nNumber of unique users = ", num_users)
        print("Number of docs = ", num_docs)

        print("\nLength user2id = ", len(self.user2id))
        print("Length doc2id = ", len(self.doc2id))

        # Creating and filling the adjacency matrix (doc-user edges)
        adj_matrix = lil_matrix((num_docs + num_users, num_users + num_docs))
        self.n_total = adj_matrix.shape[0]

        edge_type = lil_matrix((num_docs + num_users, num_users + num_docs))

        # adj_matrix = np.zeros((num_docs+num_users, num_users+num_docs))
        # adj_matrix_file = './data/complete_data/adj_matrix_pheme.npz'
        # adj_matrix = load_npz(adj_matrix_file)
        # adj_matrix = lil_matrix(adj_matrix)

        # Creating self-loops for each node (diagonals are 1's)
        for i in range(adj_matrix.shape[0]):
            adj_matrix[i, i] = 1
            edge_type[i, i] = 1

        print(f"\nSize of adjacency matrix = {adj_matrix.shape} \nPrinting every  {int(num_docs / 10)} docs")
        start = time.time()

        edge_list = []

        print("\nPreparing entries for doc-user pairs...")

        not_found = 0
        for root, dirs, files in os.walk(src_dir):
            for count, file_name in enumerate(files):
                doc_key = str(file_name.split(".")[0])
                if doc_key == '':
                    continue
                src_file = json.load(open(os.path.join(root, file_name), 'r'))
                users = map(str, src_file['users'])
                for user in users:
                    if doc_key in self.doc2id and user in self.user2id and doc_key not in self.test_docs:

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

        end = time.time()
        hrs, mins, secs = self.calc_elapsed_time(start, end)
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not Found users = {not_found}")
        print(f"Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Non-zero entries edge_type = {edge_type.getnnz()}")
        # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))

        # Filling the adjacency matrix (user-user edges)
        start = time.time()
        key_errors, not_found, overlaps = 0, 0, 0
        print("\nPreparing entries for user-user pairs...")
        print(f"Printing every {int(num_users / 10)}  users done")

        for user_context in USER_CONTEXTS_FILTERED:
            print(f"\n    - from {user_context} folder...")
            user_context_src_dir = self.data_raw_path(user_context)
            for root, dirs, files in os.walk(user_context_src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    # user_id = src_file_path.split(".")[0]
                    src_file = json.load(open(src_file_path, 'r'))
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
        print("Saving edge list in :", edge_list_file)
        with open(edge_list_file, 'w+') as j:
            json.dump(edge_list, j)

        hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not found user_ids = {not_found}")
        print(f"Total Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Total Non-zero entries edge_type = {edge_type.getnnz()}")

        self.save_adj_matrix(adj_matrix, edge_type)

    def save_adj_matrix(self, adj_matrix, edge_type):

        adj_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k)
        print(f"\nMatrix construction done! Saving in  {adj_file}")
        save_npz(adj_file, adj_matrix.tocsr())

        edge_type_file = self.data_complete_path(EDGE_TYPE_FILE_NAME % self.top_k)
        print(f"\nEdge type construction done! Saving in  {edge_type_file}")
        save_npz(edge_type_file, edge_type.tocsr())

        # Creating an edge_list matrix of the adj_matrix as required by some GCN frameworks
        print("\nCreating edge_index format of adj_matrix...")
        # G = nx.DiGraph(adj_matrix.tocsr())
        # temp_matrix = adj_matrix.toarray()
        # rows, cols = np.nonzero(temp_matrix)
        rows, cols = adj_matrix.nonzero()

        edge_index = np.vstack((np.array(rows), np.array(cols)))
        print("Edge index shape = ", edge_index.shape)

        edge_matrix_file = self.data_complete_path(ADJACENCY_MATRIX_FILE % self.top_k + '_edge.npy')
        print("saving edge_list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)

        edge_index = edge_type[edge_type.nonzero()]
        edge_index = edge_index.toarray()
        edge_index = edge_index.squeeze(0)
        print("edge_type shape = ", edge_index.shape)
        edge_matrix_file = self.data_complete_path(EDGE_INDEX_FILE_NAME % self.top_k)
        print("saving edge_type list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)

    def create_fea_matrix(self, file_contents):
        self.print_step("Processing feature matrix for ")

        self.maybe_load_id_mappings()
        self.maybe_load_doc_splits()

        vocab = self.build_vocab(self.train_docs, self.doc2id, file_contents)
        vocab_size = len(vocab)
        stop_words = set(stopwords.words('english'))

        n_total = len(self.train_docs) + len(self.val_docs) + len(self.user2id)
        feat_matrix = lil_matrix((n_total, vocab_size))
        print("\nSize of feature matrix = ", feat_matrix.shape)
        print("\nCreating feat_matrix entries for docs nodes...")

        start = time.time()
        split_docs = self.train_docs + self.val_docs

        for count, (doc_id, content_file) in enumerate(file_contents.items()):
            print_iter = int(len(file_contents) / 5)
            # doc_name = content_file.split('\\')[-1].split('.')[0]
            if str(doc_id) in split_docs:
                if doc_id in self.doc2id and doc_id not in self.test_docs:
                    # feat_matrix[doc2id[str(doc_name)], :] = np.random.random(len(vocab)) > 0.99

                    with open(content_file, 'r') as f:
                        vector = np.zeros(len(vocab))
                        file_content = json.load(f)
                        text = file_content['text'].replace('\n', ' ')[:1500]
                        text = text.replace('\t', ' ').lower()
                        text = re.sub(r'#[\w-]+', 'hashtag', text)
                        text = re.sub(r'https?://\S+', 'url', text)
                        # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                        text = nltk.word_tokenize(text)
                        text_filtered = [w for w in text if w not in stop_words]
                        for token in text_filtered:
                            if token in vocab.keys():
                                vector[vocab[token]] = 1
                        feat_matrix[self.doc2id[doc_id], :] = vector

            if count % print_iter == 0:
                print("{} / {} done..".format(count + 1, len(file_contents)))

        hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        feat_matrix_sum = np.array(feat_matrix.sum(axis=1)).squeeze(1)
        # print(feat_matrix_sum.shape)
        idx = np.where(feat_matrix_sum == 0)
        # print(len(idx[0]))

        print("\nCreating feat_matrix entries for users nodes...")
        start = time.time()
        not_found, use = 0, 0
        # user_splits = json.load(open('./data/complete_data/{}/user_splits.json'.format(dataset), 'r'))
        # train_users = user_splits['train_users']

        src_dir = self.data_raw_path('engagements', 'complete', self.dataset)
        for root, dirs, files in os.walk(src_dir):
            for count, file in enumerate(files):
                print_iter = int(len(files) / 10)
                src_file_path = os.path.join(root, file)
                src_file = json.load(open(src_file_path, 'r'))
                users = src_file['users']
                doc_key = file.split(".")[0]
                # if str(doc_key) in self.train_docs:
                # Each user of this doc has its features as the features of the doc
                if (str(doc_key) in split_docs) and str(doc_key) in self.doc2id:
                    for user in users:
                        if str(user) in self.user2id:
                            feat_matrix[self.user2id[str(user)], :] += feat_matrix[self.doc2id[str(doc_key)], :]

                if count % print_iter == 0:
                    print(" {} / {} done..".format(count + 1, len(files)))
                    print(datetime.datetime.now())

        hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        # print(not_found, use)
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        feat_matrix = feat_matrix >= 1
        feat_matrix = feat_matrix.astype(int)

        # Sanity Checks
        feat_matrix_sum = np.array(feat_matrix.sum(axis=1)).squeeze(1)
        # print(feat_matrix_sum.shape)
        idx = np.where(feat_matrix_sum == 0)
        # print(len(idx[0]))

        filename = self.data_complete_path(FEAT_MATRIX_FILE_NAME % self.top_k)
        print("Matrix construction done! Saving in: {}".format(filename))
        save_npz(filename, feat_matrix.tocsr())

    def build_vocab(self, train_docs, doc2id, content_files):

        vocab_file = self.data_complete_path('vocab.json')
        if os.path.isfile(vocab_file):
            print("\nReading vocabulary from:  ", vocab_file)
            return json.load(open(vocab_file, 'r'))

        print("\nBuilding vocabulary...")
        vocab = {}
        stop_words = set(stopwords.words('english'))
        start = time.time()

        for doc_id, content_file in content_files.items():
            if doc_id in train_docs and doc_id in doc2id:
                with open(content_file, 'r') as f:
                    file_content = json.load(f)
                    text = file_content['text'].lower()[:500]
                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                    text = re.sub(r'https?://\S+', 'url', text)
                    # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                    text = nltk.word_tokenize(text)
                    text = [w for w in text if not w in stop_words]
                    for token in text:
                        if token not in vocab.keys():
                            vocab[token] = len(vocab)

        hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        print("Size of vocab =  ", len(vocab))
        print(f"Saving vocab for {self.dataset} at: {vocab_file}")
        with open(vocab_file, 'w+') as v:
            json.dump(vocab, v)

        return vocab

    @abc.abstractmethod
    def create_labels(self):
        raise NotImplementedError

    # @abc.abstractmethod
    # def create_split_masks(self):
    #     raise NotImplementedError

    @abc.abstractmethod
    def create_dgl_graph(self):
        raise NotImplementedError

    @staticmethod
    def np_converter(obj):
        """
        A converter which can be used when dumping JSON strings to files, as JSON can not work with numpy data types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    def create_split_masks(self):

        self.print_step("Creating split masks")

        self.maybe_load_id_mappings()
        self.maybe_load_doc_splits()

        if self.n_total is None:
            adj_matrix_train = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k))
            self.n_total = adj_matrix_train.shape[0]
            del adj_matrix_train

        train_mask, val_mask, test_mask = np.zeros(self.n_total), np.zeros(self.n_total), np.zeros(self.n_total)
        representation_mask = np.ones(self.n_total)

        not_in_train_or_val = 0
        for doc, doc_id in self.doc2id.items():
            doc_n = str(doc)
            if doc_n in self.train_docs:
                train_mask[doc_id] = 1
            elif doc_n in self.val_docs:
                val_mask[doc_id] = 1
                representation_mask[doc_id] = 0
            elif doc_n in self.test_docs:
                # TODO: there are many more items in the doc2dict than length of masks --> Why? --> because adjacency matrix only contains length for val and train docs
                if doc_id < test_mask.shape[0]:
                    test_mask[doc_id] = 1
            else:
                not_in_train_or_val += 1

        print("\nNot in train or val = ", not_in_train_or_val)
        print("train_mask sum = ", int(sum(train_mask)))
        print("val_mask sum = ", int(sum(val_mask)))
        print("test_mask sum = ", int(sum(test_mask)))

        mask_dict = {'train_mask': list(train_mask), 'val_mask': list(val_mask), 'test_mask': list(test_mask),
                     'repr_mask': list(representation_mask)}
        split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME % self.top_k)
        print("\nWriting split mask file in : ", split_mask_file)
        with open(split_mask_file, 'w+') as j:
            json.dump(mask_dict, j)


class DataPreprocessor(GraphIO):

    @staticmethod
    def print_step(step_title):
        print(f'\n{"=" * 50}\n \t\t{step_title}\n{"=" * 50}')

    def create_data_splits_standard(self, max_len=5000):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution.
        """
        self.print_step("Creating Data Splits")

        print(f"\nPreparing {self.dataset} ...")
        x_data, y_data, doc_data = [], [], []

        # Reading the dataset into workable lists
        removed, lens = [], []
        data_file = os.path.join(self.data_tsv_dir, self.dataset, CONTENT_INFO_FILE_NAME)

        with open(data_file, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                if isinstance(row['text'], str) and len(row['text']) > 5:
                    text = row['text'].replace('\n', ' ')
                    text = text.replace('\t', ' ')
                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                    text = re.sub(r'https?://\S+', 'url', text)

                    x_data.append(str(text[:max_len]))
                    lens.append(len(text[:max_len]))
                    y_data.append(int(row['label']))
                    doc_data.append(str(row['id']))
                else:
                    removed.append(row['id'])

        print("Average length = ", sum(lens) / len(lens))
        print("Maximum length = ", max(lens))
        print("Minimum Length = ", min(lens))
        print("Total data points removed (length < 6) = ", len(removed))

        # Creating train-val-test split with same/similar label distribution in each split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=21)
        x_rest, x_test, y_rest, y_test = [], [], [], []
        doc_rest, doc_id_test = [], []

        for train_index, test_index in sss.split(x_data, y_data):
            for idx in train_index:
                x_rest.append(x_data[idx])
                y_rest.append(y_data[idx])
                doc_rest.append(doc_data[idx])

            for idx in test_index:
                article_text = x_data[idx]
                # if article_text not in x_test:
                x_test.append(article_text)
                y_test.append(y_data[idx])
                doc_id_test.append(doc_data[idx])

        # TODO: may there be duplicates in train/test/val documents? Is this intended?
        # assert len(set(doc_id_test)) == len(doc_id_test), "doc_id_test contains duplicate doc IDs!"

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=21)

        for fold, (train_index, val_index) in enumerate(sss.split(x_rest, y_rest)):
            # TODO: why is this in the inner loop?
            x_train, x_val, y_train, y_val = [], [], [], []
            doc_id_train, doc_id_val = [], []

            for idx in train_index:
                article_text = x_rest[idx]
                # TODO: check if we allow duplicates or not
                # if article_text not in x_train:
                x_train.append(article_text)
                y_train.append(y_rest[idx])
                doc_id_train.append(doc_rest[idx])

            for idx in val_index:
                article_text = x_rest[idx]
                # TODO: check if we allow duplicates or not
                # if article_text not in x_val:
                x_val.append(article_text)
                y_val.append(y_rest[idx])
                doc_id_val.append(doc_rest[idx])

            # TODO: may there be duplicates in train/test/val documents? Is this intended?
            # assert len(set(doc_id_train)) == len(doc_id_train), "doc_id_train contains duplicate doc IDs!"
            # assert len(set(doc_id_val)) == len(doc_id_val), "doc_id_val contains duplicate doc IDs!"

        fake, real = self.get_label_distribution(y_train)
        print(f"\nFake labels in train split  = {fake * 100:.2f} %")
        print(f"Real labels in train split  = {real * 100:.2f} %")

        fake, real = self.get_label_distribution(y_val)
        print(f"\nFake labels in val split  = {fake * 100:.2f} %")
        print(f"Real labels in val split  = {real * 100:.2f} %")

        fake, real = self.get_label_distribution(y_test)
        print(f"\nFake labels in test split = {fake * 100:.2f} %")
        print(f"Real labels in test split  = {real * 100:.2f} %")

        print("\nWriting train-val-test files..")
        splits = ['train', 'val', 'test']
        for split in splits:
            if split == 'train':
                x = x_train
                y = y_train
                id_list = doc_id_train
            elif split == 'val':
                x = x_val
                y = y_val
                id_list = doc_id_val
            else:
                x = x_test
                y = y_test
                id_list = doc_id_test

            data_tsv_dir = os.path.join(self.data_tsv_dir, self.dataset, 'splits')
            if not os.path.exists(data_tsv_dir):
                os.makedirs(data_tsv_dir)

            data_tsv_dir = os.path.join(data_tsv_dir, f'{split}.tsv')
            print(f"{split} file in : {data_tsv_dir}")

            with open(data_tsv_dir, 'a', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                # csv_writer.writerow(['text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([x[i], y[i], id_list[i]])

        doc_splits_file = os.path.join(self.data_tsv_dir, self.dataset, 'docSplits.json')
        print("Writing doc_splits in : ", doc_splits_file)

        print("\nTotal train = ", len(doc_id_train))
        print("Total test = ", len(doc_id_test))
        print("Total val = ", len(doc_id_val))

        # print("\nDuplicates train = ", str(len(doc_id_train) - len(set(doc_id_train))))

        temp_dict = {'test_docs': doc_id_test, 'train_docs': doc_id_train, 'val_docs': doc_id_val}
        json.dump(temp_dict, open(doc_splits_file, 'w+'))

    @staticmethod
    def get_label_distribution(labels):
        fake = labels.count(1)
        real = labels.count(0)
        denom = fake + real
        return fake / denom, real / denom


def load_json_file(file_name):
    return json.load(open(file_name, 'r'))
