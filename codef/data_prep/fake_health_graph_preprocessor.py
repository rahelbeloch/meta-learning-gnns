import argparse
import datetime
import glob
import json
import os
import re
import time
from collections import defaultdict

import nltk

nltk.download('stopwords')

import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import lil_matrix, save_npz, load_npz

from data_prep.config import *
from data_prep.graph_io import GraphIO

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']


def load_json_file(file_name):
    return json.load(open(file_name, 'r'))


class GraphDataPreprocessor(GraphIO):
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

    def __init__(self, config):
        super().__init__(config['data_set'], config['data_raw_dir'], config['data_tsv_dir'],
                         config['data_complete_dir'])

        self.exclude_frequent_users = config['exclude_frequent']

        # temporary attributes for data which has been loaded and will be reused
        self.n_total = None
        self.user2id = None
        self.doc2id = None
        self.train_docs, self.val_docs, self.test_docs = None, None, None

        # self.aggregate_user_contexts()
        # self.create_doc_user_splits()
        # self.create_doc_id_dicts()
        # self.filter_user_contexts()
        # self.create_adjacency_matrix()
        # self.create_feature_matrix()
        # self.create_labels()
        self.create_split_masks()

        # self.create_dgl_graph()

    def save_labels(self, doc2labels):

        doc2labels_file = self.data_complete_path(DOC_2_LABELS_FILE_NAME)
        print(f"Saving doc2labels for {self.dataset} at: {doc2labels_file}")
        with open(doc2labels_file, 'w+') as v:
            json.dump(doc2labels, v)

        labels_list = np.zeros(self.n_total, dtype=int)
        for key, value in doc2labels.items():
            labels_list[self.doc2id[str(key)]] = value

        # Sanity Checks
        # print(sum(labels_list))
        # print(len(labels_list))
        # print(sum(labels_list[2402:]))
        # print(sum(labels_list[:2402]))

        labels_file = self.data_complete_path(LABELS_FILE_NAME)
        print(f"\nLabels list construction done! Saving in : {labels_file}")
        with open(labels_file, 'w+') as v:
            json.dump({'labels_list': list(labels_list)}, v, default=self.np_converter)

        # Create the all_labels file
        all_labels = np.zeros(self.n_total, dtype=int)
        all_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
        for doc in doc2labels.keys():
            all_labels[self.doc2id[str(doc)]] = doc2labels[str(doc)]

        print("\nSum of labels this test set = ", int(sum(all_labels)))
        print("Len of labels = ", len(all_labels))

        print(f"\nall_labels list construction done! Saving in : {all_labels_file}")
        with open(all_labels_file, 'w+') as j:
            json.dump({'all_labels': list(all_labels)}, j, default=self.np_converter)

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

    def maybe_load_id_mappings(self):
        if self.user2id is None:
            self.user2id = json.load(open(self.data_complete_path(USER_2_ID_FILE_NAME), 'r'))
        if self.doc2id is None:
            self.doc2id = json.load(open(self.data_complete_path(DOC_2_ID_FILE_NAME), 'r'))

    def maybe_load_doc_splits(self):
        doc_splits = load_json_file(self.data_tsv_path(DOC_SPLITS_FILE_NAME))
        self.train_docs = doc_splits['train_docs']
        self.test_docs = doc_splits['test_docs']
        self.val_docs = doc_splits['val_docs']

    def build_vocab(self, vocab_file, train_docs, doc2id):

        if os.path.isfile(vocab_file):
            print("\nReading vocabulary from:  ", vocab_file)
            return json.load(open(vocab_file, 'r'))

        print("\nBuilding vocabulary...")
        vocab = {}
        stop_words = set(stopwords.words('english'))
        # start = time.time()
        # if self.dataset in ['gossipcop', 'politifact']:
        #     labels = ['fake', 'real']
        #     for label in labels:
        #         src_doc_dir = os.path.join(self.data_dir, 'base_data', dataset, label)
        #         for root, dirs, files in os.walk(src_doc_dir):
        #             for file in files:
        #                 doc = file.split('.')[0]
        #                 if str(doc) in train_docs and str(doc) in doc2id:
        #                     src_file_path = os.path.join(root, file)
        #                     with open(src_file_path, 'r') as f:
        #                         file_content = json.load(f)
        #                         text = file_content['text'].lower()[:500]
        #                         text = re.sub(r'#[\w-]+', 'hashtag', text)
        #                         text = re.sub(r'https?://\S+', 'url', text)
        #                         # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
        #                         text = nltk.word_tokenize(text)
        #                         text = [w for w in text if not w in stop_words]
        #                         for token in text:
        #                             if token not in vocab.keys():
        #                                 vocab[token] = len(vocab)
        #
        #         hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        #         print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        #         print("Size of vocab =  ", len(vocab))
        #         print("Saving vocab for  {}  at:  {}".format(dataset, vocab_file))
        #         with open(vocab_file, 'w+') as v:
        #             json.dump(vocab, v)

        if self.dataset in ['HealthStory', 'HelathRelease']:
            src_doc_dir = self.data_raw_path('content', self.dataset + "/*.json")
            all_files = glob.glob(src_doc_dir)
            for file in all_files:
                with open(file, 'r') as f:
                    file_content = json.load(f)
                    text = file_content['text'].replace('\n', ' ')[:1500]
                    text = text.replace('\t', ' ').lower()
                    text = re.sub(r'#[\w-]+', 'hashtag', text)
                    text = re.sub(r'https?://\S+', 'url', text)
                    # text = re.sub(r"[^A-Za-z(),!?\'`]", " ", text)
                    text = nltk.word_tokenize(text)
                    text = [w for w in text if w not in stop_words]
                    for token in text:
                        if token not in vocab.keys():
                            vocab[token] = len(vocab)

            print(f"Saving vocab for {self.dataset} at: {vocab_file}")
            with open(vocab_file, 'w+') as v:
                json.dump(vocab, v)

        return vocab

    def save_adj_matrix(self, adj_matrix, edge_type):

        adj_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME)
        print(f"\nMatrix construction done! Saving in  {adj_file}")
        save_npz(adj_file, adj_matrix.tocsr())

        edge_type_file = self.data_complete_path(EDGE_TYPE_FILE_NAME)
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

        edge_matrix_file = self.data_complete_path(ADJACENCY_MATRIX_FILE + '_edge.npy')
        print("saving edge_list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)

        edge_index = edge_type[edge_type.nonzero()]
        edge_index = edge_index.toarray()
        edge_index = edge_index.squeeze(0)
        print("edge_type shape = ", edge_index.shape)
        edge_matrix_file = self.data_complete_path(EDGE_INDEX_FILE_NAME)
        print("saving edge_type list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)

    def aggregate_user_contexts(self):
        """
        Aggregates only user IDs from different folders of tweets/retweets to a single place. Creates a folder which
        has files named after document Ids. Each file contains all the users that interacted with it.
        """

        self.print_step("Aggregating follower/ing relations")

        src_dir = self.data_raw_path("engagements", self.dataset)
        if not os.path.exists(src_dir):
            raise ValueError(f'Source directory {src_dir} does not exist!')

        dest_dir = self.data_complete_path("engagements")
        if not os.path.exists(dest_dir):
            print(f"Creating destination dir:  {dest_dir}\n")
            os.makedirs(dest_dir)

        c = 0
        docs_users = defaultdict(list)
        for root, dirs, files in os.walk(src_dir):
            if root.endswith("replies"):
                continue
            for count, file in enumerate(files):
                if file.startswith('.'):
                    continue
                c += 1
                doc = root.split('/')[-2]

                src_file = load_json_file(os.path.join(root, file))
                user = src_file['user']['id']

                docs_users[doc].append(user)
                if c % 10000 == 0:
                    print(f"{c} done")

        print(f"\nTotal tweets/re-tweets in the data set = {c}")
        print(f"\nWriting all the info in the dir: {dest_dir}")

        for doc, user_list in docs_users.items():
            document_user_file = os.path.join(dest_dir, f'{str(doc)}.json')
            with open(document_user_file, 'w+') as j:
                temp_dict = {'users': list(set(user_list))}
                json.dump(temp_dict, j)

        print("\nDONE..!!")

    def create_doc_user_splits(self):
        """

        :return:
        """

        self.print_step("Creating doc and user splits")

        self.maybe_load_doc_splits()

        print("\nCreating users in splits file..")

        src_dir = self.data_complete_path('engagements')

        train_users, val_users, test_users = set(), set(), set()
        for root, dirs, files in os.walk(src_dir):
            for count, file in enumerate(files):
                doc_key = file.split('.')[0]
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

        doc2id_file = self.data_complete_path(DOC_2_ID_FILE_NAME)
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
            restricted_users_file = self.data_tsv_path('restricted_users_5.json')
            valid_users_file = self.data_tsv_path('valid_users_top50.json')

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

        user2id_train = {}
        for count, user in enumerate(all_users):
            user2id_train[str(user)] = count + len(self.doc2id)
            node_type.append(2)

        user2id_train_file = self.data_complete_path(USER_2_ID_FILE_NAME)
        print("\nuser2id size = ", len(user2id_train))
        print("Saving user2id_train in : ", user2id_train_file)
        with open(user2id_train_file, 'w+') as j:
            json.dump(user2id_train, j)

        node2id = self.doc2id.copy()
        node2id.update(user2id_train)

        assert len(node2id) == len(user2id_train) + len(
            self.doc2id), "Length of node2id is not the sum of doc2id and user2id length!"

        print("\nnode2id size = ", len(node2id))
        node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME)
        print("Saving node2id_lr_train in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        # node type already contains train and val docs and train, val and test users
        assert len(node_type) == len(self.val_docs) + len(self.train_docs) + len(all_users)

        print("\nNode type size = ", len(node_type))
        node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME)
        node_type = np.array(node_type)
        print("Saving node_type in :", node_type_file)
        np.save(node_type_file, node_type, allow_pickle=True)

        print("\nAdding test docs..")
        orig_doc2id_len = len(self.doc2id)
        for test_count, doc in enumerate(self.test_docs):
            self.doc2id[str(doc)] = test_count + len(user2id_train) + orig_doc2id_len
        print("Test docs = ", len(self.test_docs))
        print("doc2id = ", len(self.doc2id))
        with open(self.data_complete_path(DOC_2_ID_FILE_NAME), 'w+') as j:
            json.dump(self.doc2id, j)

        node2id = self.doc2id.copy()
        node2id.update(user2id_train)
        print("node2id size = ", len(node2id))
        node2id_file = self.data_complete_path('node2id_lr_top50.json')
        print("Saving node2id_lr in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        print("\nDone ! All files written..")

    def filter_user_contexts(self):
        """
        Reduces the follower/ing data to users that are among the top k users.

        """

        self.print_step("Creating filtered follower-following")

        with open(self.data_complete_path(USER_2_ID_FILE_NAME), 'r') as j:
            all_users = json.load(j)

        # print_iter = int(len(all_users) / 10)
        print("Total users in this dataset = ", len(all_users))

        for user_context in USER_CONTEXTS:
            print(f"\n    - from {user_context}  folder...")

            dest_dir = self.data_raw_path(f'{user_context}_filtered')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            user_context_src_dir = self.data_raw_path(user_context)
            for root, dirs, files in os.walk(user_context_src_dir):
                for count, file in enumerate(files):
                    print_iter = int(len(files) / 10)
                    if count == 0:
                        print("Total user files = ", len(files))
                        print("Writing filtered lists in : ", dest_dir)
                        print("Printing every: ", print_iter)
                    src_file_path = os.path.join(root, file)
                    user_id = file.split(".")[0]
                    if os.path.isfile(os.path.join(dest_dir, str(user_id) + '.json')):
                        continue
                    if str(user_id) in all_users:
                        src_file = json.load(open(src_file_path, 'r'))
                        dest_file_path = os.path.join(dest_dir, str(user_id) + '.json')
                        temp = set()
                        followers = src_file['ids']
                        followers = list(map(int, followers))
                        for follower in followers:
                            if str(follower) in all_users:
                                temp.update([follower])
                        temp_dict = {'user_id': user_id}
                        name = 'followers' if user_context == 'user_followers' else 'following'
                        temp_dict[name] = list(temp)
                        with open(dest_file_path, 'w+') as v:
                            json.dump(temp_dict, v)

                    if count % print_iter == 0:
                        # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                        print(f"{count + 1} done..")

    def create_adjacency_matrix(self):
        """

        :return:
        """

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
        src_dir = self.data_complete_path('engagements')
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

        edge_list_file = self.data_complete_path(EDGE_LIST_FILE_NAME)
        print("Saving edge list in :", edge_list_file)
        with open(edge_list_file, 'w+') as j:
            json.dump(edge_list, j)

        hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not found user_ids = {not_found}")
        print(f"Total Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Total Non-zero entries edge_type = {edge_type.getnnz()}")

        self.save_adj_matrix(adj_matrix, edge_type)

    def create_feature_matrix(self):
        self.print_step("Processing feature matrix for ")

        self.maybe_load_id_mappings()

        self.maybe_load_doc_splits()

        vocab_file = self.data_complete_path('vocab.json')
        vocab = self.build_vocab(vocab_file, self.train_docs, self.doc2id)
        vocab_size = len(vocab)
        stop_words = set(stopwords.words('english'))

        n_total = len(self.train_docs) + len(self.val_docs) + len(self.user2id)
        feat_matrix = lil_matrix((n_total, vocab_size))
        print("\nSize of feature matrix = ", feat_matrix.shape)
        print("\nCreating feat_matrix entries for docs nodes...")

        start = time.time()
        split_docs = self.train_docs + self.val_docs

        src_doc_dir = self.data_raw_path('content', self.dataset + "/*.json")
        all_files = glob.glob(src_doc_dir)
        for count, file in enumerate(all_files):
            print_iter = int(len(all_files) / 5)
            doc_name = file.split('\\')[-1].split('.')[0]
            if str(doc_name) in split_docs:
                if str(doc_name) in self.doc2id and str(doc_name) not in self.test_docs:
                    # feat_matrix[doc2id[str(doc_name)], :] = np.random.random(len(vocab)) > 0.99

                    with open(file, 'r') as f:
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
                        feat_matrix[self.doc2id[str(doc_name)], :] = vector

            if count % print_iter == 0:
                print("{} / {} done..".format(count + 1, len(all_files)))

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

        filename = self.data_complete_path(FEAT_MATRIX_FILE_NAME)
        print("Matrix construction done! Saving in: {}".format(filename))
        save_npz(filename, feat_matrix.tocsr())

    def create_labels(self):
        """
        Create labels for each node of the graph
        """
        self.print_step("Creating labels")

        self.maybe_load_id_mappings()

        src_dir = self.data_raw_path('reviews', self.dataset + '.json')
        doc_labels = json.load(open(src_dir, 'r'))

        if self.n_total is None:
            adj_matrix = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME))
            self.n_total = adj_matrix.shape[0]
            del adj_matrix

        self.maybe_load_doc_splits()
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

    def create_split_masks(self):

        self.print_step("Creating split masks")

        self.maybe_load_id_mappings()
        self.maybe_load_doc_splits()

        if self.n_total is None:
            adj_matrix_train = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME))
            self.n_total = adj_matrix_train.shape[0]
            del adj_matrix_train

        train_mask, val_mask = np.zeros(self.n_total), np.zeros(self.n_total)  # np.zeros(test_n)
        representation_mask = np.ones(self.n_total)

        not_in_train_or_val = 0
        for doc, doc_id in self.doc2id.items():
            if str(doc) in self.train_docs:
                train_mask[doc_id] = 1
            elif str(doc) in self.val_docs:
                val_mask[doc_id] = 1
                representation_mask[doc_id] = 0
            else:
                not_in_train_or_val += 1

        print("\nNot in train or val = ", not_in_train_or_val)
        print("train_mask sum = ", int(sum(train_mask)))
        print("val_mask sum = ", int(sum(val_mask)))

        mask_dict = {'train_mask': list(train_mask), 'val_mask': list(val_mask), 'repr_mask': list(representation_mask)}
        split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
        print("\nWriting split mask file in : ", split_mask_file)
        with open(split_mask_file, 'w+') as j:
            json.dump(mask_dict, j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_raw_dir', type=str, default=RAW_DIR,
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=COMPLETE_DIR,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=TSV_DIR,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='HealthStory', help='TODO')

    parser.add_argument('--exclude_frequent', type=bool, default=False, help='TODO')

    args, unparsed = parser.parse_known_args()

    preprocessor = GraphDataPreprocessor(args.__dict__)
