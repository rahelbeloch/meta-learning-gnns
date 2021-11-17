import abc
import csv
import datetime
import os
import time
from collections import defaultdict
from json import JSONDecodeError

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from scipy.sparse import lil_matrix, save_npz
from data_preprocess_utils import *

from config import *

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']
LABELS = {0: 'fake', 1: 'real'}


class GraphIO:

    def __init__(self, dataset, raw_dir=RAW_DIR, tsv_dir=TSV_DIR, complete_dir=COMPLETE_DIR):
        self.dataset = dataset

        full_raw_path = os.path.join(raw_dir, dataset)
        if not os.path.exists(full_raw_path):
            raise ValueError(f"Wanting to preprocess data for dataset '{dataset}', but raw data in path"
                             f" with raw data '{full_raw_path}' does not exist!")

        self.data_raw_dir = raw_dir
        self.data_tsv_dir = self.create_dir(tsv_dir)
        self.data_complete_dir = self.create_dir(complete_dir)

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

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


class GraphPreprocessor(GraphIO):

    def __init__(self, config):
        super().__init__(config['data_set'], config['data_raw_dir'], config['data_tsv_dir'],
                         config['data_complete_dir'])
        self.exclude_frequent_users = config['exclude_frequent']
        self.top_k = config['top_k']

        # temporary attributes for data which has been loaded and will be reused
        self.doc2id, self.user2id = None, None
        self.n_total = None
        self.train_docs, self.test_docs, self.val_docs = None, None, None

    def doc_used(self, doc_id):
        return doc_id in self.train_docs or doc_id in self.test_docs or doc_id in self.val_docs

    def maybe_load_doc_splits(self):
        doc_splits = load_json_file(self.data_tsv_path(DOC_SPLITS_FILE_NAME))
        self.train_docs = doc_splits['train_docs']
        self.test_docs = doc_splits['test_docs']
        self.val_docs = doc_splits['val_docs']

    def maybe_load_id_mappings(self):
        if self.user2id is None:
            user2id_file = self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k)
            if os.path.exists(user2id_file):
                self.user2id = json.load(open(user2id_file, 'r'))
        if self.doc2id is None:
            doc2id_file = self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k)
            if os.path.exists(doc2id_file):
                self.doc2id = json.load(open(doc2id_file, 'r'))

        self.n_total = len(self.user2id) + len(self.doc2id)

    def save_user_doc_engagements(self, docs_users):

        dest_dir = self.data_complete_path('engagements')
        if not os.path.exists(dest_dir):
            print(f"Creating destination dir:  {dest_dir}\n")
            os.makedirs(dest_dir)

        print(f"\nWriting engagement info in the dir: {dest_dir}")

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

    def filter_restricted_users(self):

        doc2labels = load_json_file(self.data_tsv_path('doc2labels.json'))
        user_stats = defaultdict(lambda: {'fake': 0, 'real': 0})

        restriction_docs = 0
        for root, dirs, files in os.walk(self.data_complete_path('engagements')):
            for count, file in enumerate(files):

                # only restrict users interacting with this document ID if we actually use this doc in our splits
                doc_key = self.get_doc_key(file, name_type='file')
                if not self.doc_used(doc_key):
                    continue

                restriction_docs += 1

                try:
                    src_file = load_json_file(os.path.join(root, file))
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
                        user_stats[u][LABELS[doc2labels[doc_key]]] += 1

        # based on user stats, exclude some
        n_docs = len(self.train_docs) + len(self.test_docs) + len(self.val_docs)

        assert n_docs == restriction_docs, "Total nr of documents used does not equal the number of docs for " \
                                           "which we restrict users!"

        user_stats_avg = user_stats.copy()
        for user, stat in user_stats_avg.items():
            for label in LABELS.values():
                stat[label] = stat[label] / n_docs

        # filter for 30% in either one of the classes
        restricted_users = []
        for user, stat in user_stats_avg.items():
            restricted = stat['fake'] >= 0.3 or stat['real'] >= 0.3
            if restricted:
                restricted_users.append(user)

        print('Restricted users done.')

    def create_user_splits(self):
        """

        :return:
        """

        self.print_step("Creating doc and user splits")

        print("\nCreating users in splits file..")

        train_users, val_users, test_users = set(), set(), set()

        # walk through user-doc engagement files created before
        for root, dirs, files in os.walk(self.data_complete_path('engagements')):
            for count, file in enumerate(files):
                src_file_path = os.path.join(root, file)
                try:
                    src_file = json.load(open(src_file_path, 'r'))
                except UnicodeDecodeError:
                    # TODO: fix this error / keep track of files for which this happens
                    print("Exception")
                    continue

                doc_key = self.get_doc_key(file, name_type='file')
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

        self.print_step("Creating doc2id and user2id dicts")

        n_train = len(self.train_docs)
        print("Train docs = ", n_train)
        doc2id_train, node_type_train = self.doc_node_info(self.train_docs)

        n_val = len(self.val_docs)
        print("Val docs = ", n_val)
        doc2id_val, node_type_val = self.doc_node_info(self.val_docs)

        self.doc2id = doc2id_train | doc2id_val

        splits = load_json_file(self.data_complete_path(USER_SPLITS_FILE_NAME))
        train_users, val_users, test_users = splits['train_users'], splits['val_users'], splits['test_users']

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

            # find restricted users (shared more than 30% of articles in either class)

            # for each
            # walk through doc-user engagements

            train_users = [u for u in train_users if str(u) not in restricted_users and str(u) in valid_users]
            val_users = [u for u in val_users if str(u) not in restricted_users and str(u) in valid_users]
            test_users = [u for u in test_users if str(u) not in restricted_users and str(u) in valid_users]
            # train_users = list(set(train_users)-set(done_users['done_users'])-set(restricted_users['restricted_users']))

        all_users = list(set(train_users + val_users + test_users))

        print('\nTrain users = ', len(train_users))
        print("Test users = ", len(test_users))
        print("Val users = ", len(val_users))
        print("All users = ", len(all_users))

        a = set(train_users + val_users)
        b = set(test_users)
        print("\nUsers common between train/val and test = ", len(a.intersection(b)))

        node_type_user = []
        self.user2id = {}

        for count, user in enumerate(all_users):
            self.user2id[str(user)] = count + len(self.doc2id)
            node_type_user.append(2)

        print("\nUser2id size = ", len(self.user2id))
        user2id_train_file = self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k)
        print("Saving user2id_train in : ", user2id_train_file)
        with open(user2id_train_file, 'w+') as j:
            json.dump(self.user2id, j)

        # node type should contain train and val docs and train, val and test users
        node_type = node_type_train + node_type_val + node_type_user
        assert len(node_type) == n_val + n_train + len(all_users)

        print(f"\nNode type size = {len(node_type)}")
        node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME % self.top_k)
        node_type = np.array(node_type)
        print(f"Saving node type in : {node_type_file}")
        np.save(node_type_file, node_type, allow_pickle=True)

        print("\nAdding test docs..")
        n_test = len(self.test_docs)
        print("Test docs = ", n_test)
        orig_doc2id_len = len(self.doc2id)

        for test_count, doc in enumerate(self.test_docs):
            # n_test + len(self.user2id) + orig_doc2id_len
            self.doc2id[doc] = test_count + len(self.user2id) + orig_doc2id_len

        assert len(self.doc2id) == (n_val + n_train + n_test), "Doc2id does not contain all documents!"
        print("New doc2id len including test docs = ", len(self.doc2id))
        with open(self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k), 'w+') as j:
            json.dump(self.doc2id, j)

        node2id = self.doc2id.copy()
        node2id.update(self.user2id)
        assert len(node2id) == len(self.user2id) + len(self.doc2id), \
            "Length of node2id is not the sum of doc2id and user2id length!"

        print("\nNode2id size = ", len(node2id))
        node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
        print("Saving node2id_lr in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        print("\nDone ! All files written.")

    def doc_node_info(self, docs):
        doc2id, node_type = {}, []

        for doc_count, doc_name in enumerate(docs):
            doc2id[doc_name] = doc_count
            node_type.append(1)

        assert len(docs) == len(doc2id) == len(node_type), \
            "doc2id and node type for train to not match number of train documents!"

        return doc2id, node_type

    def filter_contexts(self, follower_key=None):
        """
        Reduces the follower/ing data to users that are among the top k users.
        """

        self.print_step("Creating filtered follower-following")

        with open(self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k), 'r') as j:
            all_users = json.load(j)

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

                    src_file = json.load(open(os.path.join(root, file), 'r'))

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
                        print(f"{count + 1} done..")

    def create_adj_matrix(self):

        self.print_step(f"Processing {self.dataset} dataset for adj_matrix")

        self.maybe_load_id_mappings()

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
        for root, dirs, files in os.walk(self.data_complete_path('engagements')):
            for count, file_name in enumerate(files):
                doc_key = str(file_name.split(".")[0])
                if doc_key == '':
                    continue
                src_file = load_json_file(os.path.join(root, file_name))
                users = map(str, src_file['users'])
                for user in users:
                    if doc_key in self.test_docs:
                        # no connections between users and test documents!
                        continue

                    if doc_key in self.doc2id and user in self.user2id and doc_key:
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
            user_context_src_dir = self.data_raw_path(user_context)
            for root, dirs, files in os.walk(user_context_src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    # user_id = src_file_path.split(".")[0]
                    src_file = load_json_file(src_file_path)
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

    def create_fea_matrix(self, file_contents):
        self.print_step("Creating feature matrix")

        self.maybe_load_id_mappings()

        vocab = self.build_vocab(file_contents)
        feat_matrix = lil_matrix((self.n_total, len(vocab)))
        print("\nSize of feature matrix = ", feat_matrix.shape)
        print("\nCreating feat_matrix entries for docs nodes...")

        start = time.time()
        split_docs = self.train_docs + self.val_docs
        stop_words = set(stopwords.words('english'))

        for count, (doc_name, content_file) in enumerate(file_contents.items()):
            print_iter = int(len(file_contents) / 5)
            if doc_name in split_docs:
                if doc_name in self.doc2id and doc_name not in self.test_docs:
                    # feat_matrix[doc2id[str(doc_name)], :] = np.random.random(len(vocab)) > 0.99

                    with open(content_file, 'r') as f:
                        text = nltk.word_tokenize(sanitize_text(json.load(f)['text'], 1500))
                        text_filtered = [w for w in text if w not in stop_words]

                        vector = np.zeros(len(vocab))
                        for token in text_filtered:
                            if token in vocab.keys():
                                vector[vocab[token]] = 1
                        feat_matrix[self.doc2id[doc_name], :] = vector

            if count % print_iter == 0:
                print("{} / {} done..".format(count + 1, len(file_contents)))

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        # feat_matrix_sum = np.array(feat_matrix.sum(axis=1)).squeeze(1)

        print("\nCreating feat_matrix entries for users nodes...")
        start = time.time()

        src_dir = self.data_complete_path('engagements')
        for root, dirs, files in os.walk(src_dir):
            print_iter = int(len(files) / 10)

            for count, file in enumerate(files):
                src_file = load_json_file(os.path.join(root, file))
                doc_key = file.split(".")[0]
                # if str(doc_key) in self.train_docs:
                # Each user of this doc has its features as the features of the doc
                if doc_key in split_docs and doc_key in self.doc2id:
                    for user in src_file['users']:
                        if str(user) in self.user2id:
                            feat_matrix[self.user2id[str(user)], :] += feat_matrix[self.doc2id[str(doc_key)], :]

                if count % print_iter == 0:
                    print(" {} / {} done..".format(count + 1, len(files)))
                    print(datetime.datetime.now())

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")

        feat_matrix = feat_matrix >= 1
        feat_matrix = feat_matrix.astype(int)

        # Sanity Checks
        # feat_matrix_sum = np.array(feat_matrix.sum(axis=1)).squeeze(1)

        filename = self.data_complete_path(FEAT_MATRIX_FILE_NAME % self.top_k)
        print("Matrix construction done! Saving in: {}".format(filename))
        save_npz(filename, feat_matrix.tocsr())

    def build_vocab(self, content_files):

        vocab_file = self.data_complete_path('vocab.json')
        if os.path.isfile(vocab_file):
            print("\nReading vocabulary from:  ", vocab_file)
            return json.load(open(vocab_file, 'r'))

        print("\nBuilding vocabulary...")
        vocab = {}
        stop_words = set(stopwords.words('english'))
        start = time.time()

        for doc_id, content_file in content_files.items():
            if doc_id in self.train_docs and doc_id in self.doc2id:
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

        hrs, mins, secs = calc_elapsed_time(start, time.time())
        print("Done. Took {}hrs and {}mins and {}secs\n".format(hrs, mins, secs))
        print("Size of vocab =  ", len(vocab))
        print(f"Saving vocab for {self.dataset} at: {vocab_file}")
        with open(vocab_file, 'w+') as v:
            json.dump(vocab, v)

        return vocab

    @abc.abstractmethod
    def create_labels(self):
        raise NotImplementedError

    def create_split_masks(self):

        self.print_step("Creating split masks")

        self.maybe_load_id_mappings()

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
                # TODO: there are many more items in the doc2dict than length of masks -->
                #  Why? --> because adjacency matrix only contains length for val and train docs
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

    def preprocess(self, min_len=6, max_len=5000):
        """
        Applies some preprocessing to the data, e.g. replacing special characters, filters non-requiured articles out.
        :param min_len: Minimum required length for articles.
        :param max_len: Maximum required length for articles.
        :return: Numpy arrays for article tests (x_data), article labels (y_data), and article names (doc_names).
        """

        x_data, y_data, doc_names, x_lengths, invalid = [], [], [], [], []

        data_file = os.path.join(self.data_tsv_dir, self.dataset, CONTENT_INFO_FILE_NAME)

        with open(data_file, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                if isinstance(row['text'], str) and len(row['text']) >= min_len:
                    text = sanitize_text(row['text'], max_len)
                    x_data.append(text)
                    x_lengths.append(len(text))
                    y_data.append(int(row['label']))
                    doc_names.append(str(row['id']))
                else:
                    invalid.append(row['id'])

        print(f"Average length = {sum(x_lengths) / len(x_lengths)}")
        print(f"Maximum length = {max(x_lengths)}")
        print(f"Minimum Length = {min(x_lengths)}")
        print(f"Total data points invalid and therefore removed (length < {min_len}) = {len(invalid)}")

        return np.array(x_data), np.array(y_data), np.array(doc_names)

    def create_data_splits(self, test_size=0.2, val_size=0.1, splits=1, duplicate_stats=False):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution. Currently only supports splitting data in 1 split for each set.

        :param test_size: Size of the test split compared to the whole data.
        :param val_size: Size of the val split compared to the whole data.
        :param splits: Number of splits.
        """

        self.print_step("Creating Data Splits")

        data = self.preprocess()

        if duplicate_stats:
            # counting duplicates in test set
            texts = []
            duplicates = defaultdict(lambda: {'counts': 1, 'd_names': []})

            for i in range(len(data[0])):
                d_text = data[0][i]
                if d_text in texts:
                    duplicates[d_text]['counts'] += 1
                    duplicates[d_text]['d_names'].append(data[2][i])
                else:
                    texts.append(d_text)

            duplicates_file = self.data_tsv_path('duplicates_info.json')
            json.dump(duplicates, open(duplicates_file, 'w+'), default=self.np_converter)

        # Creating train-val-test split with same/similar label distribution in each split

        # one tuple is one split and contains: (x, y, doc_names)
        rest_split, test_split = split_data(splits, test_size, data)

        assert len(set(test_split[2])) == len(test_split[2]), "Test split contains duplicate doc names!"

        # split rest data into validation and train splits
        train_split, val_split = split_data(splits, val_size, rest_split)

        assert len(set(val_split[2])) == len(val_split[2]), "Validation split contains duplicate doc names!"
        assert len(set(train_split[2])) == len(train_split[2]), "Train split contains duplicate doc names!"

        print_label_distribution(train_split[1])
        print_label_distribution(val_split[1])
        print_label_distribution(test_split[1])

        print("\nWriting train-val-test files..")

        splits = {'train': train_split, 'val': val_split, 'test': test_split}
        for split, data in splits.items():
            x, y, name_list = data

            split_path = self.data_tsv_path('splits')
            if not os.path.exists(split_path):
                os.makedirs(split_path)

            split_file_path = os.path.join(split_path, f'{split}.tsv')
            print(f"{split} file in : {split_file_path}")

            with open(split_file_path, 'a', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                # csv_writer.writerow(['text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([x[i], y[i], name_list[i]])

        doc_splits_file = self.data_tsv_path(DOC_SPLITS_FILE_NAME)
        print("Writing doc_splits in : ", doc_splits_file)

        doc_names_train, doc_names_test, doc_names_val = train_split[2], test_split[2], val_split[2]
        print("\nTotal train = ", len(doc_names_train))
        print("Total test = ", len(doc_names_test))
        print("Total val = ", len(doc_names_val))

        split_dict = {'test_docs': doc_names_test, 'train_docs': doc_names_train, 'val_docs': doc_names_val}
        json.dump(split_dict, open(doc_splits_file, 'w+'), default=self.np_converter)

    def store_doc2labels(self, doc2labels):
        """
        Stores the doc2label dictionary as JSON file in the respective directory.

        :param doc2labels: Dictionary containing document names mapped to the label for that document.
        """

        print(f"Total docs : {len(doc2labels)}")
        doc2labels_file = self.data_tsv_path('doc2labels.json')

        print(f"Writing doc2labels JSON in :  {doc2labels_file}")
        json.dump(doc2labels, open(doc2labels_file, 'w+'))

    def store_doc_contents(self, contents):
        """
        Stores document contents (doc name/id, title, text and label) as a TSV file.

        :param contents: List of entries (doc name/id, title, text and label) for every article.
        """
        print("\nCreating the data corpus file for: ", self.dataset)

        content_dest_file = self.data_tsv_path(CONTENT_INFO_FILE_NAME)
        # TODO: should this be append or just always be overwritten?
        if os.path.isfile(content_dest_file):
            print(f"\nTarget data file '{content_dest_file}' already exists, overwriting it.")
            open_mode = 'w'
        else:
            open_mode = 'a'

        with open(content_dest_file, open_mode, encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['id', 'title', 'text', 'label'])
            for file_content in contents:
                csv_writer.writerow(file_content)

        print("Final file written in :  ", content_dest_file)

    def print_step(self, step_title):
        print(f'\n{"-" * 50}\n \t\t {step_title} for {self.dataset} dataset.\n{"-" * 50}')
