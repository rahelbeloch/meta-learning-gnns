import argparse
import datetime
import glob
from collections import defaultdict

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from scipy.sparse import load_npz

from data_prep.graph_io import *


class FakeHealthGraphPreprocessor(GraphPreprocessor):
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

    @staticmethod
    def get_doc_key(name, name_type):
        return name.split('.')[0]

    def __init__(self, config):
        super().__init__(config)

        # temporary attributes for data which has been loaded and will be reused
        self.n_total = None
        self.user2id = None

        # self.aggregate_user_contexts()
        # self.create_doc_user_splits()
        # self.create_doc_id_dicts()
        # self.filter_user_contexts()
        # self.create_adjacency_matrix()
        # self.create_feature_matrix()
        # self.create_labels()
        # self.create_split_masks()

        # self.create_dgl_graph()

    def save_labels(self, doc2labels):

        doc2labels_file = self.data_complete_path(DOC_2_LABELS_FILE_NAME % self.top_k)
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

        labels_file = self.data_complete_path(LABELS_FILE_NAME % self.top_k)
        print(f"\nLabels list construction done! Saving in : {labels_file}")
        with open(labels_file, 'w+') as v:
            json.dump({'labels_list': list(labels_list)}, v, default=self.np_converter)

        # Create the all_labels file
        all_labels = np.zeros(self.n_total, dtype=int)
        all_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME % self.top_k)
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

        docs_users = defaultdict(list)
        count = 0
        for root, dirs, files in os.walk(src_dir):
            if root.endswith("replies"):
                continue
            for count, file in enumerate(files):
                if file.startswith('.'):
                    continue

                src_file = load_json_file(os.path.join(root, file))
                user = src_file['user']['id']

                doc = root.split('/')[-2]
                docs_users[doc].append(user)
                if count % 10000 == 0:
                    print(f"{count} done")

        self.save_user_docs(count, dest_dir, docs_users)

    def create_doc_user_splits(self):
        self.create_user_splits(src_dir=self.data_complete_path('engagements'))

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

        assert len(node2id) == len(user2id_train) + len(
            self.doc2id), "Length of node2id is not the sum of doc2id and user2id length!"

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
        node2id_file = self.data_complete_path('node2id_lr_top50.json')
        print("Saving node2id_lr in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        print("\nDone ! All files written..")

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

        filename = self.data_complete_path(FEAT_MATRIX_FILE_NAME % self.top_k)
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
            adj_matrix = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k))
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
            adj_matrix_train = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k))
            self.n_total = adj_matrix_train.shape[0]
            del adj_matrix_train

        train_mask, val_mask, test_mask = np.zeros(self.n_total), np.zeros(self.n_total), np.zeros(self.n_total)
        representation_mask = np.ones(self.n_total)

        # not_in_train_or_val = 0
        for doc, doc_id in self.doc2id.items():
            doc_n = str(doc)
            if doc_n in self.train_docs:
                train_mask[doc_id] = 1
            elif doc_n in self.val_docs:
                val_mask[doc_id] = 1
                representation_mask[doc_id] = 0
            elif doc_n in self.test_docs:
                test_mask[doc_id] = 1
            # else:
            #     not_in_train_or_val += 1

        # print("\nNot in train or val = ", not_in_train_or_val)
        print("train_mask sum = ", int(sum(train_mask)))
        print("val_mask sum = ", int(sum(val_mask)))
        print("test_mask sum = ", int(sum(test_mask)))

        mask_dict = {'train_mask': list(train_mask), 'val_mask': list(val_mask), 'test_mask': list(test_mask),
                     'repr_mask': list(representation_mask)}
        split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
        print("\nWriting split mask file in : ", split_mask_file)
        with open(split_mask_file, 'w+') as j:
            json.dump(mask_dict, j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_raw_dir', type=str, default=RAW_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=COMPLETE_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=TSV_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='HealthStory',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_k', type=int, default=50, help='Number of top users.')

    parser.add_argument('--exclude_frequent', type=bool, default=True, help='TODO')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeHealthGraphPreprocessor(args.__dict__)
