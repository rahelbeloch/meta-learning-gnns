import argparse
from collections import defaultdict

import nltk

nltk.download('stopwords')

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

        # self.aggregate_user_contexts()
        # self.create_doc_user_splits()
        # self.create_doc_id_dicts()
        # self.filter_user_contexts()
        # self.create_adjacency_matrix()
        # self.create_feature_matrix()
        # self.create_labels()
        # self.create_split_masks()

        # self.create_dgl_graph()

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
        self.create_user_splits(self.data_complete_path('engagements'))

    # def create_doc_id_dicts(self):
    #     """
    #     Creates and saves doc2id and node2id dictionaries based on the document and user splits created by
    #     `create_doc_user_splits` function. Also puts constraints on the users which are used
    #     (e.g. only top k active users).
    #     """
    #
    #     self.print_step("Creating dicts")
    #     print("\nCreating doc2id and user2id dicts....\n")
    #
    #     user_splits_file = self.data_complete_path(USER_SPLITS_FILE_NAME)
    #     usr_splits = json.load(open(user_splits_file, 'r'))
    #     train_users, val_users, test_users = usr_splits['train_users'], usr_splits['val_users'], usr_splits[
    #         'test_users']
    #
    #     self.maybe_load_doc_splits()
    #
    #     doc2id = {}
    #     node_type = []
    #
    #     for train_count, doc in enumerate(self.train_docs):
    #         doc2id[str(doc)] = train_count
    #         node_type.append(1)
    #
    #     print("Train docs = ", len(self.train_docs))
    #     print("doc2id train = ", len(doc2id))
    #     print("Node type = ", len(node_type))
    #
    #     # assert len(self.train_docs) == len(node_type), "Length of train docs is not the same as length of node type!"
    #     # assert len(self.train_docs) == len(self.doc2id), "Length of train docs is not the same as length of doc2ids!"
    #
    #     for val_count, doc in enumerate(self.val_docs):
    #         doc2id[str(doc)] = val_count + len(self.train_docs)
    #         node_type.append(1)
    #
    #     print("\nVal docs = ", len(self.val_docs))
    #     print("doc2id train = ", len(doc2id))
    #     print("Node type = ", len(node_type))
    #
    #     assert len(self.train_docs) + len(self.val_docs) == len(node_type), \
    #         "Sum of train docs and val docs length is not the same as length of node type!"
    #
    #     doc2id_file = self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k)
    #     print("Saving doc2id dict in :", doc2id_file)
    #     with open(doc2id_file, 'w+') as j:
    #         json.dump(doc2id, j)
    #
    #     # print('\nTrain users = ', len(train_users))
    #     # print("Test users = ", len(test_users))
    #     # print("Val users = ", len(val_users))
    #     # print("All users = ", len(all_users))
    #
    #     if self.exclude_frequent_users:
    #         print("\nRestricting users ... ")
    #
    #         # Exclude most frequent users
    #         restricted_users_file = self.data_tsv_path(RESTRICTED_USERS)
    #         valid_users_file = self.data_tsv_path(VALID_USERS % self.top_k)
    #
    #         restricted_users, valid_users = [], []
    #         try:
    #             restricted_users = json.load(open(restricted_users_file, 'r'))['restricted_users']
    #             valid_users = json.load(open(valid_users_file, 'r'))['valid_users']
    #         except FileNotFoundError:
    #             print("\nDid not find file for restricted and valid users although they should be excluded!\n")
    #
    #         train_users = [u for u in train_users if str(u) not in restricted_users and str(u) in valid_users]
    #         val_users = [u for u in val_users if str(u) not in restricted_users and str(u) in valid_users]
    #         test_users = [u for u in test_users if str(u) not in restricted_users and str(u) in valid_users]
    #     else:
    #         train_users = [u for u in train_users]
    #         val_users = [u for u in val_users]
    #         test_users = [u for u in test_users]
    #
    #     # train_users = list(set(train_users)-set(done_users['done_users'])-set(restricted_users['restricted_users']))
    #
    #     all_users = list(set(train_users + val_users + test_users))
    #
    #     print('\nTrain users = ', len(train_users))
    #     print("Test users = ", len(test_users))
    #     print("Val users = ", len(val_users))
    #     print("All users = ", len(all_users))
    #
    #     a = set(train_users + val_users)
    #     b = set(test_users)
    #     print("\nUsers common between train/val and test = ", len(a.intersection(b)))
    #
    #     user2id_train = {}
    #     for count, user in enumerate(all_users):
    #         user2id_train[str(user)] = count + len(self.doc2id)
    #         node_type.append(2)
    #
    #     user2id_train_file = self.data_complete_path(USER_2_ID_FILE_NAME % self.top_k)
    #     print("\nuser2id size = ", len(user2id_train))
    #     print("Saving user2id_train in : ", user2id_train_file)
    #     with open(user2id_train_file, 'w+') as j:
    #         json.dump(user2id_train, j)
    #
    #     node2id = self.doc2id.copy()
    #     node2id.update(user2id_train)
    #
    #     assert len(node2id) == len(user2id_train) + len(
    #         self.doc2id), "Length of node2id is not the sum of doc2id and user2id length!"
    #
    #     print("\nnode2id size = ", len(node2id))
    #     node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
    #     print("Saving node2id_lr_train in : ", node2id_file)
    #     with open(node2id_file, 'w+') as json_file:
    #         json.dump(node2id, json_file)
    #
    #     # node type already contains train and val docs and train, val and test users
    #     assert len(node_type) == len(self.val_docs) + len(self.train_docs) + len(all_users)
    #
    #     print("\nNode type size = ", len(node_type))
    #     node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME % self.top_k)
    #     node_type = np.array(node_type)
    #     print("Saving node_type in :", node_type_file)
    #     np.save(node_type_file, node_type, allow_pickle=True)
    #
    #     print("\nAdding test docs..")
    #     orig_doc2id_len = len(self.doc2id)
    #     for test_count, doc in enumerate(self.test_docs):
    #         self.doc2id[str(doc)] = test_count + len(user2id_train) + orig_doc2id_len
    #     print("Test docs = ", len(self.test_docs))
    #     print("doc2id = ", len(self.doc2id))
    #     with open(self.data_complete_path(DOC_2_ID_FILE_NAME % self.top_k), 'w+') as j:
    #         json.dump(self.doc2id, j)
    #
    #     node2id = self.doc2id.copy()
    #     node2id.update(user2id_train)
    #     print("node2id size = ", len(node2id))
    #     node2id_file = self.data_complete_path('node2id_lr_top50.json')
    #     print("Saving node2id_lr in : ", node2id_file)
    #     with open(node2id_file, 'w+') as json_file:
    #         json.dump(node2id, json_file)
    #
    #     print("\nDone ! All files written..")

    def filter_user_contexts(self):
        self.filter_contexts('ids')

    def create_adjacency_matrix(self):
        self.create_adj_matrix(self.data_complete_path('engagements'))

    def create_feature_matrix(self):

        src_doc_dir = self.data_raw_path('content', self.dataset + "/*.json")
        all_files = glob.glob(src_doc_dir)

        self.create_fea_matrix(all_files)

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
