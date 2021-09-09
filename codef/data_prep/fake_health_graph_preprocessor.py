import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import lil_matrix, save_npz

from data_utils import load_json_file

USER_CONTEXTS = ['user_followers', 'user_following']
USER_CONTEXTS_FILTERED = ['user_followers_filtered', 'user_following_filtered']

USER_2_ID_FILE_NAME = 'user2id_lr_top50_train.json'
DOC_2_ID_FILE_NAME = 'doc2id_lr_top50_train.json'
NODE_2_ID_FILE_NAME = 'node2id_lr_top50_train.json'
NODE_TYPE_FILE_NAME = 'node_type_lr_top50_train.npy'
DOC_SPLITS_FILE_NAME = 'docSplits.json'
USER_SPLITS_FILE_NAME = 'userSplits.json'
ADJACENCY_MATRIX_FILE = 'adj_matrix_lr_top50_train'
ADJACENCY_MATRIX_FILE_NAME = ADJACENCY_MATRIX_FILE + '.npz'
EDGE_TYPE_FILE = 'edge_type_lr_top50'
EDGE_TYPE_FILE_NAME = '%s.npz' % EDGE_TYPE_FILE


class GraphDataPreprocessor:
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
        self.dataset = config['data_set']
        self.data_raw_dir = config['data_raw_dir']
        self.data_complete_dir = config['data_complete_dir']
        self.data_tsv_dir = config['data_tsv_dir']
        self.exclude_frequent_users = config['exclude_frequent']

        self.aggregate_user_contexts()
        # self.create_doc_user_splits()
        # self.create_doc_id_dicts()
        # self.filter_user_contexts()
        # self.create_adjacency_matrix()

    @staticmethod
    def calc_elapsed_time(start, end):
        hours, rem = divmod(end - start, 3600)
        time_hours, time_rem = divmod(end, 3600)
        minutes, seconds = divmod(rem, 60)
        time_mins, _ = divmod(time_rem, 60)
        return int(hours), int(minutes), int(seconds)

    def data_src_path(self, *parts):
        return os.path.join(self.data_raw_dir, *parts)

    def data_intermediate_path(self, *parts):
        return os.path.join(self.data_tsv_dir, self.dataset, *parts)

    def data_dest_path(self, *parts):
        return os.path.join(self.data_complete_dir, self.dataset, *parts)

    def print_step(self, step_title):
        print(f'\n{"-" * 100}\n \t\t {step_title} for {self.dataset} dataset.\n{"-" * 100}')

    def save_adj_matrix(self, adj_matrix, edge_type):

        adj_file = self.data_dest_path(ADJACENCY_MATRIX_FILE_NAME)
        print(f"\nMatrix construction done! Saving in  {adj_file}")
        save_npz(adj_file, adj_matrix.tocsr())

        edge_type_file = self.data_dest_path(EDGE_TYPE_FILE_NAME)
        print(f"\nedge_type construction done! Saving in  {edge_type_file}")
        save_npz(edge_type_file, edge_type.tocsr())

        # Creating an edge_list matrix of the adj_matrix as required by some GCN frameworks
        print("\nCreating edge_index format of adj_matrix...")
        # G = nx.DiGraph(adj_matrix.tocsr())
        # temp_matrix = adj_matrix.toarray()
        # rows, cols = np.nonzero(temp_matrix)
        rows, cols = adj_matrix.nonzero()

        edge_index = np.vstack((np.array(rows), np.array(cols)))
        print("Edge index shape = ", edge_index.shape)

        edge_matrix_file = self.data_dest_path(ADJACENCY_MATRIX_FILE + '_edge.npy')
        print("saving edge_list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)

        edge_index = edge_type[edge_type.nonzero()]
        edge_index = edge_index.toarray()
        edge_index = edge_index.squeeze(0)
        print("edge_type shape = ", edge_index.shape)
        edge_matrix_file = self.data_dest_path(EDGE_TYPE_FILE + '_edge.npy')
        print("saving edge_type edge list format in :  ", edge_matrix_file)
        np.save(edge_matrix_file, edge_index, allow_pickle=True)

    def aggregate_user_contexts(self):
        """
        Aggregates only user IDs from different folders of tweets/retweets to a single place. Creates a folder which
        has files named after document Ids. Each file contains all the users that interacted with it.
        """

        self.print_step("Aggregating follower/ing relations")

        src_dir = self.data_src_path("engagements", self.dataset)
        if not os.path.exists(src_dir):
            raise ValueError(f'Source directory {src_dir} does not exist!')

        dest_dir = self.data_dest_path("engagements")
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

        print("\nTotal tweets/re-tweets in the data set = ", c)
        print("\nWriting all the info in the dir: ", dest_dir)

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

        doc_splits_file = self.data_intermediate_path(DOC_SPLITS_FILE_NAME)
        doc_splits = load_json_file(doc_splits_file)

        train_docs = doc_splits['train_docs']
        test_docs = doc_splits['test_docs']
        val_docs = doc_splits['val_docs']

        print("\nCreating users in splits file..")

        src_dir = self.data_dest_path('engagements')

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
                if str(doc_key) in train_docs:
                    train_users.update(users)
                if str(doc_key) in val_docs:
                    val_users.update(users)
                if str(doc_key) in test_docs:
                    test_users.update(users)

        user_splits_file = self.data_dest_path(USER_SPLITS_FILE_NAME)
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

        user_splits_file = self.data_dest_path(USER_SPLITS_FILE_NAME)
        usr_splits = json.load(open(user_splits_file, 'r'))
        doc_splits_file = self.data_intermediate_path(DOC_SPLITS_FILE_NAME)
        doc_splits = json.load(open(doc_splits_file, 'r'))

        train_users, val_users, test_users = usr_splits['train_users'], usr_splits['val_users'], usr_splits[
            'test_users']
        train_docs, val_docs, test_docs = doc_splits['train_docs'], doc_splits['val_docs'], doc_splits['test_docs']

        doc2Id = {}
        node_type = []

        for train_count, doc in enumerate(train_docs):
            doc2Id[str(doc)] = train_count
            node_type.append(1)

        print("Train docs = ", len(train_docs))
        print("doc2id train = ", len(doc2Id))
        print("Node type = ", len(node_type))

        # assert len(train_docs) == len(node_type), "Length of train docs is not the same as length of node type!"
        # assert len(train_docs) == len(doc2Id), "Length of train docs is not the same as length of doc2Ids!"

        for val_count, doc in enumerate(val_docs):
            doc2Id[str(doc)] = val_count + len(train_docs)
            node_type.append(1)

        print("\nVal docs = ", len(val_docs))
        print("doc2id train = ", len(doc2Id))
        print("Node type = ", len(node_type))

        assert len(train_docs) + len(val_docs) == len(node_type), \
            "Sum of train docs and val docs length is not the same as length of node type!"

        doc2id_file = self.data_dest_path(DOC_2_ID_FILE_NAME)
        print("Saving doc2id dict in :", doc2id_file)
        with open(doc2id_file, 'w+') as j:
            json.dump(doc2Id, j)

        # print('\nTrain users = ', len(train_users))
        # print("Test users = ", len(test_users))
        # print("Val users = ", len(val_users))
        # print("All users = ", len(all_users))

        if self.exclude_frequent_users:
            print("\nRestricting users ... ")

            # Exclude most frequent users
            restricted_users_file = self.data_intermediate_path('restricted_users_5.json')
            valid_users_file = self.data_intermediate_path('valid_users_top50.json')

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
            user2id_train[str(user)] = count + len(doc2Id)
            node_type.append(2)

        user2id_train_file = self.data_dest_path(USER_2_ID_FILE_NAME)
        print("\nuser2id size = ", len(user2id_train))
        print("Saving user2id_train in : ", user2id_train_file)
        with open(user2id_train_file, 'w+') as j:
            json.dump(user2id_train, j)

        node2id = doc2Id.copy()
        node2id.update(user2id_train)

        assert len(node2id) == len(user2id_train) + len(
            doc2Id), "Length of node2id is not the sum of doc2id and user2id length!"

        print("\nnode2id size = ", len(node2id))
        node2id_file = self.data_dest_path(NODE_2_ID_FILE_NAME)
        print("Saving node2id_lr_train in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        # node type already contains train and val docs and train, val and test users
        assert len(node_type) == len(val_docs) + len(train_docs) + len(all_users)

        print("\nNode type size = ", len(node_type))
        node_type_file = self.data_dest_path(NODE_TYPE_FILE_NAME)
        node_type = np.array(node_type)
        print("Saving node_type in :", node_type_file)
        np.save(node_type_file, node_type, allow_pickle=True)

        print("\nAdding test docs..")
        orig_doc2id_len = len(doc2Id)
        for test_count, doc in enumerate(test_docs):
            doc2Id[str(doc)] = test_count + len(user2id_train) + orig_doc2id_len
        print("Test docs = ", len(test_docs))
        print("doc2Id = ", len(doc2Id))
        with open(self.data_dest_path(DOC_2_ID_FILE_NAME), 'w+') as j:
            json.dump(doc2Id, j)

        node2id = doc2Id.copy()
        node2id.update(user2id_train)
        print("node2id size = ", len(node2id))
        node2id_file = self.data_dest_path('node2id_lr_top50.json')
        print("Saving node2id_lr in : ", node2id_file)
        with open(node2id_file, 'w+') as json_file:
            json.dump(node2id, json_file)

        print("\nDone ! All files written..")

    def filter_user_contexts(self):
        """
        Reduces the follower/ing data to users that are among the top k users.

        """

        self.print_step("Creating filtered follower-following")

        with open(self.data_dest_path(USER_2_ID_FILE_NAME), 'r') as j:
            all_users = json.load(j)

        # print_iter = int(len(all_users) / 10)
        print("Total users in this dataset = ", len(all_users))

        for user_context in USER_CONTEXTS:
            print(f"\n    - from {user_context}  folder...")

            dest_dir = self.data_src_path(f'{user_context}_filtered')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            user_context_src_dir = self.data_src_path(user_context)
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

        user2id_file = self.data_dest_path(USER_2_ID_FILE_NAME)
        user2id = json.load(open(user2id_file, 'r'))

        doc2id_file = self.data_dest_path(DOC_2_ID_FILE_NAME)
        doc2id = json.load(open(doc2id_file, 'r'))

        doc_splits_file = self.data_intermediate_path(DOC_SPLITS_FILE_NAME)
        doc_splits = json.load(open(doc_splits_file, 'r'))
        test_docs = doc_splits['test_docs']

        # this assumes users and documents are unique in all data sets!
        num_users, num_docs = len(user2id), len(doc2id) - len(test_docs)
        print("\nNumber of unique users = ", num_users)
        print("Number of docs = ", num_docs)

        print("\nLength user2id = ", len(user2id))
        print("Length doc2id = ", len(doc2id))

        # Creating the adjacency matrix (doc-user edges)
        adj_matrix = lil_matrix((num_docs + num_users, num_users + num_docs))
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

        print("\nPreparing entries for doc-user pairs...")
        src_dir = self.data_dest_path('engagements')
        not_found = 0
        for root, dirs, files in os.walk(src_dir):
            for count, file in enumerate(files):
                src_file_path = os.path.join(root, file)
                doc_key = file.split(".")[0]
                src_file = json.load(open(src_file_path, 'r'))
                users = src_file['users']
                for user in users:
                    if str(doc_key) in doc2id and str(user) in user2id and str(doc_key) not in test_docs:
                        try:
                            adj_matrix[doc2id[str(doc_key)], user2id[str(user)]] = 1
                            adj_matrix[user2id[str(user)], doc2id[str(doc_key)]] = 1
                            edge_type[doc2id[str(doc_key)], user2id[str(user)]] = 2
                            edge_type[user2id[str(user)], doc2id[str(doc_key)]] = 2
                        except IndexError:
                            print(f"Index Error! Skipping user with ID {user2id[str(user)]}.")
                            continue
                    else:
                        not_found += 1

        end = time.time()
        hrs, mins, secs = self.calc_elapsed_time(start, end)
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not Found users = {not_found}")
        print(f"Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Non-zero entries edge_type = {edge_type.getnnz()}")
        # print("Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))

        # Creating the adjacency matrix (user-user edges)
        start = time.time()
        key_errors, not_found, overlaps = 0, 0, 0
        print("\nPreparing entries for user-user pairs...")
        print(f"Printing every {int(num_users / 10)}  users done")

        # TODO: we need and expect doc IDs from 0 - x and user IDs from x to y without gabs

        for user_context in USER_CONTEXTS_FILTERED:
            print(f"\n    - from {user_context}  folder...")
            user_context_src_dir = self.data_src_path(user_context)
            for root, dirs, files in os.walk(user_context_src_dir):
                for count, file in enumerate(files):
                    src_file_path = os.path.join(root, file)
                    # user_id = src_file_path.split(".")[0]
                    src_file = json.load(open(src_file_path, 'r'))
                    user_id = int(src_file['user_id'])
                    if str(user_id) in user2id:
                        followers = src_file['followers'] if user_context == 'user_followers_filtered' else src_file[
                            'following']
                        followers = list(map(int, followers))
                        for follower in followers:
                            if str(follower) in user2id:
                                adj_matrix[user2id[str(user_id)], user2id[str(follower)]] = 1
                                adj_matrix[user2id[str(follower)], user2id[str(user_id)]] = 1
                                edge_type[user2id[str(user_id)], user2id[str(follower)]] = 3
                                edge_type[user2id[str(follower)], user2id[str(user_id)]] = 3

                    else:
                        not_found += 1
                    # if count%print_iter==0:
                    #     # print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, adj_matrix.getnnz()))
                    #     print("{}/{} done..  Non-zeros =  {}".format(count+1, num_users, len(np.nonzero(adj_matrix)[0])))

        hrs, mins, secs = self.calc_elapsed_time(start, time.time())
        print(f"Done. Took {hrs}hrs and {mins}mins and {secs}secs\n")
        print(f"Not found user_ids = {not_found}")
        print(f"Total Non-zero entries = {adj_matrix.getnnz()}")
        print(f"Total Non-zero entries edge_type = {edge_type.getnnz()}")
        # print("Total Non-zero entries = ", len(np.nonzero(adj_matrix)[0]))

        self.save_adj_matrix(adj_matrix, edge_type)

    def create_feature_matrix(self):
        pass

    def create_split_masks(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_raw_dir', type=str, default='../../data/raw/FakeHealth',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default='../../data/complete/FakeHealth',
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default='../../data/tsv/FakeHealth',
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='HealthStory', help='TODO')

    parser.add_argument('--exclude_frequent', type=bool, default=False, help='TODO')

    args, unparsed = parser.parse_known_args()

    preprocessor = GraphDataPreprocessor(args.__dict__)
