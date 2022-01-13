import copy
import csv
import json
import random
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

from data_prep.config import *
from data_prep.data_preprocess_utils import save_json_file, print_label_distribution, split_data, load_json_file
from data_prep.graph_io import GraphIO, NIV_IDX


class DataPreprocessor(GraphIO):

    def __init__(self, config, data_dir, tsv_dir, complete_dir):
        super().__init__(config, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

        self.user_doc_threshold = config['user_doc_threshold']
        self.top_k = config['top_k']

        self.valid_users = None

    def maybe_load_valid_users(self):
        if self.valid_users is None:
            self.valid_users = self.load_if_exists(self.data_complete_path(VALID_USERS % self.top_k))

    def maybe_load_non_interaction_docs(self):
        if self.non_interaction_docs is None:
            self.non_interaction_docs = self.load_if_exists(self.data_tsv_path('nonInteractionDocs.json'))

    def is_restricted(self, statistics):
        for label, percentage in statistics.items():
            if percentage >= self.user_doc_threshold:
                return True
        return False

    def aggregate_user_contexts(self):
        """
        Aggregates only user IDs from different folders of tweets/retweets to a single place. Creates a folder which
        has files named after document Ids. Each file contains all the users that interacted with it.
        """

        self.print_step("Aggregating follower/ing relations")

        docs_users = defaultdict(set)

        count = 0
        for user_context in ['tweets', 'retweets']:
            print("\nIterating over : ", user_context)

            src_dir = self.data_raw_path(self.dataset, user_context)
            if not src_dir.exists():
                raise ValueError(f'Source directory {src_dir} does not exist!')

            for count, file_path in enumerate(src_dir.glob('*')):

                # need to differentiate between how to read them because retweets are stored as JSONs in CSV!
                if user_context == 'tweets':
                    user_ids = pd.read_csv(file_path)['user_id']
                elif user_context == 'retweets':
                    user_ids = []
                    with open(file_path, encoding='utf-8', newline='') as csv_file:
                        lines = csv_file.readlines()
                        for line in lines:
                            json_str = json.loads(line)
                            user_ids.append(json_str['user']['id'])
                else:
                    raise ValueError(f'Unknown user context {user_context}!')

                user_ids = list(set([s for s in user_ids if isinstance(s, int)]))

                doc_id = file_path.stem

                # only include this document ID in doc_users if we actually use it in our splits
                # if not self.doc_used(doc_id):
                #     continue

                docs_users[doc_id].update(user_ids)
                if count == 0:
                    print(doc_id, docs_users[doc_id])
                if count % 2000 == 0:
                    print(f"{count} done")

        print(f"\nTotal tweets/re-tweets in the data set = {count}")

        # filter out documents which do not have any interaction with any user
        self.non_interaction_docs = list(set(dict(filter(lambda f: f[1] <= 0, dict(
            zip(docs_users, map(lambda x: len(x[1]), docs_users.items()))).items())).keys()))

        non_interaction_docs_file = self.data_tsv_path('nonInteractionDocs.json')
        print(f"\nNon-interaction documents: {len(self.non_interaction_docs)}")
        print(f"Saving in the dir: {non_interaction_docs_file}")
        save_json_file(self.non_interaction_docs, non_interaction_docs_file)

        docs_users = {k: v for k, v in docs_users.items() if k not in self.non_interaction_docs}

        self.save_user_doc_engagements(docs_users)

    def filter_users(self, user_stats, n_docs):
        """
        Counts how many document each user interacted with and filter users:
            - identify users that shared at least X% of the articles of any class
            - exclude top 3% sharing users (these are bots that have shared almost everything)
            - pick from the remaining the top K active users
        """

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

        # dict with user_ids and total shared/interacted docs
        users_shared_sorted = dict(sorted(user_stats.items(), key=lambda it: sum(it[1].values()), reverse=True))

        bot_users = []
        if self.dataset == 'gossipcop':
            bot_percentage = 0.01
            print(f"\nFiltering top sharing users (bots) : {bot_percentage}")

            # calculate total number of document shares per user
            total_user_shares = np.array([sum(values.values()) for _, values in users_shared_sorted.items()])

            # set 1% of the top sharing users as bot users

            num_bots = round(len(total_user_shares) * bot_percentage)

            # get users which are considered bots
            bot_users = np.array(list(users_shared_sorted.keys()))[:num_bots]

            # max_bin = total_user_shares.max()
            # mu = total_user_shares.mean()
            # sigma = np.var(total_user_shares)
            #
            # fig, ax = plt.subplots()
            #
            # # ax.set_ylim([0.0, 0.03])
            #
            # probs, bins, patches = ax.hist(total_user_shares, max_bin, density=True)
            #
            # # add a 'best fit' line
            # # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            # #      np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
            # # ax.plot(bins, y, '--')
            #
            # ax.set_xlabel('Nr. of Articles shared')
            # ax.set_ylabel('Probability density')
            #
            # # noinspection PyTypeChecker
            # ax.set_title(f"Dataset '{self.dataset}' user shares: $mean={round(mu, 2)}$, $var={round(sigma, 2)}$")
            #
            # # Tweak spacing to prevent clipping of y label
            # fig.tight_layout()
            # plt.show()

            print(f'Nr. of bot users : {len(bot_users)}')
            bot_users_file = self.data_complete_path(BOT_USERS % bot_percentage)
            save_json_file(bot_users, bot_users_file, converter=self.np_converter)
            print(f"Bot users stored in : {bot_users_file}")

        print(f"\nCollecting top K users as valid users : {self.top_k * 1000}")

        # remove users that we have already restricted before
        users_total_shared = OrderedDict()
        for key, value in users_shared_sorted.items():
            if key not in restricted_users and key not in bot_users:
                users_total_shared[key] = value

        # select the top k
        valid_users = list(users_total_shared.keys())[:self.top_k * 1000]

        valid_users_file = self.data_complete_path(VALID_USERS % self.top_k)
        save_json_file(valid_users, valid_users_file)
        print(f"Valid/top k users stored in : {valid_users_file}\n")

        self.valid_users = set(valid_users)

    def save_user_doc_engagements(self, docs_users):
        """
        For all documents that were shared by any user save a file which contains IDs of the users who shared it.
        :param docs_users: Dictionary where keys are document names and values are lists of user IDs.
        """

        dest_dir = self.data_tsv_path('engagements')
        if not dest_dir.exists():
            print(f"Creating destination dir:  {dest_dir}\n")
            dest_dir.mkdir()

        print(f"\nSaving engagement info for nr of docs: {len(docs_users)}")
        print(f"Writing in the dir: {dest_dir}")

        for doc, user_list in docs_users.items():
            document_user_file = dest_dir.joinpath(f'{str(doc)}.json')
            save_json_file({'users': list(user_list)}, document_user_file)

        print("\nDONE..!!")

    def preprocess(self, num_train_nodes):
        """
        Applies some preprocessing to the data, e.g. replacing special characters, filters non-required articles out.


        :param num_train_nodes: Maximum amount of documents to use.
        :return: Numpy arrays for article tests (x_data), article labels (y_data), and article names (doc_names).
        """

        data_file = self.data_tsv_path(CONTENT_INFO_FILE_NAME)
        data_dict = {}

        with open(data_file, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                doc_key = row['key']
                tokens = row['tokens'].split('_')
                data_dict[doc_key] = (tokens, int(row['label']))

        # Get the vocab we would have from these texts
        vocabulary, _ = self.get_vocab_token2idx({key: value[0] for (key, value) in data_dict.items()})

        # Filter out documents for which we do not have features any ways
        invalid_only_niv = {}
        x_data, y_data, doc_names, x_lengths = [], [], [], []

        for doc_key, doc_data in data_dict.items():
            tokens = doc_data[0]

            if self.feature_type == 'one-hot':
                indices = self.as_vocab_indices(vocabulary, tokens)
            elif 'glove' in self.feature_type:
                vocab_stoi = defaultdict(lambda: NIV_IDX[0], vocabulary.stoi)
                indices = [vocab_stoi[token] for token in tokens]
            else:
                raise ValueError(f"Trying to create features of type {self.feature_type} which is not unknown!")

            if all(v == NIV_IDX[0] for v in indices):
                # if only NIV tokens for this document, skip it
                invalid_only_niv[doc_key] = tokens
                continue

            x_data.append(tokens)
            y_data.append(doc_data[1])
            doc_names.append(doc_key)
            x_lengths.append(len(tokens))

        print(f"Average length = {sum(x_lengths) / len(x_lengths)}")
        print(f"Shortest Length = {min(x_lengths)}")
        print(f"Longest Length = {max(x_lengths)}")

        print(f"Total data points invalid and removed (only NIV tokens) = {len(set(invalid_only_niv.keys()))}")

        if len(invalid_only_niv) > 0:
            # save the invalid files with their words
            invalid_docs_file = self.data_complete_path('invalid-docs-niv.txt')
            with open(invalid_docs_file, mode='w+', encoding='utf-8') as file:
                invalid = {doc_key: ' '.join(tokens) for doc_key, tokens in invalid_only_niv.items()}
                invalid = [': '.join(entry) for entry in invalid.items()]
                file.write('\n'.join(invalid))

        x_data = np.array(x_data, dtype=object)
        y_data = np.array(y_data)
        doc_names = np.array(doc_names)

        if num_train_nodes is None:
            return x_data, y_data, doc_names

        # select only as many as we want
        per_class = int(num_train_nodes / len(self.labels))

        sampled_indices = []
        for c in self.labels.keys():
            sampled = random.sample(np.where(y_data == c)[0].tolist(), per_class)
            sampled_indices += sampled

        return x_data[sampled_indices], y_data[sampled_indices], doc_names[sampled_indices]

    def preprocess_documents(self, num_train_nodes=None, duplicate_stats=False):
        """
        :param num_train_nodes: Maximum amount of train documents to use.
        :param duplicate_stats: If the statistics about duplicate texts should be collected or not.
        """

        self.print_step("Filtering out invalid documents")

        data = self.preprocess(num_train_nodes)

        if duplicate_stats:
            # counting duplicates in test set
            texts = []
            duplicates = defaultdict(lambda: {'counts': 1, 'd_names': {'real': [], 'fake': []}, 'classes': set()})

            for i in range(len(data[0])):
                d_text = data[0][i]
                if d_text in texts:
                    duplicates[d_text]['counts'] += 1
                    duplicates[d_text]['d_names'][self.labels[data[1][i]]].append(data[2][i])
                else:
                    texts.append(d_text)

            duplicates_file = self.data_tsv_path('duplicates_info.json')
            save_json_file(duplicates, duplicates_file, converter=self.np_converter)

        return data

    def create_document_splits(self, data, train_size, test_size, val_size, splits=1):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution. Currently, only supports splitting data in 1 split for each set.

        :param test_size: Size of the test split compared to the whole data.
        :param val_size: Size of the val split compared to the whole data.
        :param splits: Number of splits.
        """

        self.print_step("Creating Data Splits")

        # Creating train-val-test split with same/similar label distribution in each split
        split_dict = {}

        if test_size > 0:
            # one tuple is one split and contains: (x, y, doc_names)
            rest_split, test_split = split_data(splits, test_size, data)
            assert len(set(test_split[2])) == len(test_split[2]), "Test split contains duplicate doc names!"
            print_label_distribution(test_split[1], 'test')
            split_dict['test'] = test_split
        else:
            rest_split = data

        if val_size > 0:
            # split rest data into validation and train splits
            train_split, val_split = split_data(splits, val_size, rest_split)
            assert len(set(val_split[2])) == len(val_split[2]), "Validation split contains duplicate doc names!"
            print_label_distribution(val_split[1], 'val')
            split_dict['val'] = val_split
        else:
            train_split = data

        assert len(set(train_split[2])) == len(train_split[2]), "Train split contains duplicate doc names!"
        print_label_distribution(train_split[1], 'train')
        split_dict['train'] = train_split

        print("\nWriting train-val-test files...\n")

        folder_name = DOC_SPLITS_FOLDER_NAME % (self.feature_type, self.max_vocab, train_size, val_size, test_size)
        split_path = self.data_tsv_path(folder_name)

        doc_names_split_dict = {}
        for split, data in split_dict.items():
            x, y, name_list = data

            if not split_path.exists():
                split_path.mkdir()

            split_file_path = split_path.joinpath(f'{split}.tsv')
            print(f"{split} file in : {split_file_path}")

            with open(split_file_path, 'w', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                csv_writer.writerow(['id', 'text', 'label'])
                for i in range(len(x)):
                    text = ' '.join(x[i])  # join for better reading it later
                    csv_writer.writerow([name_list[i], text, y[i]])

            print(f"Total {split} = ", len(name_list))

            doc_names_split_dict[f'{split}_docs'] = name_list

        file_name = DOC_SPLITS_FILE_NAME % (self.feature_type, self.max_vocab, train_size, val_size, test_size)
        doc_splits_file = self.data_tsv_path(file_name)
        print("\nWriting doc splits in : ", doc_splits_file)
        save_json_file(doc_names_split_dict, doc_splits_file, converter=self.np_converter)

    def create_user_splits(self, max_users):
        """
        Walks through all valid users and divides them on train/val/test splits.
        """

        self.print_step("Creating user splits")

        self.maybe_load_valid_users()
        self.load_doc_splits()

        doc2labels = load_json_file(self.data_complete_path(DOC_2_LABELS_FILE_NAME))

        print("\nCollecting users for splits file..")

        train_users, val_users, test_users = set(), set(), set()

        # walk through user-doc engagement files created before
        files = list(self.get_engagement_files())
        print_iter = int(len(files) / 20)

        for count, file_path in enumerate(files):
            if max_users is not None and (len(train_users) + len(test_users) + len(val_users)) >= max_users:
                break

            doc_key = file_path.stem
            if doc_key not in doc2labels:
                continue

            if count % print_iter == 0:
                print("{} / {} done..".format(count + 1, len(files)))

            if doc_key in self.train_docs:
                user_set = train_users
            if doc_key in self.val_docs:
                user_set = val_users
            if doc_key in self.test_docs:
                user_set = test_users

            users = load_json_file(file_path)['users']
            users_filtered = [u for u in users if self.valid_user(u)]
            # noinspection PyUnboundLocalVariable
            user_set.update(users_filtered)

        self.store_user_splits(train_users, test_users, val_users)

    def store_doc2label(self, doc2labels):
        """
        Stores the doc2label dictionary as JSON file in the respective directory.

        :param doc2labels: Dictionary containing document names mapped to the label for that document.
        """

        self.create_dir(self.data_complete_path())

        print(f"Total docs : {len(doc2labels)}")
        doc2labels_file = self.data_complete_path(DOC_2_LABELS_FILE_NAME)

        print(f"Writing doc2labels JSON in :  {doc2labels_file}")
        save_json_file(doc2labels, doc2labels_file, converter=self.np_converter)

    def store_doc_contents(self, contents):
        """
        Stores document contents (doc name/id, title, text and label) as a TSV file.

        :param contents: List of entries (doc name/id, title, text and label) for every article.
        """

        print("\nCreating the data corpus file for: ", self.dataset)

        content_dest_file = self.data_tsv_path(CONTENT_INFO_FILE_NAME)
        with open(content_dest_file, 'w+', encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['key', 'tokens', 'label'])
            for file_content in contents:
                csv_writer.writerow(file_content)

        print("Final file written in :  ", content_dest_file)

    def store_user_splits(self, train_users, test_users, val_users):
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
