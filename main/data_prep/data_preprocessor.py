import csv
import json
import random
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
import torch

from data_prep.config import *
from data_prep.data_preprocess_utils import save_json_file, sanitize_text, print_label_distribution, split_data
from data_prep.graph_io import GraphIO


class DataPreprocessor(GraphIO):

    def __init__(self, dataset, feature_type, max_vocab, data_dir, tsv_dir, complete_dir):
        super().__init__(dataset, feature_type, max_vocab, data_dir=data_dir, tsv_dir=tsv_dir,
                         complete_dir=complete_dir)

    def maybe_load_non_interaction_docs(self):
        if self.non_interaction_docs is None:
            self.non_interaction_docs = self.load_if_exists(self.data_tsv_path('nonInteractionDocs.json'))

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
        self.non_interaction_docs = list(set(dict(filter(lambda f: f[1] > 0, dict(
            zip(docs_users, map(lambda x: len(x[1]), docs_users.items()))).items())).keys()))

        non_interaction_docs_file = self.data_tsv_path('nonInteractionDocs.json')
        print(f"\nNon-interaction documents: {len(self.non_interaction_docs)}")
        print(f"Saving in the dir: {non_interaction_docs_file}")
        save_json_file(self.non_interaction_docs, non_interaction_docs_file)

        docs_users = {k: v for k, v in docs_users.items() if k in self.non_interaction_docs}

        self.save_user_doc_engagements(docs_users)

    def save_user_doc_engagements(self, docs_users):

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

    def preprocess(self, num_train_nodes, min_len):
        """
        Applies some preprocessing to the data, e.g. replacing special characters, filters non-required articles out.
        :param min_len: Minimum required length for articles.
        :param num_train_nodes: Maximum amount of documents to use.
        :return: Numpy arrays for article tests (x_data), article labels (y_data), and article names (doc_names).
        """

        invalid_min_length = []
        data_dict = {}

        data_file = self.data_tsv_path(CONTENT_INFO_FILE_NAME)

        with open(data_file, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:
                text = sanitize_text(row['text'])
                doc_key = str(row['id'])
                tokens = set(nltk.word_tokenize(text))
                if len(tokens) >= min_len:
                    data_dict[doc_key] = (tokens, int(row['label']))
                else:
                    invalid_min_length.append(doc_key)

        # Get the vocab we would have from these texts
        token2idx, _, _ = self.get_vocab_token2idx({key: value[0] for (key, value) in data_dict.items()})

        # Filter out documents for which we do not have features any ways
        invalid_no_features = {}
        x_data, y_data, doc_names, x_lengths = [], [], [], []

        for doc_key, doc_data in data_dict.items():
            tokens = doc_data[0]

            # check if we would end up having features for this text
            indices = torch.tensor([token2idx[token] for token in tokens if token in token2idx])
            if len(indices[indices < self.max_vocab]) == 0:
                invalid_no_features[doc_key] = tokens
                continue

            x_data.append(tokens)
            y_data.append(doc_data[1])
            doc_names.append(doc_key)
            x_lengths.append(len(tokens))

        print(f"Average length = {sum(x_lengths) / len(x_lengths)}")
        print(f"Shortest Length = {min(x_lengths)}")
        print(f"Longest Length = {max(x_lengths)}")
        print(f"Total data points invalid and removed (length < {min_len}) = {len(set(invalid_min_length))}")
        print(f"Total data points invalid and removed (no features) = {len(set(invalid_no_features.keys()))}")

        if len(invalid_no_features) > 0:
            # save the invalid files with their words
            invalid_docs_file = self.data_complete_path('invalid-docs-no-features.txt')
            with open(invalid_docs_file, mode='w+', encoding='utf-8') as file:
                invalid = {doc_key: ' '.join(tokens) for doc_key, tokens in invalid_no_features.items()}
                invalid = [': '.join(entry) for entry in invalid.items()]
                file.write('\n'.join(invalid))

        x_data = np.array(x_data)
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

    def create_data_splits(self, num_train_nodes=None, min_length=None, test_size=0.2, val_size=0.1, splits=1,
                           duplicate_stats=False):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution. Currently only supports splitting data in 1 split for each set.

        :param num_train_nodes: Maximum amount of train documents to use.
        :param test_size: Size of the test split compared to the whole data.
        :param val_size: Size of the val split compared to the whole data.
        :param splits: Number of splits.
        :param duplicate_stats: If the statistics about duplicate texts should be collected or not.
        """

        self.print_step("Creating Data Splits")

        data = self.preprocess(num_train_nodes, min_length)

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

        split_path = self.data_tsv_path(f'splits-{self.feature_type}-{self.max_vocab}')

        for split, data in {'train': train_split, 'val': val_split, 'test': test_split}.items():
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

        doc_splits_file = self.data_tsv_path(DOC_SPLITS_FILE_NAME % self.feature_type)
        print("Writing doc_splits in : ", doc_splits_file)

        doc_names_train, doc_names_test, doc_names_val = train_split[2], test_split[2], val_split[2]
        print("\nTotal train = ", len(doc_names_train))
        print("Total test = ", len(doc_names_test))
        print("Total val = ", len(doc_names_val))

        split_dict = {'test_docs': doc_names_test, 'train_docs': doc_names_train, 'val_docs': doc_names_val}
        save_json_file(split_dict, doc_splits_file, converter=self.np_converter)

    def store_doc2labels(self, doc2labels):
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
            csv_writer.writerow(['id', 'text', 'label'])
            for file_content in contents:
                csv_writer.writerow(file_content)

        print("Final file written in :  ", content_dest_file)
