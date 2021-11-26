import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from data_prep.config import *
from data_prep.data_preprocess_utils import save_json_file, sanitize_text, print_label_distribution, split_data
from data_prep.graph_io import GraphIO


class DataPreprocessor(GraphIO):

    def maybe_load_non_interaction_docs(self):
        if self.non_interaction_docs is None:
            self.non_interaction_docs = self.load_if_exists('nonInteractionDocs.json')

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
            if not os.path.exists(src_dir):
                raise ValueError(f'Source directory {src_dir} does not exist!')

            for root, _, files in os.walk(src_dir):
                for count, file in enumerate(files):
                    file_path = os.path.join(root, file)

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

                    doc_id = file.split('.')[0]

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
        self.non_interaction_docs = list(set(dict(filter(lambda f: f[1] > 0,
                                                         dict(zip(docs_users, map(lambda x: len(x[1]),
                                                                                  docs_users.items()))).items())).keys()))

        non_interaction_docs_file = self.data_tsv_path('nonInteractionDocs.json')
        print(f"\nNon-interaction documents: {len(self.non_interaction_docs)}")
        print(f"Saving in the dir: {non_interaction_docs_file}")
        save_json_file(self.non_interaction_docs, non_interaction_docs_file)

        docs_users = {k: v for k, v in docs_users.items() if k in self.non_interaction_docs}

        self.save_user_doc_engagements(docs_users)

    def save_user_doc_engagements(self, docs_users):

        dest_dir = self.data_tsv_path('engagements')
        if not os.path.exists(dest_dir):
            print(f"Creating destination dir:  {dest_dir}\n")
            os.makedirs(dest_dir)

        print(f"\nSaving engagement info for nr of docs: {len(docs_users)}")
        print(f"Writing in the dir: {dest_dir}")

        for doc, user_list in docs_users.items():
            document_user_file = os.path.join(dest_dir, f'{str(doc)}.json')
            save_json_file({'users': list(user_list)}, document_user_file)

        print("\nDONE..!!")

    def preprocess(self, max_data_points, min_len=6):
        """
        Applies some preprocessing to the data, e.g. replacing special characters, filters non-required articles out.
        :param min_len: Minimum required length for articles.
        :return: Numpy arrays for article tests (x_data), article labels (y_data), and article names (doc_names).
        """

        x_data, y_data, doc_names, x_lengths, invalid = [], [], [], [], []

        data_file = os.path.join(self.data_tsv_dir, self.dataset, CONTENT_INFO_FILE_NAME)

        with open(data_file, encoding='utf-8') as data:
            reader = csv.DictReader(data, delimiter='\t')
            for row in reader:

                # if max_data_points is not None and len(x_data) > max_data_points:
                #     break

                if isinstance(row['text'], str) and len(row['text']) >= min_len:
                    text = sanitize_text(row['text'])
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

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        doc_names = np.array(doc_names)

        if max_data_points is None:
            return x_data, y_data, doc_names

        # select only as many as we want
        per_class = int(max_data_points / 2)

        sampled_indices = []
        for c in [0, 1]:
            sampled = random.sample(np.where(y_data == c)[0].tolist(), per_class)
            sampled_indices += sampled

        return x_data[sampled_indices], y_data[sampled_indices], doc_names[sampled_indices]

    def create_data_splits(self, max_data_points=None, test_size=0.2, val_size=0.1, splits=1, duplicate_stats=False):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution. Currently only supports splitting data in 1 split for each set.

        :param test_size: Size of the test split compared to the whole data.
        :param val_size: Size of the val split compared to the whole data.
        :param splits: Number of splits.
        :param duplicate_stats: If the statistics about duplicate texts should be collected or not.
        """

        self.print_step("Creating Data Splits")

        data = self.preprocess(max_data_points)

        if duplicate_stats:
            # counting duplicates in test set
            texts = []
            duplicates = defaultdict(lambda: {'counts': 1, 'd_names': {'real': [], 'fake': []}, 'classes': set()})

            for i in range(len(data[0])):
                d_text = data[0][i]
                if d_text in texts:
                    duplicates[d_text]['counts'] += 1
                    duplicates[d_text]['d_names'][self.labels()[data[1][i]]].append(data[2][i])
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

        splits = {'train': train_split, 'val': val_split, 'test': test_split}
        for split, data in splits.items():
            x, y, name_list = data

            split_path = self.data_tsv_path('splits')
            if not os.path.exists(split_path):
                os.makedirs(split_path)

            split_file_path = os.path.join(split_path, f'{split}.tsv')
            print(f"{split} file in : {split_file_path}")

            with open(split_file_path, 'w', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                csv_writer.writerow(['id', 'text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([name_list[i], x[i], y[i]])

        doc_splits_file = self.data_tsv_path(DOC_SPLITS_FILE_NAME)
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
        with open(content_dest_file, 'w', encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['id', 'title', 'text', 'label'])
            for file_content in contents:
                csv_writer.writerow(file_content)

        print("Final file written in :  ", content_dest_file)
