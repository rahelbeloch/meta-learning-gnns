import csv
import glob
import json
import os
import re

from sklearn.model_selection import StratifiedShuffleSplit

from data_utils import load_json_file, print_step

DATASETS = ['HealthRelease', 'HealthStory']

BASE_DIR = '../data/raw/FakeHealth'
DEST_DIR = '../data/tsv/FakeHealth'


class TSVPreprocessor:
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for HealthStory or HealthRelease ()
    """

    def __init__(self, base_dir, dest_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        self.base_dir = base_dir

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        self.dest_dir = dest_dir

        self.corpus_to_tsv()
        self.create_data_splits_standard()
        # self.get_data_size()

    @staticmethod
    def print_step(step_title):
        print(f'\n{"=" * 50}\n \t\t{step_title}\n{"=" * 50}')

    @staticmethod
    def get_label_distribution(labels):
        fake = labels.count(1)
        real = labels.count(0)
        denom = fake + real
        return fake / denom, real / denom

    def corpus_to_tsv(self):
        """
        This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
        The .tsv file contains the fields: ID, article title, article content and the label.
        """

        self.print_step("Preparing Data Corpus")

        for dataset in DATASETS:

            print("\nCreating doc2labels for:  ", dataset)
            doc_labels_src_dir = os.path.join(self.base_dir, 'reviews', f'{dataset}.json')
            doc2labels = {}
            count = 0

            for count, doc in enumerate(load_json_file(doc_labels_src_dir)):
                label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
                doc2labels[str(doc['news_id'])] = label

            dest_dir = os.path.join(self.dest_dir, dataset)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            print("Total docs : ", count)
            doc2labels_file = os.path.join(dest_dir, 'doc2labels.json')
            print("\nWriting doc2labels file in :  ", doc2labels_file)
            json.dump(doc2labels, open(doc2labels_file, 'w+'))

            print("\nCreating the data corpus file for: ", dataset)

            # TODO: loading necessary?
            doc2labels = load_json_file(doc2labels_file)
            content_src_dir = os.path.join(self.base_dir, 'content', dataset + "/*.json")
            final_data_file = os.path.join(self.dest_dir, dataset, 'docsMetaInformation.tsv')
            all_files = glob.glob(content_src_dir)

            with open(final_data_file, 'a', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                csv_writer.writerow(['id', 'title', 'text', 'label'])
                for file in all_files:
                    with open(file, 'r') as f:
                        content = json.load(f)
                        file_id = file.split('/')[-1]
                        file_id = file_id.split('.')[0]
                        csv_writer.writerow(
                            [file_id, content['title'], content['text'].replace('\n', " "), doc2labels[str(file_id)]])

            print("Final file written in :  ", final_data_file)

    def create_data_splits_standard(self):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution.
        """
        self.print_step("Creating Data Splits")

        for dataset in DATASETS:
            print(f"\nPreparing {dataset} ...")
            x_data, y_data, doc_data = [], [], []

            # Reading the dataset into workable lists
            removed, lens = 0, []
            data_file = os.path.join(self.dest_dir, dataset, 'docsMetaInformation.tsv')

            with open(data_file, encoding='utf-8') as data:
                reader = csv.DictReader(data, delimiter='\t')
                for row in reader:
                    if isinstance(row['text'], str) and len(row['text']) > 5:
                        text = row['text'].replace('\n', ' ')
                        text = text.replace('\t', ' ')
                        text = re.sub(r'#[\w-]+', 'hashtag', text)
                        text = re.sub(r'https?://\S+', 'url', text)

                        x_data.append(str(text[:5000]))
                        lens.append(len(text[:5000]))
                        y_data.append(int(row['label']))
                        doc_data.append(str(row['id']))
                    else:
                        removed += 1

            print("avg lens = ", sum(lens) / len(lens))
            print("max lens = ", max(lens))
            print("minimum lens = ", min(lens))
            print("Total data points removed = ", removed)

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
                    x_test.append(x_data[idx])
                    y_test.append(y_data[idx])
                    doc_id_test.append(doc_data[idx])

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=21)
            x_train, x_val, y_train, y_val = [], [], [], []
            doc_id_train, doc_id_val = [], []

            for fold, (train_index, val_index) in enumerate(sss.split(x_rest, y_rest)):
                # TODO: why is this in the inneer loop?
                x_train, x_val, y_train, y_val = [], [], [], []
                doc_id_train, doc_id_val = [], []

                for idx in train_index:
                    x_train.append(x_rest[idx])
                    y_train.append(y_rest[idx])
                    doc_id_train.append(doc_rest[idx])

                for idx in val_index:
                    x_val.append(x_rest[idx])
                    y_val.append(y_rest[idx])
                    doc_id_val.append(doc_rest[idx])

            if len(y_train) == 0:
                raise ValueError('No training documents were added.')

            fake, real = self.get_label_distribution(y_train)
            print("\nFake labels in train split  = {:.2f} %".format(fake * 100))
            print("Real labels in train split  = {:.2f} %".format(real * 100))

            fake, real = self.get_label_distribution(y_val)
            print("\nFake labels in val split  = {:.2f} %".format(fake * 100))
            print("Real labels in val split  = {:.2f} %".format(real * 100))

            fake, real = self.get_label_distribution(y_test)
            print("\nFake labels in test split = {:.2f} %".format(fake * 100))
            print("Real labels in test split  = {:.2f} %".format(real * 100))

            print("\nWriting train-val-test files..")
            splits = ['train', 'val', 'test']
            for split in splits:
                if split == 'train':
                    x, y, id_list = x_train, y_train, doc_id_train
                elif split == 'val':
                    x, y, id_list = x_val, y_val, doc_id_val
                else:
                    x, y, id_list = x_test, y_test, doc_id_test

                dest_dir = os.path.join(self.dest_dir, dataset, 'splits')
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                dest_dir = os.path.join(dest_dir, f'{split}.tsv')
                print(f"{split} file in : {dest_dir}")

                with open(dest_dir, 'a', encoding='utf-8', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter='\t')
                    # csv_writer.writerow(['text', 'label'])
                    for i in range(len(x)):
                        csv_writer.writerow([x[i], y[i], id_list[i]])

            doc_splits_file = os.path.join(self.dest_dir, dataset, 'docSplits.json')
            print("Writing doc_splits in : ", doc_splits_file)

            temp_dict = {'test_docs': doc_id_test, 'train_docs': doc_id_train, 'val_docs': doc_id_val}
            json.dump(temp_dict, open(doc_splits_file, 'w+'))


if __name__ == '__main__':
    TSVPreprocessor(BASE_DIR, DEST_DIR)
