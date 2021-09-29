import csv
import glob
import json
import os
import re

from sklearn.model_selection import StratifiedShuffleSplit

from config import CONTENT_INFO_FILE_NAME, RAW_DIR, TSV_DIR
from graph_io import GraphIO


# from data_utils import load_json_file
def load_json_file(file_name):
    return json.load(open(file_name, 'r'))


DATASETS = ['HealthRelease', 'HealthStory']


class TSVPreprocessor(GraphIO):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for HealthStory or HealthRelease ()
    """

    def __init__(self, dataset, data_raw_dir=RAW_DIR, data_tsv_dir=TSV_DIR):
        super().__init__(dataset=dataset, raw_dir=data_raw_dir, tsv_dir=data_tsv_dir)

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

        self.print_step_1("Preparing Data Corpus")

        print("\nCreating doc2labels for:  ", self.dataset)
        doc_labels_src_dir = os.path.join(self.data_raw_dir, 'reviews', f'{self.dataset}.json')
        doc2labels = {}
        count = 0

        for count, doc in enumerate(load_json_file(doc_labels_src_dir)):
            label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
            doc2labels[str(doc['news_id'])] = label

        data_tsv_dir = os.path.join(self.data_tsv_dir, self.dataset)
        if not os.path.exists(data_tsv_dir):
            os.makedirs(data_tsv_dir)

        print("Total docs : ", count)
        doc2labels_file = os.path.join(data_tsv_dir, 'doc2labels.json')
        print("\nWriting doc2labels file in :  ", doc2labels_file)
        json.dump(doc2labels, open(doc2labels_file, 'w+'))

        print("\nCreating the data corpus file for: ", self.dataset)

        # TODO: loading necessary?
        doc2labels = load_json_file(doc2labels_file)
        final_data_file = os.path.join(self.data_tsv_dir, dataset, CONTENT_INFO_FILE_NAME)

        content_src_dir = os.path.join(self.data_raw_dir, 'content', dataset + "/*.json")
        all_files = glob.glob(content_src_dir)

        # TODO: should this be append?
        if os.path.isfile(final_data_file):
            print(f"\nTarget data file '{final_data_file}' already exists, overwriting it.")
            open_mode = 'w'
        else:
            open_mode = 'a'

        with open(final_data_file, open_mode, encoding='utf-8', newline='') as csv_file:
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

    def create_data_splits_standard(self, dateset):
        """
        Creates train, val and test splits via random splitting of the dataset in a stratified fashion to ensure
        similar data distribution.
        """
        self.print_step_1("Creating Data Splits")

        print(f"\nPreparing {dataset} ...")
        x_data, y_data, doc_data = [], [], []

        # Reading the dataset into workable lists
        removed, lens = [], []
        data_file = os.path.join(self.data_tsv_dir, dataset, CONTENT_INFO_FILE_NAME)

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
                    removed.append(row['id'])

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

        # if len(y_train) == 0:
        #     raise ValueError('No training documents were added.')

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

            data_tsv_dir = os.path.join(self.data_tsv_dir, dataset, 'splits')
            if not os.path.exists(data_tsv_dir):
                os.makedirs(data_tsv_dir)

            data_tsv_dir = os.path.join(data_tsv_dir, f'{split}.tsv')
            print(f"{split} file in : {data_tsv_dir}")

            with open(data_tsv_dir, 'a', encoding='utf-8', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                # csv_writer.writerow(['text', 'label'])
                for i in range(len(x)):
                    csv_writer.writerow([x[i], y[i], id_list[i]])

        doc_splits_file = os.path.join(self.data_tsv_dir, dataset, 'docSplits.json')
        print("Writing doc_splits in : ", doc_splits_file)

        temp_dict = {'test_docs': doc_id_test, 'train_docs': doc_id_train, 'val_docs': doc_id_val}
        json.dump(temp_dict, open(doc_splits_file, 'w+'))


if __name__ == '__main__':

    for dataset in DATASETS:
        preprocessor = TSVPreprocessor(dataset)
        # preprocessor.corpus_to_tsv()
        preprocessor.create_data_splits_standard()
        # preprocessor.get_data_size()
