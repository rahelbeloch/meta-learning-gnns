import csv
import glob
import json
import os

from config import CONTENT_INFO_FILE_NAME
from graph_io import DataPreprocessor, load_json_file

DATASETS = ['HealthRelease', 'HealthStory']


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for HealthStory or HealthRelease.
    """

    def __init__(self, dataset):
        super().__init__(dataset)

    def corpus_to_tsv(self):
        """
        This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
        The .tsv file contains the fields: ID, article title, article content and the label.
        """

        self.print_step("Preparing Data Corpus")

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

        content_src_dir = os.path.join(self.data_raw_dir, 'content', self.dataset + "/*.json")
        content_dest_dir = os.path.join(self.data_tsv_dir, self.dataset, CONTENT_INFO_FILE_NAME)

        # TODO: should this be append?
        if os.path.isfile(content_dest_dir):
            print(f"\nTarget data file '{content_dest_dir}' already exists, overwriting it.")
            open_mode = 'w'
        else:
            open_mode = 'a'

        with open(content_dest_dir, open_mode, encoding='utf-8', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['id', 'title', 'text', 'label'])
            for file in glob.glob(content_src_dir):
                with open(file, 'r') as f:
                    content = json.load(f)
                    file_id = file.split('/')[-1]
                    file_id = file_id.split('.')[0]
                    csv_writer.writerow(
                        [file_id, content['title'], content['text'].replace('\n', " "), doc2labels[str(file_id)]])

        print("Final file written in :  ", content_dest_dir)


if __name__ == '__main__':

    for data in DATASETS:
        preprocessor = TSVPreprocessor(data)
        # preprocessor.corpus_to_tsv()
        preprocessor.create_data_splits_standard()
