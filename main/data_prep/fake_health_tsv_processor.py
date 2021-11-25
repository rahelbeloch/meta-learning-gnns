import glob
import json
import os

from config import RAW_DIR
from data_preprocess_utils import load_json_file
from data_preprocessor import DataPreprocessor

DATASETS = ['HealthRelease', 'HealthStory']


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for HealthStory or HealthRelease.
    """

    def __init__(self, dataset, raw_dir):
        super().__init__(dataset, raw_dir=raw_dir)

    def labels(self):
        return None

    def corpus_to_tsv(self):
        """
        This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
        The .tsv file contains the fields: ID, article title, article content and the label.
        """

        self.print_step("Preparing Data Corpus")

        print("\nCreating doc2labels...")
        doc_labels_src_dir = os.path.join(self.data_raw_dir, 'reviews', f'{self.dataset}.json')
        doc2labels = {}

        for count, doc in enumerate(load_json_file(doc_labels_src_dir)):
            label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
            doc2labels[str(doc['news_id'])] = label

        self.store_doc2labels(doc2labels)

        print("\nCollecting doc contents...")
        contents = []
        content_file_paths = os.path.join(self.data_raw_dir, 'content', self.dataset + "/*.json")
        for file in glob.glob(content_file_paths):
            with open(file, 'r') as f:
                doc_name = file.split('/')[-1].split('.')[0]
                content = json.load(f)
                contents.append([doc_name, content['title'], content['text'], doc2labels[str(doc_name)]])

        self.store_doc_contents(contents)


if __name__ == '__main__':
    data = 'FakeHealth'
    preprocessor = TSVPreprocessor(data, RAW_DIR)
    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits()
