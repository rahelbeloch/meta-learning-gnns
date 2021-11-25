import glob
import json
import os.path

from config import RAW_DIR, TSV_small_DIR, COMPLETE_DIR, COMPLETE_small_DIR
from data_preprocessor import DataPreprocessor

LABELS = {0: 'fake', 1: 'real'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for gossipcop.
    """

    def __init__(self, dataset, raw_dir, tsv_dir, complete_dir):
        super().__init__(dataset, raw_dir=raw_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

    def labels(self):
        return LABELS

    def corpus_to_tsv(self):
        """
        This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
        The .tsv file contains the fields: ID, article title, article content and the label.
        """

        self.print_step("Preparing Data Corpus")

        print("\nCreating doc2labels and collecting doc contents...")
        doc2labels = {}
        contents = []
        no_content = 0

        labels = {'real': 0, 'fake': 1}
        for label in labels.keys():
            # load all files from this label folder
            content_files = os.path.join(self.data_raw_dir, self.dataset, label, '*')
            for folder_name in glob.glob(content_files):
                file_contents = folder_name + "/news content.json"
                if not os.path.exists(file_contents):
                    no_content += 1
                    continue

                doc_name = folder_name.split('/')[-1]

                if doc_name not in self.valid_docs:
                    continue

                doc2labels[doc_name] = labels[label]

                with open(file_contents, 'r') as f:
                    doc = json.load(f)
                    contents.append([doc_name, doc['title'], doc['text'], labels[label]])

        print(f"Total docs without content : {no_content}")

        self.store_doc2labels(doc2labels)
        self.store_doc_contents(contents)


if __name__ == '__main__':
    data = 'gossipcop'
    preprocessor = TSVPreprocessor(data, "../" + RAW_DIR, TSV_small_DIR, COMPLETE_small_DIR)
    preprocessor.aggregate_user_contexts()
    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits(max_data_points=50, duplicate_stats=False)
