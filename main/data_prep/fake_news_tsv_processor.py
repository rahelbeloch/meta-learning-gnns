import json

from data_prep.config import *
from data_prep.data_preprocessor import DataPreprocessor

LABELS = {0: 'fake', 1: 'real'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for gossipcop.
    """

    def __init__(self, dataset, f_type, vocab_size, data_dir, tsv_dir, comp_dir):
        super().__init__(dataset, f_type, vocab_size, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=comp_dir)

    @property
    def labels(self):
        return LABELS

    def corpus_to_tsv(self):
        """
        This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
        The .tsv file contains the fields: ID, article title, article content and the label.
        """

        self.print_step("Preparing Data Corpus")

        self.maybe_load_non_interaction_docs()

        print("\nCreating doc2labels and collecting doc contents...")
        doc2labels = {}
        contents = []
        no_content = 0

        for label in LABELS.keys():
            # load all files from this label folder
            for folder_name in self.data_raw_path(self.dataset, LABELS[label]).rglob('*'):
                file_contents = folder_name / 'news content.json'
                if not file_contents.exists():
                    no_content += 1
                    continue

                doc_name = folder_name.stem
                if doc_name not in self.non_interaction_docs:
                    continue

                doc2labels[doc_name] = label

                with open(file_contents, 'r') as f:
                    doc = json.load(f)
                    contents.append([doc_name, doc['text'], label])

        print(f"Total docs without content : {no_content}")

        self.store_doc2labels(doc2labels)
        self.store_doc_contents(contents)


if __name__ == '__main__':
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_train_nodes = int(COMPLETE_small_DIR.split('-')[1])

    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    num_train_nodes = None

    feature_type = 'one-hot'
    max_vocab = 10000
    data = 'gossipcop'

    preprocessor = TSVPreprocessor(data, feature_type, max_vocab, 'data', tsv_dir, complete_dir)
    preprocessor.aggregate_user_contexts()
    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits(num_train_nodes=num_train_nodes, min_length=25, duplicate_stats=False)
