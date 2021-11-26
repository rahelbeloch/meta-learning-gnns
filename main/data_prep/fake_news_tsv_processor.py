import glob
import json
import os.path

from data_prep.data_preprocessor import DataPreprocessor

LABELS = {0: 'fake', 1: 'real'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for gossipcop.
    """

    def __init__(self, dataset, data_dir, tsv_dir, complete_dir):
        super().__init__(dataset, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

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

        labels = {'real': 0, 'fake': 1}
        for label in labels.keys():
            # load all files from this label folder
            for folder_name in self.data_raw_path(self.dataset, label).rglob('*'):
                file_contents = folder_name / 'news content.json'
                if not file_contents.exists():
                    no_content += 1
                    continue

                doc_name = folder_name.stem
                if doc_name not in self.non_interaction_docs:
                    continue

                doc2labels[doc_name] = labels[label]

                with open(file_contents, 'r') as f:
                    doc = json.load(f)
                    contents.append([doc_name, doc['title'], doc['text'], labels[label]])

        print(f"Total docs without content : {no_content}")

        self.store_doc2labels(doc2labels)
        self.store_doc_contents(contents)


if __name__ == '__main__':
    tsv_dir = "tsv-50"
    complete_dir = "complete-50"
    max_doc_nodes = 500

    # tsv_dir = TSV_DIR
    # complete_dir = COMPLETE_DIR
    # max_doc_nodes = None

    data = 'gossipcop'
    preprocessor = TSVPreprocessor(data, 'data', tsv_dir, complete_dir)
    preprocessor.aggregate_user_contexts()
    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits(max_data_points=max_doc_nodes, duplicate_stats=False)
