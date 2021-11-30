import csv

from data_prep.config import *
from data_prep.data_preprocessor import DataPreprocessor

LABELS = {0: 'racism', 1: 'sexism', 2: 'none'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for twitter hate speech.
    """

    def __init__(self, dataset, data_dir, tsv_dir, complete_dir, content_file):
        super().__init__(dataset, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

        self.content_file = content_file

    def labels(self):
        return LABELS

    def corpus_to_tsv(self):
        """
        This method reads all the individual JSON files of both the datasets and creates separate .tsv files for each.
        The .tsv file contains the fields: ID, article title, article content and the label.
        """

        self.print_step("Preparing Data Corpus")

        # TODO check if this doc has 0 interactions
        # self.maybe_load_non_interaction_docs()

        print("\nCreating doc2labels and collecting doc contents...")
        doc2labels = {}
        no_content = []
        contents = []

        data_file = self.data_raw_path(self.dataset, self.content_file)
        with open(data_file, encoding='utf-8') as content_data:
            reader = csv.DictReader(content_data)
            # skip the header
            next(reader, None)
            for row in reader:

                tweet_id = row['id']
                tweet_content = row['tweet']
                annotation = row['annotation']

                if len(tweet_content) == 0:
                    no_content.append(tweet_id)
                    continue

                # label = LABELS[int(annotation)]
                doc2labels[tweet_id] = int(annotation)
                contents.append([tweet_id, tweet_content, int(annotation)])

        print(f"Total docs without content : {str(len(no_content))}")

        self.store_doc2labels(doc2labels)
        self.store_doc_contents(contents)


if __name__ == '__main__':
    # tsv_dir = "tsv-50"
    # complete_dir = "complete-50"
    # max_doc_nodes = 500

    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    max_doc_nodes = None

    data = 'twitterHateSpeech'
    preprocessor = TSVPreprocessor(data, 'data', tsv_dir, complete_dir, 'twitter_data_waseem_hovy.csv')

    # already done in author.txt
    # preprocessor.aggregate_user_contexts()

    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits(max_data_points=max_doc_nodes, duplicate_stats=False)
