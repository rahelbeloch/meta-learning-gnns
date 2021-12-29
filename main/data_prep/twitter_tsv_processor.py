import argparse
import csv

from data_prep.config import *
from data_prep.data_preprocessor import DataPreprocessor
from data_prep.graph_io import FEATURE_TYPES

LABELS = {0: 'racism', 1: 'sexism', 2: 'none'}


class TSVPreprocessor(DataPreprocessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for twitter hate speech.
    """

    def __init__(self, dataset, f_type, max_vocab, data_dir, tsv_dir, comp_dir, content_file):
        super().__init__(dataset, f_type, max_vocab, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=comp_dir)

        self.content_file = content_file

    @property
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
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_train_nodes = int(COMPLETE_small_DIR.split('-')[1])

    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    num_train_nodes = None

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=complete_dir,
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=tsv_dir,
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data', type=str, default='twitterHateSpeech',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_k', type=int, default=30, help='Number (in K) of top users.')

    parser.add_argument('--user_doc_threshold', type=float, default=0.3, help='Threshold defining how many articles '
                                                                              'of any class users may max have shared '
                                                                              'to be included in the graph.')

    parser.add_argument('--valid_users', type=bool, default=True, help='Flag if only top K and users not sharing '
                                                                       'more than X% of any class should be used.')

    parser.add_argument('--feature_type', type=str, default='one-hot', help='The type of features to use.',
                        choices=FEATURE_TYPES)

    parser.add_argument('--max_vocab', type=int, default=10000, help='Size of the vocabulary used (if one-hot).')

    parser.add_argument('--train_size', type=float, default=0.0, help='Size of train split.')

    parser.add_argument('--val_size', type=float, default=0.25, help='Size of validation split.')

    parser.add_argument('--test_size', type=float, default=0.75, help='Size of train split.')

    args, unparsed = parser.parse_known_args()
    args = args.__dict__

    preprocessor = TSVPreprocessor(args['data'], args['feature_type'], args['max_vocab'], args['data_dir'],
                                   args['data_tsv_dir'], args['data_complete_dir'], 'twitter_data_waseem_hovy.csv')

    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits(train_size=args['train_size'],
                                    test_size=args['test_size'],
                                    val_size=args['val_size'],
                                    num_train_nodes=num_train_nodes,
                                    min_length=6)
