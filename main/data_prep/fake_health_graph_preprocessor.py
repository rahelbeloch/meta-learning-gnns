import argparse
import glob
from collections import defaultdict

import nltk

nltk.download('stopwords')

from data_prep.graph_io import *
from data_preprocess_utils import load_json_file


class FakeHealthGraphPreprocessor(GraphPreprocessor):
    """
    Does all the preprocessing work to later be able to quickly load graph datasets from existing files.
    This includes creating and storing (in json files) the following components:
        - Engagement information (following/follower)
        - Document and user splits
        - Doc2id and id2doc dictionaries
        - Followers and following list of only the users who exist in the dataset
        - Adjacency matrix
        - Feature matrix
        - Labels
        - Split masks
    """

    @staticmethod
    def get_doc_key(name, name_type):
        return name.split('.')[0]

    def __init__(self, config):
        super().__init__(config)

        # self.aggregate_user_contexts()
        # self.create_user_splits()
        # self.create_doc_id_dicts()
        # self.filter_contexts('ids')
        # self.create_adj_matrix()
        # self.create_feature_matrix()
        # self.create_labels()
        self.create_split_masks()

    def aggregate_user_contexts(self):
        self.print_step("Aggregating follower/ing relations")

        src_dir = self.data_raw_path("engagements", self.dataset)
        if not os.path.exists(src_dir):
            raise ValueError(f'Source directory {src_dir} does not exist!')

        docs_users = defaultdict(set)
        count = 0
        for root, dirs, files in os.walk(src_dir):
            if root.endswith("replies"):
                continue
            for count, file in enumerate(files):
                if file.startswith('.'):
                    continue

                src_file = load_json_file(os.path.join(root, file))
                doc_name = root.split('/')[-2]
                docs_users[doc_name].update(src_file['user']['id'])
                if count % 10000 == 0:
                    print(f"{count} done")

        print(f"\nTotal tweets/re-tweets in the data set = {count}")
        self.save_user_doc_engagements(docs_users)

    def create_feature_matrix(self):
        src_doc_dir = self.data_raw_path('content', self.dataset + "/*.json")
        self.create_fea_matrix(glob.glob(src_doc_dir))

    def create_labels(self):
        """
        Create labels for each node of the graph
        """
        self.print_step("Creating labels")

        self.maybe_load_id_mappings()

        src_dir = self.data_raw_path('reviews', self.dataset + '.json')
        doc_labels = json.load(open(src_dir, 'r'))

        if self.n_total is None:
            adj_matrix = load_npz(self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k))
            self.n_total = adj_matrix.shape[0]
            del adj_matrix

        self.maybe_load_doc_splits()
        split_docs = self.train_docs + self.val_docs

        print("\nCreating doc2labels dictionary...")
        doc2labels = {}

        for count, doc in enumerate(doc_labels):
            if str(doc['news_id']) in split_docs:
                label = 1 if doc['rating'] < 3 else 0  # rating less than 3 is fake
                doc2labels[str(doc['news_id'])] = label

        # print(len(doc2labels.keys()))
        # print(len(doc2id.keys()) - len(doc_splits['test_docs']))
        assert len(doc2labels.keys()) == len(self.doc2id.keys()) - len(self.test_docs)
        print(f"\nLen of doc2labels = {len(doc2labels)}")

        self.save_labels(doc2labels)

    def save_labels(self, doc2labels):

        doc2labels_file = self.data_complete_path(DOC_2_LABELS_FILE_NAME % self.top_k)
        print(f"Saving doc2labels for {self.dataset} at: {doc2labels_file}")
        with open(doc2labels_file, 'w+') as v:
            json.dump(doc2labels, v)

        labels_list = np.zeros(self.n_total, dtype=int)
        for key, value in doc2labels.items():
            labels_list[self.doc2id[str(key)]] = value

        # Sanity Checks
        # print(sum(labels_list))
        # print(len(labels_list))
        # print(sum(labels_list[2402:]))
        # print(sum(labels_list[:2402]))

        labels_file = self.data_complete_path(LABELS_FILE_NAME % self.top_k)
        print(f"\nLabels list construction done! Saving in : {labels_file}")
        with open(labels_file, 'w+') as v:
            json.dump({'labels_list': list(labels_list)}, v, default=self.np_converter)

        # Create the all_labels file
        all_labels = np.zeros(self.n_total, dtype=int)
        all_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME % self.top_k)
        for doc in doc2labels.keys():
            all_labels[self.doc2id[str(doc)]] = doc2labels[str(doc)]

        print("\nSum of labels this test set = ", int(sum(all_labels)))
        print("Len of labels = ", len(all_labels))

        print(f"\nall_labels list construction done! Saving in : {all_labels_file}")
        with open(all_labels_file, 'w+') as j:
            json.dump({'all_labels': list(all_labels)}, j, default=self.np_converter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_raw_dir', type=str, default=RAW_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the raw data.')

    parser.add_argument('--data_complete_dir', type=str, default=COMPLETE_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the complete data.')

    parser.add_argument('--data_tsv_dir', type=str, default=TSV_DIR + 'FakeHealth',
                        help='Dataset folder path that contains the folders to the intermediate data.')

    parser.add_argument('--data_set', type=str, default='HealthStory',
                        help='The name of the dataset we want to process.')

    parser.add_argument('--top_k', type=int, default=50, help='Number of top users.')

    parser.add_argument('--exclude_frequent', type=bool, default=True, help='TODO')

    args, unparsed = parser.parse_known_args()

    preprocessor = FakeHealthGraphPreprocessor(args.__dict__)
