import glob
import os.path

from graph_io import *


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
        The .tsv file contains the fields: ID, article title, article content and the label
        """

        self.print_step("Preparing Data Corpus")

        print("\nCreating doc2labels for:  ", self.dataset)
        doc2labels = {}
        contents = []
        count, no_content = 0, 0

        labels = {'real': 0, 'fake': 1}
        for label in labels.keys():
            # load all files from this label folder
            content_files = os.path.join(self.data_raw_dir, self.dataset, label, '*')
            for folder_name in glob.glob(content_files):
                file_contents = folder_name + "/news content.json"
                if not os.path.exists(file_contents):
                    no_content += 1
                    continue

                doc_id = folder_name.split('/')[-1]
                doc2labels[doc_id] = labels[label]
                count += 1

                with open(file_contents, 'r') as f:
                    doc = json.load(f)
                    contents.append([doc_id, doc['title'], doc['text'], labels[label]])

        data_tsv_dir = os.path.join(self.data_tsv_dir, self.dataset)
        if not os.path.exists(data_tsv_dir):
            os.makedirs(data_tsv_dir)

        print("Total docs : ", count)
        print("Total docs without content : ", no_content)
        doc2labels_file = os.path.join(data_tsv_dir, 'doc2labels.json')
        print("Writing doc2labels file in :  ", doc2labels_file)
        json.dump(doc2labels, open(doc2labels_file, 'w+'))

        print("\nCreating the data corpus file for: ", self.dataset)

        content_dest_dir = os.path.join(self.data_tsv_dir, self.dataset, CONTENT_INFO_FILE_NAME)
        if os.path.isfile(content_dest_dir):
            print(f"\nTarget data file '{content_dest_dir}' already exists, overwriting it.")
            open_mode = 'w'
        else:
            open_mode = 'a'

        with open(content_dest_dir, open_mode, encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            csv_writer.writerow(['id', 'title', 'text', 'label'])
            for file_content in contents:
                csv_writer.writerow(file_content)

        print("Final file written in :  ", content_dest_dir)


if __name__ == '__main__':
    data = 'gossipcop'
    preprocessor = TSVPreprocessor(data)
    preprocessor.corpus_to_tsv()
    preprocessor.create_data_splits_standard()
