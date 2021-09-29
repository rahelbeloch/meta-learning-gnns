import csv
import os.path

from main.data_prep.fake_health_tsv_processor import TSVPreprocessor

RAW_DIR = '../data/raw/FakeHealth'
TSV_DIR = '../data/tsv/FakeHealth'


def corpus_to_tsv(dataset):
    """

    :return:
    """

    # Given
    content_file_name = f"{TSV_DIR}/{dataset}/docsContentInformation.tsv"
    content_files_path = f"{RAW_DIR}/content/{dataset}/"

    nr_articles = len(
        [f for f in os.listdir(content_files_path) if os.path.isfile(os.path.join(content_files_path, f))])

    preprocessor = TSVPreprocessor(dataset, RAW_DIR, TSV_DIR)

    # When
    preprocessor.corpus_to_tsv()

    # Then
    nr_csv_entries = len(list(csv.reader(open(content_file_name))))
    assert nr_csv_entries == nr_articles + 1, "Number of health story articles is not equal number of CSV entries!"


def test_corpus_to_tsv_health_story():
    """

    :return:
    """
    corpus_to_tsv('HealthStory')


def test_corpus_to_tsv_health_release():
    """

    :return:
    """
    corpus_to_tsv('HealthRelease')
