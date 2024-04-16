import csv
from collections import defaultdict

from data_prep.content_processing import ContentProcessor


class TwitterHateSpeechContentProcessor(ContentProcessor):
    """
    Does all the preprocessing work from the original corpus to individual TSV files. Needs to be run before the
    GraphPreprocessor can do its work.

    This includes creating and storing the following components:
        - TSV files for twitter hate speech.
    """

    def __init__(self, config, content_file, **super_kwargs):
        super().__init__(
            config,
            **super_kwargs,
        )

        assert self.dataset == "twitterHateSpeech"
        self.log(
            f"{'=' * 100}\n\t\t\tContent Processor for {self.dataset} \n{'=' * 100}"
        )

        self.content_file = content_file

    def load_content(self, invalid_docs):
        self.summary["File not found"] = 0
        self.summary["Empty file"] = 0

        doc2labels = dict()
        doc2content = dict()

        data_file = self.data_raw_path(self.dataset, self.content_file)
        with open(data_file, encoding="utf-8") as content_data:
            reader = csv.DictReader(content_data)

            # skip the header
            next(reader, None)

            for row in reader:
                tweet_id = row["id"]
                tweet_content = row["tweet"]
                annotation = int(row["annotation"])

                if len(tweet_content) == 0:
                    invalid_docs.add(tweet_id)
                    self.summary["Empty file"] += 1
                    continue

                doc2content[tweet_id] = row["tweet"]
                doc2labels[tweet_id] = annotation

        return invalid_docs, doc2content, doc2labels

    def load_doc_interactions(self, invalid_users):
        doc2users = defaultdict(set)
        user2docs = defaultdict(set)

        authors_file = self.data_raw_path(self.dataset, "authors.txt")
        with open(authors_file, "r") as f:
            authorship = f.read().split("\n")
            for count, author_entry in enumerate(authorship):
                if len(author_entry) == 0:
                    continue

                doc_key, user_key = author_entry.split()
                user_key = int(user_key)

                doc2users[doc_key].add(user_key)
                user2docs[user_key].add(doc_key)

        doc2users = dict(doc2users)
        user2docs = dict(user2docs)

        return invalid_users, doc2users, user2docs
