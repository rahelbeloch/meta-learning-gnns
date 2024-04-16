import json
from collections import defaultdict

import pandas as pd

from data_prep.content_processing import ContentProcessor
from utils.io import load_json_file

USER_CONTEXTS = ["tweets", "retweets"]


class GossipcopContentProcessor(ContentProcessor):
    def __init__(self, config, **super_kwargs):
        super().__init__(
            config,
            **super_kwargs,
        )

        assert self.dataset == "gossipcop"
        self.log(
            f"{'=' * 100}\n\t\t\tContent Processor for {self.dataset} \n{'=' * 100}"
        )

    def load_content(self, invalid_docs):
        self.summary["File not found"] = 0
        self.summary["Empty file"] = 0

        doc2content = dict()
        doc2labels = dict()

        for label in self.labels:
            for folder_name in self.data_raw_path(
                self.dataset, self.labels[label]
            ).glob("*"):
                doc_name = folder_name.stem

                file_contents = folder_name / "news content.json"
                if not file_contents.exists():
                    invalid_docs.add(doc_name)
                    self.summary["File not found"] += 1
                    continue

                text = load_json_file(file_contents)["text"]

                if len(text) == 0:
                    invalid_docs.add(doc_name)
                    self.summary["Empty file"] += 1
                    continue

                doc2content[doc_name] = text
                doc2labels[doc_name] = label

        return invalid_docs, doc2content, doc2labels

    def load_doc_interactions(self, invalid_users):
        doc2users = defaultdict(set)
        user2docs = defaultdict(set)

        count = 0
        for user_context in USER_CONTEXTS:
            self.log(f"\nIterating over : {user_context}...")

            src_dir = self.data_raw_path(self.dataset, user_context)
            if not src_dir.exists():
                raise ValueError(f"Source directory {src_dir} does not exist!")

            file_paths = list(src_dir.glob("*"))

            for count, file_path in enumerate(file_paths):
                # need to differentiate between how to read them because retweets are stored as JSONs in CSV!
                if user_context == "tweets":
                    user_ids = pd.read_csv(file_path)["user_id"]

                elif user_context == "retweets":
                    user_ids = []

                    with open(file_path, encoding="utf-8", newline="") as csv_file:
                        lines = csv_file.readlines()
                        for line in lines:
                            json_str = json.loads(line)
                            user_ids.append(json_str["user"]["id"])

                else:
                    raise ValueError(f"Unknown user context {user_context}!")

                user_ids = {s for s in user_ids if isinstance(s, int)}

                doc_id = file_path.stem

                doc2users[doc_id].update(user_ids)
                for user_id in user_ids:
                    user2docs[user_id].add(doc_id)

                if (
                    count == 0
                    or count % (len(file_paths) // 10) == 0
                    or count == len(file_paths) - 1
                ):
                    self.log(
                        f"{count+1}/{len(file_paths)} [{round((count+1)/len(file_paths)*100):d}%]"
                    )

        doc2users = dict(doc2users)
        user2docs = dict(user2docs)

        return invalid_users, doc2users, user2docs
