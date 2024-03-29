from collections import defaultdict

from data_prep.content_processing import ContentProcessor
from utils.io import load_json_file

USER_CONTEXTS = ["tweets", "retweets"]


class HealthStoryContentProcessor(ContentProcessor):
    def __init__(self, args, **super_kwargs):
        super().__init__(
            args,
            enforce_raw=False,
            **super_kwargs,
        )

        self.fake_margin = args["fake_margin"]

        assert self.dataset == "HealthStory"
        self.log(
            f"{'=' * 100}\n\t\t\tContent Processor for {self.dataset} \n{'=' * 100}"
        )

    def load_content(self, invalid_docs):
        label_to_id = {v: k for k, v in self.labels.items()}

        self.summary["File not found"] = 0
        self.summary["Empty file"] = 0

        reviews_fp = self.data_raw_path("FakeHealth", "reviews", self.dataset + ".json")

        doc_reviews = load_json_file(reviews_fp)

        doc2content = dict()
        doc2labels = dict()

        for review in doc_reviews:
            label = (
                label_to_id["fake"]
                if review["rating"] < self.fake_margin
                else label_to_id["real"]
            )

            doc_id = review["news_id"]

            content_fp = self.data_raw_path(
                "FakeHealth", "content", self.dataset, doc_id + ".json"
            )

            if not content_fp.exists():
                invalid_docs.add(doc_id)
                self.summary["File not found"] += 1
                continue

            text = load_json_file(content_fp)["text"]

            if len(text) == 0:
                invalid_docs.add(doc_id)
                self.summary["Empty file"] += 1
                continue

            doc2content[doc_id] = text
            doc2labels[doc_id] = label

            self.save_file("doc2labels", doc2labels)

        return invalid_docs, doc2content, doc2labels

    def load_doc_interactions(self, invalid_users):
        doc2labels = self.load_file("doc2labels")

        doc2users = defaultdict(set)
        user2docs = defaultdict(set)

        engagements_dir = self.data_raw_path("FakeHealth", "engagements", self.dataset)

        for user_context in USER_CONTEXTS:
            self.log(f"\nIterating over : {user_context}...")

            for i, doc_id in enumerate(doc2labels):
                src_dir = engagements_dir / doc_id / user_context

                if not src_dir.exists():
                    raise ValueError(f"Source directory {src_dir} does not exist!")

                file_paths = list(src_dir.glob("*"))

                for file_path in file_paths:
                    user_id = int(load_json_file(file_path)["user"]["id"])

                    if user_id in invalid_users:
                        continue

                    doc2users[doc_id].add(user_id)
                    user2docs[user_id].add(doc_id)

                if (
                    i == 0
                    or i % (len(doc2labels) // 10) == 0
                    or i == len(doc2labels) - 1
                ):
                    self.log(
                        f"{i+1}/{len(doc2labels)} [{round((i+1)/len(doc2labels)*100):d}%]"
                    )

        doc2users = dict(doc2users)
        user2docs = dict(user2docs)

        return invalid_users, doc2users, user2docs
