import csv
import json
from collections import defaultdict

from data_prep.content_processing import ContentProcessor
from utils.io import load_json_file, save_json_file


class CoaidContentProcessor(ContentProcessor):
    def __init__(self, config, **super_kwargs):
        super().__init__(
            config,
            **super_kwargs,
        )

        self.keep_non_articles = config["keep_non_articles"]
        self.remove_unicode = config["remove_unicode"]
        self.append_title = config["append_title"]

        assert self.dataset == "CoAID"
        self.log(
            f"{'=' * 100}\n\t\t\tContent Processor for {self.dataset} \n{'=' * 100}"
        )

    def prep(self):
        def combine_news_records(news_fp):
            news_records = []
            with open(news_fp, "r") as f:
                article_tweets_reader = csv.reader(f)

                header_row = next(article_tweets_reader)
                header_row[0] = "id"

                for row in article_tweets_reader:
                    record = {k: v for k, v in zip(header_row, row)}

                    news_records.append(record)

            return news_records

        def combine_news_tweets(news_tweets_fp):
            news_tweets = defaultdict(set)
            with open(news_tweets_fp, "r") as f:
                article_tweets_reader = csv.reader(f)

                header_row = next(article_tweets_reader)

                for row in article_tweets_reader:
                    news_tweets[row[0]].add(row[1])

            news_tweets = dict(news_tweets)
            for k, v in news_tweets.items():
                news_tweets[k] = list(v)

            return news_tweets

        subdirs = sorted(
            [
                fp
                for fp in self.data_raw_path(self.dataset, "main").glob("*")
                if fp.is_dir()
            ]
        )

        real_news_records = []
        real_news_tweets = dict()
        fake_news_records = []
        fake_news_tweets = dict()

        for subdir in subdirs:
            real_news_fp = subdir / "NewsRealCOVID-19.csv"
            real_news_records.extend(combine_news_records(real_news_fp))

            real_news_tweets_fp = subdir / "NewsRealCOVID-19_tweets.csv"
            real_news_tweets.update(combine_news_tweets(real_news_tweets_fp))

            fake_news_fp = subdir / "NewsFakeCOVID-19.csv"
            fake_news_records.extend(combine_news_records(fake_news_fp))

            fake_news_tweets_fp = subdir / "NewsFakeCOVID-19_tweets.csv"
            fake_news_tweets.update(combine_news_tweets(fake_news_tweets_fp))

        save_json_file(
            real_news_records,
            self.data_raw_path(self.dataset, "main", "news_real_combined_new.json"),
        )
        save_json_file(
            real_news_tweets,
            self.data_raw_path(self.dataset, "main", "tweets_real_combined_new.json"),
        )
        save_json_file(
            fake_news_records,
            self.data_raw_path(self.dataset, "main", "news_fake_combined_new.json"),
        )
        save_json_file(
            fake_news_tweets,
            self.data_raw_path(self.dataset, "main", "tweets_fake_combined_new.json"),
        )

        super().prep()

    def load_content(self, invalid_docs):
        self.summary["File not found"] = 0
        self.summary["Empty file"] = 0
        self.summary["Non-articles"] = 0

        label2docs = defaultdict(list)
        doc2content = dict()
        doc2labels = dict()

        for label, label_class in self.labels.items():
            doc_file_path = self.data_raw_path(
                self.dataset, "main", f"news_{label_class}_combined_new.json"
            )

            all_contents = load_json_file(doc_file_path)

            for article in all_contents:
                article_id = article["id"]
                doc_name = f"{label_class}_{article_id}"
                article_type = article["type"]

                if article_type != "article":
                    self.summary["Non-articles"] += 1

                    if not self.keep_non_articles:
                        invalid_docs.add(doc_name)
                        continue

                if self.append_title:
                    content = article["title"].lower() + " " + article["content"]

                else:
                    content = article["content"]

                if self.remove_unicode:
                    content = content.encode("ascii", "ignore")
                    content = content.decode()

                if len(content) == 0:
                    invalid_docs.add(doc_name)
                    self.summary["Empty file"] += 1
                    continue

                doc2content[doc_name] = content
                doc2labels[doc_name] = label

                label2docs[label] += [article_id]

            docids_with_interactions = load_json_file(
                self.data_raw_path(
                    self.dataset, "main", f"tweets_{label_class}_combined_new.json"
                )
            ).keys()

            for article_id in list(docids_with_interactions):
                doc_name = f"{label_class}_{article_id}"

                if doc_name not in doc2content:
                    self.summary["File not found"] += 1
                    invalid_docs.add(doc_name)

        self.log(f"Found {self.summary['Non-articles']} non-articles.")

        doc2content = dict(doc2content)
        doc2labels = dict(doc2labels)

        return invalid_docs, doc2content, doc2labels

    def load_doc_interactions(self, invalid_users):
        invalid_docs = self.load_file("invalid_docs")

        article2tweets = defaultdict(set)

        user2tweets = defaultdict(set)
        tweet2users = defaultdict(set)
        tweet2label = dict()

        self.log(f"\nIterating over tweets and retweets...")
        for article_tweets_fp in self.data_raw_path(self.dataset, "tweets").glob("*"):
            article_id = int(article_tweets_fp.stem)

            with open(article_tweets_fp, "r") as f:
                article_tweets_reader = csv.reader(f)

                header_row = next(article_tweets_reader)

                for row in article_tweets_reader:
                    record = {k: v for k, v in zip(header_row, row)}

                    user_id = int(record["user_id"])

                    article2tweets[article_id].add(record["tweet_id"])

                    tweet2label[record["tweet_id"]] = int(record["fake"])

                    tweet2users[record["tweet_id"]].add(user_id)
                    user2tweets[user_id].add(record["tweet_id"])

            retweets_fp = self.data_raw_path(
                self.dataset, "retweets", f"{article_id}.csv"
            )

            if retweets_fp.exists():
                with open(retweets_fp, "r") as f:
                    retweets = f.readlines()

                for row in retweets:
                    record = json.loads(row)

                    user_id = int(record["user"]["id_str"])
                    orig_tweet_id = record["retweeted_status"]["id_str"]

                    user2tweets[user_id].add(orig_tweet_id)
                    tweet2users[orig_tweet_id].add(user_id)

        self.log(f"\nFolding tweets and retweets into doc-user interactions...")

        doc2users = {
            doc_id: set() for doc_id, _ in self.load_file("doc2labels").items()
        }
        user2docs = defaultdict(set)

        for article_id, tweet_ids in article2tweets.items():
            for tweet_id in tweet_ids:
                label = self.labels[tweet2label[tweet_id]]
                doc_id = f"{label}_{article_id}"

                if doc_id in invalid_docs:
                    continue

                interacting_users = tweet2users[tweet_id]

                doc2users[doc_id].update(interacting_users)

                for user_id in interacting_users:
                    user2docs[user_id].add(doc_id)

        doc2users = dict(doc2users)
        user2docs = dict(user2docs)

        return invalid_users, doc2users, user2docs
