from collections import defaultdict

from data_prep.graph_processing import GraphProcessor


class TwitterHateSpeechGraphProcessor(GraphProcessor):
    def __init__(self, args, **super_kwargs):
        super().__init__(args, **super_kwargs)

    def get_user2docs(self):
        users2docs = self.load_file("user2docs")

        return users2docs

    def get_user2users(self, user2docs):
        invalid_docs = self.load_file("invalid_docs")
        invalid_users = self.load_file("invalid_users")

        # ======================================================================
        # User2users
        # ======================================================================
        user2users = defaultdict(set)

        authors_file = self.data_raw_path(self.dataset, "authors.edgelist")
        with open(authors_file, "r") as f:
            author_entries = f.read().split("\n")

        for author_entry in author_entries:
            if len(author_entry) == 0:
                continue

            user1, user2 = author_entry.split()

            user1 = int(user1)
            user2 = int(user2)

            if user1 in invalid_users or user2 in invalid_users:
                continue

            user2users[user1].add(user2)
            user2users[user2].add(user1)

        user2users = dict(user2users)

        # ======================================================================
        # Degree
        # ======================================================================
        user_degree = defaultdict(int)

        for user_id, docs in user2docs.items():
            if user_id in invalid_users:
                continue

            user_degree[user_id] += len(docs - invalid_docs)

        for user_id, users in user2users.items():
            if user_id in invalid_users:
                continue

            user_degree[user_id] += len(users - invalid_users)

        user_degree = dict(user_degree)

        user_degree = sorted(user_degree.items(), key=lambda x: x[1], reverse=True)

        # ======================================================================
        # Truncation
        # ======================================================================
        num_users = len(user_degree)
        self.log(f"\nFound a total of {num_users} users with an edge")

        # Left truncation ======================================================
        self.log(f"\nRemoving top {self.top_users_excluded}%")
        num_truncated_users = int(self.top_users_excluded * len(user_degree) / 100)
        invalid_users.update(
            list(map(lambda x: x[0], user_degree[:num_truncated_users]))
        )
        user_degree = user_degree[num_truncated_users:]

        self.log(f"Truncated {num_truncated_users} users from left")
        self.summary["Num left truncated users"] = num_truncated_users

        # Right truncation =====================================================
        self.log(f"\nKeeping top {self.top_users}k")
        removed_users = user_degree[self.top_users * 1000 :]
        kept_users = user_degree[: self.top_users * 1000]
        invalid_users.update(list(map(lambda x: x[0], removed_users)))

        self.log(f"Truncated {len(removed_users)} users from right")
        self.summary["Num right truncated users"] = len(removed_users)

        self.summary["Invalid users (post truncation)"] = len(invalid_users)

        sorted_users, degrees = list(map(list, zip(*kept_users)))

        self.save_file("invalid_docs", invalid_docs)
        self.save_file("invalid_users", invalid_users)

        self.save_file("user2docs", user2docs)
        self.save_file("user2users", user2docs)

        return user2docs, user2users, sorted_users, degrees
