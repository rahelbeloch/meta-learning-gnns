from collections import defaultdict

import ujson

from data_prep.graph_processing import GraphProcessor

USER_CONTEXTS = ["followers", "following"]


class CoaidGraphProcessor(GraphProcessor):
    def __init__(self, args, **super_kwargs):
        super().__init__(args, **super_kwargs)

    def get_user2docs(self):
        users2docs = self.load_file("user2docs")

        return users2docs

    def get_user2users(self, user2docs):
        invalid_docs = self.load_file("invalid_docs")
        invalid_users = self.load_file("invalid_users")

        # ======================================================================
        # Discover users
        # ======================================================================
        # Find which users are presenet (or not)
        self.log("Discovering users...")

        user_files = defaultdict(dict)

        for user_context in USER_CONTEXTS:
            user_context_src_dir = self.data_raw_path(
                self.dataset, "user_" + user_context
            )

            if not user_context_src_dir.exists():
                raise ValueError(
                    f"User context {'user_' + user_context} folder does not exist!"
                )

            if len(list(user_context_src_dir.glob("*"))) == 0:
                raise ValueError(
                    f"User context {'user_' + user_context} folder is empty!"
                )

            user_fps = set(user_context_src_dir.glob("*"))

            for i, file_path in enumerate(user_fps):
                user_id = int(file_path.stem)

                if user_id in invalid_users:
                    continue

                user_files[user_id][user_context] = file_path

        user_files = dict(user_files)

        # ======================================================================
        # Degree
        # ======================================================================
        # Compute the degrees to decide which users to keep
        self.log("\nComputing user degree...")

        user_degree = defaultdict(int)
        for user_id, incident_docs in user2docs.items():
            user_id = int(user_id)

            if user_id in invalid_users:
                continue

            user_degree[user_id] = len(incident_docs - invalid_docs)

        for i, (user_id, user_fps) in enumerate(user_files.items()):
            user_neighbours = set()

            for user_context, file_path in user_fps.items():
                with open(file_path, "r") as f:
                    context_neighbours = ujson.load(f)[user_context]
                user_neighbours.update(context_neighbours)

            user_neighbours = set(map(int, user_neighbours)) - invalid_users

            user_degree[user_id] += len(user_neighbours)

            if i == 0 or i % (len(user_files) // 10) == 0 or i == len(user_files) - 1:
                self.log(
                    f"{i+1}/{len(user_files)} [{round((i+1)/len(user_files)*100):d}%] Users: {len(user_degree)}"
                )

        user_degree = dict(user_degree)

        # Sort the degrees and truncate to topk
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
            list(map(lambda x: int(x[0]), user_degree[:num_truncated_users]))
        )
        user_degree = user_degree[num_truncated_users:]

        self.log(f"Truncated {num_truncated_users} users from left")
        self.summary["Num left truncated users"] = num_truncated_users

        # Right truncation =====================================================
        self.log(f"\nKeeping top {self.top_users}k")
        removed_users = user_degree[self.top_users * 1000 :]
        kept_users = user_degree[: self.top_users * 1000]
        invalid_users.update(list(map(lambda x: int(x[0]), removed_users)))

        self.log(f"Truncated {len(removed_users)} users from right")
        self.summary["Num right truncated users"] = len(removed_users)

        self.summary["Invalid users (post truncation)"] = len(invalid_users)

        sorted_users, degrees = list(map(list, zip(*kept_users)))

        # ======================================================================
        # User2users
        # ======================================================================
        self.log("\nBuilding edge list...")
        valid_users = set(sorted_users)

        user2users = defaultdict(set)

        isolated_user = 0
        for i, user_a_id in enumerate(sorted_users):
            user_a_neighbours = set()

            try:
                for user_context, file_path in user_files[user_a_id].items():
                    with open(file_path, "r") as f:
                        context_neighbours = ujson.load(f)[user_context]
                    user_a_neighbours.update(context_neighbours)
            except KeyError:
                self.log(
                    f"User id: {user_a_id}, "
                    + f"dType: {type(user_a_id)}, "
                    + f"In user files as str: {str(user_a_id) in user_files}, "
                    + f"In user files as int: {int(user_a_id) in user_files}"
                )
                raise KeyboardInterrupt

            user_a_neighbours = set.intersection(user_a_neighbours, valid_users)

            user_a_neighbours = set(map(int, user_a_neighbours))

            if len(user_a_neighbours) == 0:
                invalid_users.add(int(user_a_id))
                isolated_user += 1
            else:
                user2users[user_a_id].update(user_a_neighbours)
                for user_b_id in user_a_neighbours:
                    user2users[int(user_b_id)].add(user_a_id)

            if (
                i == 0
                or i % ((len(valid_users) - 1) // 10) == 0
                or i == len(valid_users) - 1
            ):
                self.log(
                    f"{i+1}/{len(valid_users)} [{round((i+1)/len(valid_users)*100):d}%] Users: {len(user2users)} [{len(user2users)/len(valid_users)*100:.2f}%]"
                )

        user2users = dict(user2users)

        self.log(f"Found {isolated_user} isolated users")
        self.summary["Isolated users"] = isolated_user

        self.save_file("invalid_docs", invalid_docs)
        self.save_file("invalid_users", invalid_users)

        self.save_file("user2docs", user2docs)
        self.save_file("user2users", user2docs)

        return user2docs, user2users, sorted_users, degrees
