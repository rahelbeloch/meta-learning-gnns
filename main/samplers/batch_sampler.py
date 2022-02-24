import itertools
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class FewShotSampler(Sampler):

    def __init__(self, targets, mask, n_query, n_way=2, k_shot=5, shuffle=True, shuffle_once=False):
        """
        Support sets should contain n_way * k_shot examples. So, e.g. 2 * 5 = 10 sub graphs.
        Query set is of same size ...

        Inputs:
            targets - Tensor containing all targets of the graph.
            n_way - Number of classes to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
            shuffle - If True, examples and classes are newly shuffled in each
                      iteration (for training).
            shuffle_once - If True, examples and classes are shuffled once in
                           the beginning, but kept constant across iterations
                           (for validation).
        """
        super().__init__(None)

        self.n_way = n_way
        self.k_shot = k_shot
        self.batch_size = self.n_way * self.k_shot  # Number of overall samples per query and support batch

        # number of samples which actually can be used with n classes and k shot
        n_samples_used = int(len(targets[mask]) / self.batch_size) * self.batch_size + 1
        # self.data_targets = targets[mask][:n_samples_used]
        self.data_targets = targets[mask]
        self.total_samples = self.data_targets.shape[0]

        self.n_query = n_query
        # self.n_support = self.total_samples - n_query

        self.shuffle = shuffle

        # Organize examples by set type and class
        self.sets = ['query', 'support']
        self.classes = torch.unique(self.data_targets).tolist()
        self.num_classes = len(self.classes)

        # Number of K-shot batches that each class can provide
        self.batches_per_class = dict(support={}, query={})
        self.indices_per_class = dict(support={}, query={})

        n_query_temp = 0

        for c in self.classes:
            # noinspection PyTypeChecker
            class_indices = torch.where(self.data_targets == c)[0]
            class_n_samples = len(class_indices)

            # calculate how many query and support samples from this class
            percentage_class_indices = round(class_n_samples / self.total_samples, 1)
            n_query_class = int(percentage_class_indices * self.n_query)
            n_query_temp += n_query_class
            # make sure the n_query_class keeps being evenly divided by k_shot
            n_query_class = self.k_shot * round(n_query_class / self.k_shot)

            n_support_class = class_n_samples - n_query_class
            assert n_support_class + n_query_class == class_n_samples

            self.indices_per_class['support'][c] = class_indices[:n_support_class]
            query_samples = class_indices[n_support_class:]
            assert query_samples.shape[0] == n_query_class
            self.indices_per_class['query'][c] = query_samples

            for s in self.sets:
                # nr of examples we have per class // nr of shots -> amount of batches we can create from this class
                # noinspection PyUnresolvedReferences
                self.batches_per_class[s][c] = self.indices_per_class[s][c].shape[0] // self.k_shot

        # verify that we have the exact amount of query examples which we defined/need
        nr_query_samples = sum([len(indices) for indices in self.indices_per_class['query'].values()])
        assert nr_query_samples == self.n_query
        assert (nr_query_samples // self.k_shot) == sum(self.batches_per_class['query'].values())

        # Create a list of classes from which we select the N classes per batch
        query_batches = sum(self.batches_per_class['query'].values()) // self.n_way
        support_batches = sum(self.batches_per_class['support'].values()) // self.n_way

        print(f"\nBatch sampler generated {support_batches} support batches and {query_batches} query batches.")

        self.num_batches = min(query_batches, support_batches)

        print(f"Batch sampler can only create {self.num_batches} episodes, leaving out "
              f"{abs(query_batches - support_batches) * self.k_shot * self.n_way} samples.")

        # verify that we really used all samples
        assert query_batches * self.k_shot * self.n_way == self.n_query

        self.target_lists = {}
        for s in self.sets:
            self.target_lists[s] = [c for c in self.classes for _ in range(self.batches_per_class[s][c])]

        if shuffle_once or self.shuffle:
            self.shuffle_data()
        else:
            for s in self.sets:
                # For testing, we iterate over classes instead of shuffling them
                sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in
                             range(self.batches_per_class[s][c])]

                self.target_lists[s] = np.array(self.target_lists[s])[np.argsort(sort_idxs)].tolist()

    def shuffle_data(self):
        # Shuffle the examples per class
        for c, s in itertools.product(self.classes, self.sets):
            perm = torch.randperm(self.indices_per_class[s][c].shape[0])
            self.indices_per_class[s][c] = self.indices_per_class[s][c][perm]

        # Shuffle the target list from which we sample. Note that this way of shuffling
        # does not prevent to choose the same class twice in a batch. However, for
        # training and validation, this is not a problem.
        for s in self.sets:
            random.shuffle(self.target_lists[s])

    @property
    def query_samples(self):
        return self.indices_per_class['query']

    def __iter__(self):
        # Shuffle data
        if self.shuffle:
            self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(lambda: defaultdict(int))
        for it in range(self.num_batches):
            index_batches = dict()
            for s in self.sets:

                # Select N classes for the batch
                class_batch = self.target_lists[s][it * self.n_way:(it + 1) * self.n_way]
                set_index_batch = []
                for c in class_batch:  # For each class, select the next K examples and add them to the batch
                    set_index_batch.extend(
                        self.indices_per_class[s][c][start_index[s][c]:start_index[s][c] + self.k_shot])
                    start_index[s][c] += self.k_shot
                index_batches[s] = set_index_batch

            full_batch = index_batches['support'] + index_batches['query']
            yield full_batch

    def __len__(self):
        return self.num_batches


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]
