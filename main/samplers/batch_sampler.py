import itertools
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler


class FewShotSampler(Sampler):

    def __init__(self, targets, mask, n_way=2, k_shot=5, shuffle=True, shuffle_once=False):
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
        self.data_targets = targets[mask]
        # self.mask = mask  # the mask for the idx to be used for this split
        self.n_way = n_way
        self.k_shot = k_shot
        self.shuffle = shuffle
        self.batch_size = self.n_way * self.k_shot  # Number of overall samples per query and support batch

        # Organize examples by class
        self.sets = ['query', 'support']
        self.classes = torch.unique(self.data_targets).tolist()
        self.num_classes = len(self.classes)

        # Number of K-shot batches that each class can provide
        self.batches_per_class = dict(support={}, query={})
        self.indices_per_class = dict(support={}, query={})

        for c in self.classes:
            class_indices = torch.where(self.data_targets == c)[0]
            len_indices = int(len(class_indices) / 2)

            self.indices_per_class['support'][c] = class_indices[:len_indices]
            self.indices_per_class['query'][c] = class_indices[len_indices:]

            for s in self.sets:
                # nr of examples we have per class // nr of shots -> amount of batches we can create from this class
                # noinspection PyUnresolvedReferences
                self.batches_per_class[s][c] = self.indices_per_class[s][c].shape[0] // self.k_shot

        # Create a list of classes from which we select the N classes per batch
        # self.num_batches = sum([x for xs in self.batches_per_class.values() for x in xs.values()]) // self.n_way
        self.num_batches = sum(self.batches_per_class['support'].values()) // self.n_way
        # self.num_batches = sum(self.batches_per_class.values()) // self.n_way  # total batches we can create

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
