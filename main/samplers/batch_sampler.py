import itertools
import random
from collections import defaultdict
from math import floor

import numpy as np
import torch
from torch.utils.data import Sampler

SHOTS = [5, 10, 20, 40]


class FewShotSampler(Sampler):

    def __init__(self, targets, max_n_query, mode, n_way=2, k_shot=5, shuffle=True, shuffle_once=False, verbose=False):
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

        self.verbose = verbose
        self.n_way = n_way
        self.k_shot = k_shot
        self.batch_size = self.n_way * self.k_shot  # Number of overall samples per query and support batch

        # number of samples which actually can be used with n classes and k shot
        self.data_targets = targets
        self.total_samples = self.data_targets.shape[0]

        # is the maximum number of query examples which should be used
        self.max_n_query = max_n_query

        self.shuffle = shuffle

        # Organize examples by set type and class
        self.sets = ['query', 'support']
        self.classes = torch.unique(self.data_targets).tolist()
        self.num_classes = len(self.classes)

        # Number of K-shot batches that each class can provide
        self.batches_per_class, self.indices_per_class = dict(support={}, query={}), dict(support={}, query={})

        for c in self.classes:
            # noinspection PyTypeChecker
            class_indices = torch.where(self.data_targets == c)[0]
            n_class = len(class_indices)

            # calculate how many query samples to take from this class
            c_max_n_query = int(round(n_class / self.total_samples, 1) * self.max_n_query)
            n_query_class = get_n_query_for_samples(c_max_n_query, self.n_way)

            # divide the samples we have for this class into support and query samples
            n_support_class = n_class - n_query_class
            self.indices_per_class['support'][c] = class_indices[:n_support_class]

            query_samples = class_indices[n_support_class:]
            assert query_samples.shape[0] == n_query_class
            self.indices_per_class['query'][c] = query_samples

            if self.verbose:
                print(
                    f"{mode} sampler nr of samples for shot '{self.k_shot}' and class '{c}': {n_support_class}"
                    f" (support), {n_query_class} (query).")

            for s in self.sets:
                # nr of examples we have per class // nr of shots -> amount of batches we can create from this class
                # noinspection PyUnresolvedReferences
                self.batches_per_class[s][c] = self.indices_per_class[s][c].shape[0] // self.k_shot

        # some validation
        nr_query_samples = sum([len(indices) for indices in self.indices_per_class['query'].values()])
        assert nr_query_samples <= self.max_n_query, \
            "The number of query examples we are using exceeds the max query nr!"

        # dividing all query samples into batches/episodes should be the number of batches we have for the query set
        assert (nr_query_samples // self.k_shot) == sum(self.batches_per_class['query'].values())

        # Create a list of classes from which we select the N classes per batch
        query_batches = sum(self.batches_per_class['query'].values()) // self.n_way
        support_batches = sum(self.batches_per_class['support'].values()) // self.n_way

        if self.verbose:
            print(f"\n{mode} sampler support batches: {support_batches}")
            print(f"\n{mode} sampler query batches: {query_batches}")

        self.num_batches = min(query_batches, support_batches)

        if self.verbose:
            print(f"{mode} sampler episodes/batches: {self.num_batches}")

        print(f"{mode} sampler query samples: {nr_query_samples}")

        # verify that we used only up to n_query query examples
        assert query_batches * self.k_shot * self.n_way <= self.max_n_query

        self.batches_target_lists = {}
        for s in self.sets:
            self.batches_target_lists[s] = [c for c in self.classes for _ in range(self.batches_per_class[s][c])]

        # if shuffle_once or self.shuffle:
        #     self.shuffle_data()
        # else:
        for s in self.sets:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in
                         range(self.batches_per_class[s][c])]

            self.batches_target_lists[s] = np.array(self.batches_target_lists[s])[np.argsort(sort_idxs)].tolist()

    # def shuffle_data(self):
    #     # Shuffle the examples per class
    #     for c, s in itertools.product(self.classes, self.sets):
    #         perm = torch.randperm(self.indices_per_class[s][c].shape[0])
    #         self.indices_per_class[s][c] = self.indices_per_class[s][c][perm]
    #
    #     # Shuffle the target list from which we sample. Note that this way of shuffling
    #     # does not prevent to choose the same class twice in a batch. However, for
    #     # training and validation, this is not a problem.
    #     for s in self.sets:
    #         random.shuffle(self.batches_target_lists[s])

    @property
    def query_samples(self):
        return self.indices_per_class['query']

    @property
    def b_size(self):
        return self.batch_size

    def __iter__(self):
        yield from self._iter(self.num_batches, self.k_shot)

    def _iter(self, n_batches, offset):
        # Shuffle data
        # if self.shuffle:
        #     self.shuffle_data()

        # Sample few-shot batches
        start_index = defaultdict(lambda: defaultdict(int))
        for it in range(n_batches):
            index_batches = dict()
            for s in self.sets:

                # Select N classes for the batch
                class_batch = self.batches_target_lists[s][it * self.n_way:(it + 1) * self.n_way]
                set_index_batch = []
                for c in class_batch:  # For each class, select the next K examples and add them to the batch
                    set_index_batch.extend(
                        self.indices_per_class[s][c][start_index[s][c]:start_index[s][c] + offset])
                    start_index[s][c] += offset
                index_batches[s] = set_index_batch

            full_batch = index_batches['support'] + index_batches['query']
            yield full_batch

    def __len__(self):
        return self.num_batches


class BatchSampler(FewShotSampler):
    """
    Sampler which uses batches created by FewShotSampler and concatenates them to bigger batches.
    Useful for non-meta baselines for more stable training with bigger batches.
    """

    def __init__(self, targets, max_n_query, mode, n_way=2, k_shot=5, shuffle=True, shuffle_once=False, verbose=False):
        super().__init__(targets, max_n_query, mode, n_way, k_shot, shuffle, shuffle_once, verbose)

        # TODO: must be divisible by self.k_shot * self.n_way
        # self.new_b_size = 688   # --> 5 batches
        self.new_b_size = 344   # --> 10 batches

        self.n_old_batches = super(BatchSampler, self).__len__()

        n_new_batches = 2 * self.n_old_batches * self.k_shot / self.new_b_size
        assert n_new_batches % 1 == 0, f"New batch size {self.new_b_size} can not create even number of batches."

        self.n_new_batches = int(n_new_batches)

    def __iter__(self):
        offset = int(self.new_b_size / 4)
        yield from super(BatchSampler, self)._iter(self.n_new_batches, offset)

    def __len__(self):
        return self.n_new_batches

    @property
    def b_size(self):
        return self.new_b_size


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def get_n_query_for_samples(max_samples, n_class, max_shot=max(SHOTS)):
    """
    First determines the maximum amount of query examples based on the number of classes and total samples available.
    Subsequently, define a number which is divisible by the shot int and the number of classes.
    """

    # make sure it is still evenly divisible
    max_n_query = get_max_n(max_samples, n_class, max_shot)

    # test to verify this number is divisible by shot int and number of classes
    for shot in SHOTS:
        assert (max_n_query / shot / n_class) % 1 == 0

    return max_n_query


def get_max_n(n_query, n_class, max_shot=max(SHOTS)):
    return (max_shot * n_class) * floor(n_query / (max_shot * n_class))
