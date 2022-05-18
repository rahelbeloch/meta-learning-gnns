from collections import defaultdict
from math import floor

import numpy as np
import torch
from torch.utils.data import Sampler

from train_config import SHOTS


def get_random_oversampled(majority_class_idx, minority_class_idx, indices):
    n_sample_minority = indices[majority_class_idx].shape[0] - indices[minority_class_idx].shape[0]
    oversampled_minority = torch.multinomial(indices[majority_class_idx].float(), n_sample_minority)
    return torch.cat([indices[minority_class_idx], oversampled_minority])


class FewShotEpisodeSampler(Sampler):

    def __init__(self, targets, max_n_query, mode, n_way=2, k_shot=5, verbose=True):
        """
        Support sets should contain n_way * k_shot examples. So, e.g. 2 * 5 = 10 sub graphs.
        Query set is of same size ...

        Inputs:
            targets - Tensor containing all targets of the graph.
            n_way - Number of classes to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
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

        # Organize examples by set type and class
        self.sets = ['query', 'support']
        self.classes = torch.unique(self.data_targets).tolist()
        self.num_classes = len(self.classes)

        # Number of K-shot batches that each class can provide
        self.batches_per_class, self.indices_per_class = dict(support={}, query={}), dict(support={}, query={})

        n_support = 0

        if self.verbose:
            print(f"{mode} sampler:")

        for c in self.classes:
            # noinspection PyTypeChecker
            class_indices = torch.where(self.data_targets == c)[0]
            n_class = len(class_indices)

            # calculate how many query samples to take from this class
            c_max_n_query = int(round(n_class / self.total_samples, 1) * self.max_n_query)
            n_query_class = get_max_nr_for_shots(c_max_n_query, self.n_way)

            # divide the samples we have for this class into support and query samples
            n_support_class = n_class - n_query_class
            self.indices_per_class['support'][c] = class_indices[:n_support_class]
            n_support += n_support_class

            query_samples = class_indices[n_support_class:]
            assert query_samples.shape[0] == n_query_class
            self.indices_per_class['query'][c] = query_samples

            if self.verbose:
                print(f" samples for shot '{self.k_shot}' and class '{c}': {n_support_class} (support),"
                      f" {n_query_class} (query).")

        # random oversampling for the support sets for both classes
        self.indices_per_class['support'][1] = get_random_oversampled(0, 1, self.indices_per_class['support'])

        for c in self.classes:
            for s in self.sets:
                # nr of examples we have per class // nr of shots -> amount of batches we can create from this class
                # noinspection PyUnresolvedReferences
                self.batches_per_class[s][c] = self.indices_per_class[s][c].shape[0] // self.k_shot

        # some validation
        nr_query_samples = self.count_samples('query')
        assert nr_query_samples <= self.max_n_query, "Number of query examples we are using exceeds the max query nr!"

        # dividing all query samples into batches/episodes should be the number of batches we have for the query set
        assert (nr_query_samples // self.k_shot) == sum(self.batches_per_class['query'].values())

        # Create a list of classes from which we select the N classes per batch
        query_batches = sum(self.batches_per_class['query'].values()) // self.n_way
        support_batches = sum(self.batches_per_class['support'].values()) // self.n_way

        self.num_batches = min(query_batches, support_batches)
        nr_support_samples = self.count_samples('support')

        if self.verbose:
            print(f" support batches: {support_batches}")
            print(f" query batches: {query_batches}")
            print(f" final episodes/batches used: {self.num_batches}")
            print(f" support samples: {nr_support_samples}")
            print(f" query samples: {nr_query_samples}")

        # verify that we used only up to n_query query examples
        assert query_batches * self.k_shot * self.n_way <= self.max_n_query

        self.batches_target_lists = {}
        for s in self.sets:
            self.batches_target_lists[s] = [c for c in self.classes for _ in range(self.batches_per_class[s][c])]

        for s in self.sets:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in
                         range(self.batches_per_class[s][c])]

            self.batches_target_lists[s] = np.array(self.batches_target_lists[s])[np.argsort(sort_idxs)].tolist()

    def count_samples(self, set_type):
        return sum([len(indices) for indices in self.indices_per_class[set_type].values()])

    @property
    def query_samples(self):
        return self.indices_per_class['query']

    @property
    def b_size(self):
        return self.batch_size

    def __iter__(self):
        yield from self._iter(self.num_batches, self.k_shot)

    def _iter(self, n_batches, offset):
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


class NonMetaFewShotEpisodeSampler(FewShotEpisodeSampler):
    """
    Sampler which uses batches created by FewShotSampler and concatenates them to bigger batches.
    Useful for non-meta baselines for more stable training with bigger batches.
    """

    def __init__(self, targets, max_n_query, mode, batch_size, n_way=2, k_shot=5, verbose=False):
        super().__init__(targets, max_n_query, mode, n_way, k_shot, verbose)

        # assert (batch_size / (self.k_shot * self.n_way)) % 1 == 0, "Batch size not divisible by n way and k shot."
        # must be divisible by self.k_shot * self.n_way

        # unbalanced dataset (gossipcop)
        # self.new_b_size = 688   # --> 5 batches
        # self.new_b_size = 344  # --> 10 batches

        # balanced dataset (gossipcop)
        # self.new_b_size = 497  # --> balanced dataset, 9 batches

        self.new_b_size = batch_size
        self.n_old_batches = super(NonMetaFewShotEpisodeSampler, self).__len__()

        n_new_batches = 2 * self.n_old_batches * self.k_shot / self.new_b_size
        # assert n_new_batches % 1 == 0, f"New batch size {self.new_b_size} can not create even number of batches."

        self.n_new_batches = int(n_new_batches)

    def __iter__(self):
        offset = int(self.new_b_size / 4)
        yield from super(NonMetaFewShotEpisodeSampler, self)._iter(self.n_new_batches, offset)

    def __len__(self):
        return self.n_new_batches

    @property
    def b_size(self):
        return self.new_b_size


class MetaFewShotEpisodeSampler(FewShotEpisodeSampler):

    def __init__(self, dataset_targets, max_n_query, mode, n_way, k_shot, batch_size=None):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            batch_size - Number of tasks to aggregate in a batch
            n_way - Number of classes to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
        """
        super().__init__(dataset_targets, max_n_query, mode, n_way, k_shot)

        # for training, we want to pass the whole data as 1 batch
        self.task_batch_size = batch_size if batch_size is not None else self.num_batches
        # self.task_batch_size = batch_size

        self.local_batch_size = self.batch_size * 2  # to account for support and query

    def __iter__(self):
        # Aggregate multiple batches before returning the indices
        batch_list = []
        for batch_idx, batch in enumerate(super().__iter__()):
            batch_list.extend(batch)
            if (batch_idx + 1) % self.task_batch_size == 0:
                yield batch_list
                batch_list = []

    def __len__(self):
        return self.num_batches // self.task_batch_size

    # def get_collate_fn(self, _):
    #     """
    #     This collate function converts list of all given samples (e.g. (local batch size * task batch size) x 3)
    #     into a list of batches (task batch size x 3 x local batch size). Makes it easier to process in the PL Maml.
    #     """
    #
    #     def collate_fn(batch_samples):
    #         _, graphs, targets = list(map(list, zip(*batch_samples)))
    #
    #         targets = torch.stack(targets)
    #
    #         assert targets.shape[0] == (self.local_batch_size * self.task_batch_size)
    #         assert len(graphs) == (self.local_batch_size * self.task_batch_size)
    #
    #         targets = targets.chunk(self.task_batch_size, dim=0)
    #
    #         # no torch chunking for list of graphs
    #         graphs = [graphs[i:i + self.local_batch_size] for i in range(0, len(graphs), self.local_batch_size)]
    #
    #         # should have size: 16 (self.task_batch_size)
    #         assert len(targets) == self.task_batch_size
    #         assert len(graphs) == self.task_batch_size
    #         assert len(graphs[0]) == self.local_batch_size  # should have size: 30 (self.local_batch_size)
    #         assert len(targets[0]) == self.local_batch_size  # should have size: 30 (self.local_batch_size)
    #
    #         return list(zip(graphs, targets))
    #
    #     return collate_fn


def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


def get_max_nr_for_shots(max_samples, n_class, max_shot=max(SHOTS)):
    """
    First determines the maximum amount of examples we should use based on the number of classes
    and total samples available. Subsequently, searches for the closest number which is divisible by the shot int
    and the number of classes.
    """

    # make sure it is still evenly divisible
    max_nr = get_max_n(max_samples, n_class, max_shot)

    # test to verify this number is divisible by shot int and number of classes
    for shot in SHOTS:
        assert (max_nr / shot / n_class) % 1 == 0

    return max_nr


def get_max_n(nr, n_class, max_shot=max(SHOTS)):
    return (max_shot * n_class) * floor(nr / (max_shot * n_class))
