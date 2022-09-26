from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler

from samplers.episode_sampler import get_max_nr_for_shots


class FewShotEpisodeSampler(Sampler):

    def __init__(self, targets, n_support, mode, n_way=2, k_shot=5, verbose=False):
        super().__init__(None)

        self.verbose = verbose
        self.n_way = n_way
        self.k_shot = k_shot
        self.batch_size = self.n_way * self.k_shot  # Number of overall samples per query and support batch

        # number of samples which actually can be used with n classes and k shot
        self.data_targets = targets
        self.total_samples = self.data_targets.shape[0]

        # total nr of samples that should be used for support sets
        self.n_support = n_support

        # Organize examples by set type and class
        self.sets = ['query', 'support']
        self.classes = torch.unique(self.data_targets).tolist()
        self.num_classes = len(self.classes)

        # Number of K-shot batches that each class can provide
        self.batches_per_class, self.indices_per_class = dict(support={}, query={}), dict(support={}, query={})

        # some verification

        for c in self.classes:
            # noinspection PyTypeChecker
            class_indices = torch.where(self.data_targets == c)[0]
            n_class = len(class_indices)

            # calculate how many support samples to take from this class
            c_max_n_support = int(round(n_class / self.total_samples, 1) * self.n_support)
            n_support_class = get_max_nr_for_shots(c_max_n_support, self.n_way)

            # divide the samples we have for this class into support and query samples
            n_query_class = n_class - n_support_class
            self.indices_per_class['query'][c] = class_indices[:n_query_class]

            support_samples = class_indices[n_query_class:]
            assert support_samples.shape[0] == n_query_class
            self.indices_per_class['support'][c] = support_samples

            if self.verbose:
                print(
                    f"{mode} sampler nr of samples for shot '{self.k_shot}' and class '{c}': {n_support_class}"
                    f" (support), {n_query_class} (query).")

            for s in self.sets:
                # nr of examples we have per class // nr of shots -> amount of batches we can create from this class
                # noinspection PyUnresolvedReferences
                self.batches_per_class[s][c] = self.indices_per_class[s][c].shape[0] // self.k_shot

        # some validation
        nr_support_samples = sum([len(indices) for indices in self.indices_per_class['support'].values()])
        assert nr_support_samples <= self.n_support, \
            "The number of query examples we are using exceeds the max query nr!"

        # dividing all query samples into batches/episodes should be the number of batches we have for the query set
        assert (nr_support_samples // self.k_shot) == sum(self.batches_per_class['support'].values())

        # Create a list of classes from which we select the N classes per batch
        query_batches = sum(self.batches_per_class['query'].values()) // self.n_way
        support_batches = sum(self.batches_per_class['support'].values()) // self.n_way

        if self.verbose:
            print(f"\n{mode} sampler support batches: {support_batches}")
            print(f"\n{mode} sampler query batches: {query_batches}")

        self.num_batches = min(query_batches, support_batches)

        if self.verbose:
            print(f"{mode} sampler episodes/batches: {self.num_batches}")

        nr_support_samples = sum([len(indices) for indices in self.indices_per_class['support'].values()])
        nr_query_samples = sum([len(indices) for indices in self.indices_per_class['query'].values()])
        print(f"{mode} sampler support samples: {nr_support_samples}")
        print(f"{mode} sampler query samples: {nr_query_samples}")

        # verify that we used only up to n_query query examples
        assert support_batches * self.k_shot * self.n_way <= self.n_support

        self.batches_target_lists = {}
        for s in self.sets:
            self.batches_target_lists[s] = [c for c in self.classes for _ in range(self.batches_per_class[s][c])]

        for s in self.sets:
            # For testing, we iterate over classes instead of shuffling them
            sort_idxs = [i + p * self.num_classes for i, c in enumerate(self.classes) for p in
                         range(self.batches_per_class[s][c])]

            self.batches_target_lists[s] = np.array(self.batches_target_lists[s])[np.argsort(sort_idxs)].tolist()

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
