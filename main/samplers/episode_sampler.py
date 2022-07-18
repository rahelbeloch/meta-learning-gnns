from collections import defaultdict
from math import floor

import numpy as np
import torch
from torch.utils.data import Sampler

from train_config import SHOTS


def get_random_oversampled(majority_class_idx, minority_class_idx, indices):
    n_sample_minority = indices[majority_class_idx].shape[0] - indices[minority_class_idx].shape[0]
    oversampled = torch.multinomial(indices[minority_class_idx].float(), n_sample_minority, replacement=True)
    return torch.cat([indices[minority_class_idx], indices[minority_class_idx][oversampled]])


class FewShotEpisodeSampler(Sampler):

    def __init__(self, indices, targets, max_n_query, mode, n_way=2, k_shot=5, verbose=True, oversample=False):
        """
        Support sets should contain n_way * k_shot examples. So, e.g. 2 * 5 = 10 sub graphs.
        Query set is of same size ...

        Inputs:
            indices - Tensor containing all node indices which can be used by this sampler.
            targets - Tensor containing all targets of the elements in 'indices'.
            n_way - Number of classes to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
        """
        super().__init__(None)

        self.verbose = verbose
        self.n_way = n_way
        self.k_shot_support = k_shot

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

        majority_class, largest_n_support = None, 0

        if self.verbose:
            print(f"{mode} sampler:")

        for c in self.classes:
            # noinspection PyTypeChecker
            class_indices = indices[torch.where(self.data_targets == c)[0]]
            n_class = len(class_indices)

            # if mode != 'test':
            # make sure we take the same amount for query sets
            assert self.max_n_query is not None, f"Episode sampler for mode {mode}, but max_n_query is not defined!"

            # calculate how many query samples to take from this class
            c_max_n_query = int((n_class / self.total_samples) * self.max_n_query)
            # c_max_n_query = int(round(n_class / self.total_samples, 1) * self.max_n_query)
            n_query_class = get_max_nr_for_shots(c_max_n_query, self.n_way)
            # else:
            #     # meta test query does not follow any constraints --> Simply divide samples by 2 and upsample support
            #     assert self.max_n_query is None, f"Episode sampler for mode {mode}, but max_n_query is defined!"
            #
            #     n_query_class = int(n_class / 2)

            # divide the samples we have for this class into support and query samples
            n_support_class = n_class - n_query_class
            self.indices_per_class['support'][c] = class_indices[:n_support_class]

            if majority_class is None or n_support_class > largest_n_support:
                # keep track of which class of all is the majority class for upsampling later!
                majority_class, largest_n_support = c, n_support_class

            query_samples = class_indices[n_support_class:]
            assert query_samples.shape[0] == n_query_class
            self.indices_per_class['query'][c] = query_samples

            if self.verbose:
                print(f" samples for shot '{self.k_shot_support}' and class '{c}': {n_support_class} (support),"
                      f" {n_query_class} (query).")

        if oversample:
            for c in self.classes:
                if c == majority_class:
                    # no need to upsample the majority class
                    continue

                self.indices_per_class['support'][c] = get_random_oversampled(majority_class, c,
                                                                              self.indices_per_class['support'])
        # some verification that we now really have balanced support set samples!!
        n_support_samples = []
        for c in self.classes:
            n_support_samples.append(self.indices_per_class['support'][c].shape[0])

        assert len(set(n_support_samples)) == 1, "We should have the same amount of support samples across all classes!"

        # assert 1 not in self.data_targets[self.indices_per_class['support'][0]] \
        #        and 0 not in self.data_targets[self.indices_per_class['support'][1]] \
        #        and 1 not in self.data_targets[self.indices_per_class['query'][0]] \
        #        and 0 not in self.data_targets[self.indices_per_class['query'][1]], \
        #     "There are wrong data targets in the indices per class stored!"

        # we use the number of batches we can create from support and align the query batch sizes accordingly
        for c in self.classes:
            # nr of examples we have per class // nr of shots -> amount of batches we can create from this class
            # noinspection PyUnresolvedReferences
            self.batches_per_class['support'][c] = self.indices_per_class['support'][c].shape[0] // self.k_shot_support

        support_batches = self.batches_per_class['support'][0]
        self.num_batches = support_batches

        self.k_shot_query = {}
        for c in self.classes:
            self.k_shot_query[c] = self.indices_per_class['query'][c].shape[0] / support_batches

        if self.verbose:
            print(f" Query shots before rounding: {self.k_shot_query}")

        for c in self.classes:
            self.k_shot_query[c] = int(self.k_shot_query[c])

        if self.verbose:
            print(f" Query shots after rounding: {self.k_shot_query}")

        # Number of overall samples per query and support batch
        self.batch_size = self.k_shot_support * self.n_way + sum(self.k_shot_query.values())

        # some validation
        # if mode == 'test' and self.max_n_query is not None:
        nr_query_samples = self.count_samples('query')
        assert nr_query_samples <= self.max_n_query, "Query examples number we are using exceeds the max query nr!"

        # dividing all query samples into batches/episodes should be the number of batches we have for the query set
        # assert (nr_query_samples // self.k_shot_support) == sum(self.batches_per_class['query'].values())

        # dividing query samples according to k-shots for each class should yield at least as many support batches
        for c in self.classes:
            query_class_batches = int(self.indices_per_class['query'][c].shape[0] // self.k_shot_query[c])
            assert query_class_batches >= support_batches, "Number of query batches is smaller than support batches!"
            self.batches_per_class['query'][c] = query_class_batches

        # nr_support_samples = self.count_samples('support')

        if self.verbose:
            print(f" batches: {self.batches_per_class}")
            print(f" final episodes/batches used: {self.num_batches}")
            # print(f" support samples: {nr_support_samples}")
            # print(f" query samples: {nr_query_samples}")

        # verify that we used only up to n_query query examples
        # assert query_batches * self.k_shot_support * self.n_way <= self.max_n_query

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
        yield from self._iter(self.num_batches, self.k_shot_support, self.k_shot_query)

    def _iter(self, n_batches, offset_support, offset_query):
        # Sample few-shot batches
        start_index = defaultdict(lambda: defaultdict(int))
        for it in range(n_batches):
            index_batches = dict()
            for s in self.sets:
                # Select N classes for the batch
                class_batch = self.batches_target_lists[s][it * self.n_way:(it + 1) * self.n_way]
                set_index_batch = []

                for c in class_batch:
                    # For each class, select the next K examples and add them to the batch
                    offset = offset_support if s == 'support' else offset_query[c]
                    set_index_batch.extend(self.indices_per_class[s][c][start_index[s][c]:start_index[s][c] + offset])
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

    def __init__(self, indices, targets, max_n_query, mode, gat_train_batches, n_way=2, k_shot=5,
                 verbose=False,
                 oversample=False):
        super().__init__(indices, targets, max_n_query, mode, n_way, k_shot, verbose, oversample)

        # different shot sizes change the number of samples which can be used.
        # For comparability, GAT should be trained with 3 batches for each shot size...

        gossipcop_shot_batch_size_map = {
            5: {1: 8112, 2: 4056, 3: 2704, 13: 624},
            8: {1: 8086, 2: 4043, 13: 622},
            10: {1: 8349, 3: 2783, 11: 759},
            16: {1: 8215, 5: 1643},
            20: {2: 4221, 3: 2814, 6: 1407, 9: 938, 14: 603},
            40: {1: 8442, 2: 4221, 3: 2814, 7: 1206, 9: 938},
        }

        # twitter_shot_batch_size_map = {8: {1: 8034, 3: 2678}, 16: {1: 8316, 2: 4158, 7: 378}}

        shot_batch_size_map = gossipcop_shot_batch_size_map
        assert k_shot in shot_batch_size_map and gat_train_batches in shot_batch_size_map[k_shot], \
            f"K-Shot {k_shot} not in shot batch map or nr of gat train batches ({gat_train_batches}) not " \
            f"in shot batch map: {shot_batch_size_map[k_shot]}."

        # unbalanced dataset (gossipcop): shots 5
        # self.new_b_size = 8112   # --> 1 batch
        # self.new_b_size = 4056   # --> 2 batches
        # self.new_b_size = 2704   # --> 3 batches
        # self.new_b_size = 624  # --> 13 batches

        # unbalanced dataset (gossipcop): shots 10
        # self.new_b_size = 8349   # --> 1 batch
        # self.new_b_size = 2783   # --> 3 batches
        # self.new_b_size = 759  # --> 11 batches

        # unbalanced dataset (gossipcop): shots 20
        # self.new_b_size = 4221   # --> 2 batch
        # self.new_b_size = 2814   # --> 3 batches
        # self.new_b_size = 1407  # --> 6 batches
        # self.new_b_size = 938  # --> 9 batches
        # self.new_b_size = 603  # --> 14 batches

        # unbalanced dataset (gossipcop): shots 40
        # self.new_b_size = 8442   # --> 1 batch
        # self.new_b_size = 4221   # --> 2 batches
        # self.new_b_size = 2814   # --> 3 batches
        # self.new_b_size = 1206  # --> 7 batches
        # self.new_b_size = 938  # --> 9 batches

        # unbalanced dataset (twitterHateSpeech): shots 8
        # self.new_b_size = 8034   # --> 1 batch
        # self.new_b_size = 2678   # --> 3 batches

        # unbalanced dataset (twitterHateSpeech): shots 16
        # self.new_b_size = 8316   # --> 1 batch
        # self.new_b_size = 4158   # --> 2 batches
        # self.new_b_size = 378    # --> 7 batches

        # balanced dataset (gossipcop)
        # self.new_b_size = 497  # --> balanced dataset, 9 batches

        self.new_b_size = shot_batch_size_map[k_shot][gat_train_batches]

        used_samples = self.num_batches * self.batch_size
        n_new_batches = used_samples / self.new_b_size
        assert n_new_batches % 1 == 0, f"New batch size {self.new_b_size} can not create even number of batches out of {used_samples} used samples."

        self.n_new_batches = int(n_new_batches)

    def __iter__(self):
        """
        Aggregates multiple batches until
        :return:
        """

        batch_list = []
        for batch_idx, batch in enumerate(super().__iter__()):
            batch_list.extend(batch)

            # Aggregate as many batches until we reach the new batch size
            if len(batch_list) == self.new_b_size:
                yield batch_list
                batch_list = []

    def __len__(self):
        return self.n_new_batches

    @property
    def b_size(self):
        return self.new_b_size


class MetaFewShotEpisodeSampler(FewShotEpisodeSampler):

    def __init__(self, indices, targets, max_n_query, mode, n_way, k_shot, oversample=False):
        """
        Inputs:
            dataset_targets - PyTorch tensor of the labels of the data elements.
            batch_size - Number of tasks to aggregate in a batch
            n_way - Number of classes to sample per batch.
            k_shot - Number of examples to sample per class in the batch.
        """
        super().__init__(indices, targets, max_n_query, mode, n_way, k_shot, oversample=oversample)

        self.task_batch_size = self.num_batches if mode == 'train' else 2

        self.local_batch_size = self.batch_size  # already includes both classes for support and query

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


def split_list(a_list, num_support):
    return a_list[:num_support], a_list[num_support:]


def get_max_nr_for_shots(max_samples, n_class, max_shot=max(SHOTS)):
    """
    First determines the maximum amount of examples we should use based on the number of classes
    and total samples available. Subsequently, searches for the closest number which is divisible by the shot int
    and the number of classes.
    """

    # make sure it is still evenly divisible
    max_nr = get_max_n(max_samples, n_class, max_shot)

    # test to verify this number is divisible by shot int and number of classes
    # for shot in SHOTS:
    #     assert (max_nr / shot / n_class) % 1 == 0

    return max_nr


def get_max_n(nr, n_class, max_shot=max(SHOTS)):
    return (max_shot * n_class) * floor(nr / (max_shot * n_class))
