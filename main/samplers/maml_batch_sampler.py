from samplers.batch_sampler import FewShotSampler


# class FewShotMamlSampler(Sampler):
#
#     def __init__(self, dataset_targets, mask, n_way, k_shot, batch_size=16, include_query=False, shuffle=True):
#         """
#         Inputs:
#             dataset_targets - PyTorch tensor of the labels of the data elements.
#             batch_size - Number of tasks to aggregate in a batch
#             n_way - Number of classes to sample per batch.
#             k_shot - Number of examples to sample per class in the batch.
#             include_query - If True, returns batch of size N_way*K_shot*2, which
#                             can be split into support and query set. Simplifies
#                             the implementation of sampling the same classes but
#                             distinct examples for support and query set.
#             shuffle - If True, examples and classes are newly shuffled in each
#                       iteration (for training)
#         """
#         super().__init__(None)
#         self.batch_sampler = FewShotSampler(dataset_targets, mask, n_way, k_shot, include_query, shuffle)
#         self.task_batch_size = batch_size
#         self.local_batch_size = self.batch_sampler.batch_size
#
#     def __iter__(self):
#         # Aggregate multiple batches before returning the indices
#         batch_list = []
#         for batch_idx, batch in enumerate(self.batch_sampler):
#             batch_list.extend(batch)
#             if (batch_idx + 1) % self.task_batch_size == 0:
#                 yield batch_list
#                 batch_list = []
#
#     def __len__(self):
#         return len(self.batch_sampler) // self.task_batch_size

class FewShotMamlSampler(FewShotSampler):

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
