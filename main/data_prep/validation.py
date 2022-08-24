import itertools

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from data_prep.data_utils import get_data, get_loader, get_max_n_query
from data_prep.graph_dataset import TorchGeomGraphDataset
from samplers.episode_sampler import NonMetaFewShotEpisodeSampler, FewShotEpisodeSampler
from train_config import SHOTS

data_train = 'gossipcop'
data_eval = 'gossipcop'
model_name = 'maml'
h_size = 2
top_users, top_users_excluded = 30, 1
k_shot = 5
train_split_size, eval_split_size = (0.7, 0.1, 0.2), None
feature_type = 'one-hot'
vocab_size = 10000
dirs = "data", "../data/tsv", "../data/complete"
num_workers = 0


def sub_graphs_loader_validation():
    """
    Validates that the center index of each subgraph is stored iin the correct indices_per_class of the batch sampler.
    """
    loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                top_users_excluded, 5, train_split_size, eval_split_size,
                                                feature_type, vocab_size, dirs, num_workers)

    for loader in loaders:
        for data_list, targets in loader:
            for i, data in enumerate(data_list):
                assert data.orig_center_idx in loader.b_sampler.indices_per_class[data.set_type][targets[i].item()], \
                    'Center idx stored in wrong index per class!'


gat_train_batches = 1


def validate_query_set_equal():
    query_shot_nodes = dict()
    for k in SHOTS:
        loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                    top_users_excluded, k, train_split_size, eval_split_size,
                                                    feature_type, vocab_size, dirs, gat_train_batches, num_workers)

        train_loader, train_val_loader, test_loader, test_val_loader = loaders

        query_nodes, support_nodes = [], []
        for episode in iter(test_loader):
            # if model_name == 'gat' and mode != 'text':
            #     # sub graphs are organizes as: [s,s,s,s,s... q,q,q,q, ..., s,s,s,s, ...q,q,q,q ...]
            #     sub_graphs, targets = episode
            #     for i, g in enumerate(sub_graphs):
            #         if g.set_type == 'query':
            #             query_nodes.append((g.orig_center_idx, targets[i].item(), g.set_type))
            # else:
            support_sub_graphs, query_sub_graphs, support_targets, query_targets = episode
            for i, g in enumerate(query_sub_graphs):
                query_nodes.append((g.orig_center_idx, query_targets[i].item(), g.set_type))

        query_shot_nodes[k] = query_nodes
        print(f"\nCollected {len(query_nodes)} query nodes for shot '{k}'")

    difference_5_10 = set(query_shot_nodes[4]).symmetric_difference(set(query_shot_nodes[8]))
    difference_5_20 = set(query_shot_nodes[4]).symmetric_difference(set(query_shot_nodes[12]))
    difference_5_40 = set(query_shot_nodes[4]).symmetric_difference(set(query_shot_nodes[16]))

    difference_10_20 = set(query_shot_nodes[8]).symmetric_difference(set(query_shot_nodes[12]))
    difference_10_40 = set(query_shot_nodes[8]).symmetric_difference(set(query_shot_nodes[16]))

    difference_20_40 = set(query_shot_nodes[12]).symmetric_difference(set(query_shot_nodes[16]))

    assert len(difference_5_10) == 0
    assert len(difference_5_20) == 0
    assert len(difference_5_40) == 0
    assert len(difference_10_20) == 0
    assert len(difference_10_40) == 0
    assert len(difference_20_40) == 0

    print(f"Difference 4 and 8: {difference_5_10}")
    print(f"Difference 4 and 12: {difference_5_20}")
    print(f"Difference 4 and 16: {difference_5_40}")
    print(f"Difference 8 and 12: {difference_10_20}")
    print(f"Difference 8 and 16: {difference_10_40}")
    print(f"Difference 12 and 16: {difference_20_40}")


def check_train_loader_query_samples():
    data_config = {'top_users': top_users, 'top_users_excluded': top_users_excluded, 'feature_type': feature_type,
                   'vocab_size': vocab_size}
    train_config = {**data_config, **{'data_set': data_train, 'train_size': train_split_size[0],
                                      'val_size': train_split_size[1], 'test_size': train_split_size[2]}}
    graph_data_train = TorchGeomGraphDataset(train_config, train_split_size, *dirs)

    n_queries = get_max_n_query(graph_data_train)

    train_loader_5_shot = get_loader(graph_data_train, model_name, h_size, 5, num_workers, 'train', n_queries, 8112)
    concatenated_5 = get_query_indices(train_loader_5_shot)

    train_loader_10_shot = get_loader(graph_data_train, model_name, h_size, 10, num_workers, 'train', n_queries, 8349)
    concatenated_10 = get_query_indices(train_loader_10_shot)

    train_loader_20_shot = get_loader(graph_data_train, model_name, h_size, 20, num_workers, 'train', n_queries, 4221)
    concatenated_20 = get_query_indices(train_loader_20_shot)

    difference_5_10 = set(concatenated_5).symmetric_difference(set(concatenated_10))
    difference_5_20 = set(concatenated_5).symmetric_difference(set(concatenated_20))
    difference_10_20 = set(concatenated_10).symmetric_difference(set(concatenated_20))

    assert len(difference_5_10) == 0
    assert len(difference_5_20) == 0
    assert len(difference_10_20) == 0


def get_query_indices(loader):
    class_query_samples = loader.b_sampler.query_samples
    return torch.cat([class_query_samples[0].unsqueeze(dim=0), class_query_samples[1].unsqueeze(dim=0)],
                     dim=1).squeeze().tolist()


def unused_samples_stats():
    """
    Print how many samples stay unused for each loader we can create.
    """
    loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                top_users_excluded, 5, train_split_size, eval_split_size,
                                                feature_type, vocab_size, dirs, num_workers)

    for loader in loaders:
        n_indices = 0

        for c, s in itertools.product([0, 1], ['support', 'query']):
            n_indices += loader.b_sampler.indices_per_class[s][c].shape[0]

        n_query = sum([len(indices) for indices in loader.b_sampler.indices_per_class['query'].values()])
        n_support = sum([len(indices) for indices in loader.b_sampler.indices_per_class['support'].values()])

        # multiplied with 2 because of support and query
        n_used = loader.b_sampler.batch_size * loader.b_sampler.num_batches * 2

        print(f'{loader.mode} loader total samples: {n_indices}')
        print(f'{loader.mode} loader unused query samples: {int(n_query - n_used / 2)}')
        print(f'{loader.mode} loader unused support samples: {int(n_support - n_used / 2)}\n')


def visualize_subgraphs():
    loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                top_users_excluded, 5, train_split_size, eval_split_size,
                                                feature_type, vocab_size, dirs, num_workers=num_workers,
                                                batch_size=344)

    for loader in loaders:
        for episode in iter(loader):
            support_graphs, query_graphs, support_targets, query_targets = episode

            for i, graph in enumerate(support_graphs):
                print(f"Label: {support_targets[i]}")
                nx_graph = to_networkx(graph)
                nx.draw_networkx(nx_graph)
                plt.show()


def verify_node_indices(loader):
    """
    Verifies, that all node indices which a sampler has, are indeed from the correct data split.
    """
    expected_indices = torch.where(loader.mask == True)[0]
    start_idx, end_idx = expected_indices[0], expected_indices[-1]

    for batch in loader:
        if loader.mode == 'test' or (type(loader.b_sampler) == FewShotEpisodeSampler and model_name == 'gat'):
            support_graphs, query_graphs, _, _ = batch
            center_indices_in_split(start_idx, end_idx, support_graphs + query_graphs, loader.mode)
        elif type(loader.b_sampler) == NonMetaFewShotEpisodeSampler:
            center_indices_in_split(start_idx, end_idx, batch[0], loader.mode)
        else:
            for graphs, _ in batch:
                center_indices_in_split(start_idx, end_idx, graphs, loader.mode)


def center_indices_in_split(start_idx, end_idx, graphs, mode):
    center_indices = [graph.orig_center_idx for graph in graphs]
    all_in = all(start_idx <= x <= end_idx for x in center_indices)
    assert all_in, f"Not all node indices for this loader are from the correct split for loader {mode}!!"


def node_indices_belong_to_split():
    settings = [('gat', 5, 2704), ('gat', 10, 759), ('gat', 20, 1407), ('gat', 40, 2814),
                ('maml', 5, None), ('maml', 10, None), ('maml', 20, None), ('maml', 40, None),
                ('proto-maml', 5, None), ('proto-maml', 10, None), ('proto-maml', 20, None), ('proto-maml', 40, None), ]

    for s in settings:
        model, shots, batch_size = s
        loaders, train_graph, eval_graph = get_data(data_train, data_eval, model, h_size, top_users,
                                                    top_users_excluded, shots, train_split_size, eval_split_size,
                                                    feature_type, vocab_size, dirs, num_workers=num_workers,
                                                    batch_size=batch_size)
        for loader in loaders:
            verify_node_indices(loader)


if __name__ == '__main__':
    # check_train_loader_query_samples()

    validate_query_set_equal()

    # data_config = {'top_users': top_users, 'top_users_excluded': top_users_excluded, 'feature_type': feature_type,
    #                'vocab_size': vocab_size}
    # train_config = {**data_config, **{'data_set': data_train, 'train_size': train_split_size[0],
    #                                   'val_size': train_split_size[1], 'test_size': train_split_size[2]}}
    # graph_data_train = TorchGeomGraphDataset(train_config, train_split_size, *dirs)

    # sub_graphs_loader_validation()

    # unused_samples_stats()

    # get_max_n()

    # visualize_subgraphs()

    # node_indices_belong_to_split()
