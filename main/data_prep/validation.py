import itertools

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx

from data_prep.data_utils import get_data, get_loader, get_max_n_query
from data_prep.graph_dataset import TorchGeomGraphDataset
from samplers.batch_sampler import SHOTS

data_train = 'gossipcop'
data_eval = 'gossipcop'
model_name = 'gat'
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


def validate_query_set_equal():
    query_shot_nodes = dict()
    for k in SHOTS:
        loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                    top_users_excluded, k, train_split_size, eval_split_size,
                                                    feature_type, vocab_size, dirs, num_workers)

        train_loader, train_val_loader, test_loader, test_val_loader = loaders

        query_nodes, support_nodes = [], []
        for episode in iter(train_loader):
            if model_name == 'gat':
                sub_graphs, _ = episode
                half_len = int(len(sub_graphs) / 2)
                query_sub_graphs = sub_graphs[half_len:]
            else:
                support_sub_graphs, query_sub_graphs, _, _ = episode
            query_nodes += [(graph.orig_center_idx, graph.target, graph.set_type) for graph in query_sub_graphs]
        query_shot_nodes[k] = query_nodes
        print(f"\nCollected {len(query_nodes)} query nodes for shot '{k}'")

    difference_5_10 = set(query_shot_nodes[5]).symmetric_difference(set(query_shot_nodes[10]))
    difference_5_20 = set(query_shot_nodes[5]).symmetric_difference(set(query_shot_nodes[20]))
    difference_5_40 = set(query_shot_nodes[5]).symmetric_difference(set(query_shot_nodes[40]))

    difference_10_20 = set(query_shot_nodes[10]).symmetric_difference(set(query_shot_nodes[20]))
    difference_10_40 = set(query_shot_nodes[10]).symmetric_difference(set(query_shot_nodes[40]))

    difference_20_40 = set(query_shot_nodes[20]).symmetric_difference(set(query_shot_nodes[40]))

    assert len(difference_5_10) == 0
    assert len(difference_5_20) == 0
    assert len(difference_5_40) == 0
    assert len(difference_10_20) == 0
    assert len(difference_10_40) == 0
    assert len(difference_20_40) == 0


def check_train_loader_query_samples():
    data_config = {'top_users': top_users, 'top_users_excluded': top_users_excluded, 'feature_type': feature_type,
                   'vocab_size': vocab_size}
    train_config = {**data_config, **{'data_set': data_train, 'train_size': train_split_size[0],
                                      'val_size': train_split_size[1], 'test_size': train_split_size[2]}}
    graph_data_train = TorchGeomGraphDataset(train_config, train_split_size, *dirs)

    n_queries = get_max_n_query(graph_data_train)

    train_loader_5_shot = get_loader(graph_data_train, model_name, h_size, 5, num_workers, 'train', n_queries)
    concatenated_5 = get_query_indices(train_loader_5_shot)

    train_loader_10_shot = get_loader(graph_data_train, model_name, h_size, 10, num_workers, 'train', n_queries)
    concatenated_10 = get_query_indices(train_loader_10_shot)

    train_loader_20_shot = get_loader(graph_data_train, model_name, h_size, 20, num_workers, 'train', n_queries)
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
                                                batch_size=344, oversample_fake=False)

    for loader in loaders:
        for episode in iter(loader):
            support_graphs, query_graphs, support_targets, query_targets = episode

            for i, graph in enumerate(support_graphs):

                print(f"Label: {support_targets[i]}")
                nx_graph = to_networkx(graph)
                nx.draw_networkx(nx_graph)
                plt.show()


if __name__ == '__main__':
    # check_train_loader_query_samples()

    # validate_query_set_equal()

    # data_config = {'top_users': top_users, 'top_users_excluded': top_users_excluded, 'feature_type': feature_type,
    #                'vocab_size': vocab_size}
    # train_config = {**data_config, **{'data_set': data_train, 'train_size': train_split_size[0],
    #                                   'val_size': train_split_size[1], 'test_size': train_split_size[2]}}
    # graph_data_train = TorchGeomGraphDataset(train_config, train_split_size, *dirs)
    #
    # print(f"\n{get_max_n_query(graph_data_train)}")

    # sub_graphs_loader_validation()

    # unused_samples_stats()

    # get_max_n()

    visualize_subgraphs()
