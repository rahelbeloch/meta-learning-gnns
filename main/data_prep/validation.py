import torch

from data_prep.data_utils import get_data, get_loader
from data_prep.graph_dataset import TorchGeomGraphDataset

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


def validate_query_set_equal():
    expected_n_query_samples = 3553

    query_shot_nodes = dict()
    for k in [5, 10, 20, 40]:
        loaders, b_size, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                            top_users_excluded, k_shot, train_split_size,
                                                            eval_split_size, feature_type, vocab_size, dirs,
                                                            num_workers)

        train_loader, train_val_loader, test_loader, test_val_loader = loaders

        query_nodes, support_nodes = [], []
        for episode in iter(train_loader):
            if model_name == 'gat':
                sub_graphs, _ = episode
                half_len = int(len(sub_graphs) / 2)
                query_sub_graphs = sub_graphs[half_len:]
                support_sub_graphs = sub_graphs[:half_len]
            else:
                support_sub_graphs, query_sub_graphs, _, _ = episode
            support_nodes += [graph.orig_center_idx for graph in support_sub_graphs]
            query_nodes += [graph.orig_center_idx for graph in query_sub_graphs]
        query_shot_nodes[k] = query_nodes
        print(f"\nCollected query nodes for shot '{k}'")

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

    train_loader_5_shot = get_loader(graph_data_train, model_name, h_size, 5, num_workers, 'train')
    concatenated_5 = get_query_indices(train_loader_5_shot)

    train_loader_10_shot = get_loader(graph_data_train, model_name, h_size, 10, num_workers, 'train')
    concatenated_10 = get_query_indices(train_loader_10_shot)

    train_loader_20_shot = get_loader(graph_data_train, model_name, h_size, 20, num_workers, 'train')
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


if __name__ == '__main__':
    # check_train_loader_query_samples()

    validate_query_set_equal()
