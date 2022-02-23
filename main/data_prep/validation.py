from data_prep.data_utils import get_data

if __name__ == '__main__':

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
    num_workers = None

    shot_nodes = dict()
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
                query_sub_graphs = sub_graphs[:half_len]
                support_sub_graphs = sub_graphs[half_len:]
            else:
                support_sub_graphs, query_sub_graphs, _, _ = episode
            query_nodes += [graph.orig_center_idx.item() for graph in query_sub_graphs]
            support_nodes += [graph.orig_center_idx.item() for graph in support_sub_graphs]
        shot_nodes[k] = query_nodes
        print(f"Collected query nodes for shot '{k}'")

    # compute intersection
    intersection1 = list(set(shot_nodes[5]) & set(shot_nodes[10]))
    intersection2 = list(set(shot_nodes[5]) & set(shot_nodes[20]))
    intersection3 = list(set(shot_nodes[5]) & set(shot_nodes[40]))

    # 7100 sub graphs in total --> half for query set, half for support set
    expected_len_5 = len(train_loader) * 5 * 2
    expected_len_10 = len(train_loader) * 10 * 2
    expected_len_20 = len(train_loader) * 20 * 2
    expected_len_40 = len(train_loader) * 40 * 2