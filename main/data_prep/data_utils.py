import torch.cuda
from torch.utils.data import DataLoader

from data_prep.graph_dataset import DGLSubGraphs, DglGraphDataset, as_dataloader, collate_fn_proto
from models.batch_sampler import FewShotSubgraphSampler

SUPPORTED_DATASETS = ['HealthStory', 'gossipcop']


def get_data(data_name, model, data_dir, batch_size, hop_size, top_k, k_shot):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_name (str): Name of the data corpus which should be used.
        model (str): Name of the model should be used.
        data_dir (str): Path to the data (full & complete) to be used to create the graph (feature file, edge file etc.)
        batch_size (int): Size of one batch.
        hop_size (int): Number of hops used to create sub graphs.
        top_k (int): Number of top users to be used in graph.
        k_shot (int): Number of examples used per task/batch.
    Raises:
        Exception: if the data_name is not in SUPPORTED_DATASETS.
    """

    if data_name not in SUPPORTED_DATASETS:
        raise ValueError("Data with name '%s' is not supported." % data_name)

    # graph_data = TorchGeomGraphDataset(data_name)
    graph_data = DglGraphDataset(data_name, top_k, data_dir)

    train_graphs = DGLSubGraphs(graph_data, 'train_mask', b_size=batch_size, h_size=hop_size, meta=model != 'gat')
    val_graphs = DGLSubGraphs(graph_data, 'val_mask', b_size=batch_size, h_size=hop_size, meta=model != 'gat')
    test_graphs = DGLSubGraphs(graph_data, 'test_mask', b_size=batch_size, h_size=hop_size, meta=model != 'gat')

    num_workers = 6 if torch.cuda.is_available() else 1     # mac has 8 CPUs

    if model == 'gat':
        # load the whole graph once (it internally has the train/val/test masks)

        # loader, vocab_size, data = graph_data.initialize_graph_data()
        # train_sub_graphs = TorchGeomSubGraphs(graph_data, 'train', b_size=128, h_size=2)
        # val_sub_graphs = TorchGeomSubGraphs(graph_data, 'val', b_size=128, h_size=2)
        # test_sub_graphs = TorchGeomSubGraphs(graph_data, 'test', b_size=128, h_size=2)

        train_loader = as_dataloader(train_graphs, num_workers)
        val_loader = as_dataloader(val_graphs, num_workers)
        test_loader = as_dataloader(test_graphs, num_workers)

    elif model == 'prototypical' or model == 'gmeta':

        train_sampler = FewShotSubgraphSampler(train_graphs, include_query=True, k_shot=int(k_shot / 2))
        train_loader = DataLoader(train_graphs, batch_sampler=train_sampler, num_workers=num_workers,
                                  collate_fn=collate_fn_proto)

        val_sampler = FewShotSubgraphSampler(val_graphs, include_query=True, k_shot=int(k_shot / 2))
        val_loader = DataLoader(val_graphs, batch_sampler=val_sampler, num_workers=num_workers,
                                collate_fn=collate_fn_proto)

        test_sampler = FewShotSubgraphSampler(test_graphs, include_query=True, k_shot=int(k_shot / 2))
        test_loader = DataLoader(test_graphs, batch_sampler=test_sampler, num_workers=num_workers,
                                 collate_fn=collate_fn_proto)
    else:
        raise ValueError("Don't know model name '%s'." % model)

    return train_loader, val_loader, test_loader, graph_data.num_features

    # if data_name == 'HealthRelease':
    #     # return HealthReleaseData(val_size=val_size)
    #     return GraphDataset(data_name)
    # elif data_name == 'HealthStory':
    #     return HealthStoryData(val_size=val_size)
    # else:
    #     raise ValueError("Data with name '%s' is not supported." % data_name)

# def sample_sub_graphs():
#     """
#     G-Meta samples sub graphs based on a train/test/val distribution of labels, where every label appears exactly once in
#     one of the splits. The assignment is done randomized. We can not do this (with HealthStory), because we have
#     only labels 0 and 1. Therefore, we need to rely on the randomized assignment of nodes to the different splits.
#     We are already doing this --> we can immediately generate subgraphs on the flight for each node.
#     """
#     # number of unique labels, e.g. 2
#     num_of_labels = 30
#
#     # number of labels for each label set, ideally << num_of_labels so that each task can
#     # consist out of different permutation of labels
#     num_label_set = 5
#
#     labels_file = "../../data/complete/FakeHealth/HealthStory/" + ALL_LABELS_FILE_NAME
#     labels = json.load(open(labels_file, 'r'))
#
#     all_labels = torch.LongTensor(labels['all_labels'])
#     labels = np.unique(list(range(num_of_labels)))
#
#     test_labels = np.random.choice(labels, num_label_set, False)
#     labels_left = [i for i in labels if i not in test_labels]
#     val_labels = np.random.choice(labels_left, num_label_set, False)
#     train_labels = [i for i in labels_left if i not in val_labels]
#
#     print("foo")
#
# def sample_g_meta():
#     label_list = list(range(100))
#
#     num_of_labels = 30
#     num_label_set = 5
#
#     info = {}
#
#     # G is a dgl graph
#     for j in range(len(label_list)):
#         info[str(0) + '_' + str(j)] = label_list[j]
#
#     df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})
#
#     labels = np.unique(list(range(num_of_labels)))
#
#     test_labels = np.random.choice(labels, num_label_set, False)
#     labels_left = [i for i in labels if i not in test_labels]
#     val_labels = np.random.choice(labels_left, num_label_set, False)
#     train_labels = [i for i in labels_left if i not in val_labels]
#
#     path = "../../data/gmeta"
#
#     df[df.label.isin(train_labels)].reset_index(drop=True).to_csv(path + '/train.csv')
#     df[df.label.isin(val_labels)].reset_index(drop=True).to_csv(path + '/val.csv')
#     df[df.label.isin(test_labels)].reset_index(drop=True).to_csv(path + '/test.csv')
#
#     # with open(path + '/graph_dgl.pkl', 'wb') as f:
#     #     pickle.dump(dgl_Gs, f)
#
#     with open(path + '/label.pkl', 'wb') as f:
#         pickle.dump(info, f)
#
#     # np.save(path + '/features.npy', np.array(feature_map))
#
#
# if __name__ == '__main__':
#     # sample_g_meta()
#     train, val, test = get_data("HealthStory", "gat")
#     train.generate_subgraph(5)
#     print("foo")
