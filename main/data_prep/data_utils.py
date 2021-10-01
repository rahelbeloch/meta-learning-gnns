import json
import pickle

import numpy as np
import pandas as pd
import torch

from data_prep.config import ALL_LABELS_FILE_NAME
from data_prep.graph_dataset import DGLSubGraphs, DglGraphDataset, as_dataloader

SUPPORTED_DATASETS = ['HealthRelease', 'HealthStory']


# def load_json_file(file_name):
#     return json.load(open(file_name, 'r'))


def get_data(data_name, model, data_dir):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_name (str): Name of the data corpus which should be used.
        model (str): Name of the model should be used.
    Raises:
        Exception: if the data_name is not in SUPPORTED_DATASETS.
    """

    if data_name not in SUPPORTED_DATASETS:
        raise ValueError("Data with name '%s' is not supported." % data_name)

    if model == 'gat':
        # TODO: create batch of sub graphs (only via node indices) --> every data point can have a different label

        # load the whole graph once (it internally has the train/val/test masks)
        # graph_data = TorchGeomGraphDataset(data_name)
        # loader, vocab_size, data = graph_data.initialize_graph_data()
        # train_sub_graphs = TorchGeomSubGraphs(graph_data, 'train', b_size=128, h_size=2)
        # val_sub_graphs = TorchGeomSubGraphs(graph_data, 'val', b_size=128, h_size=2)
        # test_sub_graphs = TorchGeomSubGraphs(graph_data, 'test', b_size=128, h_size=2)

        graph_data = DglGraphDataset(data_name, data_dir)
        train_sub_graphs = as_dataloader(DGLSubGraphs(graph_data, 'train', b_size=6, h_size=2))
        val_sub_graphs = as_dataloader(DGLSubGraphs(graph_data, 'val', b_size=6, h_size=2))
        test_sub_graphs = as_dataloader(DGLSubGraphs(graph_data, 'test', b_size=6, h_size=2))

        return train_sub_graphs, val_sub_graphs, test_sub_graphs, graph_data.num_features

    # if data_name == 'HealthRelease':
    #     # return HealthReleaseData(val_size=val_size)
    #     return GraphDataset(data_name)
    # elif data_name == 'HealthStory':
    #     return HealthStoryData(val_size=val_size)
    # else:
    #     raise ValueError("Data with name '%s' is not supported." % data_name)


def sample_sub_graphs():
    """
    G-Meta samples subgraphs based on a train/test/val distribution of labels, where every label appears exactly once in
    one of the splits. The assignment is done randomized. We can not do this (with HealthStory), because we have
    only labels 0 and 1. Therefore, we need to rely on the randomized assignment of nodes to the different splits.
    We are already doing this --> we can immediately generate subgraphs on the flight for each node.
    """
    # number of unique labels, e.g. 2
    num_of_labels = 30

    # number of labels for each label set, ideally << num_of_labels so that each task can
    # consist out of different permutation of labels
    num_label_set = 5

    labels_file = "../../data/complete/FakeHealth/HealthStory/" + ALL_LABELS_FILE_NAME
    labels = json.load(open(labels_file, 'r'))

    all_labels = torch.LongTensor(labels['all_labels'])
    labels = np.unique(list(range(num_of_labels)))

    test_labels = np.random.choice(labels, num_label_set, False)
    labels_left = [i for i in labels if i not in test_labels]
    val_labels = np.random.choice(labels_left, num_label_set, False)
    train_labels = [i for i in labels_left if i not in val_labels]

    print("foo")


def sample_g_meta():
    label_list = list(range(100))

    num_of_labels = 30
    num_label_set = 5

    info = {}

    # G is a dgl graph
    for j in range(len(label_list)):
        info[str(0) + '_' + str(j)] = label_list[j]

    df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})

    labels = np.unique(list(range(num_of_labels)))

    test_labels = np.random.choice(labels, num_label_set, False)
    labels_left = [i for i in labels if i not in test_labels]
    val_labels = np.random.choice(labels_left, num_label_set, False)
    train_labels = [i for i in labels_left if i not in val_labels]

    path = "../../data/gmeta"

    df[df.label.isin(train_labels)].reset_index(drop=True).to_csv(path + '/train.csv')
    df[df.label.isin(val_labels)].reset_index(drop=True).to_csv(path + '/val.csv')
    df[df.label.isin(test_labels)].reset_index(drop=True).to_csv(path + '/test.csv')

    # with open(path + '/graph_dgl.pkl', 'wb') as f:
    #     pickle.dump(dgl_Gs, f)

    with open(path + '/label.pkl', 'wb') as f:
        pickle.dump(info, f)

    # np.save(path + '/features.npy', np.array(feature_map))


if __name__ == '__main__':
    # sample_g_meta()
    train, val, test = get_data("HealthStory", "gat")
    train.generate_subgraph(5)
    print("foo")
