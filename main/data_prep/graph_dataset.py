import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import load_npz
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.utils import contains_isolated_nodes, remove_isolated_nodes

import data_prep.fake_news_tsv_processor
import data_prep.twitter_tsv_processor
from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file
from data_prep.graph_io import GraphIO


def get_adj_matrix(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    for edge in edge_index.T:
        adj[edge[0], edge[1]] = 1
    for i in range(num_nodes):
        adj[i, i] = 1
    return adj


class TorchGeomGraphDataset(GraphIO, GeometricDataset):
    """
    Parent class for graph datasets. It loads the graph from respective files.
    """

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def __init__(self, config, split_size, data_dir, tsv_dir, complete_dir, verbose=False, analyse_node_degrees=False):
        super().__init__(config, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir, enforce_raw=False)

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._verbose = verbose
        self._analyse_node_degrees = analyse_node_degrees

        self.top_users = config['top_users']
        self.class_ratio = None
        self.train_size, self.val_size, self.test_size = split_size
        self._data = None
        self._labels = None

        self.x_data, self.y_data = None, None
        self.edge_index, self.edge_type = None, None
        self.adj = None
        self.doc2id = None
        self.split_masks = None

        self.read_files()

        # some verification of isolated nodes
        # should not occur in the train or val split, only in the test split
        isolated_nodes = contains_isolated_nodes(edge_index=self.edge_index)
        print(f"Contains isolated nodes: {isolated_nodes}")
        self.max_doc_id = list(self.doc2id.values())[-1]
        first_test_doc = torch.where(self.split_masks['test_mask'] == True)[0][0].item()

        _, _, isolated_mask = remove_isolated_nodes(edge_index=self.edge_index)
        isolated_train_val_docs = isolated_mask[:self.max_doc_id][:first_test_doc]
        isolated_train_val_nodes = torch.where(isolated_train_val_docs == False)[0]
        assert isolated_train_val_nodes.shape[0] == 0, "Edge index contains isolated nodes which are not test nodes!"

        isolated_train_val_indices = torch.where(torch.where(self.adj.sum(dim=1) == 1)[0] < first_test_doc)[0]
        assert isolated_train_val_indices.shape[0] == 0, "Adjacency contains isolated nodes which are not test nodes!"
        print(f"Nodes from train/val splits with only self connections: {isolated_train_val_indices}")

        # if self._analyse_node_degrees or self.dataset == 'gossipcop':
        #     print('\nAnalysing node degrees ..........')

        # Checking node degree distribution
        # node_degrees, probs = self.plot_node_degree_dist(self.adj)

        # if self.dataset == 'gossipcop':
        #     print(f"\nFixing node degrees for dataset '{self.dataset}'..........")
        #     self.fix_node_degree_distribution(node_degrees, probs)
        #
        #     if self._verbose:
        #         # compute and plot the new node distribution
        #         self.plot_node_degree_dist(self.adj)

        # check that no overlap between train/test and train/val

        # all node data
        # self.x_data

        # all edge data
        # self.edge_index
        # self.adj

        # all masks for splits
        # self.split_masks['train_mask']
        # self.split_masks['val_mask']
        # self.split_masks['test_mask']

        # 1. check that no classification node intersection between train/test/val

        # 2. any node that appears in any subgraph from test, may not appear in any subgraph for train

        # 3.

        self.initialize_graph()

    def read_files(self):
        """
        Reads all the files necessary for the graph. These are:
            - adjacency matrix (edge index) & edge type
            - feature matrix
            - labels
            - node2id dict
            - node type map
            - split mask
        """

        self.print_step("Reading files for Torch Geometric Graph")

        doc2id_file = self.data_complete_path(
            DOC_2_ID_FILE_NAME % (self.feature_type, self.vocab_size, self.train_size, self.val_size, self.test_size))
        self.doc2id = json.load(open(doc2id_file, 'r'))

        feat_matrix_file = self.data_complete_path(self.get_file_name(FEAT_MATRIX_FILE_NAME))
        if not feat_matrix_file.exists():
            raise ValueError(f"Feature matrix file does not exist: {feat_matrix_file}")
        self.x_data = torch.from_numpy(load_npz(feat_matrix_file).toarray())
        num_nodes, self.vocab_size = self.x_data.shape

        labels_file = self.data_complete_path(self.get_file_name(ALL_LABELS_FILE_NAME))
        if not labels_file.exists():
            raise ValueError(f"Labels file does not exist: {labels_file}")
        self.y_data = torch.LongTensor(load_json_file(labels_file)['all_labels'])
        self._labels = self.y_data.unique()

        # calculate class imbalance for the loss function
        self.compute_class_ratio()

        edge_list_file = self.data_complete_path(self.get_file_name(EDGE_LIST_FILE_NAME))
        if not edge_list_file.exists():
            raise ValueError(f"Edge list file does not exist: {edge_list_file}")
        edge_list = load_json_file(edge_list_file)
        self.edge_index = torch.tensor(edge_list).t().contiguous()

        # load adjacency matrix
        adj_matrix_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME)
        if adj_matrix_file.exists():
            self.adj = torch.from_numpy(load_npz(adj_matrix_file).toarray()).long()
        else:
            # create from edge_list file
            self.adj = get_adj_matrix(self.edge_index, num_nodes)

        # node_type_file = self.data_complete_path(NODE_TYPE_FILE_NAME % self.top_users)
        # node_type = np.load(node_type_file)
        # node_type = torch.from_numpy(node_type).float()

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.

        split_mask_file = self.data_complete_path(self.get_file_name(SPLIT_MASK_FILE_NAME))
        if not split_mask_file.exists():
            raise ValueError(f"Split masks file does not exist: {split_mask_file}")
        split_masks = load_json_file(split_mask_file)
        self.split_masks = {
            'train_mask': torch.BoolTensor(split_masks['train_mask']),
            'val_mask': torch.BoolTensor(split_masks['val_mask']),
            'test_mask': torch.BoolTensor(split_masks['test_mask'])
        }

        true_values = [torch.unique(self.split_masks[key], return_counts=True)[1][1].item() for key in self.split_masks]
        assert sum(true_values) == self.y_data.shape[0], "Split masks have more True values than there are labels!"

        if self._verbose:
            self.print_step("Statistics")
            print(f"Vocabulary size: {self.vocab_size}")
            print(f'No. of nodes in graph: {num_nodes}')

    def compute_class_ratio(self):
        self.class_ratio = torch.bincount(self.y_data) / self.y_data.shape[0]

        for i in range(len(self.class_ratio)):
            print(f"Ratio class '{self.label_names[i]}': {round(self.class_ratio[i].item(), 3)}")

    def fix_node_degree_distribution(self, node_degrees, probs, keep_threshold=0.97):
        """
        Find the bin (number of neighboring nodes) where at least 97% of the graph nodes fall in and discard the rest
        of the nodes.
        """

        # TODO: filter only user nodes out and not document nodes!!
        max_degree = np.argwhere(np.cumsum(probs) <= keep_threshold).squeeze().max()
        outliers = np.where(node_degrees > max_degree)[0].shape[0]

        # only use these nodes that have up to max_bin neighboring nodes
        new_node_indices = torch.from_numpy(np.argwhere(node_degrees <= max_degree).squeeze())
        num_new_nodes = new_node_indices.shape[0]

        if self._verbose:
            print(f"\nFound outliers ({round(1 - keep_threshold, 2) * 100}% of all nodes have more than "
                  f"{max_degree} and up to {node_degrees.max()} neighbors): {outliers}")
            print(f'Selecting the nodes that have less than {max_degree} neighbors...')
            print(f'\nNew no. of nodes in graph: {num_new_nodes}')

        row_select = torch.index_select(self.adj, 0, new_node_indices)
        new_adj = torch.index_select(row_select, 1, new_node_indices)

        # check that no node degree now is greater than the max_bin we choose earlier
        # noinspection PyTypeChecker
        assert not torch.any(new_adj.sum(dim=0) > max_degree).item()
        assert new_adj.shape[0] == num_new_nodes and new_adj.shape[1] == num_new_nodes

        # noinspection PyTypeChecker
        edges = torch.where(new_adj == 1)
        self.edge_index = torch.tensor(list(zip(list(edges[0]), list(edges[1])))).t()

        # noinspection PyTypeChecker
        old_max_doc_id = self.y_data.shape[0]
        filtered_doc_labels = (new_node_indices <= old_max_doc_id)[:old_max_doc_id]
        self.y_data = self.y_data[filtered_doc_labels]

        self._labels = self.y_data.unique()

        # calculate class imbalance for the loss function
        self.compute_class_ratio()

        # TODO: reset the features (also recalculate them for the user nodes??)
        filename = self.data_complete_path(self.get_file_name(USER_2_DOC_ID_FILE_NAME))
        user_doc_id_mapping = load_json_file(filename)

        new_max_doc_id = self.y_data.shape[0]
        old_max_user_id = self.x_data.shape[0]

        user_features = []
        for user_id in range(old_max_doc_id, old_max_user_id):
            user_docs = torch.tensor(user_doc_id_mapping[str(user_id)])
            new_user_docs = user_docs[user_docs < new_max_doc_id]
            new_user_feature = self.x_data[new_user_docs].sum(axis=0)
            if self.feature_type == 'one-hot':
                new_user_feature = new_user_feature >= 1
                new_user_feature = new_user_feature.int()
            user_features.append(new_user_feature)

        # first copy the document features
        self.x_data = self.x_data[new_node_indices]

        # then copy the user features
        self.x_data[self.y_data.shape[0]:] = torch.stack(user_features)

        num_nodes, self.vocab_size = self.x_data.shape

        # reset split masks
        self.split_masks['test_mask'] = self.split_masks['test_mask'][new_node_indices]
        self.split_masks['val_mask'] = self.split_masks['val_mask'][new_node_indices]
        self.split_masks['train_mask'] = self.split_masks['train_mask'][new_node_indices]

        self.adj = new_adj

    def plot_node_degree_dist(self, adj):
        node_degrees = torch.sum(adj, dim=0).numpy()

        max_bin = node_degrees.max()

        mu = node_degrees.mean()
        sigma = np.var(node_degrees)

        fig, ax = plt.subplots()

        ax.set_ylim([0.0, 0.03])

        probs, bins, patches = ax.hist(node_degrees, max_bin, density=True, stacked=True)

        if self._verbose:
            # add a 'best fit' line
            # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            #      np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2))
            # ax.plot(bins, y, '--')

            ax.set_xlabel('Nr. of neighboring nodes // Degree')
            ax.set_ylabel('Probability density')

            # noinspection PyTypeChecker
            ax.set_title(f"Graph '{self.dataset}' node degrees: $mean={round(mu, 2)}$, $var={round(sigma, 2)}$")

            # Tweak spacing to prevent clipping of y label
            fig.tight_layout()
            plt.show()

        return node_degrees, probs

    def initialize_graph(self):

        self.print_step("Initializing TorchGeom graph")

        # x_data should be float, edge_index should be long
        self._data = Data(x=self.x_data, edge_index=self.edge_index, edge_attr=self.edge_type, y=self.y_data)

        new_num_nodes, _ = self._data.x.shape

        self._data.train_mask = self.split_masks['train_mask']
        self._data.val_mask = self.split_masks['val_mask']
        self._data.test_mask = self.split_masks['val_mask']

        if self._verbose:
            print(f"No. of edges in graph = {self._data.num_edges}")
            print(f"\nNo. of train instances = {self._data.train_mask.sum().item()}")
            print(f"No. of val instances = {self._data.val_mask.sum().item()}", )
            print(f"No. of test instances = {self._data.test_mask.sum().item()}")

    @property
    def labels(self):
        return self._labels

    @property
    def f1_target_label(self):
        if self.dataset == 'gossipcop':
            d = data_prep.fake_news_tsv_processor.LABELS
            key = 'fake'
        elif self.dataset == 'twitterHateSpeech':
            d = data_prep.twitter_tsv_processor.LABELS
            key = 'racism'
        else:
            raise ValueError(f"Dataset with name '{self.dataset}' does not exist.")

        # inverse the dict and return the index for the key
        return {v: k for k, v in d.items()}[key]

    @property
    def label_names(self):
        if self.dataset == 'gossipcop':
            preprocessor = data_prep.fake_news_tsv_processor
        elif self.dataset == 'twitterHateSpeech':
            preprocessor = data_prep.twitter_tsv_processor
        else:
            raise ValueError(f"Dataset with name '{self.dataset}' does not exist.")
        return preprocessor.LABELS

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self.x_data.shape

    def mask(self, mask_name):
        return torch.BoolTensor(self.split_masks[mask_name])
