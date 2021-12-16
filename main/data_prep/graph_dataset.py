import abc
import itertools

import dgl
import torch
from dgl.data import DGLDataset
from scipy.sparse import load_npz
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset
from torch_geometric.utils import k_hop_subgraph

from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file
from data_prep.graph_io import GraphIO


class TorchGeomGraphDataset(GraphIO, GeometricDataset):
    """
    Parent class for graph datasets. It loads the graph from respective files.
    """

    def __init__(self, corpus, top_k, feature_type, max_vocab, nr_train_docs, data_dir, tsv_dir, complete_dir,
                 verbose=True):
        super().__init__(corpus, feature_type, max_vocab, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._verbose = verbose

        self.top_k = top_k
        self.class_ratio = None
        self._data = None
        self._labels = None

        self.x_data, self.y_data = None, None
        self.edge_index, self.edge_type = None, None
        self.node2id = None
        self.split_masks = None
        self.vocab_size = None

        self.read_files(feature_type)
        self.initialize_graph()

    def read_files(self, feature_type):
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

        feat_matrix_file = self.data_complete_path(
            FEAT_MATRIX_FILE_NAME % (self.top_k, feature_type, self.max_vocab))
        if not feat_matrix_file.exists():
            raise ValueError(f"Feature matrix file does not exist: {feat_matrix_file}")
        self.x_data = torch.from_numpy(load_npz(feat_matrix_file).toarray())
        num_nodes, self.vocab_size = self.x_data.shape

        labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
        if not labels_file.exists():
            raise ValueError(f"Labels file does not exist: {labels_file}")
        self.y_data = torch.LongTensor(load_json_file(labels_file)['all_labels'])
        self._labels = self.y_data.unique()

        # calculate class imbalance for the loss function
        self.class_ratio = torch.bincount(self.y_data) / self.y_data.shape[0]

        # TODO: do we need this? We anyways do not do anything with nodes
        # load edge index and edge type
        # edge_index_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k)
        # self.edge_index = torch.from_numpy(load_npz(edge_index_file).toarray()).long()
        edge_list_file = self.data_complete_path(EDGE_LIST_FILE_NAME % self.top_k)
        if not edge_list_file.exists():
            raise ValueError(f"Edge list file does not exist: {edge_list_file}")
        edge_list = load_json_file(edge_list_file)
        self.edge_index = torch.tensor(edge_list).t().contiguous()

        edge_type_file = self.data_complete_path(EDGE_TYPE_FILE_NAME % self.top_k)
        self.edge_type = torch.from_numpy(load_npz(edge_type_file).toarray())

        # load node2id and node type
        # node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
        # self.node2id = json.load(open(node2id_file, 'r'))

        # node_type_file = self.data_complete_path( NODE_TYPE_FILE_NAME % self.top_k)
        # node_type = np.load(node_type_file)
        # node_type = torch.from_numpy(node_type).float()

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
        if not split_mask_file.exists():
            raise ValueError(f"Split masks file does not exist: {split_mask_file}")
        self.split_masks = load_json_file(split_mask_file)

        if self._verbose:
            self.print_step("Statistics")
            print(f"Vocabulary size: {self.vocab_size}")
            print(f'No. of nodes in graph: {num_nodes}')

    def initialize_graph(self):

        self.print_step("Initializing TorchGeom graph")

        # x_data should be float, edge_index should be long
        self._data = Data(x=self.x_data, edge_index=self.edge_index, edge_attr=self.edge_type, y=self.y_data)

        new_num_nodes, _ = self._data.x.shape

        self._data.train_mask = torch.FloatTensor(self.split_masks['train_mask'])
        self._data.val_mask = torch.FloatTensor(self.split_masks['val_mask'])
        self._data.test_mask = torch.FloatTensor(self.split_masks['val_mask'])

        # TODO: for what is this needed?
        # self._data.node2id = torch.tensor(list(self.node2id.values()))
        # self._data.node_type = self.node_type

        if self._verbose:
            print(f"No. of edges in graph = {self._data.num_edges}")
            print(f"\nNo.of train instances = {self._data.train_mask.sum().item()}")
            print(f"No.of val instances = {self._data.val_mask.sum().item()}", )
            print(f"No.of test instances = {self._data.test_mask.sum().item()}")

    def get_hop_subgraph(self, node_id, hop_size):
        """
        Finds all incoming edges and respective nodes for a given node and given hop size and creates a subgraph from it.
        :param node_id: The center node from which to start finding nodes.
        :param hop_size: Amount of hops.
        """

        node_idx, edge_index, node_mapping_idx, edge_mask = k_hop_subgraph(node_id.unsqueeze(dim=0), hop_size,
                                                                           self._data.edge_index, relabel_nodes=True,
                                                                           flow="target_to_source")

        # select the respective features to the subgraph
        # TODO: check more efficient way and make sure the features in the original graph are updated
        node_feats = self._data.x[node_idx]

        return TorchGeomSubGraph(node_id.item(), node_idx, node_feats, edge_index, node_mapping_idx, edge_mask)

    @property
    def labels(self):
        return self._labels

    @property
    def size(self):
        return self.x_data.shape


class DglGraphDataset(GraphIO, DGLDataset):
    """
    Parent class for graph datasets. It loads the graph from respective files.
    """

    def __init__(self, corpus, top_k, feature_type, max_vocab, nr_train_docs, data_dir, tsv_dir, complete_dir):
        super().__init__(corpus, feature_type, max_vocab, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

        self.top_k = top_k
        self.graph = None
        self.class_ratio = None

        self.initialize_graph(feature_type, nr_train_docs)

    def initialize_graph(self, feature_type, nr_train_docs):
        print('\nInitializing DGL graph ..........')

        # check if a DGL graph exists already for this dataset
        graph_file = self.data_complete_path(
            DGL_GRAPH_FILE % (self.dataset, nr_train_docs, feature_type, self.max_vocab))

        if graph_file.exists():
            print(f'Graph file exists, loading graph from it: {graph_file}')
            (g,), _ = dgl.load_graphs(str(graph_file))
            self.graph = g
        else:
            print(f'Graph does not exist, creating it.')

            feat_matrix_file = self.data_complete_path(
                FEAT_MATRIX_FILE_NAME % (self.top_k, feature_type, self.max_vocab))
            if not feat_matrix_file.exists():
                raise ValueError(f"Feature matrix file does not exist: {feat_matrix_file}")
            feat_matrix = torch.from_numpy(load_npz(feat_matrix_file).toarray())

            n_nodes = feat_matrix.shape[0]

            edge_list_file = self.data_complete_path(EDGE_LIST_FILE_NAME % self.top_k)
            if not edge_list_file.exists():
                raise ValueError(f"Edge list file does not exist: {edge_list_file}")
            edge_list = load_json_file(edge_list_file)
            src, dst = tuple(zip(*edge_list))

            g = dgl.graph((src, dst), num_nodes=n_nodes)
            g = dgl.add_reverse_edges(g)

            # adjacency_matrix_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k)
            # adj_matrix = torch.from_numpy(load_npz(adjacency_matrix_file).toarray())

            # coo = g.adj(scipy_fmt='coo')
            # values = coo.data
            # indices = np.vstack((coo.row, coo.col))

            # i = torch.LongTensor(indices)
            # v = torch.FloatTensor(values)
            # shape = coo.shape

            # torch_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

            g.ndata['feat'] = feat_matrix

            labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
            if not labels_file.exists():
                raise ValueError(f"Labels file does not exist: {labels_file}")
            y_labels = torch.LongTensor(load_json_file(labels_file)['all_labels'])
            g.ndata['label'] = y_labels

            # calculate class imbalance for the loss function
            self.class_ratio = torch.bincount(y_labels) / y_labels.shape[0]

            # If your dataset is a node classification dataset, you will need to assign
            # masks indicating whether a node belongs to training, validation, and test set.
            split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
            if not split_mask_file.exists():
                raise ValueError(f"Split masks file does not exist: {split_mask_file}")
            split_masks = load_json_file(split_mask_file)

            g.ndata['train_mask'] = torch.tensor(split_masks['train_mask'])
            g.ndata['val_mask'] = torch.tensor(split_masks['val_mask'])
            g.ndata['test_mask'] = torch.tensor(split_masks['test_mask'])

            # we do not have edge weights / features
            # g.edata['weight'] = edge_features

            print(f"Created DGL graph, saving it in: {graph_file}")
            dgl.save_graphs(str(graph_file), g)

            self.graph = g

        # calculate class imbalance for the loss function
        self.class_ratio = self.get_class_ratio()

        # make the features a sparse vector
        # g.ndata['feat'] = g.ndata['feat'].to_sparse()

        # 1st dimension number of tensor dimensions, 2nd dimension number of non-zero; should be shape of the whole tensor
        # indices = torch.nonzero(features)
        # features = torch.sparse_coo_tensor(indices, values, features.shape)

    def get_class_ratio(self):
        y_labels = self.graph.ndata['label']
        return torch.bincount(y_labels) / y_labels.shape[0]

    @property
    def labels(self):
        return self.graph.ndata['label'].unique()

    @property
    def size(self):
        return self.graph.ndata['feat'].shape

    @property
    def num_nodes(self):
        return self.graph.ndata['feat'].shape[0]

    def process(self):
        pass

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class SubGraphs(Dataset):
    """
    Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
    """

    def __init__(self, full_graph, mask_name, h_size):
        super().__init__()

        self.graph = full_graph
        self.hop_size = h_size
        self.mask_name = mask_name

    @property
    @abc.abstractmethod
    def mask(self):
        raise NotImplementedError

    @abc.abstractmethod
    def generate_subgraph(self, node_id):
        """
        Generate sub graphs on the flight.
        """
        raise NotImplementedError


class DGLSubGraphs(SubGraphs):
    """
    Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
    """

    def __init__(self, full_graph, mask_name, h_size, meta=False):
        super().__init__(full_graph, mask_name, h_size)

        self.meta = meta

    def generate_subgraph(self, node_id):
        """
        Generate sub graphs on the flight using DGL graphs.
        """

        # instead of calculating shortest distance, we find the following ways to get sub graphs are quicker
        sub_graph = self.get_hop_subgraph(node_id)

        # create mask for the classification; which node we want to classify
        classification_mask = torch.BoolTensor(sub_graph.num_nodes())
        # noinspection PyTypeChecker
        classification_mask[torch.where(sub_graph.ndata[dgl.NID] == node_id)[0]] = True
        sub_graph.ndata['classification_mask'] = classification_mask  # mask tensors must be bool

        return sub_graph

    def get_hop_subgraph(self, node_id):

        # find all incoming edges and the respective nodes

        one_hop_nodes = [n.item() for n in self.graph.graph.in_edges(node_id)[0]]

        if self.hop_size == 1:
            h_hop_neighbors = torch.tensor(list(set(one_hop_nodes + [node_id]))).numpy()
        elif self.hop_size == 2:
            # find one hops of the one hops
            two_hop_nodes = []
            for node in one_hop_nodes:
                one_hop_nodes = [n.item() for n in self.graph.graph.in_edges(node)[0] if n.item()]
                two_hop_nodes.append(one_hop_nodes)

            h_hop_neighbors = torch.tensor(list(itertools.chain(*two_hop_nodes)) + one_hop_nodes + [node_id]).unique()
        else:
            raise ValueError("Max 2 hop subgraphs supported for DGL dataset.")

        return dgl.node_subgraph(self.graph.graph, h_hop_neighbors, store_ids=True)

    def __getitem__(self, node_id):
        """
        Loads and returns a sample (a subgraph) from the dataset at the given index.
        :param node_id: Index for node ID from self.node_ids defining the node ID for which a subgraph should be created.
        :return:
        """

        subgraph = self.generate_subgraph(node_id)
        label = self.graph.graph.ndata['label'][node_id]
        return subgraph, label

    @property
    def targets(self):
        return self.graph.graph.ndata['label']

    @property
    def mask(self):
        return self.graph.graph.ndata[self.mask_name].bool()

    def as_dataloader(self, sampler, num_workers, collate_fn):
        return DataLoader(self, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)


class TorchGeomSubGraphs(SubGraphs):
    """
    Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
    """

    def __init__(self, full_graph, mask_name, h_size, meta=False):
        super().__init__(full_graph, mask_name, h_size)

        self.meta = meta

    def __getitem__(self, node_id):
        """
        Loads and returns a sample (a subgraph) from the dataset at the given index.
        :param node_id: Index for node ID from self.node_ids defining the node ID for which a subgraph should be created.
        :return:
        """

        subgraph = self.graph.get_hop_subgraph(node_id, self.hop_size)
        label = self.graph.y_data[node_id]

        return subgraph, label

    @property
    def targets(self):
        return self.graph.y_data

    @property
    def mask(self):
        return torch.BoolTensor(self.graph.split_masks[self.mask_name])

    def as_dataloader(self, sampler, num_workers, collate_fn):
        return DataLoader(self, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)


class TorchGeomSubGraph:
    """
    Class for a smaller graph constructed through the k-hop neighbors of one centroid node.
    """

    def __init__(self, center_node, node_idx, node_feats, edge_index, _, edge_mask):
        super().__init__()

        self.center_node = center_node
        self.node_idx = node_idx
        self.node_feats = node_feats
        self.edge_index = edge_index
        self.edge_mask = edge_mask

        self.node_mapping_idx = dict()
        for idx in range(node_idx.shape[0]):
            self.node_mapping_idx[node_idx[idx].item()] = idx

        # create mask for the classification; which node we want to classify
        self.classification_mask = torch.BoolTensor(node_idx.shape[0])
        # noinspection PyTypeChecker
        self.classification_mask[torch.where(node_idx == center_node)[0]] = True

    @property
    def num_nodes(self):
        return self.node_idx.shape[0]
