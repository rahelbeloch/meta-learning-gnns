import torch
from scipy.sparse import load_npz
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset

from data_prep.config import *
from data_prep.data_preprocess_utils import load_json_file
from data_prep.graph_io import GraphIO


class TorchGeomGraphDataset(GraphIO, GeometricDataset):
    """
    Parent class for graph datasets. It loads the graph from respective files.
    """

    def __init__(self, corpus, top_k, feature_type, max_vocab, split_size, data_dir, tsv_dir, complete_dir,
                 verbose=True):
        super().__init__(corpus, feature_type, max_vocab, data_dir=data_dir, tsv_dir=tsv_dir, complete_dir=complete_dir)

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._verbose = verbose

        self.top_k = top_k
        self.feature_type = feature_type
        self.class_ratio = None
        self.train_size, self.val_size, self.test_size = split_size
        self._data = None
        self._labels = None

        self.x_data, self.y_data = None, None
        self.edge_index, self.edge_type = None, None
        self.node2id = None
        self.split_masks = None
        self.vocab_size = None

        self.read_files()
        self.initialize_graph()

    def get_file_name(self, file):
        return file % (self.top_k, self.feature_type, self.max_vocab, self.train_size, self.val_size, self.test_size)

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
        self.class_ratio = torch.bincount(self.y_data) / self.y_data.shape[0]

        # TODO: do we need this? We anyways do not do anything with nodes
        # load edge index and edge type
        # edge_index_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k)
        # self.edge_index = torch.from_numpy(load_npz(edge_index_file).toarray()).long()
        edge_list_file = self.data_complete_path(self.get_file_name(EDGE_LIST_FILE_NAME))
        if not edge_list_file.exists():
            raise ValueError(f"Edge list file does not exist: {edge_list_file}")
        edge_list = load_json_file(edge_list_file)
        self.edge_index = torch.tensor(edge_list).t().contiguous()

        edge_type_file = self.data_complete_path(self.get_file_name(EDGE_TYPE_FILE_NAME))
        self.edge_type = torch.from_numpy(load_npz(edge_type_file).toarray())

        # load node2id and node type
        # node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
        # self.node2id = json.load(open(node2id_file, 'r'))

        # node_type_file = self.data_complete_path( NODE_TYPE_FILE_NAME % self.top_k)
        # node_type = np.load(node_type_file)
        # node_type = torch.from_numpy(node_type).float()

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.

        split_mask_file = self.data_complete_path(self.get_file_name(SPLIT_MASK_FILE_NAME))
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

        self._data.train_mask = torch.BoolTensor(self.split_masks['train_mask'])
        self._data.val_mask = torch.BoolTensor(self.split_masks['val_mask'])
        self._data.test_mask = torch.BoolTensor(self.split_masks['val_mask'])

        # TODO: for what is this needed?
        # self._data.node2id = torch.tensor(list(self.node2id.values()))
        # self._data.node_type = self.node_type

        if self._verbose:
            print(f"No. of edges in graph = {self._data.num_edges}")
            print(f"\nNo. of train instances = {self._data.train_mask.sum().item()}")
            print(f"No. of val instances = {self._data.val_mask.sum().item()}", )
            print(f"No. of test instances = {self._data.test_mask.sum().item()}")

    # def get_hop_subgraph(self, node_id, hop_size):
    #     """
    #     Finds all incoming edges and respective nodes for a given node and given hop size and creates a subgraph from it.
    #     :param node_id: The center node from which to start finding nodes.
    #     :param hop_size: Amount of hops.
    #     """
    #
    #     node_idx, edge_index, node_mapping_idx, edge_mask = k_hop_subgraph(node_id.unsqueeze(dim=0), hop_size,
    #                                                                        self._data.edge_index, relabel_nodes=True,
    #                                                                        flow="target_to_source")
    #
    #     # select the respective features to the subgraph
    #     # TODO: check more efficient way and make sure the features in the original graph are updated
    #     node_feats = self._data.x[node_idx]
    #
    #     return TorchGeomSubGraph(node_id.item(), node_idx, node_feats, edge_index, node_mapping_idx, edge_mask)

    @property
    def labels(self):
        return self._labels

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self.x_data.shape

    def mask(self, mask_name):
        return torch.BoolTensor(self.split_masks[mask_name])

# class SubGraphs(GeometricDataset):
#     """
#     Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
#     """
#
#     def __init__(self, full_graph, mask_name, h_size):
#         super().__init__()
#
#         self.graph = full_graph
#         self.hop_size = h_size
#         self.mask_name = mask_name
#
#     @property
#     @abc.abstractmethod
#     def mask(self):
#         raise NotImplementedError
#
#     @abc.abstractmethod
#     def generate_subgraph(self, node_id):
#         """
#         Generate sub graphs on the flight.
#         """
#         raise NotImplementedError

# class TorchGeomSubGraphs(SubGraphs):
#     """
#     Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
#     """
#
#     def __init__(self, full_graph, mask_name, h_size, meta=False):
#         super().__init__(full_graph, mask_name, h_size)
#
#         self.meta = meta
#
#     # def __getitem__(self, subgraph):
#     #     """
#     #     Loads and returns a sample (a subgraph) from the dataset at the given index.
#     #     :param node_id: Index for node ID from self.node_ids defining the node ID for which a subgraph should be created.
#     #     :return:
#     #     """
#     #
#     #     # this was before SaintGraphSampler
#     #     # subgraph = self.graph.get_hop_subgraph(node_id, self.hop_size)
#     #     # label = self.graph.y_data[node_id]
#     #
#     #     node_ids = subgraph[0]
#     #
#     #     # assuming the first node in the list is the center node
#     #     label = self.graph.y_data[node_ids[0]]
#     #     node_feats = self.graph.x_data[node_ids]
#     #
#     #     return TorchGeomSubGraph(node_ids, subgraph[1], node_feats), label
#
#     @property
#     def targets(self):
#         return self.graph.y_data
#
#     @property
#     def mask(self):
#         return torch.BoolTensor(self.graph.split_masks[self.mask_name])
#
#     def as_dataloader(self, sampler, num_workers, collate_fn):
#         return DataLoader(self, batch_sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)


# class TorchGeomSubGraph:
#     """
#     Class for a smaller graph constructed through the k-hop neighbors of one centroid node.
#     """
#
#     def __init__(self, center_node, node_idx, node_feats, edge_index, _, edge_mask):
#         super().__init__()
#
#         self.center_node_orig = center_node
#         self.node_idx = node_idx
#         self.node_feats = node_feats
#         self.edge_index = edge_index
#         self.edge_mask = edge_mask
#
#         self.node_mapping_idx = dict()
#         for idx in range(node_idx.shape[0]):
#             self.node_mapping_idx[node_idx[idx].item()] = idx
#
#         self.center_node = self.node_mapping_idx[center_node]
#
#         # create mask for the classification; which node we want to classify
#         # self.classification_mask = torch.BoolTensor(node_idx.shape[0])
#         # # noinspection PyTypeChecker
#         # self.classification_mask[torch.where(node_idx == center_node)[0]] = True
#
#     @property
#     def num_nodes(self):
#         return self.node_idx.shape[0]

# class TorchGeomSubGraph:
#     """
#     Class for a smaller graph constructed through the k-hop neighbors of one centroid node.
#     """
#
#     def __init__(self, node_idx, node_feats, adjs):
#         super().__init__()
#
#         self.center_node_orig = node_idx[0]
#         self.node_idx = node_idx
#         self.node_feats = node_feats.to_dense().to_sparse()
#         self.adjs = adjs
#
#     @property
#     def num_nodes(self):
#         return self.node_idx.shape[0]
