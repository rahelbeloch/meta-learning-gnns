import abc
import itertools

import dgl
import torch
from dgl.data import DGLDataset
from torch.utils.data import DataLoader

from data_prep.fake_health_graph_preprocessor import *
from data_prep.graph_io import GraphIO


# from torch_geometric.data import Data


# class TorchGeomGraphDataset(GraphIO):
#     """
#     Parent class for graph datasets. It loads the graph from respective files.
#     """
#
#     def __init__(self, corpus, verbose=True):
#         super().__init__(dataset=corpus, complete_dir=COMPLETE_DIR)
#
#         self.verbose = verbose
#
#         self.x_data, self.y_data = None, None
#         self.edge_index_data = None
#         self.node2id = None
#         self.split_masks = None
#         self.vocab_size = None
#         self.data = None
#         self.edge_type_data = None
#         self.loader = None
#
#         self.read_files()
#
#     def read_files(self):
#         """
#         Reads all the files necessary for the graph. These are:
#             - adjacency matrix
#             - feature matrix
#             - labels
#             - node2id dict
#             - node type map
#             - split mask
#         """
#
#         self.print_step("Reading files for Torch Geom Graph")
#
#         start = time.time()
#
#         x_feat_matrix_file = self.data_complete_path(FEAT_MATRIX_FILE_NAME % self.top_k)
#         self.x_data = torch.from_numpy(load_npz(x_feat_matrix_file).toarray())
#         num_nodes, self.vocab_size = self.x_data.shape
#
#         y_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
#         y_labels = json.load(open(y_labels_file, 'r'))
#         self.y_data = torch.LongTensor(y_labels['all_labels'])
#
#         edge_index_file = self.data_complete_path(ADJACENCY_MATRIX_FILE_NAME % self.top_k)
#         self.edge_index_data = torch.from_numpy(np.load(edge_index_file)).long()
#
#         node2id_file = self.data_complete_path(NODE_2_ID_FILE_NAME % self.top_k)
#         self.node2id = json.load(open(node2id_file, 'r'))
#
#         # TODO: node type
#         # node_type_file = self.data_complete_path( NODE_TYPE_FILE_NAME % self.top_k)
#         # node_type = np.load(node_type_file)
#         # node_type = torch.from_numpy(node_type).float()
#
#         split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
#         self.split_masks = json.load(open(split_mask_file, 'r'))
#
#         # if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
#         #     edge_type_file = self.data_complete_path(
#         #                                   'edge_type_lr_train_30_5_edge.npy'.format(self.config['data_name']))
#
#         # if self.config['model_name'] in ['rgcn', 'rgat', 'rsage']:
#         #     self.edge_type_data = np.load(edge_type_file)
#         #     self.edge_type_data = torch.from_numpy(self.edge_type_data).long()
#
#         if self.verbose:
#             self.print_header("DATA STATISTICS:")
#             # if self.config['model_name'] != 'HGCN':
#             #     print("Contains isolated nodes = ", isolated_nodes)
#             #     print("Contains self loops = ", self_loops)
#             print("Vocabulary size = ", self.vocab_size)
#             print('No. of nodes in graph = ', num_nodes)
#
#             # print('No. of nodes after removing isolated nodes = ', new_num_nodes)
#
#             # TODO: print this only once data is loaded!
#             print("No. of edges in graph = ", self.data.num_edges)
#             print("\nNo.of train instances = ", self.data.train_mask.sum().item())
#             print("No.of val instances = ", self.data.val_mask.sum().item())
#             nr_test_instances = num_nodes - self.data.train_mask.sum().item() - self.data.val_mask.sum().item()
#             print(f"No.of test instances = {nr_test_instances}")
#
#             hours, minutes, seconds = self.calc_elapsed_time(start, time.time())
#             self.print_header(f'Took  {hours:0>2} hours: {minutes:0>2} mins: {seconds:05.2f} secs  to Prepare Data')
#
#     def initialize_graph(self):
#
#         self.print_step("Initializing TorchGeom graph")
#
#         if self.verbose:
#             print("\n\n==>> Clustering the graph and preparing dataloader....")
#
#         self.data = Data(x=self.x_data.float(), edge_index=self.edge_index_data.long(),
#                          edge_attr=self.edge_type_data, y=self.y_data)
#         new_num_nodes, _ = self.data.x.shape
#
#         self.data.train_mask = torch.FloatTensor(self.split_masks['train_mask'])
#         self.data.val_mask = torch.FloatTensor(self.split_masks['val_mask'])
#         self.data.representation_mask = torch.FloatTensor(self.split_masks['repr_mask'])
#
#         self.data.node2id = torch.tensor(list(self.node2id.values()))
#         # self.data.node_type = self.node_type
#
#         # if not self.config['full_graph']:
#         #     if self.config['cluster']:
#         #         cluster_data = ClusterData(self.data, num_parts=self.config['clusters'], recursive=False)
#         #         self.loader = ClusterLoader(cluster_data, batch_size=self.config['batch_size'],
#         #                                     shuffle=self.config['shuffle'], num_workers=0)
#         #     elif self.config['saint'] == 'random_walk':
#         #         self.loader = GraphSAINTRandomWalkSampler(self.data, batch_size=6000, walk_length=2, num_steps=5,
#         #                                                   sample_coverage=100, num_workers=0)
#         #     elif self.config['saint'] == 'node':
#         #         self.loader = GraphSAINTNodeSampler(self.data, batch_size=6000, num_steps=5, sample_coverage=100,
#         #                                             num_workers=0)
#         #     elif self.config['saint'] == 'edge':
#         #         self.loader = GraphSAINTEdgeSampler(self.data, batch_size=6000, num_steps=5, sample_coverage=100,
#         #                                             num_workers=0)
#         # else:
#         self.loader = None
#
#         return self.loader, self.vocab_size, self.data


class DglGraphDataset(GraphIO, DGLDataset):
    """
    Parent class for graph datasets. It loads the graph from respective files.
    """

    def __init__(self, corpus, top_k, complete_dir=COMPLETE_DIR):
        super().__init__(dataset=corpus, complete_dir=complete_dir)

        self.top_k = top_k
        self.num_features = None

        self.initialize_graph()

    def initialize_graph(self):
        """

        :return:
        """

        self.print_step("Initializing DGL graph")

        # check if a DGL graph exists already for this dataset
        graph_file = self.data_complete_path(DGL_GRAPH_FILE % self.dataset)
        if os.path.exists(graph_file):
            print(f'Graph file exists, loading graph from it: {graph_file}')
            (g,), _ = dgl.load_graphs(graph_file)
            self.graph = g
            return

        print(f'Graph does not exist, creating it.')

        feat_matrix_file = self.data_complete_path(FEAT_MATRIX_FILE_NAME % self.top_k)
        feat_matrix = torch.from_numpy(load_npz(feat_matrix_file).toarray())
        n_nodes = feat_matrix.shape[0]
        self.num_features = feat_matrix.shape[1]

        edge_list_file = self.data_complete_path(EDGE_LIST_FILE_NAME % self.top_k)
        edge_list = load_json_file(edge_list_file)
        src, dst = tuple(zip(*edge_list))

        g = dgl.graph((src, dst), num_nodes=n_nodes)
        g = dgl.add_reverse_edges(g)

        g.ndata['feat'] = feat_matrix

        y_labels_file = self.data_complete_path(ALL_LABELS_FILE_NAME)
        y_labels = load_json_file(y_labels_file)['all_labels']
        g.ndata['label'] = torch.LongTensor(y_labels)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        split_mask_file = self.data_complete_path(SPLIT_MASK_FILE_NAME)
        split_masks = load_json_file(split_mask_file)

        g.ndata['train_mask'] = torch.tensor(split_masks['train_mask'])
        g.ndata['val_mask'] = torch.tensor(split_masks['val_mask'])
        g.ndata['test_mask'] = torch.tensor(split_masks['test_mask'])

        # we do not have edge weights / features
        # g.edata['weight'] = edge_features

        self.graph = g
        # print(f"Created DGL graph, saving it in: {graph_file}")
        # dgl.save_graphs(graph_file, g)

    def process(self):
        pass

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1


class SubGraphs:
    """
    Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
    """

    def __init__(self, full_graph, mask, b_size, h_size):
        super().__init__()

        self.graph = full_graph

        self.batch_size = b_size
        self.hop_size = h_size

        # we need all node IDs for this split
        self.node_ids = torch.where(full_graph.graph.ndata[mask].bool())[0]

        self.sub_graphs = {}

        # from the node_ids which are available in this graph, create a batch (batch_size node IDs)
        # sub graphs are created on the flight during training
        self.batch_node_ids = self.create_batch()

        print(f'\nNode IDs used for {mask.split("_")[0]}: {self.batch_node_ids}\n')

    def __len__(self):
        """
        This dataloader is of size batch size as we initially sampled batch_size number of node IDs.
        :return:
        """
        return self.batch_size

    def create_batch(self):
        """
        Create the entire set of batches (node IDs, sub graphs are generated on the flight).
        """

        sampled_node_ids = []

        # sample a node IDs for sample in the batch
        for x in range(self.batch_size):
            # select 1 node from the respective split
            selected_node = np.random.choice(self.node_ids, 1, False)[0]  # no duplicate
            # np.random.shuffle(selected_nodes)

            sampled_node_ids.append(selected_node)

        return sampled_node_ids

    @abc.abstractmethod
    def generate_subgraph(self, node_id):
        """
        Generate sub graphs on the flight.
        """
        raise NotImplementedError


# outside of the subgraph class because the collate function can not be pickled
def as_dataloader(sub_graph, shuffle=False):
    """

    :param sub_graph:
    :param shuffle:
    :return:
    """
    # num_workers = 0  # if gpu else 24; somehow no multiprocessing on GPU
    num_workers = 6
    # TODO: shuffle should be True?

    return DataLoader(sub_graph, batch_size=sub_graph.batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, collate_fn=collate_fn)


def collate_fn(batch_samples):
    """
    Receives a batch of samples (subgraphs and labels) node IDs for which sub graphs need to be generated on the flight.
    :param batch_samples: List of pairs where each pair is: (graph, label)
    """
    graphs, labels = list(map(list, zip(*batch_samples)))
    return dgl.batch(graphs), torch.LongTensor(labels),  # center, node_idx, graph_idx


class DGLSubGraphs(SubGraphs):
    """
    Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
    """

    def __init__(self, full_graph, mode, b_size, h_size):
        super().__init__(full_graph, mode, b_size, h_size)

        self.subgraph_statistics = None

    def generate_subgraph(self, node_id):
        """
        Generate sub graphs on the flight using DGL graphs.
        """

        # check if we have already generated a subgraph for this node
        if node_id not in self.sub_graphs.keys():
            # Takes super long; try to find alternative
            # subgraph = dgl.transform.khop_graph(self.graph.graph, 1, copy_ndata=False)

            # instead of calculating shortest distance, we find the following ways to get sub graphs are quicker

            h_hop_neighbors = self.get_hop_neighbors(node_id)

            # sub_graph = self.graph.graph.subgraph(h_hop_neighbors)
            sub_graph = dgl.node_subgraph(self.graph.graph, h_hop_neighbors, store_ids=True)

            # create mask for the classification; which node we want to classify
            classification_mask = torch.zeros(sub_graph.num_nodes())
            # noinspection PyTypeChecker
            classification_mask[torch.where(sub_graph.ndata[dgl.NID] == node_id)[0]] = 1
            sub_graph.ndata['classification_mask'] = classification_mask.bool()  # mask tensors must be bool

            # h_c = list(sub_graph.parent_nid.numpy())
            # dict_ = dict(zip(h_c, list(range(len(h_c)))))

            # self.subgraphs[node_id] = (sub_graph, dict_[node_id], h_c)
            self.sub_graphs[node_id] = sub_graph

        if self.subgraph_statistics is None:
            self.subgraph_statistics = {}
            for node_id, subgraph in self.sub_graphs.items():
                self.subgraph_statistics[node_id] = {'num_nodes': subgraph.num_nodes, 'num_edges': subgraph.num_edges}

            print(f"Subgraph statistics: {str(self.subgraph_statistics)}")

        return self.sub_graphs[node_id]

    def get_hop_neighbors(self, node_id):

        # find all incoming edges and the respective nodes

        # one_hop_neighbors = np.argwhere(self.graph.edge_index_data[:, node_id] == 1)
        # one_hop_neighbors = [one_hop_neighbors]

        one_hop_neighbors = [n.item() for n in self.graph.graph.in_edges(node_id)[0]]
        # row_idx =  np.argwhere(self.graph.edge_index_data[node_id, :]==1)

        if self.hop_size == 1:
            return torch.tensor(list(set(one_hop_neighbors + [node_id]))).numpy()
        elif self.hop_size == 2:
            # find one hops of the one hops
            two_hop_nodes = []
            for node in one_hop_neighbors:
                # one_hop_neighbors = np.argwhere(self.graph.edge_index_data[:, node] == 1)
                one_hop_nodes = [n.item() for n in self.graph.graph.in_edges(node)[0]]
                two_hop_nodes.append(one_hop_nodes)

            # f_hop = [n.item() for n in G.in_edges(i)[0]]     # list of edge IDs
            # n_l = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]

            two_hop_neighbors = torch.tensor(
                list(set(list(itertools.chain(*two_hop_nodes)) + one_hop_neighbors + [node_id]))).numpy()
            return two_hop_neighbors

        # elif self.hop_size == 3:
        # f_hop = [n.item() for n in G.in_edges(i)[0]]
        # n_2 = [[n.item() for n in G.in_edges(i)[0]] for i in f_hop]
        # n_3 = [[n.item() for n in G.in_edges(i)[0]] for i in list(itertools.chain(*n_2))]
        # h_hops_neighbor = torch.tensor(
        #     list(set(list(itertools.chain(*n_2)) + list(itertools.chain(*n_3)) + f_hop + [i]))).numpy()

        # if h_hops_neighbor.reshape(-1, ).shape[0] > self.sample_nodes:
        #     h_hops_neighbor = np.random.choice(h_hops_neighbor, self.sample_nodes, replace=False)
        #     h_hops_neighbor = np.unique(np.append(h_hops_neighbor, [i]))

    def __getitem__(self, node_id_idx):
        """
        Loads and returns a sample (a subgraph) from the dataset at the given index.
        :param node_id_idx: Index for node ID from self.node_ids defining the node ID for which a subgraph should be created.
        :return:
        """
        node_id = self.batch_node_ids[node_id_idx]
        subgraph = self.generate_subgraph(node_id)
        label = self.graph.graph.ndata['label'][node_id]
        return subgraph, label

    # return batched_graph_spt, torch.LongTensor(support_y_relative), batched_graph_qry, torch.LongTensor(
    #     query_y_relative), torch.LongTensor(support_center), torch.LongTensor(
    #     query_center), support_node_idx, query_node_idx, support_graph_idx, query_graph_idx

    # def get(self, idx):
    #     return self._data

# class TorchGeomSubGraphs(SubGraphs):
#     """
#     Sub graphs class for a smaller graph constructed through the k-hop neighbors of one centroid node.
#     """
#
#     def __init__(self, full_graph, mode, b_size, h_size):
#         super().__init__(full_graph, mode, b_size, h_size)
#
#     def generate_subgraph(self, node_id):
#         """
#         Generates a sub graph using torch.geometric.
#         """
#
#         edge_matrix_file = self.data_complete_path(EDGE_INDEX_FILE_NAME % self.top_k)
#         print("saving edge_type list format in :  ", edge_matrix_file)
#         edge_index_matrix = np.load(edge_matrix_file, allow_pickle=True)
#         torch_geometric.utils.k_hop_subgraph(node_id, self.hop_size, edge_index_matrix)
