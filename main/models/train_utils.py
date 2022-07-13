import torch
from torch_geometric.data import Batch

from data_prep.data_utils import DEVICE


def get_subgraph_batch(graphs):
    batch = Batch.from_data_list(graphs).to(DEVICE)
    x = batch.x.float()

    # create the classification node mask
    cl_n_indices, n_count = [], 0
    for graph in graphs:
        cl_n_indices.append(n_count + graph.new_center_idx)
        n_count += graph.num_nodes

    return x, batch.edge_index, torch.LongTensor(cl_n_indices)


def get_predictions(logits):
    if logits.ndim == 1:
        # binary classification
        return (logits.sigmoid() > 0.5).float()
    else:
        # multiclass classification
        return torch.softmax(logits, dim=1).argmax(dim=1)
