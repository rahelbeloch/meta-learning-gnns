import torch
from torch_geometric.data import Batch


def accuracy(predictions, labels):
    # noinspection PyUnresolvedReferences
    return (labels == predictions.argmax(dim=-1)).float().mean().item()


# def evaluation_metrics(predictions, labels, f1_target_label):
#     pred_cpu = predictions.argmax(dim=-1).detach().cpu()
#     labels_cpu = labels.detach().cpu()
#
#     # TODO: use torch metrics for F1 computation: Multi batch compilation for compuuting F1 score over all batches of one episode
#     # metric.compute --> at the end finish computation
#     # only then fair comparison
#
#     # F1 score of the target class (fake for gossipcop and racism for twitter)
#
#     # We cant to report macro --> for the positive class
#     # f1_macro = f1_score(labels_cpu, pred_cpu, average='macro')
#     # f1_micro = f1_score(labels_cpu, pred_cpu, average='micro')
#
#     # recall = recall_score(labels, predictions, average='binary', pos_label=1)
#     # precision = precision_score(labels, predictions, average='binary', pos_label=1)
#
#     return f1, f1_macro, f1_micro


def get_subgraph_batch(graphs):
    batch = Batch.from_data_list(graphs)

    x = batch.x.float()
    if not x.is_sparse:
        x = x.to_sparse()

    # create the classification node mask
    cl_n_indices, n_count = [], 0
    for graph in graphs:
        cl_n_indices.append(n_count + graph.new_center_idx)
        n_count += graph.num_nodes

    return x, batch.edge_index, torch.LongTensor(cl_n_indices)
