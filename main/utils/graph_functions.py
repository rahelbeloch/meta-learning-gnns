import torch
from torch import Tensor
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor


def random_walk_subsampling_from_centernode(
    graph,
    max_nodes: int,
    walk_length: int = 5,
    bloated_budget_factor: int = 5,
    label_mask: int = -1,
):
    # If the budget is 0, just return the center node
    # and discard the rest of the graph
    if max_nodes == 0:
        labels_locs = torch.where(graph.y != label_mask)[0]

        return graph.subgraph(labels_locs)

    # Some initial statistics
    N = graph.num_nodes
    E = graph.num_edges

    labels_locs = torch.where(graph.y != label_mask)[0]
    n_labels = labels_locs.shape[0]

    # Build the adjacency matrix
    # Sparse, leverages torch_sparse
    adj = SparseTensor(
        row=graph.edge_index[0],
        col=graph.edge_index[1],
        value=torch.arange(E, device=graph.edge_index.device),
        sparse_sizes=(N, N),
    )

    # Generate the starting locations vector
    start_locs = torch.repeat_interleave(
        labels_locs, bloated_budget_factor * max_nodes
    ).chunk(n_labels)

    # Perform bloated_budget random walks
    # Add the nodes encountered to a list
    # Stop adding once budget is satisfied
    all_label_nodes = list()

    for start_loc in start_locs:
        label_nodes = set()
        walks = adj.random_walk(start_loc, walk_length).tolist()

        while len(label_nodes) < max_nodes and len(walks) > 0:
            label_nodes.update(walks.pop(-1))

        subset = torch.tensor(list(label_nodes))
        subset = torch.sort(subset).values

        all_label_nodes.append(subset)

    # Construct the subgraph from the sample nodes
    node_idx = torch.cat(all_label_nodes)
    adj, _ = adj.saint_subgraph(node_idx)

    subsampled_graph = graph.__class__()
    subsampled_graph.num_nodes = torch.tensor(node_idx.size(0))
    row, col, edge_idx = adj.coo()
    subsampled_graph.edge_index = torch.stack([row, col], dim=0)
    subsampled_graph.num_edges = torch.tensor(edge_idx.size(0))

    for k, v in graph:
        if k in ["edge_index", "adj_t", "num_nodes", "num_edges"]:
            continue
        if k == "y" and v.size(0) == graph.num_nodes:
            subsampled_graph[k] = graph.y[node_idx]
        elif isinstance(v, Tensor) and v.size(0) == graph.num_nodes:
            subsampled_graph[k] = v[node_idx]
        elif isinstance(v, Tensor) and v.size(0) == graph.num_edges:
            subsampled_graph[k] = v[edge_idx]
        else:
            subsampled_graph[k] = v

    return subsampled_graph


def avg_pool_doc_neighbours(graph):
    doc_node_ids = set(torch.where(graph.mask.bool())[0].tolist())
    user_node_ids = torch.where(~graph.mask.bool())[0]

    sparse_adj_matrix = to_torch_coo_tensor(graph.edge_index)

    for user_node in user_node_ids:
        neighbours = sparse_adj_matrix[user_node].coalesce().indices().squeeze(dim=0)

        incident_document_nodes = list(set(neighbours.tolist()) & doc_node_ids)

        if len(incident_document_nodes) == 0:
            continue

        graph.x[user_node] = graph.x[incident_document_nodes].mean(dim=(0,))

    return graph
