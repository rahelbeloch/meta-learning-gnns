import torch.cuda

from data_prep.graph_dataset import DGLSubGraphs, DglGraphDataset, collate_fn_proto, collate_fn_base
from models.batch_sampler import FewShotSubgraphSampler

SUPPORTED_DATASETS = ['HealthStory', 'gossipcop']


def get_data(data_name, model, batch_size, hop_size, top_k, k_shot, dirs):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_name (str): Name of the data corpus which should be used.
        model (str): Name of the model should be used.
        batch_size (int): Size of one batch.
        hop_size (int): Number of hops used to create sub graphs.
        top_k (int): Number of top users to be used in graph.
        k_shot (int): Number of examples used per task/batch.
        dirs (str): Path to the data (full & complete) to be used to create the graph (feature file, edge file etc.)
    Raises:
        Exception: if the data_name is not in SUPPORTED_DATASETS.
    """

    if data_name not in SUPPORTED_DATASETS:
        raise ValueError("Data with name '%s' is not supported." % data_name)

    # graph_data = TorchGeomGraphDataset(data_name)
    graph_data = DglGraphDataset(data_name, top_k, *dirs)

    train_graphs = DGLSubGraphs(graph_data, 'train_mask', b_size=batch_size, h_size=hop_size, meta=model != 'gat')
    val_graphs = DGLSubGraphs(graph_data, 'val_mask', b_size=batch_size, h_size=hop_size, meta=model != 'gat')
    test_graphs = DGLSubGraphs(graph_data, 'test_mask', b_size=batch_size, h_size=hop_size, meta=model != 'gat')

    num_workers = 6 if torch.cuda.is_available() else 0  # mac has 8 CPUs

    # load the whole graph once (it internally has the train/val/test masks)

    # loader, vocab_size, data = graph_data.initialize_graph_data()
    # train_sub_graphs = TorchGeomSubGraphs(graph_data, 'train', b_size=128, h_size=2)
    # val_sub_graphs = TorchGeomSubGraphs(graph_data, 'val', b_size=128, h_size=2)
    # test_sub_graphs = TorchGeomSubGraphs(graph_data, 'test', b_size=128, h_size=2)

    collate_fn = collate_fn_base if model == 'gat' else collate_fn_proto

    train_sampler = FewShotSubgraphSampler(train_graphs, include_query=True, k_shot=k_shot)
    train_loader = train_graphs.as_dataloader(train_sampler, num_workers, collate_fn)

    val_sampler = FewShotSubgraphSampler(val_graphs, include_query=True, k_shot=k_shot)
    val_loader = val_graphs.as_dataloader(val_sampler, num_workers, collate_fn)

    test_sampler = FewShotSubgraphSampler(test_graphs, include_query=True, k_shot=k_shot)
    test_loader = test_graphs.as_dataloader(test_sampler, num_workers, collate_fn)

    # train_sampler = FewShotSubgraphSampler(train_graphs, include_query=True, k_shot=k_shot)
    # train_loader = DataLoader(train_graphs, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)
    #
    # val_sampler = FewShotSubgraphSampler(val_graphs, include_query=True, k_shot=k_shot)
    # val_loader = DataLoader(val_graphs, batch_sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn)
    #
    # test_sampler = FewShotSubgraphSampler(test_graphs, include_query=True, k_shot=k_shot)
    # test_loader = DataLoader(test_graphs, batch_sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, graph_data.num_features
