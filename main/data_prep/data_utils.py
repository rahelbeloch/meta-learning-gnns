import torch.cuda

from data_prep.graph_dataset import DGLSubGraphs, DglGraphDataset, collate_fn_proto, collate_fn_base
from models.batch_sampler import FewShotSubgraphSampler
from models.maml_batch_sampler import FewShotMamlSubgraphSampler

SUPPORTED_DATASETS = ['HealthStory', 'gossipcop', 'twitterHateSpeech']


def get_data(data_name, model, hop_size, top_k, k_shot, nr_train_docs, feature_type, vocab_size, dirs):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_name (str): Name of the data corpus which should be used.
        model (str): Name of the model should be used.
        hop_size (int): Number of hops used to create sub graphs.
        top_k (int): Number of top users to be used in graph.
        k_shot (int): Number of examples used per task/batch.
        nr_train_docs (str): Number of total documents used for test/train/val.
        feature_type (int): Type of features that should be used.
        vocab_size (int): Size of the vocabulary.
        dirs (str): Path to the data (full & complete) to be used to create the graph (feature file, edge file etc.)
    Raises:
        Exception: if the data_name is not in SUPPORTED_DATASETS.
    """

    if data_name not in SUPPORTED_DATASETS:
        raise ValueError("Data with name '%s' is not supported." % data_name)

    # graph_data = TorchGeomGraphDataset(data_name)
    graph_data = DglGraphDataset(data_name, top_k, feature_type, vocab_size, nr_train_docs, *dirs)
    n_classes = len(graph_data.labels)

    train_graphs = DGLSubGraphs(graph_data, 'train_mask', h_size=hop_size, meta=model != 'gat')
    val_graphs = DGLSubGraphs(graph_data, 'val_mask', h_size=hop_size, meta=model != 'gat')
    test_graphs = DGLSubGraphs(graph_data, 'test_mask', h_size=hop_size, meta=model != 'gat')

    num_workers = 6 if torch.cuda.is_available() else 0  # mac has 8 CPUs

    # load the whole graph once (it internally has the train/val/test masks)

    # loader, vocab_size, data = graph_data.initialize_graph_data()
    # train_sub_graphs = TorchGeomSubGraphs(graph_data, 'train', b_size=128, h_size=2)
    # val_sub_graphs = TorchGeomSubGraphs(graph_data, 'val', b_size=128, h_size=2)
    # test_sub_graphs = TorchGeomSubGraphs(graph_data, 'test', b_size=128, h_size=2)

    collate_fn = None
    if model == 'gat':
        collate_fn = collate_fn_base
    elif model == 'prototypical':
        collate_fn = collate_fn_proto

    if model in ['gat', 'prototypical']:

        train_sampler = FewShotSubgraphSampler(train_graphs, n_way=graph_data.labels, include_query=True, k_shot=k_shot)
        print(f"\nTrain sampler amount of batches: {len(train_sampler)}")
        train_loader = train_graphs.as_dataloader(train_sampler, num_workers, collate_fn)

        val_sampler = FewShotSubgraphSampler(val_graphs, n_way=graph_data.labels, include_query=True, k_shot=k_shot)
        print(f"Val sampler amount of batches: {len(val_sampler)}")
        val_loader = val_graphs.as_dataloader(val_sampler, num_workers, collate_fn)

        test_sampler = FewShotSubgraphSampler(test_graphs, n_way=graph_data.labels, include_query=True, k_shot=k_shot)
        print(f"Test sampler amount of batches: {len(test_sampler)}")
        test_loader = test_graphs.as_dataloader(test_sampler, num_workers, collate_fn)

    elif model == 'gmeta':

        train_sampler = FewShotMamlSubgraphSampler(train_graphs, n_way=n_classes, k_shot=k_shot, include_query=True)
        print(f"\nTrain sampler amount of batches: {len(train_sampler)}")
        train_loader = train_graphs.as_dataloader(train_sampler, num_workers, train_sampler.get_collate_fn())

        val_sampler = FewShotMamlSubgraphSampler(val_graphs, n_way=n_classes, include_query=True, k_shot=k_shot)
        print(f"Val sampler amount of batches: {len(val_sampler)}")
        val_loader = val_graphs.as_dataloader(val_sampler, num_workers, val_sampler.get_collate_fn())

        test_sampler = FewShotMamlSubgraphSampler(test_graphs, n_way=n_classes, include_query=True, k_shot=k_shot)
        print(f"Test sampler amount of batches: {len(test_sampler)}")
        test_loader = test_graphs.as_dataloader(test_sampler, num_workers, test_sampler.get_collate_fn())

    else:
        raise ValueError(f"Model with name '{model}' is not supported.")

    # train_sampler = FewShotSubgraphSampler(train_graphs, include_query=True, k_shot=k_shot)
    # train_loader = DataLoader(train_graphs, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn)
    #
    # val_sampler = FewShotSubgraphSampler(val_graphs, include_query=True, k_shot=k_shot)
    # val_loader = DataLoader(val_graphs, batch_sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn)
    #
    # test_sampler = FewShotSubgraphSampler(test_graphs, include_query=True, k_shot=k_shot)
    # test_loader = DataLoader(test_graphs, batch_sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, graph_data.num_features, graph_data.num_nodes, n_classes, train_loader.batch_size
