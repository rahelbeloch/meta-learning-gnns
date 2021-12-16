import torch.cuda

from data_prep.graph_dataset import DGLSubGraphs, DglGraphDataset, TorchGeomGraphDataset, TorchGeomSubGraphs
from models.batch_sampler import FewShotSubgraphSampler
from models.maml_batch_sampler import FewShotMamlSubgraphSampler

SUPPORTED_DATASETS = ['gossipcop', 'twitterHateSpeech']


def get_data(data_train, data_eval, model, hop_size, top_k, k_shot, nr_train_docs, feature_type, vocab_size, dirs):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_train (str): Name of the data corpus which should be used for training.
        data_eval (str): Name of the data corpus which should be used for testing/evaluation.
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

    if data_train not in SUPPORTED_DATASETS:
        raise ValueError(f"Data with name '{data_train}' is not supported.")

    num_workers = 6 if torch.cuda.is_available() else 0  # mac has 8 CPUs

    # creating a train and val loader from the train dataset
    # graph_data_train = DglGraphDataset(data_train, top_k, feature_type, vocab_size, nr_train_docs, *dirs)
    graph_data_train = TorchGeomGraphDataset(data_train, top_k, feature_type, vocab_size, nr_train_docs, *dirs)

    train_loader = get_loader(graph_data_train, model, hop_size, k_shot, num_workers, 'train')
    train_val_loader = get_loader(graph_data_train, model, hop_size, k_shot, num_workers, 'val')

    eval_labels, test_val_loader = graph_data_train.labels, train_val_loader

    if data_eval is None or data_train == data_eval:
        print(f'\nData eval and data train are equal, loading graph data only once.')
        test_loader = get_loader(graph_data_train, model, hop_size, k_shot, num_workers, 'test')
    else:
        if data_eval not in SUPPORTED_DATASETS:
            raise ValueError(f"Data with name '{data_eval}' is not supported.")

        # creating a val and test loader from the eval dataset
        # graph_data_eval = DglGraphDataset(data_eval, top_k, feature_type, vocab_size, nr_train_docs, *dirs)
        graph_data_eval = TorchGeomGraphDataset(data_eval, top_k, feature_type, vocab_size, nr_train_docs, *dirs)

        test_loader = get_loader(graph_data_eval, model, hop_size, k_shot, num_workers, 'test')
        test_val_loader = get_loader(graph_data_eval, model, hop_size, k_shot, num_workers, 'val')

        assert graph_data_train.size[1] == graph_data_eval.size[1], \
            "Number of features for train and eval data is not equal!"

        eval_labels = graph_data_eval.labels

    loaders = (train_loader, train_val_loader, test_loader, test_val_loader)
    labels = (graph_data_train.labels, eval_labels)

    return loaders, graph_data_train.size, labels, train_loader.batch_size, graph_data_train.class_ratio


def get_loader(graph_data, model, hop_size, k_shot, num_workers, mode):
    # graphs = DGLSubGraphs(graph_data, f'{mode}_mask', h_size=hop_size, meta=model != 'gat')
    graphs = TorchGeomSubGraphs(graph_data, f'{mode}_mask', h_size=hop_size, meta=model != 'gat')

    n_classes = len(graph_data.labels)

    if model in ['gat', 'prototypical']:
        sampler = FewShotSubgraphSampler(graphs, n_way=n_classes, k_shot=k_shot, include_query=True)
    elif model == 'gmeta':
        sampler = FewShotMamlSubgraphSampler(graphs, n_way=n_classes, k_shot=k_shot, include_query=True)

    else:
        raise ValueError(f"Model with name '{model}' is not supported.")

    print(f"\n{mode} sampler amount of batches: {len(sampler)}")
    return graphs.as_dataloader(sampler, num_workers, sampler.get_collate_fn(model))
