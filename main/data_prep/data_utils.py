import torch.cuda

from data_prep.graph_dataset import TorchGeomGraphDataset
from data_prep.graph_preprocessor import SPLITS
from samplers.batch_sampler import FewShotSampler, get_n_query_for_samples, BatchSampler
from samplers.graph_sampler import KHopSampler
from samplers.maml_batch_sampler import FewShotMamlSampler

SUPPORTED_DATASETS = ['gossipcop', 'twitterHateSpeech']


def get_data(data_train, data_eval, model_name, hop_size, top_k, top_users_excluded, k_shot, train_split_size,
             eval_split_size, feature_type, vocab_size, dirs, num_workers=None):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_train (str): Name of the data corpus which should be used for training.
        data_eval (str): Name of the data corpus which should be used for testing/evaluation.
        model_name (str): Name of the model should be used.
        hop_size (int): Number of hops used to create sub graphs.
        top_k (int): Number (in thousands) of top users to be used in graph.
        top_users_excluded (int): Number (in percentage) of top users who should be excluded.
        k_shot (int): Number of examples used per task/batch.
        train_split_size (tuple): Floats defining the size of test/train/val for the training dataset.
        eval_split_size (tuple): Floats defining the size of test/train/val for the evaluation dataset.
        feature_type (str): Type of features that should be used.
        vocab_size (int): Size of the vocabulary.
        dirs (tuple): Path to the data (full & complete) to be used to create the graph (feature file, edge file etc.)
        num_workers (int): Amount of workers for parallel processing.
    Raises:
        Exception: if the data_name is not in SUPPORTED_DATASETS.
    """

    if data_train not in SUPPORTED_DATASETS:
        raise ValueError(f"Train data with name '{data_train}' is not supported.")

    if data_eval not in SUPPORTED_DATASETS:
        raise ValueError(f"Eval data with name '{data_eval}' is not supported.")

    if data_train == data_eval:
        assert train_split_size[0] > 0.0 and train_split_size[1] and train_split_size[2] > 0.0, \
            "Data for training and evaluation is equal and one of the split sizes is 0!"

    data_config = {'top_users': top_k, 'top_users_excluded': top_users_excluded, 'feature_type': feature_type,
                   'vocab_size': vocab_size}

    # creating a train and val loader from the train dataset
    train_config = {**data_config, **{'data_set': data_train, 'train_size': train_split_size[0],
                                      'val_size': train_split_size[1], 'test_size': train_split_size[2]}}
    graph_data_train = TorchGeomGraphDataset(train_config, train_split_size, *dirs)

    n_query_train = get_max_n_query(graph_data_train)
    print(f"\nUsing max query samples for episode creation: {n_query_train}\n")

    train_loader = get_loader(graph_data_train, model_name, hop_size, k_shot, num_workers, 'train', n_query_train)
    train_val_loader = get_loader(graph_data_train, model_name, hop_size, k_shot, num_workers, 'val', n_query_train)

    print(f"\nTrain graph size: \n num_features: {graph_data_train.size[1]}\n total_nodes: {graph_data_train.size[0]}")

    if data_train == data_eval:
        print(f'\nData eval and data train are equal, loading graph data only once.\n')
        graph_data_eval = graph_data_train
        test_val_loader = train_val_loader
        n_query_eval = n_query_train
    else:
        # creating a val and test loader from the eval dataset
        data_config['top_users_excluded'] = 0

        eval_config = {**data_config, **{'data_set': data_eval, 'train_size': eval_split_size[0],
                                         'val_size': eval_split_size[1], 'test_size': eval_split_size[2]}}

        graph_data_eval = TorchGeomGraphDataset(eval_config, eval_split_size, *dirs)
        n_query_eval = get_max_n_query(graph_data_eval)

        print(f"\nTest graph size: \n num_features: {graph_data_eval.size[1]}\n total_nodes: {graph_data_eval.size[0]}")

        test_val_loader = get_loader(graph_data_eval, model_name, hop_size, k_shot, num_workers, 'val', n_query_eval)

    test_loader = get_loader(graph_data_eval, model_name, hop_size, k_shot, num_workers, 'test', n_query_eval)

    loaders = (train_loader, train_val_loader, test_loader, test_val_loader)

    return loaders, train_loader.b_size, graph_data_train, graph_data_eval


def get_num_workers(sampler, num_workers):
    if num_workers is not None:
        return num_workers
    elif not torch.cuda.is_available():
        # mac has 8 CPUs
        return 0
    elif type(sampler) == BatchSampler:
        return 3
    elif type(sampler) in [FewShotSampler, FewShotMamlSampler]:
        return 4
    return 0


def get_loader(graph_data, model_name, hop_size, k_shot, num_workers, mode, n_queries):
    n_classes = len(graph_data.labels)

    mask = graph_data.mask(f"{mode}_mask")

    shuffle = mode == 'train'
    shuffle_once = mode == 'val'

    if model_name == 'gat' and mode == 'train':
        batch_sampler = BatchSampler(graph_data.data.y[mask], n_queries[mode], mode, n_way=n_classes, k_shot=k_shot,
                                     shuffle=shuffle, shuffle_once=shuffle_once)
    elif model_name == 'prototypical' or (model_name == 'gat' and mode != 'train'):
        batch_sampler = FewShotSampler(graph_data.data.y[mask], n_queries[mode], mode, n_way=n_classes, k_shot=k_shot,
                                       shuffle=shuffle, shuffle_once=shuffle_once)
    elif model_name == 'gmeta':
        batch_sampler = FewShotMamlSampler(graph_data.data.y, mask, n_way=n_classes, k_shot=k_shot, include_query=True,
                                           shuffle=shuffle)
    else:
        raise ValueError(f"Model with name '{model_name}' is not supported.")

    num_workers = get_num_workers(batch_sampler, num_workers)

    sampler = KHopSampler(graph_data, model_name, batch_sampler, n_classes, k_shot, hop_size, mode,
                          num_workers=num_workers)

    print(f"{mode} sampler episodes / batches: {len(sampler)}\n")

    # no need to wrap it again in a dataloader
    return sampler


def get_max_n_query(graph_data):
    """
    Calculates the number of query samples which fits the maximum configured shot number and all other available
    shot numbers. This is required in order to keep the query sets static throughout training with different shot sizes.
    """

    n_classes = len(graph_data.labels)

    n_queries = {}
    for split in SPLITS:
        # maximum amount of query samples which should be used from the total amount of samples
        samples = int(len(torch.where(graph_data.split_masks[f"{split}_mask"])[0]) / n_classes)
        n_queries[split] = get_n_query_for_samples(samples, n_classes)

    return n_queries
