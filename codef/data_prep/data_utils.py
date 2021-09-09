import json
import os

SUPPORTED_DATASETS = ['HealthRelease', 'HealthStory']


def load_json_file(file_name):
    return json.load(open(file_name, 'r'))


def get_data(data_name, val_size=0.1):
    """
    Creates and returns the correct data object depending on data_name.
    Args:
        data_name (str): Name of the data corpus which should be used.
        val_size (float, optional): Proportion of training documents to include in the validation set.
    Raises:
        Exception: if the data_name is not in ['R8', 'R52', 'AGNews', 'IMDb'].
    """

    if data_name not in SUPPORTED_DATASETS:
        raise ValueError("Data with name '%s' is not supported." % data_name)

    if data_name == 'HealthRelease':
        return HealthReleaseData(val_size=val_size)
    elif data_name == 'HealthStory':
        return HealthStoryData(val_size=val_size)
    else:
        raise ValueError("Data with name '%s' is not supported." % data_name)


# def get_data_loaders(model, b_size, data_name):
#     """
#     Initializes train, text and validation data loaders for either roberta or graph models.
#     Args:
#         model (str): The name of the model which will be used.
#         b_size (int): Batch size to be used in the data loader.
#         data_name (str): Name of the data corpus which should be used.
#     Returns:
#         train_loader (DataLoader): Training data loader.
#         val_loader (DataLoader): Validation data loader.
#         test_loader (DataLoader): Test data loader.
#         additional_params (dict): Additional parameters needed for instantiation of the actual model later.
#     Raises:
#         Exception: if the model is not in ['roberta', 'glove_gnn', 'roberta_pretrained_gnn', 'roberta_finetuned_gnn']
#     """
#     corpus = get_data(data_name)
#     additional_params = {}
#
#     if model == 'roberta':
#         train_loader = RobertaDataset(corpus.train).as_dataloader(b_size, shuffle=True)
#         test_loader = RobertaDataset(corpus.test).as_dataloader(b_size)
#         val_loader = RobertaDataset(corpus.val).as_dataloader(b_size)
#
#         additional_params['num_classes'] = corpus.num_classes
#
#         return train_loader, val_loader, test_loader, additional_params
#
#     if model == 'glove_gnn':
#         dataset = GloveGraphDataset(corpus)
#         additional_params['doc_dim'] = dataset.doc_dim
#         additional_params['word_dim'] = dataset.word_dim
#     elif model in ['roberta_pretrained_gnn', 'roberta_finetuned_gnn']:
#         dataset = RobertaGraphDataset(corpus, roberta_model)
#     else:
#         raise ValueError("Model type '%s' is not supported." % model)
#
#     if isinstance(dataset, GraphDataset):
#         additional_params['num_classes'] = corpus.num_classes
#         additional_params['gnn_output_dim'] = dataset.num_nodes
#         train_loader = val_loader = test_loader = dataset.as_dataloader()
#         return train_loader, val_loader, test_loader, additional_params

if __name__ == '__main__':
    pass
