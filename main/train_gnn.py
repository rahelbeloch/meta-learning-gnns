import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data_prep.data_utils import SUPPORTED_DATASETS
from data_prep.data_utils import get_data
from models.document_classifier import DocumentClassifier
from models.proto_net import ProtoNet

SUPPORTED_MODELS = ['gat', 'prototypical', 'gmeta']
LOG_PATH = "../logs/"


def train(model_name, seed, epochs, patience, b_size, h_size, top_k, k_shot, lr, l_rate_enc, l_rate_cl, cf_hidden_dim,
          proto_dim, data_name, data_dir, checkpoint, h_search, eval=False):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    print(f'\nConfiguration:\n mode: {"TEST" if eval else "TRAIN"}\n model_name: {model_name}\n data_name: {data_name}'
          f'\n k_shot: {k_shot} \n seed: {seed}\n batch_size: {b_size}\n checkpoint: {checkpoint}\n '
          f'max epochs: {epochs}\n patience:{patience}\n l_rate_enc: {l_rate_enc}\n '
          f'l_rate_cl: {l_rate_cl}\n cf_hidden_dim: {cf_hidden_dim}\n h_search: {h_search}\n')

    # reproducible results
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # the data preprocessing
    print('\nLoading data ..........')
    train_loader, val_loader, test_loader, num_features = get_data(data_name, model_name, data_dir, b_size, h_size,
                                                                   top_k, k_shot)

    optimizer_hparams = {"lr_enc": l_rate_enc,
                         "lr_cl": l_rate_cl,
                         "lr": lr
                         }

    num_classes = 2

    model_params = {
        'model': model_name,
        'cf_hid_dim': cf_hidden_dim,
        'input_dim': num_features,
        'output_dim': num_classes,
        'proto_dim': proto_dim
    }

    print('\nInitializing trainer ..........\n')
    trainer = initialize_trainer(epochs, patience, model_name, l_rate_enc, l_rate_cl, seed, data_name, checkpoint)

    if model_name == 'gat':
        model = DocumentClassifier(model_params, optimizer_hparams, checkpoint, h_search)
    elif model_name == 'prototypical':
        model = ProtoNet(model_params['input_dim'], model_params['cf_hid_dim'], optimizer_hparams['lr'])
    else:
        raise ValueError(f'Model name {model_name} unknown!')

    if not eval:
        # Training
        print('\nFitting model ..........\n')
        start = time.time()
        trainer.fit(model, train_loader, val_loader)

        end = time.time()
        elapsed = end - start
        print(f'\nRequired time for training: {int(elapsed / 60)} minutes.\n')

        # Load best checkpoint after training
        model_path = trainer.checkpoint_callback.best_model_path
        print(f'Best model path: {model_path}')
    else:
        raise ValueError("Wanting to evaluate, but can't as checkpoint is None.")

    model = model.load_from_checkpoint(model_path)
    test_acc, val_acc = evaluate(trainer, model, test_loader, val_loader)


def initialize_trainer(epochs, patience, model_name, l_rate_enc, l_rate_cl, seed, dataset, checkpoint):
    """
    Initializes a Lightning Trainer for respective parameters as given in the function header. Creates a proper
    folder name for the respective model files, initializes logging and early stopping.
    """

    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    if model_name == 'gat':
        version_str = f'dname={dataset}_seed={seed}_lr-enc={l_rate_enc}_lr-cl={l_rate_cl}'
    elif model_name == 'prototypical':
        version_str = f'dname={dataset}_seed={seed}_lr={l_rate_enc}'
    else:
        raise ValueError(f'Model name {model_name} unknown!')

    logger = TensorBoardLogger(LOG_PATH, name=model_name, version=version_str)

    # early_stop_callback = EarlyStopping(
    #     monitor='val_accuracy',
    #     min_delta=0.00,
    #     patience=patience,  # validation happens per default after each training epoch
    #     verbose=False,
    #     mode='max'
    # )

    trainer = pl.Trainer(log_every_n_steps=1,
                         logger=logger,
                         enable_checkpointing=True,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[model_checkpoint],
                         enable_progress_bar=True,
                         num_sanity_val_steps=0)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


def evaluate(trainer, model, test_dataloader, val_dataloader):
    """
    Tests a model on test and validation set.

    Args:
        trainer (pl.Trainer) - Lightning trainer to use.
        model (DocumentClassifier) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
        val_dataloader (DataLoader) - Data loader for the validation split.
    Returns:
        test_accuracy (float) - The achieved test accuracy.
        val_accuracy (float) - The achieved validation accuracy.
    """

    print('Testing model on validation and test ..........\n')

    test_start = time.time()

    test_result = trainer.test(model, test_dataloaders=test_dataloader, verbose=False)[0]
    test_accuracy = test_result["test_accuracy"]

    val_result = trainer.test(model, test_dataloaders=val_dataloader, verbose=False)[0]
    val_accuracy = val_result["test_accuracy"] if "val_accuracy" not in val_result else val_result["val_accuracy"]

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})'
          f'\n epochs: {trainer.current_epoch + 1}\n')

    return test_accuracy, val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8)
    parser.add_argument('--hop-size', dest='hop_size', type=int, default=2)
    parser.add_argument('--top-k', dest='top_k', type=int, default=30)
    parser.add_argument('--k-shot', dest='k_shot', type=int, default=40, help="Number of examples per task/batch.")

    parser.add_argument('--lr-enc', dest='l_rate_enc', type=float, default=0.01, help="Encoder learning rate.")
    parser.add_argument('--lr-cl', dest='l_rate_cl', type=float, default=-1, help="Classifier learning rate.")

    # META setup

    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--inner-lr', dest='inner_lr', type=float, default=0.0001, help="Inner learning rate.")
    parser.add_argument('--n_updates', dest='n_updates', type=int, default=5,
                        help="Inner gradient updates during meta learning.")
    parser.add_argument('--n_shots', dest='n_shots', type=int, default=40,
                        help="Inner gradient updates during meta learning.")

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='gossipcop', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--data-dir', dest='data_dir', default='../data/complete',
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='gat', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--cf-hidden-dim', dest='cf_hidden_dim', type=int, default=512)
    parser.add_argument('--proto-dim', dest='proto_dim', type=int, default=64)
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: None)')
    parser.add_argument('--transfer', dest='transfer', action='store_true', help='Transfer the model to new dataset.')
    parser.add_argument('--h-search', dest='h_search', action='store_true', default=False,
                        help='Flag for doing hyper parameter search (and freezing half of roberta layers) '
                             'or doing full fine tuning.')

    params = vars(parser.parse_args())

    train(
        model_name=params['model'],
        seed=params['seed'],
        epochs=params['epochs'],
        patience=params['patience'],
        b_size=params["batch_size"],
        h_size=params["hop_size"],
        top_k=params["top_k"],
        k_shot=params["k_shot"],
        lr=params["l_rate_enc"],
        l_rate_enc=params["l_rate_enc"],
        l_rate_cl=params["l_rate_cl"],
        cf_hidden_dim=params["cf_hidden_dim"],
        proto_dim=params["proto_dim"],
        data_name=params["dataset"],
        data_dir=params["data_dir"],
        checkpoint=params["checkpoint"],
        h_search=params["h_search"]
    )
