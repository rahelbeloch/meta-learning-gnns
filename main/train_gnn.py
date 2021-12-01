import argparse
import os
import time

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data_prep.config import *
from data_prep.data_utils import SUPPORTED_DATASETS
from data_prep.data_utils import get_data
from models.gat_base import GatBase
from models.proto_maml import ProtoMAML
from models.proto_net import ProtoNet

SUPPORTED_MODELS = ['gat', 'prototypical', 'gmeta']
LOG_PATH = "../logs/"


def train(model_name, seed, epochs, patience, h_size, top_k, k_shot, lr, lr_cl, lr_inner, lr_output, cf_hidden_dim,
          proto_dim, data_name, dirs, checkpoint, h_search, train_docs, feature_type, vocab_size, n_inner_updates,
          evaluation=False):
    os.makedirs(LOG_PATH, exist_ok=True)

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    nr_train_docs = 'all' if (train_docs is None or train_docs == -1) else str(train_docs)

    print(f'\nConfiguration:\n mode: {"TEST" if eval else "TRAIN"}\n model_name: {model_name}\n data_name: {data_name}'
          f'\n nr_train_docs: {nr_train_docs}\n k_shot: {k_shot}\n seed: {seed}\n '
          f' feature_type: {feature_type}\n checkpoint: {checkpoint}\n max epochs: {epochs}\n patience:{patience}\n'
          f' lr: {lr}\n lr_cl: {lr_cl}\n cf_hidden_dim: {cf_hidden_dim}\n')

    # reproducible results
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # the data preprocessing
    print('\nLoading data ..........')
    train_loader, val_loader, test_loader, num_features, n_nodes, num_classes, b_size = get_data(data_name,
                                                                                                 model_name, h_size,
                                                                                                 top_k, k_shot,
                                                                                                 nr_train_docs,
                                                                                                 feature_type,
                                                                                                 vocab_size, dirs)
    print(f'\nGraph Size:\n num_features: {num_features}\n total_nodes: {n_nodes}')

    optimizer_hparams = {
        "lr_cl": lr_cl,
        "lr": lr,
        'lr_inner': lr_inner,
        'lr_output': lr_output
    }

    model_params = {
        'model': model_name,
        'cf_hid_dim': cf_hidden_dim,
        'input_dim': num_features,
        'output_dim': num_classes,
        'proto_dim': proto_dim
    }

    print('\nInitializing trainer ..........\n')
    trainer = initialize_trainer(epochs, patience, model_name, lr, lr_cl, lr_inner, lr_output, seed, data_name, k_shot,
                                 h_size, feature_type, checkpoint)

    if model_name == 'gat':
        model = GatBase(model_params, optimizer_hparams, b_size, checkpoint)
    elif model_name == 'prototypical':
        model = ProtoNet(model_params['input_dim'], model_params['cf_hid_dim'], optimizer_hparams['lr'], b_size)
    elif model_name == 'gmeta':
        model = ProtoMAML(model_params['input_dim'], model_params['cf_hid_dim'], optimizer_hparams, n_inner_updates)
    else:
        raise ValueError(f'Model name {model_name} unknown!')

    if not evaluation:
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
    evaluate(trainer, model, test_loader, val_loader)


def initialize_trainer(epochs, patience, model_name, lr, lr_cl, lr_inner, lr_output, seed, dataset, k_shot, h_size,
                       f_type, checkpoint):
    """
    Initializes a Lightning Trainer for respective parameters as given in the function header. Creates a proper
    folder name for the respective model files, initializes logging and early stopping.
    """

    model_checkpoint = cb.ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")

    base = f'dname={dataset}_seed={seed}_kshot={k_shot}_hops={h_size}_ftype={f_type}__lr={lr}'
    if model_name == 'gat':
        version_str = f'{base}_lr-cl={lr_cl}'
    elif model_name == 'prototypical':
        version_str = f'{base}'
    elif model_name == 'gmeta':
        version_str = f'{base}_lr-inner={lr_inner}_lr-output={lr_output}'
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
        model (pl.LightningModule) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
        val_dataloader (DataLoader) - Data loader for the validation split.
    Returns:
        test_accuracy (float) - The achieved test accuracy.
        val_accuracy (float) - The achieved validation accuracy.
    """

    print('\nTesting model on validation and test ..........\n')

    test_start = time.time()

    results = trainer.test(model, dataloaders=[test_dataloader, val_dataloader], verbose=False)

    test_results = results[0]
    val_results = results[1]

    test_accuracy = test_results['test_accuracy/dataloader_idx_0']
    test_f1_macro = test_results['test_f1_macro/dataloader_idx_0']
    test_f1_micro = test_results['test_f1_micro/dataloader_idx_0']

    val_accuracy = val_results['test_accuracy/dataloader_idx_1']
    val_f1_macro = val_results['test_f1_macro/dataloader_idx_1']
    val_f1_micro = val_results['test_f1_micro/dataloader_idx_1']

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n '
          f'test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'test f1 micro: {round(test_f1_micro, 3)} ({test_f1_micro})\n '
          f'test f1 macro: {round(test_f1_macro, 3)} ({test_f1_macro})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})\n '
          f'validation f1 macro: {round(val_f1_micro, 3)} ({val_f1_micro})\n '
          f'validation f1 micro: {round(val_f1_macro, 3)} ({val_f1_macro})\n '
          f'\nepochs: {trainer.current_epoch + 1}\n')

    return test_accuracy, val_accuracy


if __name__ == "__main__":
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_nodes = int(COMPLETE_small_DIR.split('-')[1])

    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    num_nodes = -1

    # MAML setup
    # proto_dim = 64,
    # lr = 1e-3,
    # lr_inner = 0.1,
    # lr_output = 0.1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--patience', dest='patience', type=int, default=10)
    parser.add_argument('--hop-size', dest='hop_size', type=int, default=2)
    parser.add_argument('--top-k', dest='top_k', type=int, default=30)
    parser.add_argument('--k-shot', dest='k_shot', type=int, default=5, help="Number of examples per task/batch.")

    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--lr-cl', dest='lr_cl', type=float, default=-1, help="Classifier learning rate for baseline.")

    # META setup

    parser.add_argument('--output-lr', dest='lr_output', type=float, default=1e-3)
    parser.add_argument('--inner-lr', dest='lr_inner', type=float, default=0.0001)
    parser.add_argument('--n-updates', dest='n_updates', type=int, default=5,
                        help="Inner gradient updates during meta learning.")

    # CONFIGURATION

    parser.add_argument('--dataset', dest='dataset', default='twitterHateSpeech', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use.')
    parser.add_argument('--num-train-docs', dest='num_train_docs', type=int, default=num_nodes,
                        help="Inner gradient updates during meta learning.")
    parser.add_argument('--feature-type', dest='feature_type', type=str, default='one-hot',
                        help="Type of features used.")
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument('--data-dir', dest='data_dir', default='data',
                        help='Select the dataset you want to use.')
    parser.add_argument('--tsv-dir', dest='tsv_dir', default=tsv_dir,
                        help='Select the dataset you want to use.')
    parser.add_argument('--complete-dir', dest='complete_dir', default=complete_dir,
                        help='Select the dataset you want to use.')
    parser.add_argument('--model', dest='model', default='gmeta', choices=SUPPORTED_MODELS,
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
        h_size=params["hop_size"],
        top_k=params["top_k"],
        k_shot=params["k_shot"],
        lr=params["lr"],
        lr_cl=params["lr_cl"],
        lr_inner=params["lr_inner"],
        lr_output=params["lr_output"],
        cf_hidden_dim=params["cf_hidden_dim"],
        proto_dim=params["proto_dim"],
        data_name=params["dataset"],
        dirs=(params["data_dir"], params["tsv_dir"], params["complete_dir"]),
        checkpoint=params["checkpoint"],
        h_search=params["h_search"],
        train_docs=params["num_train_docs"],
        feature_type=params["feature_type"],
        vocab_size=params["vocab_size"],
        n_inner_updates=params["n_updates"]
    )
