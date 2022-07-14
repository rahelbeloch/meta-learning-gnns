import argparse
import os
import time
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_prep.config import TSV_DIR, COMPLETE_DIR
from data_prep.data_utils import get_data, SUPPORTED_DATASETS
from models.gat_base import GatBase, evaluate
from models.gmeta import GMeta
from models.maml import Maml, test_maml
from models.proto_maml import ProtoMAML, test_protomaml
from models.proto_net import ProtoNet
from train_config import LOG_PATH, SUPPORTED_MODELS, META_MODELS, SHOTS

if torch.cuda.is_available():
    torch.cuda.empty_cache()


def train(balance_data, val_loss_weight, train_loss_weight, progress_bar, model_name, seed, epochs, patience,
          patience_metric, h_size, top_users, top_users_excluded, k_shot, lr, lr_val, lr_inner, lr_output,
          hidden_dim, feat_reduce_dim, proto_dim, data_train, data_eval, dirs, checkpoint, train_split_size,
          feature_type, vocab_size, n_inner_updates, n_inner_updates_test, num_workers, gat_dropout, lin_dropout,
          attn_dropout, wb_mode, warmup, max_iters, gat_heads, batch_size, lr_decay_epochs, lr_decay_epochs_val,
          lr_decay_factor, scheduler, weight_decay, momentum, optimizer, suffix):
    os.makedirs(LOG_PATH, exist_ok=True)

    eval_split_size = (0.0, 0.25, 0.75) if data_eval != data_train else None

    if model_name not in SUPPORTED_MODELS:
        raise ValueError("Model type '%s' is not supported." % model_name)

    if checkpoint is not None and model_name not in checkpoint:
        raise ValueError(f"Can not evaluate model type '{model_name}' on a pretrained model of another type.")

    if k_shot not in SHOTS:
        raise ValueError(f"'{k_shot}' is not valid!")

    # if we only want to evaluate, model should be initialized with nr of labels from evaluation data
    evaluation = checkpoint is not None and Path(checkpoint).exists()

    print(f'\nConfiguration:\n\n balance_data: {balance_data}\n train_loss_weight: {train_loss_weight}\n '
          f'val_loss_weight: {val_loss_weight}\n mode: {"TEST" if evaluation else "TRAIN"}\n '
          f'seed: {seed}\n max epochs: {epochs}\n patience: {patience}\n patience metric: {patience_metric}\n '
          f'k_shot: {k_shot}\n\n model_name: {model_name}\n hidden_dim: {hidden_dim}\n '
          f' feat_reduce_dim: {feat_reduce_dim}\n checkpoint: {checkpoint}\n gat heads: {gat_heads}\n\n'
          f' data_train: {data_train} (splits: {str(train_split_size)})\n data_eval: {data_eval} '
          f'(splits: {str(eval_split_size)})\n hop_size: {h_size}\n '
          f'top_users: {top_users}K\n top_users_excluded: {top_users_excluded}%\n num_workers: {num_workers}\n '
          f'vocab_size: {vocab_size}\n feature_type: {feature_type}\n\n lr: {lr}\n lr_val: {lr_val}\n '
          f'lr_output: {lr_output}\n inner_lr: {lr_inner}\n n_updates: {n_inner_updates}\n proto_dim: {proto_dim}\n')

    # reproducible results
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # the data preprocessing
    print('\nLoading data ..........')

    loaders, train_graph, eval_graph = get_data(data_train, data_eval, model_name, h_size, top_users,
                                                top_users_excluded, k_shot, train_split_size, eval_split_size,
                                                feature_type, vocab_size, dirs, batch_size, num_workers,
                                                balance_data)

    train_loader, train_val_loader, test_loader, test_val_loader = loaders

    optimizer_params = {"lr": lr, "warmup": warmup,
                        "max_iters": len(train_loader) * epochs if max_iters < 0 else max_iters,
                        "lr_decay_epochs": lr_decay_epochs, "lr_decay_factor": lr_decay_factor,
                        "scheduler": scheduler, "weight_decay": weight_decay, "momentum": momentum,
                        "optimizer": optimizer}

    model_params = {
        'model': model_name,
        'hid_dim': hidden_dim,
        'feat_reduce_dim': feat_reduce_dim,
        'input_dim': train_graph.size[1],
        'output_dim': get_output_dim(model_name, proto_dim),
        'class_weight': train_graph.class_ratios,
        'gat_dropout': gat_dropout,
        'lin_dropout': lin_dropout,
        'attn_dropout': attn_dropout,
        'concat': True,
        'n_heads': gat_heads,
        'batch_size': get_batch_size(model_name, train_loader),
        'label_names': train_graph.label_names,
    }

    other_params = {'train_loss_weight': torch.tensor(train_loss_weight) if train_loss_weight is not None else None,
                    'val_loss_weight': torch.tensor(val_loss_weight) if val_loss_weight is not None else None}

    if model_name == 'gat':
        optimizer_params.update(lr_val=lr_val, lr_decay_epochs_val=lr_decay_epochs_val)
        other_params.update(val_batches=len(train_val_loader))

        model = GatBase(model_params, optimizer_params, other_params)
    elif model_name == 'prototypical':
        model = ProtoNet(model_params, optimizer_params, other_params)
    elif model_name in META_MODELS:
        model_params.update(n_inner_updates=n_inner_updates, n_inner_updates_test=n_inner_updates_test)
        optimizer_params.update(lr_inner=lr_inner)
        other_params.update(k_shot_support=k_shot)

        if model_name == 'proto-maml':
            optimizer_params.update(lr_output=lr_output)
            model = ProtoMAML(model_params, optimizer_params, other_params)
        elif model_name == 'gmeta':
            model = GMeta(model_params, optimizer_params, other_params)
        elif model_name == 'maml':
            model = Maml(model_params, optimizer_params, other_params)
    else:
        raise ValueError(f'Model name {model_name} unknown!')

    print('\nInitializing trainer ..........\n')

    wandb_config = dict(
        seed=seed,
        max_epochs=epochs,
        patience=patience,
        patience_metric=patience_metric,
        k_shot=k_shot,
        h_size=h_size,
        checkpoint=checkpoint,
        data_train=data_train,
        data_eval=data_eval,
        feature_type=feature_type,
        batch_sizes=dict(train=train_loader.b_size, val=train_val_loader.b_size, test=test_loader.b_size),
        num_batches=dict(train=len(train_loader), val=len(train_val_loader), test=len(test_loader)),
        train_splits=dict(train=train_split_size[0], val=train_split_size[1], test=train_split_size[2]),
        top_users=top_users,
        top_users_excluded=top_users_excluded,
        num_workers=num_workers,
        vocab_size=vocab_size,
        balance_data=balance_data,
        suffix=suffix
    )

    trainer = initialize_trainer(epochs, patience, patience_metric, progress_bar, wb_mode, wandb_config,
                                 suffix)

    if not evaluation:
        # Training

        print('\nFitting model ..........\n')
        start = time.time()

        # we want to train on the support set of the validation loader and validate only on query set of validation
        val_loaders = [train_val_loader, train_val_loader] if model_name == 'gat' else train_val_loader

        # noinspection PyUnboundLocalVariable
        trainer.fit(model, train_loader, val_loaders)

        end = time.time()
        elapsed = end - start
        print(f'\nRequired time for training: {int(elapsed / 60)} minutes.\n')

        # Load the best checkpoint after training
        model_path = trainer.checkpoint_callback.best_model_path
        print(f'Best model path: {model_path}')
    else:
        model_path = checkpoint

    # Evaluation
    model = model.load_from_checkpoint(model_path)

    target_classes = [0, 1, 2] if data_eval == "twitterHateSpeech" else [1]
    target_labels = [eval_graph.label_names[cls] for cls in target_classes]
    n_classes = len(eval_graph.labels)

    if data_eval != data_train and data_eval is not None:
        if model_name == 'gat':
            # model was trained on another dataset --> reinitialize some things, like classifier output or target label
            # Completely newly setting the output layer, erases all pretrained weights!
            model.model.reset_classifier_dimensions(n_classes)

        # reset the test metric with number of classes
        model.reset_test_metric(n_classes, eval_graph.label_names, target_classes)

    if model_name == 'gat':

        test_f1_queries, f1_macro_query, f1_weighted_query, elapsed = evaluate(trainer, model, test_loader,
                                                                               target_labels)
    # elif model_name == 'prototypical':
    #     (test_f1_fake, _), elapsed, _ = test_proto_net(model, eval_graph, k_shot=k_shot)
    elif model_name == 'proto-maml':
        (test_f1_queries, _), (f1_macro_query, _), (f1_weighted_query, _), elapsed = test_protomaml(model,
                                                                                                    test_loader,
                                                                                                    target_labels,
                                                                                                    len(target_classes))
    elif model_name == 'maml':
        (test_f1_queries, _), (f1_macro_query, _), (f1_weighted_query, _), elapsed = test_maml(model,
                                                                                               test_loader,
                                                                                               target_labels,
                                                                                               len(target_classes))
    # elif model_name == 'gmeta':
    #     (test_f1_fake, _), elapsed = test_gmeta(model, test_loader)
    else:
        raise ValueError(f"Model type {model_name} not supported!")

    print(f'\nRequired time for testing: {int(elapsed / 60)} minutes.\n')
    print(f'Test Results:')

    for label in target_labels:
        wandb.log({f"test/f1_fake_{label}": test_f1_queries[label]})
        print(f' test f1 {label}: {round(test_f1_queries[label], 3)} ({test_f1_queries[label]})')

    print(f' test f1 macro: {round(f1_macro_query, 3)} ({f1_macro_query})\n'
          f' test f1 weighted: {round(f1_weighted_query, 3)} ({f1_weighted_query})\n '
          f'\nepochs: {trainer.current_epoch + 1}\n')

    print(f'{trainer.current_epoch + 1}\n{get_epoch_num(model_path)}')
    for label in target_labels:
        print(f'{round_format(test_f1_queries[label])}')
    print(f'{round_format(f1_macro_query)}\n{round_format(f1_weighted_query)}\n')


def get_output_dim(model_name, proto_dim):
    if model_name in ['gat', 'maml']:
        return 1  # with binary classification, we just use one output dimension
    elif model_name in ['gmeta', 'proto-maml', 'prototypical']:
        return proto_dim
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


def get_batch_size(model_name, train_loader):
    if model_name in ['gat', 'prototypical']:
        return train_loader.b_size
    elif model_name in ['gmeta', 'proto-maml', 'maml']:
        return train_loader.b_sampler.task_batch_size
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


def get_epoch_num(model_path):
    epoch_str = 'epoch='
    start_idx = model_path.find(epoch_str) + len(epoch_str)
    expected_epoch = model_path[start_idx: start_idx + 2]
    if expected_epoch.endswith('-'):
        expected_epoch = expected_epoch[:1]
    return int(expected_epoch)


def initialize_trainer(epochs, patience, patience_metric, progress_bar, wb_mode, wandb_config, suffix=None):
    """
    Initializes a Lightning Trainer for respective parameters as given in the function header. Creates a proper
    folder name for the respective model files, initializes logging and early stopping.
    """

    if patience_metric == 'loss':
        metric, mode = 'val/query_loss', 'min'
    elif patience_metric == 'f1_macro':
        metric, mode = 'val/f1_macro', 'max'
    else:
        raise ValueError(f"Patience metric '{patience_metric}' is not supported.")

    suffix = f'_{suffix}' if suffix is not None else ''

    logger = WandbLogger(project='meta-gnn',
                         name=f"{time.strftime('%Y%m%d_%H%M', time.gmtime())}{suffix}",
                         # log_model=True if wb_mode == 'online' else False,
                         log_model=False,
                         save_dir=LOG_PATH,
                         offline=wb_mode == 'offline',
                         config=wandb_config)

    early_stop_callback = EarlyStopping(
        monitor=metric,
        patience=patience,
        verbose=False,
        mode=mode,
        check_on_train_epoch_end=False
    )

    mc_callback = cb.ModelCheckpoint(save_weights_only=True, mode=mode, monitor=metric)

    trainer = pl.Trainer(move_metrics_to_cpu=True,
                         log_every_n_steps=1,
                         logger=logger,
                         enable_checkpointing=True,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=epochs,
                         callbacks=[mc_callback, early_stop_callback, LearningRateMonitor(logging_interval='epoch')],
                         enable_progress_bar=progress_bar,
                         num_sanity_val_steps=0)

    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    return trainer


def round_format(metric):
    return f"{round(metric, 3):.3f}".replace(".", ",")


if __name__ == "__main__":
    # Small part of the dataset
    # tsv_dir = TSV_small_DIR
    # complete_dir = COMPLETE_small_DIR
    # num_nodes = int(COMPLETE_small_DIR.split('-')[1])

    # Whole dataset
    tsv_dir = TSV_DIR
    complete_dir = COMPLETE_DIR
    num_nodes = -1

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TRAINING PARAMETERS

    parser.add_argument('--progress-bar', dest='progress_bar', action='store_true')
    parser.add_argument('--no-progress-bar', dest='progress_bar', action='store_false')
    parser.set_defaults(progress_bar=True)

    parser.add_argument('--wb-mode', dest='wb_mode', type=str, default='offline')

    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--patience-metric', dest='patience_metric', type=str, default='loss')
    parser.add_argument('--patience', dest='patience', type=int, default=20)
    parser.add_argument('--gat-dropout', dest='gat_dropout', type=float, default=0.4)
    parser.add_argument('--lin-dropout', dest='lin_dropout', type=float, default=0.5)
    parser.add_argument('--attn-dropout', dest='attn_dropout', type=float, default=0.4)
    parser.add_argument('--k-shot', dest='k_shot', type=int, default=5, help="Number of examples per task/batch.",
                        choices=SHOTS)

    # OPTIMIZER
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument('--lr-val', dest='lr_val', type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--warmup", dest='warmup', type=int, default=500,
                        help="Number of steps for which we do learning rate warmup.")
    parser.add_argument("--max-iters", dest='max_iters', type=int, default=-1,
                        help='Number of iterations until the learning rate decay after warmup should last. '
                             'If not given then it is computed from the given epochs.')
    parser.add_argument('--lr-decay-epochs', dest='lr_decay_epochs', type=float, default=5,
                        help='No. of epochs after which learning rate should be decreased')
    parser.add_argument('--lr-decay-epochs-val', dest='lr_decay_epochs_val', type=float, default=2,
                        help='No. of epochs after which learning rate should be decreased')
    parser.add_argument('--lr-decay-factor', dest='lr_decay_factor', type=float, default=0.1,
                        help='Decay the learning rate of the optimizer by this multiplicative amount')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='The type of lr scheduler to use anneal learning rate: step/multi_step')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-3,
                        help='weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.8, help='Momentum for optimizer')
    parser.add_argument('--optimizer', type=str, default="Adam", help='Momentum for optimizer')

    # MODEL CONFIGURATION

    parser.add_argument('--model', dest='model', default='gat', choices=SUPPORTED_MODELS,
                        help='Select the model you want to use.')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=512)
    parser.add_argument('--gat-heads', dest='gat_heads', type=int, default=2)
    parser.add_argument('--feature-reduce-dim', dest='feat_reduce_dim', type=int, default=256)
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: None)')

    # META PARAMETERS

    parser.add_argument('--proto-dim', dest='proto_dim', type=int, default=64)
    parser.add_argument('--output-lr', dest='lr_output', type=float, default=0.01)
    parser.add_argument('--inner-lr', dest='lr_inner', type=float, default=0.01)
    parser.add_argument('--n-updates', dest='n_updates', type=int, default=5,
                        help="Inner gradient updates during meta learning.")
    parser.add_argument('--n-updates-test', dest='n_updates_test', type=int, default=10,
                        help="Inner gradient updates during meta testing.")

    # DATA CONFIGURATION
    parser.add_argument('--hop-size', dest='hop_size', type=int, default=2)
    parser.add_argument('--top-users', dest='top_users', type=int, default=30)
    parser.add_argument('--top-users-excluded', type=int, default=1,
                        help='Percentage (in %) of top sharing users that are excluded (the bot users).')
    parser.add_argument('--n-workers', dest='n_workers', type=int, default=None,
                        help="Amount of parallel data loaders.")
    parser.add_argument('--dataset-train', dest='dataset_train', default='gossipcop', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use for training. '
                             'If a checkpoint is provided we do not train again.')
    parser.add_argument('--dataset-eval', dest='dataset_eval', default='gossipcop', choices=SUPPORTED_DATASETS,
                        help='Select the dataset you want to use for evaluation.')
    parser.add_argument('--feature-type', dest='feature_type', type=str, default='one-hot',
                        help="Type of features used.")
    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=None, help="Size of batches.")
    parser.add_argument('--data-dir', dest='data_dir', default='data',
                        help='Select the dataset you want to use.')

    parser.add_argument('--balance-data', dest='no_balance_data', action='store_false')
    parser.add_argument('--no-balance-data', dest='no_balance_data', action='store_true')
    parser.set_defaults(no_balance_data=True)

    parser.add_argument('--val-loss-weight', dest='val_loss_weight', type=int, default=None,
                        help="Weight of the minority class for the validation loss function.")

    parser.add_argument('--train-loss-weight', dest='train_loss_weight', type=int, default=None,
                        help="Weight of the minority class for the training loss function.")

    # parser.add_argument('--train-size', dest='train_size', type=float, default=0.875)
    # parser.add_argument('--val-size', dest='val_size', type=float, default=0.125)
    # parser.add_argument('--test-size', dest='test_size', type=float, default=0.0)

    parser.add_argument('--train-size', dest='train_size', type=float, default=0.7)
    parser.add_argument('--val-size', dest='val_size', type=float, default=0.1)
    parser.add_argument('--test-size', dest='test_size', type=float, default=0.2)

    parser.add_argument('--tsv-dir', dest='tsv_dir', default=tsv_dir,
                        help='Select the dataset you want to use.')
    parser.add_argument('--complete-dir', dest='complete_dir', default=complete_dir,
                        help='Select the dataset you want to use.')

    parser.add_argument('--suffix', dest='suffix', default='', help='Suffix for the run name to better identify.')

    params = vars(parser.parse_args())

    os.environ["WANDB_MODE"] = params['wb_mode']

    train(balance_data=not params['no_balance_data'],
          val_loss_weight=params['val_loss_weight'],
          train_loss_weight=params['train_loss_weight'],
          progress_bar=params['progress_bar'],
          model_name=params['model'],
          seed=params['seed'],
          epochs=params['epochs'],
          patience=params['patience'],
          patience_metric=params['patience_metric'],
          h_size=params["hop_size"],
          top_users=params["top_users"],
          top_users_excluded=params["top_users_excluded"],
          k_shot=params["k_shot"],
          lr=params["lr"],
          lr_val=params["lr_val"],
          lr_inner=params["lr_inner"],
          lr_output=params["lr_output"],
          hidden_dim=params["hidden_dim"],
          feat_reduce_dim=params["feat_reduce_dim"],
          proto_dim=params["proto_dim"],
          data_train=params["dataset_train"],
          data_eval=params["dataset_eval"],
          dirs=(params["data_dir"], params["tsv_dir"], params["complete_dir"]),
          checkpoint=params["checkpoint"],
          train_split_size=(params["train_size"], params["val_size"], params["test_size"]),
          feature_type=params["feature_type"],
          vocab_size=params["vocab_size"],
          n_inner_updates=params["n_updates"],
          n_inner_updates_test=params["n_updates_test"],
          num_workers=params["n_workers"],
          gat_dropout=params["gat_dropout"],
          lin_dropout=params["lin_dropout"],
          attn_dropout=params["attn_dropout"],
          wb_mode=params['wb_mode'],
          warmup=params['warmup'],
          max_iters=params['max_iters'],
          gat_heads=params['gat_heads'],
          batch_size=params['batch_size'],
          lr_decay_epochs=params['lr_decay_epochs'],
          lr_decay_epochs_val=params['lr_decay_epochs_val'],
          lr_decay_factor=params['lr_decay_factor'],
          scheduler=params['scheduler'],
          weight_decay=params['weight_decay'],
          momentum=params['momentum'],
          optimizer=params['optimizer'],
          suffix=params['suffix'])
