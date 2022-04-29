import pytorch_lightning as pl
import torch
import torchmetrics as tm

from data_prep.graph_preprocessor import SPLITS


class GraphTrainer(pl.LightningModule):

    def __init__(self):
        super().__init__()

        type(self)

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.metrics = {
            'f1_macro': ({}, 'macro'),
            'f1_target': ({}, 'none')
        }

        # we have a binary problem
        n_classes = 1

        splits = SPLITS
        if 'GatBase' in str(type(self)):
            splits += ['val_tune']

        # Metrics from torchmetrics
        for s in splits:
            for name, (split_dict, avg) in self.metrics.items():
                metric = tm.F1 if name.startswith('f1') else None
                if metric is None:
                    raise ValueError(f"Metric with key '{name}' not supported.")
                split_dict[s] = metric(num_classes=n_classes, average=avg).to(self._device)

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        b_size = self.hparams['batch_size']
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=b_size)

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self.compute_and_log_metrics('val')

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.compute_and_log_metrics('train')

    def test_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        self.compute_and_log_metrics('test')

    def compute_and_log_metrics(self, mode, verbose=True):
        f1_fake = self.metrics['f1_target'][0][mode].compute()
        f1_macro = self.metrics['f1_macro'][0][mode].compute()

        if verbose:
            label_names = self.hparams["label_names"]

            # we are at the end of an epoch, so log now on step
            self.log_on_epoch(f'{mode}/f1_{label_names[1]}', f1_fake)
            self.log_on_epoch(f'{mode}/f1_macro', f1_macro)

        self.metrics['f1_target'][0][mode].reset()
        self.metrics['f1_macro'][0][mode].reset()
