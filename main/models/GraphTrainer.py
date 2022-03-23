import pytorch_lightning as pl
import torch
import torchmetrics as tm

from data_prep.graph_preprocessor import SPLITS


class GraphTrainer(pl.LightningModule):

    def __init__(self, n_classes):
        super().__init__()

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.metrics = {
            'f1_macro': ({}, 'macro'),
            'f1_target': ({}, 'none'),
            'accuracy': ({}, 'macro')
        }

        # Metrics from torchmetrics
        for s in SPLITS:
            for name, (split_dict, avg) in self.metrics.items():
                metric = tm.Accuracy if name == 'accuracy' else tm.F1 if name.startswith('f1') else None
                if metric is None:
                    raise ValueError(f"Metric with key '{name}' not supported.")
                split_dict[s] = metric(num_classes=n_classes, average=avg, multiclass=True).to(self._device)

        # self.f1_test = torchmetrics.F1Score

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=self.hparams['batch_size'])

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self.compute_and_log_metrics('val')

    def training_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        self.compute_and_log_metrics('train')

    def test_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.compute_and_log_metrics('test')
        # val_test_metrics = outputs[1]
        # self.log_f1(val_test_metrics, 'test_val')

    def compute_and_log_metrics(self, mode, verbose=True):
        f1_1, f1_2 = self.metrics['f1_target'][0][mode].compute()
        f1_macro = self.metrics['f1_macro'][0][mode].compute()
        accuracy = self.metrics['accuracy'][0][mode].compute()

        if verbose:
            label_names = self.hparams["label_names"]
            self.log_on_epoch(f'{mode}_f1_{label_names[0]}', f1_1)
            self.log_on_epoch(f'{mode}_f1_{label_names[1]}', f1_2)
            self.log_on_epoch(f'{mode}_f1_macro', f1_macro)
            self.log_on_epoch(f'{mode}_accuracy', accuracy)

        self.metrics['f1_target'][0][mode].reset()
        self.metrics['f1_macro'][0][mode].reset()
        self.metrics['accuracy'][0][mode].reset()
