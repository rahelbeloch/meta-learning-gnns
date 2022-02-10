import pytorch_lightning as pl
import torch
import torchmetrics as tm

from data_prep.graph_preprocessor import SPLITS


class GraphTrainer(pl.LightningModule):

    def __init__(self, n_classes):
        super().__init__()

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.f1_target = {}
        self.f1_macro = {}
        self.accuracies = {}

        # Metrics from torchmetrics
        for s in SPLITS:
            for avg, f1 in {'none': self.f1_target, 'macro': self.f1_macro}.items():
                f1[s] = tm.F1(num_classes=n_classes, average=avg, multiclass=True).to(self._device)
            self.accuracies[s] = tm.Accuracy(num_classes=n_classes, average='macro', multiclass=True).to(self._device)

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
        f1_1, f1_2 = self.f1_target[mode].compute()
        f1_macro = self.f1_macro[mode].compute()
        accuracy = self.accuracies[mode].compute()

        if verbose:
            label_names = self.hparams["label_names"]
            self.log(f'{mode}_f1_{label_names[0]}', f1_1, on_step=False, on_epoch=True)
            self.log(f'{mode}_f1_{label_names[1]}', f1_2, on_step=False, on_epoch=True)
            self.log(f'{mode}_f1_macro', f1_macro, on_step=False, on_epoch=True)
            self.log(f'{mode}_accuracy', accuracy, on_step=False, on_epoch=True)

        self.f1_target[mode].reset()
        self.f1_macro[mode].reset()
        self.accuracies[mode].reset()

        return f1_1, f1_2, f1_macro, accuracy
