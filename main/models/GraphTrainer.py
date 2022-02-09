import pytorch_lightning as pl
import torch
import torchmetrics


class GraphTrainer(pl.LightningModule):

    def __init__(self, num_classes):
        super().__init__()

        # Metrics from torchmetrics
        self.f1_scores = {
            'train': torchmetrics.F1(num_classes=num_classes),
            'test': torchmetrics.F1(num_classes=num_classes),
            'val': torchmetrics.F1(num_classes=num_classes)
        }

        # self.f1_test = torchmetrics.F1Score

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=self.hparams['batch_size'])

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self.compute_and_log_f1('val')

    def training_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        self.compute_and_log_f1('train')

    def compute_and_log_f1(self, mode):
        f1 = self.f1_scores[mode].compute()
        self.log(f'{mode}_f1', f1, on_step=False, on_epoch=True)
        self.f1_scores[mode].reset()

    def test_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.compute_and_log_f1('test')
        # val_test_metrics = outputs[1]
        # self.log_f1(val_test_metrics, 'test_val')

    def log_f1(self, outputs, mode):
        """
        Outputs is a list of dicts: {'loss': tensor(0.3119, device='cuda:0'), 'f1': 0.04878048780487806}
        """

        f1_scores = torch.FloatTensor([d['f1'] for d in outputs if d['f1'] is not None])
        # print(f"\nF1 scores: {str(f1_scores)}")
        f1_mean = f1_scores.mean()
        self.log(f'{mode}_f1', f1_mean, on_step=False, on_epoch=True)
        # print(f"Averaged F1 score for {len(f1_scores)} batches: {f1_mean}\n")
