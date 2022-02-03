import pytorch_lightning as pl
import torch


class GraphTrainer(pl.LightningModule):

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=self.hparams['batch_size'])

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self.log_f1(outputs, 'val')

    def training_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        self.log_f1(outputs, 'train')

    def test_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        test_metrics = outputs[0]
        self.log_f1(test_metrics, 'test')
        val_test_metrics = outputs[1]
        self.log_f1(val_test_metrics, 'test_val')

    def log_f1(self, outputs, mode):
        """
        Outputs is a list of dicts: {'loss': tensor(0.3119, device='cuda:0'), 'f1': 0.04878048780487806}
        """

        f1_scores = torch.FloatTensor([d['f1'] for d in outputs if d['f1'] is not None])
        # print(f"\nF1 scores: {str(f1_scores)}")
        f1_mean = f1_scores.mean()
        self.log(f'{mode}_f1', f1_mean, on_step=False, on_epoch=True)
        # print(f"Averaged F1 score for {len(f1_scores)} batches: {f1_mean}\n")
