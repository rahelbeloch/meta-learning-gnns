import pytorch_lightning as pl
import torch
import torchmetrics as tm
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, StepLR


class GraphTrainer(pl.LightningModule):

    def __init__(self, validation_sets, query_and_support=False):
        super().__init__()

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.metrics = {'f1_target': ({}, 'none')} if query_and_support is False else \
            {'f1_target_query': ({}, 'none'), 'f1_target_support': ({}, 'none')}

        # we have a binary problem
        n_classes = 1
        self.validation_sets = validation_sets
        self.query_and_support = query_and_support

        splits = ['train', 'test'] + validation_sets

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
        b_size = self.hparams["model_params"]["batch_size"]
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=b_size, add_dataloader_idx=False)

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        for s in self.validation_sets:
            self.compute_and_log_metrics(s)

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.compute_and_log_metrics('train')

    def test_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        self.compute_and_log_metrics('test')

    def update_metrics(self, mode, predictions, targets):
        for mode_dict, _ in self.metrics.values():
            mode_dict[mode].update(predictions, targets)

    def update_query(self, mode, predictions, targets):
        self.metrics['f1_target_query'][0][mode].update(predictions, targets)

    def update_support(self, mode, predictions, targets):
        self.metrics['f1_target_support'][0][mode].update(predictions, targets)

    def compute_and_log_metrics(self, mode):
        label_names = self.hparams["model_params"]["label_names"]

        # we are at the end of an epoch, so log now on step
        if not self.query_and_support:
            f1_fake = self.metrics['f1_target'][0][mode].compute()
            self.log_on_epoch(f'{mode}/f1_{label_names[1]}', f1_fake)
            self.metrics['f1_target'][0][mode].reset()
        else:
            f1_fake_support = self.metrics['f1_target_support'][0][mode].compute()
            self.log_on_epoch(f'{mode}/f1_{label_names[1]}_support', f1_fake_support)
            self.metrics['f1_target_support'][0][mode].reset()

            f1_fake_query = self.metrics['f1_target_query'][0][mode].compute()
            self.log_on_epoch(f'{mode}/f1_{label_names[1]}_query', f1_fake_query)
            self.metrics['f1_target_query'][0][mode].reset()

    def get_optimizer(self, lr, step_size, model=None, milestones=None):
        opt_params = self.hparams.optimizer_hparams

        model = self.model if model is None else model

        if opt_params['optimizer'] == 'Adam':
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=opt_params['weight_decay'])
        elif opt_params['optimizer'] == 'SGD':
            optimizer = SGD(model.parameters(), lr=lr, momentum=opt_params['momentum'],
                            weight_decay=opt_params['weight_decay'])
        else:
            raise ValueError("No optimizer name provided!")

        scheduler = None
        if opt_params['scheduler'] == 'step':
            scheduler = StepLR(optimizer, step_size=step_size, gamma=opt_params['lr_decay_factor'])
        elif opt_params['scheduler'] == 'multi_step':
            milestones = [5, 10, 15, 20, 30, 40, 55] if milestones is None else milestones
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=opt_params['lr_decay_factor'])

        return optimizer, scheduler


# noinspection PyAbstractClass
def get_loss_weight(class_weights, split):
    pos_weight = class_weights[split][0] // class_weights[split][1]
    print(f"Positive {split} weight: {pos_weight}")
    return pos_weight if pos_weight > 0 else None


def get_or_none(dikt, key):
    return dikt[key] if key in dikt else None
