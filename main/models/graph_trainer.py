import pytorch_lightning as pl
import torch
import torchmetrics as tm
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR, StepLR


class GraphTrainer(pl.LightningModule):

    def __init__(self, n_classes=1, target_classes=None, support_set=True):
        super().__init__()

        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.metrics = {'f1_target_query': ({}, 'none'),
                        'f1_macro_query': ({}, 'macro'),
                        'f1_weighted_query': ({}, 'weighted')}

        if support_set:
            self.metrics['f1_target_support'] = ({}, 'none')
            self.metrics['f1_macro_support'] = ({}, 'macro')
            self.metrics['f1_weighted_support'] = ({}, 'weighted')

        self.support_set = support_set

        # used during testing
        self.label_names, self.target_classes = None, [1] if target_classes is None else target_classes

        # we have a binary problem per default; will be adapted for testing purposes depending on dataset

        splits = ['train', 'test', 'val']

        for s in splits:
            self.initialize_metrics(s, n_classes)

    def initialize_metrics(self, split_name, n_classes):
        """
        Browses the dictionary of metrics and creates the new respective metric (currently only F1 supported)
        for the split with given name for the number of classes provided.
        """
        for name, (split_dict, avg) in self.metrics.items():
            metric = tm.F1 if name.startswith('f1') else None
            if metric is None:
                raise ValueError(f"Metric with key '{name}' not supported.")
            split_dict[split_name] = metric(num_classes=n_classes, average=avg).to(self._device)

    def reset_test_metric(self, n_classes, label_names, target_classes):
        self.initialize_metrics('test', n_classes)
        self.label_names = label_names
        self.target_classes = target_classes

    def log_on_epoch(self, metric, value):
        self.log(metric, value, on_step=False, on_epoch=True)

    def log(self, metric, value, on_step=True, on_epoch=False, **kwargs):
        b_size = self.hparams["model_params"]["batch_size"]
        super().log(metric, value, on_step=on_step, on_epoch=on_epoch, batch_size=b_size, add_dataloader_idx=False)

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        self.compute_and_log_metrics('val')

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        self.compute_and_log_metrics('train')

    def test_epoch_end(self, outputs) -> None:
        super().test_epoch_end(outputs)
        self.compute_and_log_metrics('test')

    def update_metrics(self, mode, predictions, targets, set_name=None):
        for metric_name, (mode_dict, _) in self.metrics.items():
            if set_name is None or set_name in metric_name:
                mode_dict[mode].update(predictions, targets)

    def compute_and_log_metrics(self, mode):

        if self.support_set:
            self.log_and_reset(mode, 'support')

            support_metric = self.metrics['f1_macro_support'][0][mode]
            self.log_on_epoch(f'{mode}/f1_macro_support', support_metric.compute())
            support_metric.reset()

            support_metric = self.metrics['f1_weighted_support'][0][mode]
            self.log_on_epoch(f'{mode}/f1_weighted_support', support_metric.compute())
            support_metric.reset()

        self.log_and_reset(mode, 'query')

        query_metric = self.metrics['f1_macro_query'][0][mode]
        self.log_on_epoch(f'{mode}/f1_macro_query', query_metric.compute())
        query_metric.reset()

        query_metric = self.metrics['f1_weighted_query'][0][mode]
        self.log_on_epoch(f'{mode}/f1_weighted_query', query_metric.compute())
        query_metric.reset()

    def log_and_reset(self, mode, set_name):
        metric = self.metrics[f'f1_target_{set_name}'][0][mode]
        label_names = self.hparams["model_params"]["label_names"] if self.label_names is None else self.label_names
        f1 = metric.compute()
        for t in self.target_classes:
            try:
                label_metric_value = f1[t] if len(self.target_classes) > 1 else f1[0]
            except IndexError:
                raise ValueError(f"Trying to log metric value for target {t}, but we don't have a metric value for it!")

            self.log_on_epoch(f'{mode}/f1_{label_names[t]}_{set_name}', label_metric_value)
        metric.reset()

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


def get_or_none(dikt, key):
    return dikt[key] if key in dikt else None
