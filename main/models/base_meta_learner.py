import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl

from models.utils import WarmupCosineSchedule
from utils.metrics import compute_clf_metrics, compute_aupr_metrics


class BaseMetaLearner(pl.LightningModule):
    def __init__(
        self,
        training_data_params: str,
        training_structure_params: str,
        n_classes: int,
        optimizer_hparams,
        evaluation_params,
        ignore_index: int = -1,
    ):
        super().__init__()

        self.training_data_params = training_data_params
        self.training_structure_params = training_structure_params

        # Optimizer hyperparams ================================================
        self.opt_params = optimizer_hparams

        # Algorithm hyperparameters ============================================
        self.n_classes = n_classes

        self.eval_n_inner_updates = None
        self.eval_lr_inner = None
        self.eval_head_lr_inner = None

        self.register_class_weights(
            evaluation_params["class_weights"],
            prefix="eval",
        )
        self.eval_reset_classifier = None

        # Misc =================================================================
        # Which index value to ignore when computing loss
        # Should default to -1
        self.ignore_index = ignore_index

        self.automatic_optimization = False

        self.val_prefix = "val"
        self.test_prefix = "test"

    def register_class_weights(self, weights: list, prefix: str = None):
        if weights is not None:
            if len(weights) != self.n_classes:
                raise ValueError("The class weights must match the number of classes.")

            weights = torch.tensor(weights, dtype=torch.float)

        else:
            weights = torch.ones(
                self.n_classes,
                dtype=torch.float,
            )

        if prefix is not None:
            self.register_buffer(f"{prefix}_class_weights", weights)
            print(f"Registered {prefix} class weights as: {weights}")
        else:
            print(f"Registered class weights as: {weights}")
            self.register_buffer("class_weights", weights)

    def get_optimizer(self):
        if self.opt_params["optimizer"] == "Adam":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.opt_params["lr"],
                weight_decay=self.opt_params["weight_decay"],
            )
        elif self.opt_params["optimizer"] == "SGD":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.opt_params["lr"],
                momentum=self.opt_params["momentum"],
                weight_decay=self.opt_params["weight_decay"],
            )
        else:
            raise ValueError("No optimizer name provided!")

        scheduler = None
        if self.opt_params["scheduler"] == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.opt_params["lr_decay_steps"],
                gamma=self.opt_params["lr_decay_factor"],
            )

        elif self.opt_params["scheduler"] == "cosine":
            scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=self.opt_params["warmup_steps"],
                t_total=self.opt_params["total_steps"],
            )

        return optimizer, scheduler

    def configure_optimizers(self):
        train_optimizer, train_scheduler = self.get_optimizer()

        return [train_optimizer], [train_scheduler]

    def _get_gradient_norm(self, model):
        total_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            self.opt_params["max_norm"],
            norm_type=2.0,
        )

        return total_norm

    def metrics(self, logits, gt, prefix: str = ""):
        preds = torch.argmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        metrics = compute_clf_metrics(
            preds,
            gt,
            prefix=prefix,
            num_classes=self.n_classes,
            ignore_index=self.ignore_index,
        )

        metrics |= compute_aupr_metrics(
            probs,
            gt,
            prefix=prefix,
            num_classes=self.n_classes,
            ignore_index=self.ignore_index,
        )

        return metrics, preds, gt

    def forward(self, model, graph, mode):
        # Assumes the same signature for all models
        # Reasonable?
        logits = model.forward(graph.x, graph.edge_index, mode=mode)

        return logits

    def _logits_to_preds(self, logits):
        return torch.argmax(logits, dim=1)

    def clone(self, reset_classifier: bool = False, output_dim: int = None):
        # Need to be defined by the model classes inhereting this class
        raise NotImplementedError()

    def adapt(self, model, graph, mode, updates, lr, class_weights, head_lr):
        # Need to be defined by the model classes inhereting this class
        raise NotImplementedError()

    def eval_step(self, *args, **kwargs):
        if self.eval_n_inner_updates is None or self.eval_n_inner_updates == 0:
            return self.supervised_eval_step(*args, **kwargs)

        else:
            return self.episodic_eval_step(*args, **kwargs)

    def supervised_eval_step(self, graph, prefix: str = ""):
        """
        Evaluation on full graph.
        Expects a batch to consist of a (graph, y) pair

        """

        step_metrics = dict()

        # Clone the model for adaptation =======================================
        # Not applicable

        self.model.eval()

        # Adapt the model to samples from the new task =========================
        # Not applicable

        # Evaluate the adapted model on all available data =====================
        with torch.inference_mode():
            logits = self.forward(self.model, graph=graph, mode="eval")

            loss = F.cross_entropy(
                logits,
                graph.y,
                ignore_index=self.ignore_index,
                weight=self.eval_class_weights,
            )

            loss_unweighted = F.cross_entropy(
                logits,
                graph.y,
                ignore_index=self.ignore_index,
                weight=None,
            )

        step_metrics.update(
            {
                prefix + "loss": loss,
                prefix + "loss_unweighted": loss_unweighted,
            }
        )

        metrics, preds, gt = self.metrics(
            logits=logits[graph.mask].detach().cpu(),
            gt=graph.y[graph.mask].detach().cpu(),
            prefix=prefix,
        )
        step_metrics.update(metrics)

        return step_metrics, preds, gt

    def episodic_eval_step(self, graphs, prefix: str = ""):
        """
        Evaluation on a single episode.
        Expects a batch to consist of two graphs.
        The first being the support graph, the second being the graph containing
        all other nodes.

        """
        support_graph, query_graph = graphs

        # We still have lists here due to working with subgraphs before
        # Now thst we're evaluating on full graph, not needed anymore
        query_post_adapt_loss = []
        query_post_adapt_unweighted_loss = []
        preds_gt = {"logits": [], "gt": []}
        step_metrics = dict()

        # PytorchLightning has moved to inference mode by default everywhere
        # Need to disable it in the trainer
        with torch.enable_grad():
            # Clone the model for adaptation ===================================
            # Cloning/tensor creation cannot be inside inference mode
            task_model = self.clone(
                reset_classifier=self.eval_reset_classifier, output_dim=self.n_classes
            )

            task_model.eval()

            # Adapt the model to samples from the new task =====================
            # Adapt model to the sampled task
            adapt_output = self.adapt(
                task_model,
                support_graph,
                mode="eval",
                updates=self.eval_n_inner_updates,
                lr=self.eval_lr_inner,
                class_weights=self.eval_class_weights,
                head_lr=self.eval_head_lr_inner,
            )

            if isinstance(adapt_output, int):
                if adapt_output == 0:
                    raise ValueError("NaN inner loss on first adaptation step.")

                n_updates = adapt_output - 1

                task_model = self.clone(reset_classifier=self.reset_classifer)

                adapt_output = self.adapt(
                    task_model,
                    support_graph,
                    mode="eval",
                    updates=n_updates,
                    lr=self.eval_lr_inner,
                    class_weights=self.eval_class_weights,
                    head_lr=self.eval_head_lr_inner,
                )

                if isinstance(adapt_output, int):
                    raise ValueError("Could not recover from NaN inner loss.")

            task_model, loss_logs = adapt_output

        step_metrics.update({prefix + k: v for k, v in loss_logs.items()})

        # Episodic evaluation ==================================================
        with torch.no_grad():
            # Evaluate the adapted model on the support set ====================
            # Tests improvement
            s_logits = self.forward(task_model, support_graph, "eval")

            s_loss = (
                F.cross_entropy(
                    s_logits,
                    support_graph.y,
                    ignore_index=self.ignore_index,
                    weight=self.eval_class_weights,
                )
                .detach()
                .cpu()
            )

            step_metrics.update(
                {
                    prefix + "supp_post_loss": s_loss,
                    prefix
                    + "supp_improvement": 1
                    - s_loss / loss_logs["supp_pre_loss"],
                }
            )

            # Evaluate the adapted model on all available data =================
            # Query graph should contain nodes in support graph
            # But it's nodes are not labelled, and not counted towards metrics
            q_logits = self.forward(task_model, query_graph, "eval")

            loss = F.cross_entropy(
                q_logits,
                query_graph.y,
                ignore_index=self.ignore_index,
                weight=self.eval_class_weights,
            )
            query_post_adapt_loss.append(loss.cpu())

            loss = F.cross_entropy(
                q_logits,
                query_graph.y,
                ignore_index=self.ignore_index,
                weight=None,
            )
            query_post_adapt_unweighted_loss.append(loss.cpu())

            # Get preds on center nodes only
            preds_gt["logits"].append(q_logits.detach()[query_graph.mask].cpu())
            preds_gt["gt"].append(query_graph.y[query_graph.mask].cpu())

        # Stats aggregation ====================================================
        # Want the task_model off memory ASAP
        del task_model

        metrics, preds, gt = self.metrics(
            logits=torch.cat(preds_gt["logits"]),
            gt=torch.cat(preds_gt["gt"]),
            prefix=prefix,
        )
        step_metrics.update(metrics)

        step_metrics.update(
            {
                prefix
                + "query_mean_loss": torch.mean(torch.stack(query_post_adapt_loss)),
                prefix
                + "query_mean_loss_unweighted": torch.mean(
                    torch.stack(query_post_adapt_unweighted_loss)
                ),
            }
        )

        return (
            step_metrics,
            preds,
            gt,
        )

    def on_train_batch_end(self, *args) -> None:
        # Stepwise learning rate schedulers
        if (
            self.opt_params["scheduler"] in {"cosine"}
            or self.opt_params.get("step_frequency", False) == "batch"
        ):
            train_scheduler = self.lr_schedulers()
            train_scheduler.step()

        return super().on_train_batch_end(*args)

    def on_train_epoch_end(self) -> None:
        # Epochwise learning schedulers
        if (
            self.opt_params["scheduler"] in {"step", "multi_step"}
            or self.opt_params.get("step_frequency", False) == "epoch"
        ):
            train_scheduler = self.lr_schedulers()
            train_scheduler.step()

        super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.validation_epoch_metrics = defaultdict(list)

        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        step_metrics, _, _ = self.eval_step(batch, prefix=f"{self.val_prefix}/")

        for k, v in step_metrics.items():
            self.validation_epoch_metrics[k] += [v]

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        assert all(
            map(lambda x: len(x) > 0, list(self.validation_epoch_metrics.values()))
        ), "Empty validation epoch values."

        for k, v in self.validation_epoch_metrics.items():
            v_tensor = torch.stack(v)

            mean = torch.mean(v_tensor)
            std = torch.std(v_tensor)

            self.log_dict(
                {
                    k: mean,
                    k + "_std": std,
                    k + "_se": std / math.sqrt(v_tensor.shape[0]),
                },
                on_step=False,
                on_epoch=True,
            )

        self.log(
            f"{self.val_prefix}/eval_iterations",
            float(v_tensor.shape[0]),
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_start(self) -> None:
        self.test_epoch_metrics = defaultdict(list)
        self.test_epoch_preds = list()
        self.test_epoch_gt = list()

        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        step_metrics, preds, gt = self.eval_step(batch, prefix=f"{self.test_prefix}/")

        for k, v in step_metrics.items():
            self.test_epoch_metrics[k] += [v]

        self.test_epoch_preds.append(preds)
        self.test_epoch_gt.append(gt)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        self.test_epoch_preds = torch.cat(self.test_epoch_preds)
        self.test_epoch_gt = torch.cat(self.test_epoch_gt)

        for k, v in self.test_epoch_metrics.items():
            v_tensor = torch.stack(v)

            mean = torch.mean(v_tensor)
            std = torch.std(v_tensor)

            self.log_dict(
                {
                    k: mean,
                    k + "_std": std,
                    k + "_se": std / math.sqrt(v_tensor.shape[0]),
                },
                on_step=False,
                on_epoch=True,
            )

            self.test_epoch_metrics[k] = mean.detach().item()

        self.log(
            f"{self.test_prefix}/eval_iterations",
            float(v_tensor.shape[0]),
            on_step=False,
            on_epoch=True,
        )
