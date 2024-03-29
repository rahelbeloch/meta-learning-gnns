import math
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from models.base_meta_learner import BaseMetaLearner
from models.sparse_gat import SparseGatNet
from models.pointwise_baseline import PointwiseMLP


class GatMAML(BaseMetaLearner):
    """
    First-Order MAML (FOML) which only uses first-order gradients.
    """

    def __init__(
        self,
        model_params,
        learning_hparams,
        model_architecture: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        if model_architecture is None or model_architecture == "gat":
            self.model = SparseGatNet(model_params)
        elif model_architecture == "mlp":
            self.model = PointwiseMLP(model_params)

        self.register_class_weights(learning_hparams["class_weights"])

        # Training hyperparameters =============================================
        self.n_inner_updates = learning_hparams["n_inner_updates"]
        self.lr_inner = learning_hparams["lr_inner"]
        self.head_lr_inner = learning_hparams.get("head_lr_inner", None)
        self.reset_classifer = learning_hparams["reset_classifier"]

        self.eval_n_inner_updates = self.n_inner_updates
        self.eval_lr_inner = self.lr_inner
        self.eval_head_lr_inner = self.head_lr_inner
        self.eval_reset_classifer = self.reset_classifer

    def clone(self, reset_classifier: bool = False, output_dim: int = None):
        local_model = deepcopy(self.model)

        if reset_classifier:
            local_model.reset_classifier(output_dim)

        return local_model

    def adapt(
        self,
        model,
        graph,
        mode,
        updates=None,
        lr=None,
        class_weights=None,
        head_lr=None,
    ):
        # For MAML we use adaptation during training
        if updates is None:
            updates = self.n_inner_updates
        if lr is None:
            lr = self.lr_inner
        if class_weights is None:
            class_weights = self.class_weights
        if head_lr is None:
            if self.head_lr_inner is None:
                head_lr = lr
            else:
                head_lr = self.head_lr_inner

        # The learning rate used for 10 steps should be
        # smaller than that used for 1 step adaptation
        # lr = lr / updates

        for p in model.parameters():
            p.requires_grad = True

        # Could have done this cleaner, but wasn't expecting to
        if isinstance(self.model, SparseGatNet):
            optimizer = SGD(
                [
                    {"params": model.mha_1.parameters()},
                    {"params": model.non_lin_1.parameters()},
                    {"params": model.mha_2.parameters()},
                    {"params": model.mha_collator.parameters()},
                    {"params": model.classifier.parameters(), "lr": head_lr},
                ],
                lr=lr,
            )
        else:
            optimizer = SGD(
                [
                    {"params": model.feature_extractor.parameters()},
                    {"params": model.classifier.parameters(), "lr": head_lr},
                ],
                lr=lr,
            )

        losses = []
        for i in range(updates):
            logits = self.forward(
                model,
                graph,
                mode,
            )

            loss = F.cross_entropy(
                logits,
                graph.y,
                ignore_index=self.ignore_index,
                weight=class_weights,
            )

            if torch.any(torch.isnan(loss)):
                print("NaN inner loss.")
                return i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu())

        loss_logs = {
            "supp_pre_loss": losses[0]
            if len(losses) > 0
            else torch.tensor(float("nan")),
            "supp_mean_loss": torch.mean(torch.stack(losses)),
        }

        return model, loss_logs

    def training_step(self, graphs, batch_idx):
        train_opt = self.optimizers()

        # Unpack into support and query batch
        supp_graph, query_graph = graphs

        # Copy model and adapt on support set ==================================
        task_model = self.clone(reset_classifier=self.reset_classifer)

        adapt_output = self.adapt(
            task_model,
            supp_graph,
            mode="train",
        )

        # In case we get a NaN inner loss for some reason, try and recover
        # by repeating the inner loop, but one less step than when the crash occured
        if isinstance(adapt_output, int):
            n_updates = adapt_output - 1

            task_model = self.clone(reset_classifier=self.reset_classifer)

            adapt_output = self.adapt(
                task_model,
                supp_graph,
                mode="train",
                updates=n_updates,
            )

            if isinstance(adapt_output, int):
                raise ValueError("Could not recover from NaN inner loss.")

        task_model, logs = adapt_output

        # Test on query set ====================================================
        q_logits = self.forward(
            task_model,
            query_graph,
            "train",
        )

        q_loss = F.cross_entropy(
            q_logits,
            query_graph.y,
            ignore_index=self.ignore_index,
            weight=self.class_weights,
        )

        # Optimization =========================================================
        if torch.any(torch.isnan(q_loss)):
            print(">>> NaN train loss <<<")
            grad_norm = torch.tensor(float("nan"))
            nan_loss = True

        else:
            # Backprop the query loss to the local model
            train_opt.zero_grad()
            q_loss.backward()

            # First-order approximation
            # We're merely adding meta init and adapted task gradients together
            # No grad of grad
            for p_init, p_task in zip(self.model.parameters(), task_model.parameters()):
                # If no need skip
                if p_init.requires_grad is False:
                    continue

                # If grad is empty add local grad
                if p_init.grad is None:
                    p_init.grad = p_task.grad

                # If grad already exists add local grad
                else:
                    p_init.grad += p_task.grad

            # Check gradient norm
            grad_norm = self._get_gradient_norm(self.model)
            train_opt.step()
            nan_loss = False

        # Logging ==============================================================
        logs = {"train/" + k: v for k, v in logs.items()}

        logs.update(
            {
                "train/query_post_loss": q_loss.detach().cpu(),
                "train/grad_norm": grad_norm,
            }
        )

        if batch_idx == 0 or batch_idx % 100 == 0:
            logs.update(
                {
                    # Expensive sums... but better be sure
                    "train/supp_num_nodes": float(query_graph.num_nodes),
                    "train/query_num_nodes": float(query_graph.num_nodes),
                    "train/supp_unmasked": (query_graph.y != -1).sum().float(),
                    "train/query_unmasked": (query_graph.y != -1).sum().float(),
                }
            )

        self.log_dict(
            logs,
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
            prog_bar=True,
        )

        if nan_loss:
            raise KeyboardInterrupt("Nan loss.")

        return q_loss
