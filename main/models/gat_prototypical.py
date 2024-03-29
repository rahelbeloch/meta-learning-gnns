from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from models.base_meta_learner import BaseMetaLearner
from models.sparse_gat import SparseGatNet
from models.pointwise_baseline import PointwiseMLP


class GatPrototypical(BaseMetaLearner):
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
        # No impact on ProtoMAML, so just set to True
        self.reset_classifer = True

    def clone(self, reset_classifier: bool = False, output_dim: int = None):
        task_model = deepcopy(self.model)

        return task_model

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

        # ======================================================================
        # CLF Initialization
        # ======================================================================
        # Prototypes are extracted from the meta model
        with torch.set_grad_enabled(mode == "train"):
            # If in eval, this should be done without recording gradients
            # Meta-model will not get updated
            extracted_features = self.model.extract_features(graph.x, graph.edge_index)

            # Compute prototypes
            prototypes = torch.stack(
                [
                    torch.mean(extracted_features[graph.y == l], dim=(0,))
                    for l in range(self.n_classes)
                ]
            )

            # Convert prototypes to linear layer parameters
            init_weight = 2 * prototypes
            init_bias = -torch.pow(torch.norm(prototypes, dim=1), 2)

        # Copy prototype weights to model classification head
        # Detach protopypes from computation graph
        clf_state_dict = model.classifier.state_dict()
        clf_state_dict["weight"] = init_weight.detach().requires_grad_(True)
        clf_state_dict["bias"] = init_bias.detach().requires_grad_(True)
        model.classifier.load_state_dict(clf_state_dict)

        # ======================================================================
        # Inner Loop Adaptation
        # ======================================================================
        # Enable gradients on all layers of the task model
        for p in model.parameters():
            p.requires_grad = True

        # Define an inner loop optimizer
        # Using first-order approximation -> just SGD, no monkeypatching

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

        if updates == 0:
            with torch.no_grad():
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

            losses.append(loss.detach().cpu())

        loss_logs = {
            "supp_pre_loss": losses[0]
            if len(losses) > 0
            else torch.tensor(float("nan")),
            "supp_mean_loss": torch.mean(torch.stack(losses))
            if len(losses) > 0
            else torch.tensor(float("nan")),
        }

        if mode == "train":
            return (model, init_weight, init_bias), loss_logs
        else:
            return model, loss_logs

    def training_step(self, graphs, batch_idx):
        train_opt = self.optimizers()

        # Unpack into support and query batch
        supp_graph, query_graph = graphs

        # Copy model and adapt on support set ==================================
        task_model = self.clone()

        (task_model, init_weight, init_bias), logs = self.adapt(
            task_model,
            supp_graph,
            mode="train",
        )

        # Test on query set ====================================================
        # Stitch the initialization model to the computation graph
        clf_weight = (task_model.classifier.weight - init_weight).detach() + init_weight
        clf_bias = (task_model.classifier.bias - init_bias).detach() + init_bias

        query_features = task_model.extract_features(
            query_graph.x, query_graph.edge_index
        )
        q_logits = F.linear(query_features, weight=clf_weight, bias=clf_bias)

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
                # Or, if grad is empty, skip (e.g. classification head)
                if p_init.requires_grad is False or p_init.grad is None:
                    continue

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

    def eval_step(self, *args, **kwargs):
        return self.episodic_eval_step(*args, **kwargs)
