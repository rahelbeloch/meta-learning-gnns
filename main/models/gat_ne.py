from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from models.base_meta_learner import BaseMetaLearner

from models.sparse_gat import SparseGatNet
from models.pointwise_baseline import PointwiseMLP


class GatNonEpisodic(BaseMetaLearner):
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

    def clone(self, reset_classifier: bool = False, output_dim: int = None):
        local_model = deepcopy(self.model)

        if reset_classifier:
            local_model.reset_classifier(output_dim)

        return local_model

    def adapt(self, model, graph, mode, updates=None, lr=None, class_weights=None):
        # For non-episodic, we use adaptation exclusively during evaluation
        if updates is None:
            updates = self.eval_n_inner_updates
        if lr is None:
            lr = self.eval_lr_inner
        if class_weights is None:
            class_weights = self.eval_class_weights

        optimizer = SGD(model.parameters(), lr=lr)

        losses = []
        for _ in range(updates):
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
                break

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

    def training_step(self, graph, batch_idx):
        train_opt = self.optimizers()

        logits = self.forward(
            self.model,
            graph,
            "train",
        )

        loss = F.cross_entropy(
            logits,
            graph.y,
            ignore_index=self.ignore_index,
            weight=self.class_weights,
        )

        if torch.isnan(loss):
            print("\n\n>>> NAN LOSS <<<\n\n")
            grad_norm = torch.tensor(float("nan"))
            nan_loss = True

        else:
            train_opt.zero_grad()
            self.manual_backward(loss)
            grad_norm = self._get_gradient_norm(self.model)
            train_opt.step()
            nan_loss = False

        # Logging ==============================================================
        self.log_dict(
            {
                "train/ne_loss": loss.detach(),
                "train/grad_norm": grad_norm,
            },
            on_step=True,
            on_epoch=False,
            add_dataloader_idx=False,
            prog_bar=True,
        )

        if batch_idx == 0 or batch_idx % 100 == 0:
            self.log_dict(
                {
                    # Expensive sums... but better be sure
                    "train/supp_num_nodes": float(graph.num_nodes),
                    "train/query_num_nodes": float("nan"),
                    "train/supp_unmasked": (graph.y != -1).sum().float(),
                    "train/query_unmasked": float("nan"),
                },
                on_step=True,
                on_epoch=False,
                add_dataloader_idx=False,
                prog_bar=True,
            )

        if nan_loss:
            raise KeyboardInterrupt("Nan loss.")

        return loss
