import math
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        feature_type: str,
        compression: str,
        vocab_size: int,
        compressed_size: int,
        p_mask_token: float = 0.0,
        p_dropout: float = 0.5,
        out_dim: int = 2,
    ):
        super().__init__()

        self.feature_type = feature_type
        self.compression = compression
        self.p_mask_token = p_mask_token
        self.p_dropout = p_dropout
        self.out_dim = out_dim

        if self.feature_type == "one-hot":
            self.vocab_size = vocab_size

            if self.compression in {"learned", "random"}:
                # If learning or using random projection
                # Projection method is a linear layer
                self.compressed_size = compressed_size
                self.compressor = nn.Linear(
                    in_features=self.vocab_size,
                    out_features=self.compressed_size,
                )

                self._init_linear_compressor()

                if self.compression == "random":
                    for p in self.compressor.parameters():
                        p.requires_grad = False

            elif self.compression_method == "none" or self.compression_method is None:
                # If not compressing at all, compression is just identity
                self.compressed_size = self.vocab_size
                self.compressor = nn.Identity()

        if self.feature_type == "lm-embeddings":
            self.vocab_size = None
            self.compressor = AutoModel.from_pretrained(
                self.compression,
                cache_dir="./resources/",
            )
            self.compressor_config = AutoConfig.from_pretrained(
                self.compression,
                cache_dir="./resources/",
            ).to_dict()

            self.mask_token_id = AutoTokenizer.from_pretrained(
                self.compressor.config._name_or_path,
                cache_dir="./resources/",
            ).mask_token_id

            # I hope that huggingface standardizes configs one day
            if "dim" in self.compressor_config:
                self.compressed_size = self.compressor_config["dim"]
            elif "hidden_size" in self.compressor_config:
                self.compressed_size = self.compressor_config["hidden_size"]

            for p in self.compressor.parameters():
                p.requires_grade = False

        self.clf = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Linear(
                in_features=self.compressed_size,
                out_features=self.compressed_size // 2,
            ),
            nn.ReLU(),
            nn.Dropout(p=self.p_dropout),
            nn.Linear(
                in_features=self.compressed_size // 2,
                out_features=self.out_dim,
            ),
        )

        self.hparams = {
            "feature_type": self.feature_type,
            "compression": self.compression,
            "vocab_size": self.vocab_size,
            "compressed_size": self.compressed_size,
            "p_mask_token": self.p_mask_token,
            "p_dropout": self.p_dropout,
            "out_dim": self.out_dim,
        }

    def _init_linear_compressor(self):
        # Using `kaiming_uniform` but with local seed
        # Seed stands outside of seed used for dataset processing
        gain = nn.init.calculate_gain("relu", 0)
        std = gain / math.sqrt(self.compressor.in_features)
        bound = math.sqrt(3.0) * std

        rng = torch.Generator().manual_seed(0)
        with torch.no_grad():
            self.compressor.weight.uniform_(-bound, bound, generator=rng)
            self.compressor.bias.zero_()

    @property
    def device(self):
        return next(self.parameters()).device

    def compress(self, x):
        if self.feature_type == "one-hot":
            if self.training:
                mask = torch.rand(x["model_input"].shape) < self.p_mask_token

                x["model_input"] = torch.where(
                    mask.to(self.device),
                    0.0,
                    x["model_input"],
                )

            if self.compression == "random":
                with torch.inference_mode():
                    out = self.compressor(x["model_input"])

            else:
                out = self.compressor(x["model_input"])

        elif self.feature_type == "lm-embeddings":
            if self.training:
                # Allow masking on any element that:
                #   - is used in attention
                #   - is not a special token
                maskable = (
                    torch.bitwise_and(x["attention_mask"], 1 - x["special_tokens_mask"])
                ).bool()

                mask = maskable * (
                    torch.rand_like(x["input_ids"].float()) < self.p_mask_token
                )

                x["input_ids"] = torch.where(
                    mask,
                    self.mask_token_id,
                    x["input_ids"],
                )

            with torch.inference_mode():
                out = self.compressor(
                    input_ids=x["input_ids"],
                    attention_mask=x["attention_mask"],
                    output_hidden_states=True,
                )

                out = torch.mean(
                    out.hidden_states[-2],
                    dim=1,
                )

        return out

    def forward(self, x):
        x = self.compress(x)

        logits = self.clf(x)

        return logits

    def get_state_dict(self):
        if self.feature_type == "lm-embeddings":
            # In case we're using LM embeddings, the compression is pre-trained
            # We get it from a standalone file, so no need to duplicate it
            return OrderedDict(
                [
                    (module_name, param_tensor)
                    for module_name, param_tensor in self.state_dict().items()
                    if "compressor" not in module_name
                ]
            )
        else:
            # Otherwise, the compression module is absolutely necessary to save
            return self.state_dict()
