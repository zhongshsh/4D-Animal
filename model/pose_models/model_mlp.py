# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ------------------------------------------------------------
# Modified and extended by CCVL, JHU
# Copyright (c) 2025 CCVL, JHU
# Licensed under the same terms as the original license.
# For details, see the LICENSE file.
# ------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Literal
from model.positional_embedding import PositionalEmbedding
from model.pose_models.model_util import partition_index, PoseBase


class MLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        nb_layers: int,
        embedding_dimension: int,
        output_dimension: int,
        activ: Literal["softplus", "relu", "leaky_relu"] = "relu",
    ):
        super().__init__()

        activations = {
            "softplus": nn.Softplus(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activ not in activations:
            raise ValueError(f"Unknown activation: {activ}")
        activation = activations[activ]

        self.input_dimension = input_dimension
        self.nb_layers = nb_layers
        self.embedding_dimension = embedding_dimension
        self.output_dimension = output_dimension

        if nb_layers == 1:
            layers = [nn.Linear(input_dimension, output_dimension)]
        else:
            layers = [nn.Linear(input_dimension, embedding_dimension), activation]
            for _ in range(nb_layers - 2):
                layers.extend(
                    [nn.Linear(embedding_dimension, embedding_dimension), activation]
                )
            layers.append(nn.Linear(embedding_dimension, output_dimension))

        self.model = nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def forward_partition(
        self, input: torch.Tensor, chunksize: int = 10_000
    ) -> torch.Tensor:
        BATCH, _ = input.shape
        partitions = partition_index(BATCH, chunksize)
        return torch.cat([self.model(input[a:b]) for (a, b) in partitions], dim=0)


class PoseMLP(PoseBase):
    def __init__(
        self,
        mlp_n_layers: int,
        mlp_hidd_dim: int,
        mlp_activation: Literal["softplus", "relu", "leaky_relu"],
        pos_embedding_dim: int,
        pos_embedding_mode: Literal["power", "lin"],
        init_betas: torch.Tensor | None = None,
        init_betas_limbs: torch.Tensor | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.tag = "PoseMLP"

        self.positional_embedder = PositionalEmbedding(
            L=pos_embedding_dim, mode=pos_embedding_mode
        )

        activations = {
            "softplus": nn.Softplus(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if mlp_activation not in activations:
            raise ValueError(f"Unknown activation: {mlp_activation}")
        activation = activations[mlp_activation]

        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_dim = mlp_hidd_dim
        self.num_patches = 3600
        dino_feature_dim = 384
        pos_dim = pos_embedding_dim

        self.input_dim = pos_dim + dino_feature_dim

        # Vertices model
        self.vertices_reduction_layer = nn.Linear(
            self.input_dim, mlp_hidd_dim, bias=bias
        )
        self.vertices_model = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(mlp_hidd_dim, mlp_hidd_dim, bias=bias), activation
                )
                for _ in range(mlp_n_layers)
            ],
            nn.Linear(mlp_hidd_dim, 5901, bias=bias),
        )
        nn.init.zeros_(self.vertices_model[-1].weight)
        if bias:
            nn.init.zeros_(self.vertices_model[-1].bias)

        # Shape model
        self.shape_reduction_layer = nn.Linear(self.input_dim, mlp_hidd_dim, bias=bias)
        self.shape_model = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(mlp_hidd_dim, mlp_hidd_dim, bias=bias), activation
                )
                for _ in range(mlp_n_layers)
            ],
            nn.Linear(mlp_hidd_dim, 30 + 7, bias=bias),
        )

        # Pose model
        self.pose_reduction_layer = nn.Linear(self.input_dim, mlp_hidd_dim, bias=bias)
        self.pose_model = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(mlp_hidd_dim, mlp_hidd_dim, bias=bias), activation
                )
                for _ in range(mlp_n_layers)
            ],
            nn.Linear(mlp_hidd_dim, 3 + 6 + 34 * 6, bias=bias),
        )

    def _forward_base(
        self,
        x: torch.Tensor,
        dino_feature: torch.Tensor,
        reduction_layer: nn.Linear,
        model: nn.Sequential,
    ) -> torch.Tensor:
        if x.device != dino_feature.device:
            x = x.to(dino_feature.device)
        if x.dim() == 1:
            x = x.view(-1, 1)

        embedded_x = self.positional_embedder(x)
        embedded_x_exp = embedded_x.unsqueeze(1).expand(-1, self.num_patches, -1)
        combined_feature = torch.cat([embedded_x_exp, dino_feature], dim=-1)
        reduced_feature = reduction_layer(combined_feature)
        aggregated_feature = reduced_feature.mean(dim=1)
        return model(aggregated_feature)

    def shape_forward(
        self, x: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_base(
            x, dino_feature, self.shape_reduction_layer, self.shape_model
        )

    def pose_forward(self, x: torch.Tensor, dino_feature: torch.Tensor) -> torch.Tensor:
        return self._forward_base(
            x, dino_feature, self.pose_reduction_layer, self.pose_model
        )

    def vertices_forward(
        self, x: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_base(
            x, dino_feature, self.vertices_reduction_layer, self.vertices_model
        )

    def compute_pose(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.pose_forward(X_ts, dino_feature)[..., 9:]

    def compute_orient(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.pose_forward(X_ts, dino_feature)[..., 3:9]

    def compute_transl(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ):
        pass  # To be implemented

    def compute_betas(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.shape_forward(X_ts, dino_feature)[..., :30]

    def compute_betas_limbs(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.shape_forward(X_ts, dino_feature)[..., 30:]

    def compute_vertices_off(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.vertices_forward(X_ts, dino_feature)

    def compute_global(
        self, X_ind: torch.Tensor, X_ts: torch.Tensor, dino_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.pose_forward(X_ts, dino_feature)
