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
from smal import SMAL
from model.texture_models import TextureBase
from model.pose_models import PoseBase, PoseCanonical, compute_pose
from losses import ArapLossR


class LossOptimArap:

    def __init__(self, smal: SMAL, batch_size: int, device: str):
        """
        ARAP optimizer.
        """
        self.smal = smal
        self.device = device
        self.batch_size = batch_size
        self.arap_loss = None

    def initialization(self, pose_model: PoseBase, dino_feature: torch.Tensor):
        """
        Initialize ARAP loss (with current betas/betas_limbs and canonical pose)
        """
        model_canonical = PoseCanonical(
            N=1,
            betas=pose_model.compute_betas(
                torch.tensor([0]).reshape((1,)),
                torch.tensor([0.0]).reshape((1,)),
                dino_feature,
            )
            .cpu()
            .detach()
            .clone(),
            betas_limbs=pose_model.compute_betas_limbs(
                torch.tensor([0]).reshape((1,)),
                torch.tensor([0.0]).reshape((1,)),
                dino_feature,
            )
            .cpu()
            .detach()
            .clone(),
            vert_off=pose_model.compute_vertices_off(
                torch.tensor([0]).reshape((1,)),
                torch.tensor([0.0]).reshape((1,)),
                dino_feature,
            )
            .cpu()
            .detach()
            .clone(),
        ).to(self.device)
        vertices, _, faces = compute_pose(
            self.smal,
            model_canonical,
            torch.tensor([0]).reshape((1,)),
            torch.tensor([0.0]).reshape((1,)),
            dino_feature,
        )
        vertices = vertices[0].permute(1, 0).to("cpu")
        faces = faces[0].permute(1, 0).to("cpu")

        self.arap_loss = ArapLossR(template=vertices, faces=faces).to(self.device)

    def forward_batch(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        dino_feature: torch.Tensor,
        texture_model: TextureBase,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Initialize ARAP
        if self.arap_loss is None:
            self.initialization(pose_model, dino_feature[[0]])

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(
                self.smal, pose_model, X_ind, X_ts, dino_feature
            )

        # Compute the ARAP loss
        return self.arap_loss.forward(vertices.permute(0, 2, 1)).reshape((-1, 1))
