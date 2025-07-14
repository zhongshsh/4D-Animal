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
from pytorch3d.structures import Meshes
from smal import SMAL
from losses import mesh_laplacian_smoothing
from model.texture_models import TextureBase
from model.pose_models import PoseBase, compute_pose


class LossLaplacianReg:

    def __init__(self, smal: SMAL, device: str):
        self.smal = smal
        self.device = device

    def forward_batch(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        dino_feature: torch.Tensor,
        texture_model: TextureBase,
        vertices: torch.Tensor | None = None,
        faces: torch.Tensor = None,
    ) -> torch.Tensor:

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(
                self.smal, pose_model, X_ind, X_ts, dino_feature
            )

        return mesh_laplacian_smoothing(Meshes(vertices, faces.detach()))
