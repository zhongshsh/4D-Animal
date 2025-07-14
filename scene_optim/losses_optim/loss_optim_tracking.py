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
import json
from pytorch3d.renderer import PerspectiveCameras
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase
from smal.smal_torch import SMAL
from rendering.renderer import Renderer


class LossOptimTracking:
    """
    Args:
        - tracking_points (BATCH, N_KPS, 2) TORCHFLOAT32 XY [0,1]
        - tracking_visibles (BATCH, N_KPS) TORCHFLOAT32
    """

    def __init__(
        self,
        smal: SMAL,
        device: str,
        tracking_points: torch.Tensor,
        tracking_visibles: torch.Tensor,
        cameras: PerspectiveCameras,
        renderer: Renderer,
        image_size: int,
    ):
        self.smal = smal
        self.device = device
        self.cameras = cameras
        self.image_size = image_size
        self.tracking_points = tracking_points
        self.tracking_visibles = tracking_visibles

        with open("config/smal_revised.json", "r") as f:
            smal_vert = json.load(f)

        smal_weight = {
            "head": 50.0,
            "tail": 1,
            "front_left_leg": 1.0,
            "front_right_leg": 1.0,
            "back_left_leg": 1.0,
            "back_right_leg": 1.0,
            "torso": 1,
        }
        vertex_to_body_part = {}
        for body_part, indices in smal_vert.items():
            for idx in indices:
                vertex_to_body_part[idx] = smal_weight[body_part]

        max_idx = max(vertex_to_body_part.keys(), default=0) + 1
        self.vertex_weight = torch.full(
            (max_idx,), smal_weight["torso"], dtype=torch.float, device=self.device
        )

        indices = torch.tensor(list(vertex_to_body_part.keys()), device=self.device)
        weights = torch.tensor(list(vertex_to_body_part.values()), device=self.device)
        self.vertex_weight[indices] = weights

        self.cameras = cameras
        self.renderer = renderer

    def compute_visibility_weights(
        self, X_ind: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns:
            - weights (BATCH, N_PTS) TORCHFLOAT32
        """
        # Compute visibility map for 2D points -> (BATCH, N_vertices,1) TORCHINT32
        vertex_visibility_map, faces_visibility_map = self.renderer.get_visibility_map(
            vertices, faces, self.cameras[X_ind]
        )

        return vertex_visibility_map

    def forward_batch(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        dino_feature: torch.Tensor,
        texture_model: TextureBase,
        vertices=None,
        faces=None,
    ):
        sorted_indices = torch.argsort(X_ind)
        X_ind = X_ind[sorted_indices]
        X_ts = X_ts[sorted_indices]
        dino_feature = dino_feature[sorted_indices]

        # Compute predicted 3D joint position and project on screen
        BATCH = X_ind.shape[0]
        vertices, _, faces = compute_pose(
            self.smal, pose_model, X_ind, X_ts, dino_feature
        )
        vertices_projected = (
            self.cameras[X_ind]
            .transform_points_screen(
                vertices,
                with_xyflip=True,
                image_size=(self.image_size, self.image_size),
            )[..., :2]
            .reshape((BATCH, -1, 2))
            / self.image_size
        )

        # Compute visibility weights for all sampled points on the mesh
        vertex_visibility_map = self.compute_visibility_weights(X_ind, vertices, faces)

        N = 50
        loss_values = []

        for t in range(BATCH):
            frame1_proj, frame2_proj = t, t - 1
            frame1_track, frame2_track = X_ind[t], X_ind[t - 1]

            idxs_track = torch.randperm(self.tracking_points.shape[1])[:N]
            selected_tracking_1 = self.tracking_points[
                frame1_track, idxs_track
            ]  # (N, 2)

            distances = torch.cdist(
                selected_tracking_1.unsqueeze(0),
                vertices_projected[frame1_proj]
                .to(self.tracking_points.dtype)
                .unsqueeze(0),
            ).squeeze(
                0
            )  # (N, num_vertices)
            vertex_visibility_mask = (vertex_visibility_map[frame1_proj] > 0).to(
                distances.dtype
            )  # (num_vertices)
            distances = (
                distances + (1.0 - vertex_visibility_mask) * 1e6
            )  # filter not visible points in frame 1

            idxs_proj = distances.argmin(dim=1)  # (N, )
            selected_keyp_2 = vertices_projected[frame2_proj, idxs_proj]  # (N, 2)
            selected_tracking_2 = self.tracking_points[
                frame2_track, idxs_track
            ]  # (N, 2)

            vertices_weight = self.vertex_weight[idxs_proj]
            loss_per_point = torch.norm(
                selected_tracking_2 - selected_keyp_2, dim=-1
            )  # (N,)
            loss_per_point = loss_per_point * vertices_weight

            tracking_visibility_mask = (
                self.tracking_visibles[frame1_track, idxs_track]
                * self.tracking_visibles[frame2_track, idxs_track]
            )  # (N,)
            loss_per_point = (
                loss_per_point * tracking_visibility_mask
            )  # filter not visible points in frame 2

            valid_points = tracking_visibility_mask.sum()
            if valid_points > 0:
                loss = loss_per_point.sum() / valid_points
                loss_values.append(loss)
            else:
                loss_values.append(torch.tensor(0.0, device=self.device))

        loss_values = torch.stack(loss_values).unsqueeze(-1)
        return loss_values  # batch_size, 1
