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
from pytorch3d.renderer.cameras import PerspectiveCameras, get_screen_to_ndc_transform
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase
from smal.smal_torch import SMAL
from rendering.renderer import Renderer
import json


def sample_point_cloud_from_mask(
    masks: torch.Tensor,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """
    Args:
        - masks (BATCH, H, W) TORCHFLOAT32 or TORCHINT (binary mask)
    Returns:
        - pc_gt (BATCH, MAX_L, 2) TORCHFLOAT32 (follows SCREEN coordinate system)
        - max_l_gt: int
        - length_gt: (BATCH,) TORCHINT
    """
    BATCH, H, W = masks.shape
    device = masks.device

    X_gt = masks.nonzero()  # (N_PTS, 3) — each point is (batch_idx, y, x)

    count_gt, length_gt = torch.unique(X_gt[:, 0], return_counts=True)
    max_l_gt = torch.max(length_gt)
    pc_gt = torch.zeros((BATCH, max_l_gt, 2), dtype=torch.float32, device=device)
    length_out = torch.zeros((BATCH,), dtype=length_gt.dtype, device=device)
    for k, batch_idx in enumerate(count_gt):
        pc_gt[batch_idx, : length_gt[k]] = X_gt[X_gt[:, 0] == batch_idx][
            :, [2, 1]
        ]  # (x, y)
        length_out[batch_idx] = length_gt[k]

    return pc_gt, max_l_gt, length_out


class LossOptimPartChamfer:
    """
    Args:
        - cameras PerspectiveCameras (BATCH,) [DEVICE]
        - keyp2d (BATCH, N_kps, 2) TORCHFLOAT32 [0,1]
        - closest_verts (BATCH, N_kps) TORCHINT64
        - valid_indices_cse (N_valid,) TORCHINT32 GLOBAL_INDICES
        - max_ind_csekp (BATCH,) TORCHINT64
        - smal_verts_cse_embedding (N_verts, D_CSE) TORCHFLOAT32
    """

    def __init__(
        self,
        smal: SMAL,
        device: str,
        part_masks: torch.Tensor,
        cameras: PerspectiveCameras,
        renderer: Renderer,
        image_size: int,
    ):

        self.smal = smal
        self.device = device
        self.part_masks = part_masks
        self.cameras = cameras
        self.image_size = image_size

        BATCH, H, W = self.part_masks.shape

        self.cameras = cameras
        self.renderer = renderer
        self.weight_invisible = 0.05

        with open("config/smal_revised.json", "r") as f:
            self.smal_vert = json.load(f)
        self.smal_vert["leg"] = (
            self.smal_vert["front_left_leg"]
            + self.smal_vert["front_right_leg"]
            + self.smal_vert["back_left_leg"]
            + self.smal_vert["back_right_leg"]
        )

        # Mapping from semantic part mask ID to part name
        self.idx2partname = {11: "head", 12: "torso", 13: "leg", 14: "tail"}

        smal_weight = {
            "head": 50.0,
            "tail": 0.01,
            "front_left_leg": 0.001,
            "front_right_leg": 0.001,
            "back_left_leg": 0.001,
            "back_right_leg": 0.001,
            "leg": 0.001,
            "torso": 1,
        }
        vertex_to_body_part = {}
        for body_part, indices in self.smal_vert.items():
            for idx in indices:
                vertex_to_body_part[idx] = smal_weight[body_part]

        max_idx = max(vertex_to_body_part.keys(), default=0) + 1
        self.vertex_weight = torch.full(
            (max_idx,), smal_weight["torso"], dtype=torch.float, device=self.device
        )

        indices = torch.tensor(list(vertex_to_body_part.keys()), device=self.device)
        weights = torch.tensor(list(vertex_to_body_part.values()), device=self.device)
        self.vertex_weight[indices] = weights

    def compute_visibility_weights(self, vertices, faces, part_verts_idx, X_ind):
        """
        Computes visibility weights for part-specific mesh vertices.

        Returns:
            weights: (B, N_part_verts), 1.0 if visible, else weight_invisible
        """
        B, V, _ = vertices.shape
        vertices_visibility_map, faces_visibility_map = (
            self.renderer.get_visibility_map(vertices, faces, self.cameras[X_ind])
        )  # (B, V, 1)

        visible_verts = torch.gather(
            vertices_visibility_map.squeeze(-1), dim=1, index=part_verts_idx
        )  # (B, N_part_verts)

        weights = torch.where(visible_verts > 0, 1.0, self.weight_invisible)
        return weights

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
        """
        Computes Chamfer distance loss between each part's predicted mesh region
        and the corresponding ground truth part mask.

        Returns:
            loss: (B, 1)
        """
        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(
                self.smal, pose_model, X_ind, X_ts, dino_feature
            )

        B = vertices.shape[0]
        part_losses = []

        for part_id, part_name in self.idx2partname.items():
            if part_name not in self.smal_vert:
                continue

            # Get global vertex indices for this part
            part_verts_global = torch.tensor(
                self.smal_vert[part_name], device=self.device, dtype=torch.long
            )
            part_verts_idx = part_verts_global.unsqueeze(0).expand(
                B, -1
            )  # (B, N_part_verts)

            # Extract the binary mask for this part
            mask = (self.part_masks[X_ind] == part_id).float()  # (B, H, W)

            # Filter empty part
            if mask.sum() == 0:
                continue

            pc_gt, max_l_gt, length_gt = sample_point_cloud_from_mask(
                mask > 0.5
            )  # (B, L, 2)

            # Filter empty part
            valid_mask = length_gt > 0
            if valid_mask.sum() == 0:
                continue

            # Transform ground-truth 2D points to NDC
            screen_to_ndc = get_screen_to_ndc_transform(
                self.cameras[X_ind],
                with_xyflip=True,
                image_size=(self.image_size, self.image_size),
            )
            pc_gt_ndc = screen_to_ndc.transform_points(
                torch.cat(
                    [pc_gt, torch.ones((B, max_l_gt, 1), device=self.device)], dim=-1
                )
            )[
                ..., :2
            ]  # (B, L, 2)

            # Gather corresponding 3D mesh vertices for the part
            part_verts = torch.gather(
                vertices, dim=1, index=part_verts_idx.unsqueeze(-1).expand(-1, -1, 3)
            )  # (B, N_part_verts, 3)
            part_verts_weight = self.vertex_weight[part_verts_idx]

            # Project 3D part vertices to 2D NDC
            part_verts_ndc = self.cameras[X_ind].transform_points_ndc(part_verts)[
                ..., :2
            ]  # (B, N_part_verts, 2)

            # Compute visibility-based weights
            weights = self.compute_visibility_weights(
                vertices, faces, part_verts_idx, X_ind
            )  # (B, N_part_verts)
            weights = (
                weights[valid_mask] * part_verts_weight[valid_mask]
            )  # add semantic part weight

            # Chamfer distance between projected mesh part and ground-truth mask
            loss_x, loss_y = chamfer_distance(
                x=part_verts_ndc[valid_mask],
                y=pc_gt_ndc[valid_mask],
                y_lengths=length_gt[valid_mask],
                batch_reduction=None,
                point_reduction=None,
                norm=2,
            )[
                0
            ]  # (B, N_part_verts), (B, N_mask_points)

            chamf_x = (weights * loss_x).sum(1) / weights.sum(1)
            chamf_y = loss_y.sum(1) / length_gt[valid_mask].clamp(min=1)

            part_loss = torch.zeros((B, 1), device=self.device)
            part_loss[valid_mask] = (chamf_x + chamf_y).reshape((-1, 1))
            part_losses.append(part_loss)

        if len(part_losses) == 0:
            return 0.0

        # Average over all part losses
        total_loss = torch.stack(part_losses, dim=0).mean(0)  # (B, 1)
        return total_loss


class LossOptimPartKp:
    """
    Args:
        - cameras PerspectiveCameras (BATCH,) [DEVICE]
        - keyp2d (BATCH, N_kps, 2) TORCHFLOAT32 [0,1]
        - closest_verts (BATCH, N_kps) TORCHINT64
        - valid_indices_cse (N_valid,) TORCHINT32 GLOBAL_INDICES
        - max_ind_csekp (BATCH,) TORCHINT64
        - smal_verts_cse_embedding (N_verts, D_CSE) TORCHFLOAT32
    """

    def __init__(
        self,
        smal: SMAL,
        device: str,
        mask_keypoints_xy: torch.Tensor,
        mask_keypoints_vert_id: torch.Tensor,
        cameras: PerspectiveCameras,
        renderer: Renderer,
        image_size: int,
    ):

        self.smal = smal
        self.device = device
        self.mask_keypoints_xy = mask_keypoints_xy
        self.mask_keypoints_vert_id = mask_keypoints_vert_id
        self.cameras = cameras
        self.image_size = image_size

        BATCH, N_KPS_MAX, _ = self.mask_keypoints_xy.shape

        self.cameras = cameras
        self.renderer = renderer
        self.weight_invisible = 0.05

        with open("config/smal_revised.json", "r") as f:
            smal_vert = json.load(f)
        smal_weight = {
            "head": 50.0,
            "tail": 0.1,
            "front_left_leg": 0.001,
            "front_right_leg": 0.001,
            "back_left_leg": 0.001,
            "back_right_leg": 0.001,
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

        # Compute weights
        vertex_visibility_map = torch.where(
            vertex_visibility_map > 0, 1.0, self.weight_invisible
        )
        faces_visibility_map = torch.where(
            faces_visibility_map > 0, 1.0, self.weight_invisible
        )

        return vertex_visibility_map, faces_visibility_map

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
        BATCH = X_ind.shape[0]

        # L2 loss
        total_loss = torch.zeros((X_ind.shape[0], 1), device=self.device)

        if BATCH == 0:
            return total_loss

        if (vertices is None) or (faces is None):
            vertices, _, faces = compute_pose(
                self.smal, pose_model, X_ind, X_ts, dino_feature
            )

        # Compute visibility weights for all sampled points on the mesh
        vertex_visibility_map, faces_visibility_map = self.compute_visibility_weights(
            X_ind, vertices, faces
        )  # (BATCH_P, N_Vertice)

        # Get the 3d coordinates of points on the smal mesh correspoding to cse_keypoints_xy -> (BATCH_P, N_KPS, 3)
        vert_id_expanded = (
            self.mask_keypoints_vert_id[X_ind].unsqueeze(-1).expand(-1, -1, 3)
        )
        visibility_id_expanded = self.mask_keypoints_vert_id[
            X_ind
        ]  # (BATCH_P, N_Vertice)
        keyp_3d = torch.gather(vertices, 1, vert_id_expanded)  # (BATCH_P, N_Vertice, 3)
        keyp_3d_masks = torch.gather(
            vertex_visibility_map, 1, visibility_id_expanded
        )  # (BATCH_P, N_Vertice)
        keyp_3d_weight = self.vertex_weight[
            visibility_id_expanded
        ]  # (BATCH_P, N_Vertice)

        # project on screen -> (BATCH_P, N_KPS, 2)
        keyp_3dp = (
            self.cameras[X_ind]
            .transform_points_screen(
                keyp_3d, with_xyflip=True, image_size=(self.image_size, self.image_size)
            )[..., :2]
            .reshape((BATCH, -1, 2))
            / self.image_size
        )

        loss_per_kp = (
            self.mask_keypoints_xy[X_ind] - keyp_3dp
        ) ** 2  # (BATCH_P, N_Vertice, 2)
        loss_per_kp = loss_per_kp * keyp_3d_masks.unsqueeze(
            -1
        )  # filter not visible points
        loss_per_kp = loss_per_kp * keyp_3d_weight.unsqueeze(-1)  # weight points
        total_loss = loss_per_kp.sum(axis=2).sqrt().sum(axis=1).reshape((-1, 1))

        return total_loss
