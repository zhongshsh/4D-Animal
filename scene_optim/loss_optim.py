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
from pytorch3d.renderer import PerspectiveCameras
import scene_optim.losses_optim as lopt
from smal import SMAL
from rendering.renderer import Renderer
from model.pose_models import PoseBase, compute_pose
from omegaconf import ListConfig


def piecewise_schedule(boundaries, values):
    assert len(boundaries) + 1 == len(values)

    def schedule_fn(epoch):
        for b, v in zip(boundaries, values):
            if epoch < b:
                return v
        return values[-1]

    return schedule_fn


class LossOptim:

    def __init__(
        self,
        smal: SMAL,
        device: str,
        cameras: PerspectiveCameras,
        images: torch.Tensor,
        masks: torch.Tensor,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        cse_keypoints_xy: torch.Tensor,
        cse_keypoints_vert_id: torch.Tensor,
        cse_valid_indices: torch.Tensor,
        cse_keypoints_max_indice: torch.Tensor,
        sparse_keypoints: torch.Tensor,
        sparse_keypoints_scores: torch.Tensor,
        config_chamfer: dict,
        config_color: dict,
        image_size: int,
        batch_size: int,
        weight_dict: dict,
        tracking_points: torch.Tensor,
        tracking_visibles: torch.Tensor,
        mask_keypoints_xy: torch.Tensor,
        mask_keypoints_vert_id: torch.Tensor,
        part_masks: torch.Tensor,
    ):
        """
        Args:
            - cameras PerspectiveCameras (BATCH,)

            - images (BATCH, H_render, W_render, 3) TORCHFLOAT32 [0,1]
            - mask (BATCH, H_render, W_render, 1) TORCHINT32 {0,1}

            - keyp2d (BATCH, N_kps, 2) TORCHFLOAT32 [0,1]
            - cse_embeddings (BATCH, N_kps, 16) TORCHFLOAT32
            - closest_verts (BATCH, N_kps) TORCHINT64

            - valid_indices_cse (N_valid,) TORCHINT32
            - cse_imgs (N_valid, D_CSE, H_IMG, W_IMG) TORCHFLOAT32
            - cse_masks (N_valid, H_IMG, W_IMG, 1) TORCHINT32
            - smal_verts_cse_embedding (N_verts, D_CSE) TORCHFLOAT32

            - sparse_keypoints (BATCH, N_KPS, 2) TORCHFLOAT32 [0,1]
            - sparse_keypoints_scores (BATCH, N_KPS, 1) TORCHFLOAT32 [0,1]

        """
        self.smal = smal
        self.device = device
        self.renderer = Renderer(image_size=image_size)

        self.weight_dict = {}
        for k, v in weight_dict.items():
            if isinstance(v, ListConfig):
                v = list(v)
                self.weight_dict[k] = piecewise_schedule(
                    v[0], v[1]
                )  # v: ([1000], [200.0, 800.0]) 0-1000 epoch: 200.0; > 1000: 800.0
            elif isinstance(v, float) or isinstance(v, int):
                self.weight_dict[k] = v
            else:
                raise ValueError(f"Invalid weight type for '{k}': {v}")

        self.epoch = 0

        self.loss_dict = {
            "l_optim_tracking": (
                lopt.LossOptimTracking(
                    self.smal,
                    self.device,
                    tracking_points,
                    tracking_visibles,
                    cameras,
                    self.renderer,
                    image_size,
                )
                if self.get_weight("l_optim_tracking") > 0
                else None
            ),
            "l_optim_part_kp": (
                lopt.LossOptimPartKp(
                    self.smal,
                    self.device,
                    mask_keypoints_xy,
                    mask_keypoints_vert_id,
                    cameras,
                    self.renderer,
                    image_size,
                )
                if self.get_weight("l_optim_part_kp") > 0
                else None
            ),
            "l_optim_part_chamfer": (
                lopt.LossOptimPartChamfer(
                    self.smal,
                    self.device,
                    part_masks,
                    cameras,
                    self.renderer,
                    image_size,
                )
                if self.get_weight("l_optim_part_chamfer") > 0
                else None
            ),
            "l_optim_color": lopt.LossOptimColor(
                self.smal,
                self.device,
                self.renderer,
                cameras,
                images,
                masks,
                **config_color,
            ),
            "l_optim_chamfer": (
                lopt.LossOptimChamfer(
                    masks,
                    self.smal,
                    self.device,
                    cameras,
                    self.renderer,
                    **config_chamfer,
                )
                if self.get_weight("l_optim_chamfer") > 0
                else None
            ),
            "l_optim_cse_kp": (
                lopt.LossOptimCSEKp(
                    self.smal,
                    self.device,
                    cse_keypoints_xy,
                    cse_keypoints_vert_id,
                    cse_valid_indices,
                    cse_keypoints_max_indice,
                    cameras,
                    image_size,
                )
                if self.get_weight("l_optim_cse_kp") > 0
                else None
            ),
            "l_optim_sparse_kp": (
                lopt.LossOptimSparseKp(
                    self.smal,
                    self.device,
                    sparse_keypoints,
                    sparse_keypoints_scores,
                    cameras,
                    image_size,
                )
                if self.get_weight("l_optim_sparse_kp") > 0
                else None
            ),
            "l_laplacian_reg": (
                lopt.LossLaplacianReg(self.smal, self.device)
                if self.get_weight("l_laplacian_reg") > 0
                else None
            ),
            "l_tv_reg": (
                lopt.LossOptimTVReg(X_ind, X_ts)
                if self.get_weight("l_tv_reg") > 0
                else None
            ),
            "l_arap_reg": (
                lopt.LossOptimArap(self.smal, batch_size, self.device)
                if self.get_weight("l_arap_reg") > 0
                else None
            ),
            "l_arap_fast_reg": (
                lopt.LossOptimArapFast(self.smal, batch_size, self.device)
                if self.get_weight("l_arap_fast_reg") > 0
                else None
            ),
        }

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_weight(self, key: str) -> float:
        weight_value = self.weight_dict.get(key, 0.0)
        if isinstance(weight_value, (float, int)):
            return weight_value
        elif callable(weight_value):
            return weight_value(self.epoch)
        else:
            raise ValueError(f"Invalid weight value for '{key}': {weight_value}")

    def forward_batch(self, *args, **kwargs) -> dict[str, torch.Tensor]:

        (
            vertices,
            faces,
            X_betas,
            X_betas_limbs,
            X_global_pose,
            X_transl,
            X_vertices_off,
        ) = self.compute_pose(*args, True)

        if vertices.isnan().sum() > 0:
            raise Exception("mesh vertices coords are NaN")

        kwargs.update({"vertices": vertices, "faces": faces})

        return_dict = {"total": 0.0}

        # return_dict["total"] += self.weight_dict['l_parameter'] * (torch.mean(X_betas**2, -1) + torch.mean(X_betas_limbs**2, -1) + torch.mean(X_global_pose**2, -1) + torch.mean(X_transl**2, -1) + torch.mean(X_vertices_off**2, -1)).unsqueeze(-1)
        value = (
            torch.mean(X_betas**2, -1)
            + torch.mean(X_betas_limbs**2, -1)
            + torch.mean(X_vertices_off**2, -1)
        ).unsqueeze(-1)
        return_dict["l_parameter"] = value
        return_dict["total"] += self.get_weight("l_parameter") * value

        for l_key in self.loss_dict:
            weight = self.get_weight(l_key)
            if weight > 0:
                value = self.loss_dict[l_key].forward_batch(*args, **kwargs)
                return_dict[l_key] = value
                return_dict["total"] += weight * value

        return return_dict

    def compute_pose(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        dino_feature: torch.Tensor,
        return_parameter=False,
        *args,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if return_parameter:
            (
                vertices,
                _,
                faces,
                X_betas,
                X_betas_limbs,
                X_global_pose,
                X_transl,
                X_vertices_off,
            ) = compute_pose(
                self.smal, pose_model, X_ind, X_ts, dino_feature, return_parameter
            )
            return (
                vertices,
                faces,
                X_betas,
                X_betas_limbs,
                X_global_pose,
                X_transl,
                X_vertices_off,
            )

        vertices, _, faces = compute_pose(
            self.smal, pose_model, X_ind, X_ts, dino_feature
        )
        return vertices, faces

    def no_shape_training(self) -> bool:
        """
        Should we train both models (pose, texture) or only texture model.
        """
        for key in [
            "l_optim_chamfer",
            "l_optim_cse_kp",
            "l_optim_sparse_kp",
            "l_laplacian_reg",
            "l_tv_reg",
            "l_arap_reg",
            "l_arap_fast_reg",
        ]:
            if self.get_weight(key) > 0:
                return False
        return True
