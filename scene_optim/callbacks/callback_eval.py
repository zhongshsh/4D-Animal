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
from pytorch3d.renderer.cameras import PerspectiveCameras
from model.pose_models import PoseBase, compute_pose
from model.texture_models import TextureBase
from scene_optim.evaluators import IoUEvaluator, PSNREvaluator, LPIPSEvaluator
from .callback_general import CallbackClass
from smal import SMAL


class CallbackEval(CallbackClass):
    def __init__(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        cameras: PerspectiveCameras,
        smal: SMAL,
        verts_cse_embeddings: torch.Tensor,
        image_size: int,
        device: str = "cuda",
    ):
        """
        Args:
            - images (BACTH, H_rend, W_rend, 3)  TORCHFLOAT32 [0,1]
            - masks (BATCH, IMG_H, IMG_W, 1) TORCHFLOAT32 {0,1}
            - cameras PerspectiveCameras
            - smal
            - verts_cse_embeddings (N_verts, D_CSE) TORCHFLOAT32
            - image_size INT
        """
        self.cameras = cameras
        self.smal = smal
        self.iou_evalr = IoUEvaluator(masks, image_size=image_size, device=device)
        self.psnr_evalr = PSNREvaluator(
            images, masks, verts_cse_embeddings, image_size, device
        )
        self.lpips_evalr = LPIPSEvaluator(
            images, masks, verts_cse_embeddings, image_size, device
        )

    def call(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        dino_feature: torch.Tensor,
        texture_model: TextureBase,
        return_individual: bool = False,
    ) -> tuple[dict[str, float], str]:

        with torch.no_grad():
            vertices, _, faces = compute_pose(
                self.smal, pose_model, X_ind, X_ts, dino_feature
            )
            l_iou = self.iou_evalr.evaluate(
                pose_model,
                X_ind,
                X_ts,
                dino_feature,
                self.smal,
                self.cameras[X_ind],
                vertices,
                faces,
            )
            l_psnr, l_psnrm = self.psnr_evalr.evaluate(
                pose_model,
                X_ind,
                X_ts,
                dino_feature,
                self.smal,
                self.cameras[X_ind],
                texture_model,
                vertices,
                faces,
            )
            l_lpips = self.lpips_evalr.evaluate(
                pose_model,
                X_ind,
                X_ts,
                dino_feature,
                self.smal,
                self.cameras[X_ind],
                texture_model,
            )

        l_iou_mean = torch.mean(l_iou).item()
        l_psnr_mean = torch.mean(l_psnr).item()
        l_psnrm_mean = torch.mean(l_psnrm).item()
        l_lpips_mean = torch.mean(l_lpips).item()

        percentile = int(0.05 * len(l_iou))
        l_iou_sorted, _ = torch.sort(l_iou)
        iou_w5 = torch.mean(l_iou_sorted[:percentile]).item()

        l_psnr_sorted, _ = torch.sort(l_psnr)
        psnr_w5 = torch.mean(l_psnr_sorted[:percentile]).item()

        str_log = "IoU: {:.3f} IoUw5: {:.3f} PSNR: {:.3f} PSNRw5: {:.3f} PSNRM: {:.3f} LPIPS: {:.3f}".format(
            l_iou_mean, iou_w5, l_psnr_mean, psnr_w5, l_psnrm_mean, l_lpips_mean
        )

        if return_individual:
            return {
                "l_iou": l_iou.cpu(),
                "l_psnr": l_psnr.cpu(),
                "l_psnrm": l_psnrm.cpu(),
                "l_lpips": l_lpips.cpu(),
            }, str_log
        else:
            return {
                "l_iou": l_iou_mean,
                "l_psnr": l_psnr_mean,
                "l_psnrm": l_psnrm_mean,
                "l_lpips": l_lpips_mean,
            }, str_log
