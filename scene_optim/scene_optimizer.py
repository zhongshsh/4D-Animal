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
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
import os
import time
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
from smal import SMAL
from model.texture_models import TextureBase, TextureModelTags
from model.pose_models import PoseBase, PoseModelTags
from model.logger import Logger
import scene_optim.loss_optim as lo
from scene_optim.callbacks import CallbackClass, CallbackDataClass
from omegaconf import ListConfig


def prepare_weight_dict_from_config(cfg) -> dict[str, float]:
    weight_dict_optim = {
        "l_optim_part_kp": cfg.l_optim_part_kp,
        "l_optim_part_chamfer": cfg.l_optim_part_chamfer,
        "l_optim_tracking": cfg.l_optim_tracking,
        "l_optim_chamfer": cfg.l_optim_chamfer,
        "l_optim_cse_kp": cfg.l_optim_cse_kp,
        "l_optim_sparse_kp": cfg.l_optim_sparse_kp,
        "l_optim_color": cfg.l_optim_color,
        "l_laplacian_reg": cfg.l_laplacian_reg,
        "l_tv_reg": cfg.l_tv_reg,
        "l_arap_reg": cfg.l_arap_reg,
        "l_arap_fast_reg": cfg.l_arap_fast_reg,
        "l_parameter": cfg.l_parameter,
    }
    return weight_dict_optim


"""
Class to handle the optimization of a single frame
The method implements f : f (frame_i, optim_i_step_j) -> optim_i_step_j+1
"""


class SceneOptimizer:

    def __init__(
        self,
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
        smal: SMAL,
        logger: Logger,
        device: str,
        config: dict,
        tracking_points: torch.Tensor,
        tracking_visibles: torch.Tensor,
        mask_keypoints_xy: torch.Tensor,
        mask_keypoints_vert_id: torch.Tensor,
        part_masks: torch.Tensor,
        renderer: torch.Tensor,
        texture: torch.Tensor,
    ):
        """
        Args:
            - cameras PerspectiveCameras (BATCH,)
            - images (BATCH, H_render, W_render, 3) TORCHFLOAT32 [0,1]
            - masks (BATCH, H_render, W_render, 1) TORCHINT32 {0,1}
            - cse_keypoints_xy (BATCH, N_kps, 2) TORCHFLOAT32 [0,1], (0,0) top-left
            - cse_keypoints_vert_id (BATCH, N_kps) TORCHINT64
            - cse_valid_indices (N_valid,) TORCHINT32
            - cse_max_indices (N_valid,) TORCHINT32
            - sparse_keypoints (BATCH, N_KPS, 2) TORCHFLOAT32 [0,1], (0,0) top-left
            - sparse_keypoints_scores (BATCH, N_KPS, 1) TORCHFLOAT32 [0,1]
        """

        self.image_size = config.image_size
        self.batch_size = config.batch_size

        self.lr_vertices_off = config.lr_vertices_off
        self.lr_pose = config.lr_pose
        self.lr_betas = config.lr_betas
        self.lr_texture = config.lr_texture

        self.factor_pose = config.factor_pose
        self.milestones_pose = config.milestones_pose
        self.factor_texture = config.factor_texture
        self.milestones_texture = config.milestones_texture

        self.clip_grad = config.clip_grad

        self.n_shape_steps = config.n_shape_steps

        assert (
            (len(self.milestones_pose) == 0)
            or all([type(x) == float for x in self.milestones_pose])
            or all([type(x) == int for x in self.milestones_pose])
        )
        assert (
            (len(self.milestones_texture) == 0)
            or all([type(x) == float for x in self.milestones_texture])
            or all([type(x) == int for x in self.milestones_texture])
        )

        self.logger = logger

        # Loss optim parameters
        self.device = device
        self.smal = smal
        self.cameras = cameras
        self.images = images
        self.masks = masks
        self.X_ind = X_ind
        self.X_ts = X_ts
        self.cse_keypoints_xy = cse_keypoints_xy
        self.cse_keypoints_vert_id = cse_keypoints_vert_id
        self.cse_valid_indices = cse_valid_indices
        self.cse_keypoints_max_indice = cse_keypoints_max_indice
        self.sparse_keypoints = sparse_keypoints
        self.sparse_keypoints_scores = sparse_keypoints_scores
        self.tracking_points = tracking_points
        self.tracking_visibles = tracking_visibles
        self.mask_keypoints_xy = mask_keypoints_xy
        self.mask_keypoints_vert_id = mask_keypoints_vert_id
        self.part_masks = part_masks

        self.renderer = renderer
        self.texture = texture
        # Initialize loss-optim
        self.loss_optim = self.update_loss_optim_from_config(config)
        self.config = config

        self.min_log_dict = None

    def update_loss_optim_from_config(self, config: dict) -> lo.LossOptim:

        # Extract the weight of each loss
        weight_dict = prepare_weight_dict_from_config(config)

        # Initialize the loss method
        loss_optim = lo.LossOptim(
            self.smal,
            self.device,
            self.cameras,
            self.images,
            self.masks,
            self.X_ind,
            self.X_ts,
            self.cse_keypoints_xy,
            self.cse_keypoints_vert_id,
            self.cse_valid_indices,
            self.cse_keypoints_max_indice,
            self.sparse_keypoints,
            self.sparse_keypoints_scores,
            config.l_chamfer_config,
            config.l_color_config,
            self.image_size,
            self.batch_size,
            weight_dict,
            self.tracking_points,
            self.tracking_visibles,
            self.mask_keypoints_xy,
            self.mask_keypoints_vert_id,
            self.part_masks,
        )

        return loss_optim

    def _no_shape_training(self, current_epoch: int) -> bool:
        """
        Decide if the optimization on the shape should still be performed
        """
        if current_epoch > self.n_shape_steps:
            return False
        else:
            return self.loss_optim.no_shape_training()

    def get_optimizers_and_schedulers(
        self,
        pose_model: PoseBase,
        texture_model: TextureBase,
        train_vertice_offset: bool,
    ) -> tuple[Optimizer, Optimizer, LRScheduler, LRScheduler]:

        self.shape_model = pose_model.tag
        self.texture_model = texture_model.tag

        # optimizer pose model
        if pose_model.tag == PoseModelTags.PoseMLP:
            """
            ['vertices_off', 'betas', 'betas_limbs', 'positional_embedder.freqs', 'shape_reduction_layer.weight', 'shape_reduction_layer.bias', 'shape_model.0.0.weight', 'shape_model.0.0.bias', 'shape_model.1.0.weight', 'shape_model.1.0.bias', 'shape_model.2.0.weight', 'shape_model.2.0.bias', 'shape_model.3.weight', 'shape_model.3.bias', 'pose_reduction_layer.weight', 'pose_reduction_layer.bias', 'pose_model.0.0.weight', 'pose_model.0.0.bias', 'pose_model.1.0.weight', 'pose_model.1.0.bias', 'pose_model.2.0.weight', 'pose_model.2.0.bias', 'pose_model.3.weight', 'pose_model.3.bias']
            """
            module = {
                "pose_model": (pose_model.pose_model.parameters(), self.lr_pose),
                "pose_reduction_layer": (
                    pose_model.pose_reduction_layer.parameters(),
                    self.lr_pose,
                ),
                "shape_model": (pose_model.shape_model.parameters(), self.lr_betas),
                "shape_reduction_layer": (
                    pose_model.shape_reduction_layer.parameters(),
                    self.lr_betas,
                ),
            }

            if train_vertice_offset:
                print("Train vertice offset")
                if isinstance(
                    self.lr_vertices_off, ListConfig
                ):  # v: [1e-5, [1000], [0.1, 1]] 0-1000 epoch: 1e-5*0.1; > 1000: 1e-5*1
                    self.lr_vertices_off = list(self.lr_vertices_off)
                    module["vertices_model"] = (
                        pose_model.vertices_model.parameters(),
                        self.lr_vertices_off[0],
                    )
                    module["vertices_reduction_layer"] = (
                        pose_model.vertices_reduction_layer.parameters(),
                        self.lr_vertices_off[0],
                    )
                else:
                    module["vertices_model"] = (
                        pose_model.vertices_model.parameters(),
                        self.lr_vertices_off,
                    )
                    module["vertices_reduction_layer"] = (
                        pose_model.vertices_reduction_layer.parameters(),
                        self.lr_vertices_off,
                    )

            vertices_param_indices = []
            parameters = []

            for idx, (key, (param_group, lr)) in enumerate(module.items()):
                if "vertices" in key:
                    vertices_param_indices.append(idx)
                parameters.append({"params": param_group, "lr": lr})

            optimizer_pose = Adam(parameters)

            # Custom LR scheduler: vertices group decay at epoch 10
            def lr_vertices_schedule(epoch):
                if isinstance(
                    self.lr_vertices_off, list
                ):  # v: [1e-5, [1000], [0.1, 1]] 0-1000 epoch: 1e-5*0.1; > 1000: 1e-5*1
                    for b, v in zip(self.lr_vertices_off[1], self.lr_vertices_off[2]):
                        if epoch < b:
                            return v

                    return self.lr_vertices_off[2][-1]
                else:
                    return 1.0

            lr_lambdas = []
            for idx in range(len(parameters)):
                if idx in vertices_param_indices:
                    lr_lambdas.append(lr_vertices_schedule)
                else:
                    lr_lambdas.append(lambda epoch: 1.0)

            scheduler_pose = LambdaLR(optimizer_pose, lr_lambda=lr_lambdas)

        elif pose_model.tag == PoseModelTags.PoseFixed:
            optimizer_pose = Adam(pose_model.parameters(), lr=self.lr_pose)
            # learning rate scheduler pose model
            scheduler_pose = MultiStepLR(
                optimizer_pose,
                gamma=self.factor_pose,
                milestones=self.milestones_pose,
                last_epoch=-1,
            )
        else:
            raise Exception("Unknown pose_model tag")

        # optimizer texture_model
        if texture_model.tag == TextureModelTags.TextureDuplex:
            optimizer_texture = Adam(
                [
                    {"params": [texture_model.v], "lr": self.lr_texture},
                    {
                        "params": texture_model.renderer.parameters(),
                        "lr": 0.1 * self.lr_texture,
                    },
                ]
            )
        else:
            raise Exception("Unknown texture_model tag")

        # learning rate scheduler texture model
        scheduler_texture = MultiStepLR(
            optimizer_texture,
            gamma=self.factor_texture,
            milestones=self.milestones_texture,
            last_epoch=-1,
        )

        return optimizer_pose, optimizer_texture, scheduler_pose, scheduler_texture

    def optimize_scene(
        self,
        N_epochs: int,
        pose_model: PoseBase,
        X_ind_train: torch.Tensor,
        X_ts_train: torch.Tensor,
        dino_feature: torch.Tensor,
        texture_model: TextureBase,
        callback_eval: CallbackClass,
        callback_earlystop: CallbackDataClass,
        checkpoint_freq: int,
        X_ind_test: torch.Tensor,
        X_ts_test: torch.Tensor,
        train_vertice_offset: bool,
    ):
        """
        Optimization loop on a scene.
        """

        # If callback early-stop has been triggered previously, do nothing
        if callback_earlystop.state == True:
            return

        # Get optimizers
        optimizer_pose, optimizer_texture, scheduler_pose, scheduler_texture = (
            self.get_optimizers_and_schedulers(
                pose_model, texture_model, train_vertice_offset=train_vertice_offset
            )
        )

        # Training Loop ----------------------------------------------------------------------------------------------------------------------------------
        for i in (
            pbar := tqdm(
                range(0, N_epochs),
                desc="Scene optimization ({:03d})".format(int(X_ind_train.shape[0])),
            )
        ):

            # Should we train both model (pose, texture) or texture only
            no_shape_training_ = self._no_shape_training(i)

            # Sample a batch of frames in X_ind_train
            permut = torch.randperm(X_ind_train.shape[0])[
                : min(self.batch_size, X_ind_train.shape[0])
            ]

            # Forward pass
            self.loss_optim.set_epoch(i)  # update epoch
            loss_dict = self.loss_optim.forward_batch(
                pose_model,
                X_ind_train[permut],
                X_ts_train[permut],
                dino_feature[permut],
                texture_model,
            )
            loss = loss_dict["total"]

            # Optimize [Pose]
            if not no_shape_training_:
                optimizer_pose.zero_grad()
                reduced_loss = torch.mean(loss)
                reduced_loss.backward(retain_graph=True)

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        pose_model.parameters(), max_norm=1.0
                    )

                optimizer_pose.step()

            # Optimize [Texture]
            if optimizer_texture is not None:
                if loss_dict.get("l_optim_color", None) is not None:
                    optimizer_texture.zero_grad()

                    loss_text = loss_dict["l_optim_color"]
                    reduced_loss_text = torch.mean(loss_text)

                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            texture_model.parameters(), max_norm=1.0
                        )

                    reduced_loss_text.backward()

                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            texture_model.parameters(), max_norm=1.0
                        )

                    optimizer_texture.step()

            # Evaluation
            if (i == 0) or (i == N_epochs - 1) or (i % checkpoint_freq == 0):
                (
                    eval_log_dict,
                    str_log,
                ) = callback_eval.call(
                    pose_model,
                    X_ind_test,
                    X_ts_test,
                    dino_feature[X_ind_test],
                    texture_model,
                    return_individual=True,
                )
                callback_es_dict = {"l_iou": torch.mean(eval_log_dict["l_iou"])}
                # Logging
                pbar.set_description(
                    "frame optimization ({:03d}) {}".format(
                        int(X_ind_train.shape[0]), str_log
                    )
                )
                if self.logger is not None:
                    self.logger.add_logs(eval_log_dict, i, dump_logs=False)
                    self.logger.add_tensorboard_logs(eval_log_dict, i)

                with open(
                    os.path.join(self.logger.experiment_folder, "metric.txt"), "a"
                ) as f:
                    f.write(str_log + "\n")

                if self.min_log_dict == None:
                    self.min_log_dict = eval_log_dict
            else:
                eval_log_dict = None

            # Learning rate schedulers update
            if (not no_shape_training_) and (scheduler_pose is not None):
                scheduler_pose.step()
            if scheduler_texture is not None:
                scheduler_texture.step()

            # Training log
            if (i == 0) or (i == N_epochs - 1) or (i % checkpoint_freq == 0):
                # Prepare training log dict
                log_to_add = {k: loss_dict[k].cpu().detach() for k in loss_dict}
                log_to_add.update(
                    {
                        "X_ind_train": X_ind_train[permut]
                        .cpu()
                        .detach()
                        .type(torch.int32)
                    }
                )
                # Add learning-rate optimizer_pose (shape)
                if (optimizer_pose is not None) and (not no_shape_training_):
                    log_to_add.update({"lr1": optimizer_pose.param_groups[0]["lr"]})
                # Add learning-rate optimizer_texture (texture)
                if optimizer_texture is not None:
                    log_to_add.update({"lr2": optimizer_texture.param_groups[0]["lr"]})
                # Logging
                if self.logger is not None:
                    self.logger.add_logs(log_to_add, i, dump_logs=False)
                    self.logger.add_tensorboard_logs(eval_log_dict, i)

            # Callback early-stop
            if eval_log_dict is not None:
                stop = callback_earlystop.call(i, callback_es_dict)
                if stop:
                    break

            # Checkpoint log
            if (i == 0) or (i == N_epochs - 1) or (i % checkpoint_freq == 0):
                min_iou = self.min_log_dict["l_iou"].mean()
                min_psnr = self.min_log_dict["l_psnr"].mean()
                min_score = min_iou + min_psnr * 0.01

                eval_iou = eval_log_dict["l_iou"].mean()
                eval_psnr = eval_log_dict["l_psnr"].mean()
                eval_score = eval_iou + eval_psnr * 0.01

                if min_score <= eval_score:
                    self.min_log_dict = eval_log_dict
                    self.save_models(
                        pose_model,
                        texture_model,
                        optimizer_pose,
                        optimizer_texture,
                        scheduler_pose,
                        scheduler_texture,
                        i,
                        is_best=True,
                    )
                    print(
                        f"Save best model at Epoch {i}: {min_iou:.2f}, {min_psnr:.2f} -> {eval_iou:.2f}, {eval_psnr:.2f}"
                    )

                self.save_models(
                    pose_model,
                    texture_model,
                    optimizer_pose,
                    optimizer_texture,
                    scheduler_pose,
                    scheduler_texture,
                    i,
                    is_best=False,
                )

        # Dump logs on disk-storage
        if self.logger is not None:
            self.logger.dump_logs()

    def save_models(
        self,
        pose_model: PoseBase,
        texture_model: TextureBase,
        optimizer_pose: torch.optim.Optimizer,
        optimizer_texture: torch.optim.Optimizer,
        scheduler_pose: LRScheduler,
        scheduler_texture: LRScheduler,
        epoch: int,
        is_best: bool = False,
    ):
        """
        Save checkpoints of pose and texture models.
        """
        if is_best:
            checkpoint_folder = self.logger.checkpoint_folder_best
        else:
            checkpoint_folder = self.logger.checkpoint_folder

        mlp_model_name = "pose_model_current_train.pt"
        mlp_texture_model_name = "texture_model_current_train.pt"

        # Save current state
        scene_optim_state_dict = {"current_epoch": epoch + 1, "config": self.config}
        if optimizer_pose is not None:
            scene_optim_state_dict.update(
                {"optimizer_pose": optimizer_pose.state_dict()}
            )
        if optimizer_texture is not None:
            scene_optim_state_dict.update(
                {"optimizer_texture": optimizer_texture.state_dict()}
            )
        torch.save(
            scene_optim_state_dict,
            os.path.join(checkpoint_folder, "scene_optim_current_state.pt"),
        )

        # Save mlp model
        torch.save(
            pose_model.cpu().state_dict(),
            os.path.join(checkpoint_folder, mlp_model_name),
        )
        pose_model = pose_model.to(self.device)

        # Save texture model
        torch.save(
            texture_model.cpu().state_dict(),
            os.path.join(checkpoint_folder, mlp_texture_model_name),
        )
        texture_model = texture_model.to(self.device)

        # Save logs
        self.logger.dump_logs()
        self.logger.dump_ext_logs()

        # Save time
        self.logger.logs_ext["time"][-1][1] = time.time()
