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
from hydra.utils import instantiate
from omegaconf import DictConfig
from model.pose_models import PoseBase
from model.texture_models import TextureBase
from model.pose_models.model_canonical import get_original_global


def initialize_pose_model(
    cfg: DictConfig,
    X_ind: torch.Tensor,
    X_ts: torch.Tensor,
    dino_feature: torch.Tensor,
    init_betas: torch.Tensor | None,
    init_betas_limbs: torch.Tensor | None,
    device: str,
) -> PoseBase:
    """
    Initialize a pose model from config.
    """

    if init_betas is None:
        init_betas = torch.zeros((1, 30), dtype=torch.float32)
    if init_betas_limbs is None:
        init_betas_limbs = torch.zeros((1, 7), dtype=torch.float32)

    # Make MLP model
    optim_mlp = instantiate(
        cfg.exp.mlp_model,
        init_betas=init_betas,
        init_betas_limbs=init_betas_limbs,
    ).to(device)

    # [CASE 1] Some path provided to load a model
    if cfg.exp.checkpoint_path_shape != "":
        print("External checkpoint loaded for shape model")
        optim_mlp.load_state_dict(torch.load(cfg.exp.checkpoint_path_shape))
        return optim_mlp

    # Initialize global pose (translation, orient, pose) prediction
    y_global_canonical = get_original_global(N=1, front_orient=True).to(device)

    with torch.no_grad():
        y_pred0 = optim_mlp.compute_global(X_ind[[0]], X_ts[[0]], dino_feature[[0]])

    # Replace last Linear bias with canonical T-pose
    update_state_dict = optim_mlp.state_dict()
    print(f"All module: {update_state_dict.keys()}")

    update_keys = ["pose_model.1.bias"]
    for update_key in update_keys:
        if update_state_dict.get(update_key, None) is not None:
            print(f"Replace last Linear bias with canonical T-pose, {update_key}")
            update_state_dict[update_key] = update_state_dict[update_key] + (
                y_global_canonical - y_pred0
            ).reshape((-1,))

    optim_mlp.load_state_dict(update_state_dict)

    return optim_mlp


def initialize_texture_model(cfg: DictConfig, device: str) -> TextureBase:
    """
    Initialize a texture model from config.
    """

    # Initialize texture model
    texture_model = instantiate(cfg.exp.texture_mlp_model).to(device)

    # [CASE] Some path provided to load a model
    if cfg.exp.checkpoint_path_texture != "":
        print("External checkpoint loaded for texture model")
        texture_model.load_state_dict(torch.load(cfg.exp.checkpoint_path_texture))

    return texture_model
