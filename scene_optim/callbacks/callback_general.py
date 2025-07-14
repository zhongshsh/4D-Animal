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
from model.pose_models import PoseBase
from model.texture_models import TextureBase


class CallbackClass:
    def __init__(self, *args, **kwargs):
        pass

    def call(
        self,
        pose_model: PoseBase,
        X_ind: torch.Tensor,
        X_ts: torch.Tensor,
        texture_model: TextureBase,
    ):
        raise NotImplementedError


class CallbackDataClass:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, epoch: int, data_dict: dict):
        raise NotImplementedError
