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

from .callback_general import CallbackDataClass


class CallbackEarlyStop(CallbackDataClass):

    def __init__(
        self,
        min_condition_dict: dict[str, float] | None = None,
        min_epoch: int = 0,
        bias: float = 0.3,
    ):
        self.min_condition_dict = min_condition_dict
        self.min_epoch = min_epoch
        self.bias = bias
        self.state = False

    def call(self, epoch: int, data_dict: dict) -> bool:

        if self.min_condition_dict is None:
            self.min_condition_dict = data_dict
            print("Initial minimun: ", self.min_condition_dict)
            return False

        if self.state == True:
            return self.state

        if epoch < self.min_epoch:
            return self.state

        for k in data_dict:
            if k in self.min_condition_dict and (
                data_dict[k] <= self.min_condition_dict[k] - self.bias
            ):
                print(
                    "Early stop {}:{:.3f}, because minimun is {:.3f}-{}".format(
                        k, data_dict[k], self.min_condition_dict[k], self.bias
                    )
                )
                self.state = True
                return self.state
            elif k in self.min_condition_dict and (
                data_dict[k] > self.min_condition_dict[k]
            ):
                self.min_condition_dict[k] = data_dict[k]

        return self.state
