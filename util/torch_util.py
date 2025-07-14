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

import warnings


# Define a custom context manager to suppress all warnings
class SuppressAllWarnings:
    def __enter__(self):
        # Temporarily disable all warnings
        warnings.filterwarnings("ignore")

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original warning behavior
        warnings.resetwarnings()
