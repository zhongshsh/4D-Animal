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
import random
import pickle as pk
import os
import numpy as np
from model.pose_models import PoseCanonical, compute_pose
from pnp.p_n_p_optim import get_init_from_pnp
from data.input_cop import InputCop

torch.manual_seed(1003)
random.seed(1003)
np.random.seed(1003)
torch.cuda.set_per_process_memory_fraction(0.9, device=0)


def compute_pnp_R_T_with_part(
    sequence_index: str,
    dataset_source: str,
    image_size: int,
    use_RANSAC: bool,
    N_points_per_frames: int,
    n_iter_max: int,
    thre_inl: float,
    min_inl_percentage: float,
    device: str,
    frame_limit: int = 1000,
    category: str = "dog",
    use_pixel: bool = False,
) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:
    print(f"PnP method: use RANSAC {use_RANSAC}")

    ic = InputCop(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cse_mesh_name="smal",
        N_cse_kps=N_points_per_frames,
        device=device,
        frame_limit=frame_limit,
        category=category,
    )

    N_frames = ic.N_frames_synth
    X_ind = ic.X_ind
    X_ts = ic.X_ts
    smal = ic.smal
    cameras = ic.cameras_canonical.to(device)

    pose_model = PoseCanonical(N=N_frames).to(device)

    # -------------------------------------------------------------------------------------------------
    mask_keypoints_xy, mask_keypoints_vert_id = ic.pnp_part_keypoints
    mask_keypoints_xy = (
        mask_keypoints_xy.to(device) * image_size
    )  # scale to image_size 256
    mask_keypoint_scores = torch.ones((*mask_keypoints_xy.shape[:2], 1))

    (valid_indices, cse_keypoints_xy, cse_keypoints_vert_id, cse_max_indice) = (
        ic.cse_keypoints
    )
    cse_keypoints_xy = cse_keypoints_xy.to(device) * image_size
    cse_keypoint_scores = torch.ones((N_frames, N_points_per_frames, 1))
    for i in range(N_frames):
        cse_keypoint_scores[i, cse_max_indice[i] :] = 0

    if use_pixel:
        keypoints_xy = torch.cat((cse_keypoints_xy, mask_keypoints_xy), dim=1)
        keypoints_vert_id = torch.cat(
            (cse_keypoints_vert_id, mask_keypoints_vert_id), dim=1
        )
        keypoint_scores = torch.cat((cse_keypoint_scores, mask_keypoint_scores), dim=1)
    else:
        keypoints_xy = mask_keypoints_xy
        keypoints_vert_id = mask_keypoints_vert_id
        keypoint_scores = mask_keypoint_scores

    with torch.no_grad():
        vertices, _, _ = compute_pose(
            smal=smal, pose_model=pose_model, X_ind=X_ind, X_ts=X_ts, dino_feature=None
        )
    keypoints_3d = torch.stack(
        [vertices[i][keypoints_vert_id[i]] for i in range(N_frames)], dim=0
    )
    keypoints_3d = torch.round(keypoints_3d, decimals=4)

    # Perpsective-N-point with RANSAC
    valid_ind, R_stack, T_stack, info_dict = get_init_from_pnp(
        keyp_3d=keypoints_3d,
        keyp2d_prediction=keypoints_xy,
        keyp2d_score=keypoint_scores,
        valid_indices=valid_indices,
        cameras=cameras,
        image_size=image_size,
        device=device,
        use_RANSAC=use_RANSAC,
        n_iter_max=n_iter_max,
        thre_inl=thre_inl,
        min_inl_percentage=min_inl_percentage,
    )

    if len(valid_ind) == 0:
        return valid_ind, None, None
    else:
        return valid_ind, R_stack.cpu(), T_stack.cpu()


def compute_pnp_R_T(
    sequence_index: str,
    dataset_source: str,
    image_size: int,
    use_RANSAC: bool,
    N_points_per_frames: int,
    n_iter_max: int,
    thre_inl: float,
    min_inl_percentage: float,
    device: str,
    frame_limit: int = 1000,
    category: str = "dog",
) -> tuple[list[int], torch.Tensor | None, torch.Tensor | None]:

    ic = InputCop(
        sequence_index=sequence_index,
        dataset_source=dataset_source,
        cse_mesh_name="smal",
        N_cse_kps=N_points_per_frames,
        device=device,
        frame_limit=frame_limit,
        category=category,
    )

    N_frames = ic.N_frames_synth
    X_ind = ic.X_ind
    X_ts = ic.X_ts
    smal = ic.smal
    cameras = ic.cameras_canonical.to(device)

    pose_model = PoseCanonical(N=N_frames).to(device)

    # -------------------------------------------------------------------------------------------------

    (valid_indices, cse_keypoints_xy, cse_keypoints_vert_id, cse_max_indice) = (
        ic.cse_keypoints
    )
    cse_keypoints_xy = cse_keypoints_xy.to(device) * image_size

    with torch.no_grad():
        vertices, _, _ = compute_pose(
            smal=smal, pose_model=pose_model, X_ind=X_ind, X_ts=X_ts, dino_feature=None
        )
    keypoints_3d = torch.stack(
        [vertices[i][cse_keypoints_vert_id[i]] for i in range(N_frames)], dim=0
    )
    keypoints_3d = torch.round(keypoints_3d, decimals=4)

    keypoint_scores = torch.ones((N_frames, N_points_per_frames, 1))
    for i in range(N_frames):
        keypoint_scores[i, cse_max_indice[i] :] = 0

    # Perpsective-N-point with RANSAC

    valid_ind, R_stack, T_stack, info_dict = get_init_from_pnp(
        keyp_3d=keypoints_3d,
        keyp2d_prediction=cse_keypoints_xy,
        keyp2d_score=keypoint_scores,
        valid_indices=valid_indices,
        cameras=cameras,
        image_size=image_size,
        device=device,
        use_RANSAC=use_RANSAC,
        n_iter_max=n_iter_max,
        thre_inl=thre_inl,
        min_inl_percentage=min_inl_percentage,
    )

    if len(valid_ind) == 0:
        return valid_ind, None, None
    else:
        return valid_ind, R_stack.cpu(), T_stack.cpu()


def is_already_computed_pnp(sequence_index: str, frame_limit: int, cache_path: str):
    """
    Check if the PNP solution for given sequence_index has already been calculated in cache_path
    """
    return os.path.isfile(
        os.path.join(cache_path, f"{sequence_index}_{frame_limit}_pnp_R_T.pk")
    )


def preprocess_pnp(
    sequence_index: str,
    dataset_source: str,
    cache_path: str,
    device: str = "cuda",
    frame_limit: int = 1000,
    category: str = "dog",
    use_part: bool = False,
    use_pixel: bool = False,
):
    """
    Compute PNP solution for a sequence, and save result in 'cache_path'
    """
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

    use_RANSAC = True
    save_path = os.path.join(cache_path, f"{sequence_index}_{frame_limit}_pnp_R_T.pk")
    if use_part:
        valid_indices_pnp, R_pnp, T_pnp = compute_pnp_R_T_with_part(
            sequence_index=sequence_index,
            dataset_source=dataset_source,
            image_size=256,
            use_RANSAC=use_RANSAC,
            N_points_per_frames=200,
            n_iter_max=100,
            thre_inl=0.05,
            min_inl_percentage=0.1,
            device=device,
            frame_limit=frame_limit,
            category=category,
            use_pixel=use_pixel,
        )
    else:
        valid_indices_pnp, R_pnp, T_pnp = compute_pnp_R_T(
            sequence_index=sequence_index,
            dataset_source=dataset_source,
            image_size=256,
            use_RANSAC=use_RANSAC,
            N_points_per_frames=200,
            n_iter_max=100,
            thre_inl=0.05,
            min_inl_percentage=0.1,
            device=device,
            category=category,
            frame_limit=frame_limit,
        )

    print(f"pnp Save in {save_path}")
    with open(save_path, "wb") as f:
        pk.dump(
            {"valid_indices_pnp": valid_indices_pnp, "R_PNP": R_pnp, "T_PNP": T_pnp}, f
        )
