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
import os
from omegaconf import OmegaConf
from typing import Literal
from config.keys import Keys
from data.cop3d_dataloader import COPSingleVideo, cameras_from_metadatas
from data.utils import train_test_split_frames
from cse_embedding.cse_embedder import (
    CseProcessing,
    get_closest_vertices_map_from_cse_map,
    get_cse_keypoints_from_cse_maps,
)
from cse_embedding.functional_map import get_geodesic_distance_from_name
from cse_embedding.cse_keypoints import get_cse_keypoint_cycle_masks
from pnp.camera_estimation import get_moving_cameras_from_PNP_archive
from rendering.renderer import Renderer
from smal import get_smal_model
import util.img_utils as imutil
import util.template_mesh_utils as templutil


class InputCop:
    def __init__(
        self,
        sequence_index: str,
        dataset_source: Literal["COP3D", "CUSTOM"] = "COP3D",
        cse_mesh_name: Literal["cse", "smal"] = "smal",
        frame_limit: int = 1000,
        image_size: int = 256,
        train_test_split: tuple[int, int] = (15, 5),
        N_cse_kps: int = 10_000,
        filter_cse_kps: bool = False,
        moving_camera: bool = True,
        cse_version: str = "original",
        device: str = "cuda",
        category: str = "dog",
    ):
        """
        Structure to store and cache all inputs of a scene that are used in scene optimization & visualization.
        """

        self.sequence_index = sequence_index
        self.frame_limit = frame_limit
        self.image_size = image_size
        self.cse_mesh_name = cse_mesh_name
        self.N_cse_kps = N_cse_kps
        self.filter_cse_kps = filter_cse_kps
        self.device = device
        self.dataset_source = dataset_source
        self.moving_camera = moving_camera
        self.train_test_split = train_test_split
        self.cse_version = cse_version
        self.normalize_cse = True
        self.category = category

        assert self.cse_mesh_name in ["cse", "smal"]

    @property
    def N_frames_synth(self):
        return min(self.frame_limit, self.N_frames_total)

    @property
    def N_frames_total(self):
        return int(len(self.dataset))

    @property
    def X_ind_global(self):
        return torch.arange(self.N_frames_total)

    @property
    def X_ts_global(self):
        return self.X_ind_global / self.N_frames_total

    @property
    def X_ind(self):
        return self.X_ind_global[: self.N_frames_synth]

    @property
    def X_ts(self):
        return self.X_ts_global[: self.N_frames_synth]

    @property
    def X_ind_train(self):
        if getattr(self, "_X_ind_train", None) is None:
            self._X_ind_train, self._X_ind_test = train_test_split_frames(
                self.X_ind, self.train_test_split[0], self.train_test_split[1]
            )
        return self._X_ind_train

    @property
    def X_ind_test(self):
        if getattr(self, "_X_ind_test", None) is None:
            self._X_ind_train, self._X_ind_test = train_test_split_frames(
                self.X_ind, self.train_test_split[0], self.train_test_split[1]
            )
        return self._X_ind_test

    @property
    def X_ts_train(self):
        if getattr(self, "_X_ts_train", None) is None:
            self._X_ts_train, self._X_ts_test = train_test_split_frames(
                self.X_ts, self.train_test_split[0], self.train_test_split[1]
            )
        return self._X_ts_train

    @property
    def X_ts_test(self):
        if getattr(self, "_X_ts_test", None) is None:
            self._X_ts_train, self._X_ts_test = train_test_split_frames(
                self.X_ts, self.train_test_split[0], self.train_test_split[1]
            )
        return self._X_ts_test

    @property
    def cameras_original(self):
        if getattr(self, "_cameras_original", None) is None:
            self._cameras_original = self.dataset.get_cameras(
                list(range(self.N_frames_synth))
            )
        return self._cameras_original

    @property
    def cameras_canonical(self):
        if getattr(self, "_cameras_canonical", None) is None:
            self._cameras_canonical = cameras_from_metadatas(
                self.cameras_original, device="cpu", original_cameras=False
            )[list(range(self.N_frames_synth))]
        return self._cameras_canonical

    @property
    def cameras(self):
        if getattr(self, "_cameras", None) is None:
            X_cameras_moving = cameras_from_metadatas(
                self.cameras_original, device="cpu", original_cameras=True
            )

            cameras_moving = get_moving_cameras_from_PNP_archive(
                sequence_index=self.sequence_index,
                frame_limit=self.frame_limit,
                cache_path=Keys().preprocess_path_pnp,
                cameras_original=X_cameras_moving,
            )

            if not self.moving_camera:
                for i in range(cameras_moving.R.shape[0]):
                    cameras_moving.R[i] = cameras_moving.R[0]
                    cameras_moving.T[i] = cameras_moving.T[0]

            self._cameras = cameras_moving

        return self._cameras

    @property
    def dataset(self):
        if getattr(self, "_dataset", None) is None:
            if self.dataset_source == "COP3D":
                self._dataset = COPSingleVideo(
                    cop3d_root_path=Keys().dataset_root,
                    sequence_index=self.sequence_index,
                    cop3d_cropping=False,  # must, or crop edge information
                    cop3d_resizing=True,  # must, or image size not equal
                    preload=self.frame_limit > 50,
                    frame_limit=self.frame_limit,
                    category=self.category,
                )
            else:
                raise Exception(f"Unknown dataset: {self.dataset_source}")
        return self._dataset

    @property
    def images_hr(self):
        """
        Returns:
            - images_hr (BATCH, H, W, 3) TORCHFLOAT32 [CPU] [0,1] OR List( TORCHFLOAT32(1,H_i,W_i,3)[CPU] )
        """
        if getattr(self, "_images_hr", None) is None:
            self._images_hr = self.dataset.get_imgs_rgb(
                list(range(self.N_frames_synth))
            )
            if type(self._images_hr) == list:
                self._images_hr = [
                    torch.tensor(x).type(torch.float32) for x in self._images_hr
                ]
            else:
                self._images_hr = torch.tensor(self._images_hr).type(torch.float32)
        return self._images_hr

    @property
    def original_resolution(self):
        if getattr(self, "_original_resolution", None) is None:
            self._original_resolution = (
                int(self.images_hr.shape[1]),
                int(self.images_hr.shape[2]),
            )
        return self._original_resolution

    @property
    def masks_hr(self):
        """
        Returns:
            - masks_hr (BATCH, H, W, 1) TORCHFLOAT32 [CPU] {0,1} OR List( TORCHUINT8(1,H_i,W_i,1)[CPU] )
        """
        if getattr(self, "_masks_hr", None) is None:
            self._masks_hr = self.dataset.get_masks(list(range(self.N_frames_synth)))
            if type(self._masks_hr) == list:
                self._masks_hr = [torch.tensor(x) for x in self._masks_hr]
            else:
                self._masks_hr = torch.tensor(
                    self.dataset.get_masks(list(range(self.N_frames_synth)))
                )
        return self._masks_hr

    @property
    def images(self):
        """
        (BATCH, image_size, image_size, 3) TORCHFLOAT32 [0.<->1.]
        """
        if getattr(self, "_images", None) is None:
            self._images = imutil.resize_torch(
                self.images_hr, (self.image_size, self.image_size)
            )
        return self._images

    @property
    def masks(self):
        """
        (BATCH, image_size, image_size, 1) TORCHINT32 {0,1}
        """
        if getattr(self, "_masks", None) is None:
            self._masks = (
                imutil.resize_torch(self.masks_hr, (self.image_size, self.image_size))
                > 0.7
            ).type(torch.int32)

        return self._masks

    @property
    def cproc(self):
        if getattr(self, "_cproc", None) is None:
            self._cproc = CseProcessing(
                cse_data_root_path=Keys().preprocess_path_cse,
                sequence_index=self.sequence_index,
                densepose_version=Keys().densepose_version,
                lbo_data_folder=Keys().external_data_path,
                frame_limit=self.frame_limit,
                category=self.category,
            )
        return self._cproc

    @property
    def cse_embedding(self):
        if getattr(self, "_cse_embedding", None) is None:
            self._cse_embedding = self.cproc.get_cse_embedding(
                convert_mesh_name=self.cse_mesh_name, normalize=self.normalize_cse
            )
        return self._cse_embedding

    @property
    def cse_maps(self):
        if getattr(self, "_cse_maps", None) is None:
            cse_maps, cse_masks, valid_indices = self.cproc.get_cse_maps(
                normalize=self.normalize_cse,
                output_resolution=(self.image_size, self.image_size),
            )
            cse_masks = (
                (cse_masks.argmax(1, keepdim=True) > 0)
                .type(torch.int32)
                .permute(0, 2, 3, 1)
            )

            self._cse_maps = (cse_maps, cse_masks, valid_indices)
        return self._cse_maps

    @property
    def cse_closest_verts(self):
        if getattr(self, "_cse_closest_verts", None) is None:

            (cse_maps, cse_masks, _) = self.cse_maps

            self._cse_closest_verts = get_closest_vertices_map_from_cse_map(
                cse_maps.permute(0, 2, 3, 1).to(self.device),
                cse_masks.to(self.device),
                self.cse_embedding.to(self.device),
            ).cpu()

        return self._cse_closest_verts

    @property
    def cse_keypoints(self):
        if getattr(self, "_cse_keypoints", None) is None:

            (cse_maps, cse_masks, valid_indices) = self.cse_maps
            cse_masks = cse_masks * self.masks
            cse_closest_verts = self.cse_closest_verts

            if self.filter_cse_kps:
                cse_masks = (
                    get_cse_keypoint_cycle_masks(
                        cse_maps.permute(0, 2, 3, 1), cse_masks, self.cse_embedding
                    )
                    > 0.5
                ).type(torch.int32)
            (keypoints_xy, keypoints_vert_id, cse_max_indice) = (
                get_cse_keypoints_from_cse_maps(
                    cse_closest_verts, cse_masks, valid_indices, self.N_cse_kps
                )
            )

            keypoints_xy[..., 0] = keypoints_xy[..., 0] / self.image_size
            keypoints_xy[..., 1] = keypoints_xy[..., 1] / self.image_size
            valid_indices = torch.tensor(valid_indices)

            self._cse_keypoints = (
                valid_indices,
                keypoints_xy,
                keypoints_vert_id,
                cse_max_indice,
            )

        return self._cse_keypoints

    @property
    def sparse_keypoints(self):
        if getattr(self, "_sparse_keypoints", None) is None:
            sparse_keypoints, sparse_keypoints_scores = (
                self.dataset.get_sparse_keypoints(list(range(self.N_frames_synth)))
            )
            self._sparse_keypoints = torch.tensor(sparse_keypoints), torch.tensor(
                sparse_keypoints_scores
            )
        return self._sparse_keypoints

    @property
    def renderer(self):
        if getattr(self, "_renderer", None) is None:
            self._renderer = Renderer(self.image_size)
        return self._renderer

    @property
    def smal(self):
        if getattr(self, "_smal", None) is None:
            self._smal = get_smal_model(self.device)
        return self._smal

    @property
    def texture(self):
        if getattr(self, "_texture", None) is None:
            self._texture = torch.tensor(
                templutil.get_mesh_2d_embedding_from_name(self.cse_mesh_name)
            ).type(torch.float32)
        return self._texture

    @property
    def geodesic_distance_mesh(self):
        if getattr(self, "_geodesic_distance_mesh", None) is None:
            self._geodesic_distance_mesh = torch.tensor(
                get_geodesic_distance_from_name(
                    self.cse_mesh_name, Keys().source_cse_folder
                )
            )
        return self._geodesic_distance_mesh

    @property
    def init_betas(self):
        if getattr(self, "_init_betas", None) is None:
            assert self.dataset_source == "COP3D"
            init_betas, init_betas_limbs = self.dataset.get_init_shape()
            self._init_betas = (
                torch.tensor(init_betas),
                torch.tensor(init_betas_limbs),
            )
        return self._init_betas

    @property
    def dino_feature(self):
        if getattr(self, "_dino_feature", None) is None:
            self._dino_feature = self.dataset.get_dino_feature(
                list(range(self.N_frames_synth))
            )
            if type(self._dino_feature) == list:
                self._dino_feature = [
                    torch.tensor(x).type(torch.float32) for x in self._dino_feature
                ]
            else:
                self._dino_feature = torch.tensor(self._dino_feature).type(
                    torch.float32
                )
        return self._dino_feature

    @property
    def tracking_points(self):
        if getattr(self, "_tracking_points", None) is None:
            self._tracking_points, self._tracking_visibles = (
                self.dataset.get_tracking_points(list(range(self.N_frames_synth)))
            )

        return self._tracking_points, self._tracking_visibles

    @property
    def part_keypoints(self):
        if getattr(self, "_part_keypoints", None) is None:
            self._part_keypoints = self.dataset.get_part_keypoints(
                list(range(self.N_frames_synth))
            )

        return self._part_keypoints

    @property
    def part_masks(self):
        if getattr(self, "_part_masks", None) is None:
            self._part_masks = self.dataset.get_part_masks(
                list(range(self.N_frames_synth))
            )

        return self._part_masks

    @property
    def pnp_part_keypoints(self):
        if getattr(self, "_pnp_part_keypoints", None) is None:
            self._pnp_part_keypoints = self.dataset.get_part_keypoints(
                list(range(self.N_frames_synth)), is_train=False
            )

        return self._pnp_part_keypoints


def get_input_cop_from_cfg(cfg: OmegaConf, device: str = "cuda") -> InputCop:
    """
    Initialize an InputCop instance from a config file.
    """
    ic = InputCop(
        sequence_index=cfg.exp.sequence_index,
        dataset_source=cfg.exp.dataset_source,
        cse_mesh_name=cfg.exp.cse_mesh_name,
        frame_limit=cfg.exp.frame_limit,
        image_size=cfg.exp.image_size,
        train_test_split=(cfg.exp.train_split, cfg.exp.test_split),
        N_cse_kps=cfg.exp.N_cse_kp,
        filter_cse_kps=cfg.exp.filter_cse_kps,
        moving_camera=cfg.exp.moving_camera,
        cse_version=cfg.exp.cse_version,
        device=device,
        category=cfg.exp.category,
    )
    print((cfg.exp.train_split, cfg.exp.test_split))
    return ic


def get_input_cop_from_archive(
    archive_path: str, frame_limit: int, device: str = "cuda"
) -> InputCop:
    """
    Initialize an InputCop instance from the path of a reconstruction.
    """

    path = os.path.join(archive_path, "checkpoints/args.txt")

    assert os.path.isfile(path), 'Invalid archive path (missing "checkpoints/args.txt")'

    with open(path, "rb") as f:
        cfg = OmegaConf.load(f)

    # Load default config file
    with open(Keys().default_config_path, "rb") as f:
        cfg_default = OmegaConf.load(f)
    cfg = OmegaConf.merge(cfg_default, cfg)

    cfg.exp.frame_limit = frame_limit

    ic = get_input_cop_from_cfg(cfg, device)

    return ic


def get_sequence_index_from_archive(archive_path: str) -> str:
    ic = get_input_cop_from_archive(archive_path)
    return ic.sequence_index


def get_cfg_from_archive(archive_path: str) -> OmegaConf:
    path = os.path.join(archive_path, "checkpoints/args.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        cfg = OmegaConf.load(f)
    return cfg
