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
import pickle as pk
import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms as pth_transforms
from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from util.img_utils import resize_image
from config.keys import Keys
from data.utils import array_list_to_stack


def get_keypoints(mask, smal_vert, N, is_train=True):
    plot_point_tmp = [
        [1068, 1080, 1029, 1226, 645],  # left eye
        [2660, 3030, 2675, 3038, 2567],  # right eye
        [910, 11, 5],  # mouth low
        [542, 147, 509, 200, 522],  # Left Ear
        [2507, 2174, 2122, 2126, 2474],  # Right Ear
        [1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762],  # nose tip
        [360, 1203, 1235, 1230, 298, 408, 303, 293, 384],  # front left leg, low
        [3188, 3156, 2327, 3183, 2261, 2271, 2573, 2265],  # front right leg, low
        [1976, 1974, 1980, 856, 559, 851, 556],  # back left leg, low
        [3854, 2820, 3852, 3858, 2524, 2522, 2815, 2072],  # back right leg, low
        [416, 235, 182, 440, 8, 80, 73, 112],  # front left leg, top
        [2156, 2382, 2203, 2050, 2052, 2406, 3],  # front right leg, top
        [829, 219, 218, 173, 17, 7, 279],  # back left leg, top
        [2793, 582, 140, 87, 2188, 2147, 2063],  # back right leg, top
        [384, 799, 1169, 431, 321, 314, 437, 310, 323],  # front left leg, middle
        [
            2351,
            2763,
            2397,
            3127,
            2278,
            2285,
            2282,
            2275,
            2359,
        ],  # front right leg, middle
        [221, 104, 105, 97, 103],  # back left leg, middle
        [2754, 2192, 2080, 2251, 2075, 2074],  # back right leg, middle
        [452, 1811, 63, 194, 52, 370, 64],  # tail start
        [0, 464, 465, 726, 1824, 2429, 2430, 2690],  # half tail
        [28, 474, 475, 731, 24],  # Tail tip
        [60, 114, 186, 59, 878, 130, 189, 45],  # throat, close to base of neck
        [2091, 2037, 2036, 2160, 190, 2164],  # withers (a bit lower than in reality)
        [191, 1158, 3116, 2165, 154, 653, 133, 339],  # neck
    ]

    smal_vert["head"] = [
        item
        for sublist in [
            [1068, 1080, 1029, 1226, 645],  # left eye
            [2660, 3030, 2675, 3038, 2567],  # right eye
            [910, 11, 5],  # mouth low
            [542, 147, 509, 200, 522],  # Left Ear
            [2507, 2174, 2122, 2126, 2474],  # Right Ear
            [1039, 1845, 1846, 1870, 1879, 1919, 2997, 3761, 3762],  # nose tip
        ]
        for item in sublist
    ]

    smal_vert["leg"] = [
        item
        for sublist in [
            [360, 1203, 1235, 1230, 298, 408, 303, 293, 384],  # front left leg, low
            [3188, 3156, 2327, 3183, 2261, 2271, 2573, 2265],  # front right leg, low
            [1976, 1974, 1980, 856, 559, 851, 556],  # back left leg, low
            [3854, 2820, 3852, 3858, 2524, 2522, 2815, 2072],  # back right leg, low
            [416, 235, 182, 440, 8, 80, 73, 112],  # front left leg, top
            [2156, 2382, 2203, 2050, 2052, 2406, 3],  # front right leg, top
            [829, 219, 218, 173, 17, 7, 279],  # back left leg, top
            [2793, 582, 140, 87, 2188, 2147, 2063],  # back right leg, top
            [384, 799, 1169, 431, 321, 314, 437, 310, 323],  # front left leg, middle
            [
                2351,
                2763,
                2397,
                3127,
                2278,
                2285,
                2282,
                2275,
                2359,
            ],  # front right leg, middle
            [221, 104, 105, 97, 103],  # back left leg, middle
            [2754, 2192, 2080, 2251, 2075, 2074],  # back right leg, middle
        ]
        for item in sublist
    ]

    smal_vert["tail"] = [
        item
        for sublist in [
            [452, 1811, 63, 194, 52, 370, 64],  # tail start
            [0, 464, 465, 726, 1824, 2429, 2430, 2690],  # half tail
            [28, 474, 475, 731, 24],  # Tail tip
        ]
        for item in sublist
    ]

    if is_train:
        categories = [11, 12, 13, 14]
    else:
        categories = [11, 13]

    category_names = ["head", "torso", "leg", "tail"]

    total_area = (mask > 0).sum()

    selected_points = []
    selected_smal_points = []
    for i, category in enumerate(categories):
        category_indices = np.column_stack(
            np.where(mask == category)
        )  # (num_points, 2)

        num_points_to_sample = int(
            np.round((np.count_nonzero(mask == category)) / total_area * N)
        )

        if len(category_indices) > 0:
            selected_indices = np.random.choice(
                len(category_indices), num_points_to_sample, replace=False
            )
            selected_points.extend(category_indices[selected_indices])

            selected_smal_points.extend(
                np.random.choice(
                    smal_vert[category_names[i]], len(selected_indices), replace=True
                ).tolist()
            )

    if len(selected_points) == 0:
        return None, None

    while len(selected_points) < N:
        remaining = N - len(selected_points)
        selected_points += (
            selected_points[-remaining:]
            if remaining <= len(selected_points)
            else selected_points[:remaining]
        )
        selected_smal_points += (
            selected_smal_points[-remaining:]
            if remaining <= len(selected_smal_points)
            else selected_smal_points[:remaining]
        )

    return selected_points[-N:], selected_smal_points[-N:]


def crop_to_mask(mask) -> tuple[int, int, int, int]:
    mask_nz = mask.nonzero()
    x0, x1 = int(mask_nz[:, 2].min().item()), int(mask_nz[:, 2].max().item())
    y0, y1 = int(mask_nz[:, 1].min().item()), int(mask_nz[:, 1].max().item())
    w = x1 - x0
    h = y1 - y0
    return x0, y0, w, h


def _is_valid_sequence_name(
    cop3d_dataset: SqlIndexDataset, sequence_index: str
) -> bool:
    return sequence_index in cop3d_dataset.sequence_names()


def load_cop3d_sql_dataset(
    dataset_root: str, box_crop: bool = False, resizing: bool = True
) -> SqlIndexDataset:

    metadata_file = os.path.join(dataset_root, "metadata.sqlite")
    assert os.path.isfile(metadata_file)

    # Load the COP3D dataloader https://github.com/facebookresearch/cop3d/blob/main/README.md#api-quick-start-and-tutorials
    dataset = SqlIndexDataset(
        sqlite_metadata_file=metadata_file, dataset_root=dataset_root
    )
    dataset.frame_data_builder.load_depths = (
        False  # the dataset does not provide depth maps
    )

    # To turn off cropresizing
    if box_crop:
        dataset.frame_data_builder.box_crop = True
    else:
        dataset.frame_data_builder.box_crop = False

    if resizing == False:
        dataset.frame_data_builder.image_height = None  # turns off resizing

    return dataset


def cameras_from_metadatas(
    cameras: PerspectiveCameras, device: str, original_cameras: bool = False
) -> PerspectiveCameras:

    new_cameras = cameras.clone()

    # Original intrinsics and extrinsics
    if original_cameras:
        return new_cameras.to(device)

    # Original intrinsics but fixed extrinsics
    else:
        N = new_cameras.R.shape[0]
        new_cameras.R = (
            torch.Tensor(np.eye(3)).float()[None, :, :].to(device).repeat((N, 1, 1))
        )
        new_cameras.T = torch.zeros((N, 3))
        return new_cameras.to(device)


def resize_image(
    image,
    image_height,
    image_width,
    mode: str = "bilinear",
):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image_height is None or image_width is None:
        # skip the resizing
        return image, 1.0, torch.ones_like(image[:1])

    # takes numpy array or tensor, returns pytorch tensor
    minscale = min(
        image_height / image.shape[-2],
        image_width / image.shape[-1],
    )
    imre = torch.nn.functional.interpolate(
        image[None],
        scale_factor=minscale,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        recompute_scale_factor=True,
    )[0]
    imre_ = torch.zeros(image.shape[0], image_height, image_width)
    imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
    mask = torch.zeros(1, image_height, image_width)
    mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
    return imre_, minscale, mask


class COPSingleVideo(Dataset):
    def __init__(
        self,
        cop3d_root_path: str,
        sequence_index: str,
        cop3d_cropping: bool = False,
        cop3d_resizing: bool = True,
        preload: bool = True,
        frame_limit: int = 1000,
        category: str = "dog",
    ):
        """
        COP3D Dataset for a single video indexed by 'sequence_index'.
        Args:
            - cop3d_cropping: If True, crop the image around the mask. Affect the cameras (principal_point, focal_length)
            - cop3d_resizing: If True, square-pad the image and down-resize to (800,800)
        """
        self.cop3d_root_path = cop3d_root_path
        self.sequence_index = sequence_index
        self.cop3d_cropping = cop3d_cropping
        self.cop3d_resizing = cop3d_resizing
        self.preload = preload
        self.category = category

        # Load the SQL index cop3d dataset
        self.cop3d_dataset = load_cop3d_sql_dataset(
            self.cop3d_root_path, box_crop=cop3d_cropping, resizing=self.cop3d_resizing
        )

        assert _is_valid_sequence_name(
            self.cop3d_dataset, self.sequence_index
        ), f"Invalid sequence_index: {self.sequence_index}"

        # Preload the frames
        self.sequence_frame_ids = [
            frame.frame_number
            for frame in self.cop3d_dataset.sequence_frames_in_order(
                self.sequence_index
            )
        ]

        self.sequence_frame_ids = self.sequence_frame_ids[:frame_limit]

        if self.preload:
            self.sequence_frames_dict = {
                id_: self.cop3d_dataset[self.sequence_index, id_]
                for id_ in tqdm(
                    self.sequence_frame_ids,
                    desc="Loading COP3D frames ({})".format(self.sequence_index),
                )
            }

    def __len__(self):
        return len(self.sequence_frame_ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Return:
            - cropped frame (1x3x800x800) TORCHFLOAT32 [0,1]
            - cropped mask (1x1x800x800) TORCHFLOAT32 [0,1]
            - camera metadata
            torch.Size([1, 3, 800, 800]) torch.Size([1, 1, 800, 800])
            {'camera': PerspectiveCameras(), 'camera_quality_score': tensor([[1.1306]]), 'crop_bbox': (0, 0, 448, 799), 'frame_timestamp': tensor([[0.]])}
        """
        if self.preload:
            frame_i = self.sequence_frames_dict[self.sequence_frame_ids[i]]
        else:
            frame_i = self.cop3d_dataset[
                self.sequence_index, self.sequence_frame_ids[i]
            ]

        img_ = frame_i.image_rgb.unsqueeze(0)
        mask_ = frame_i.fg_probability.unsqueeze(0)

        mask_crop = frame_i.mask_crop
        if mask_crop is not None:
            crop_bbox = crop_to_mask(mask_crop)
        else:
            crop_bbox = [0, 0, int(img_.shape[2]), int(img_.shape[3])]

        metadata = {
            "camera": frame_i.camera,
            "camera_quality_score": torch.tensor(
                [frame_i.camera_quality_score]
            ).reshape((-1, 1)),
            "crop_bbox": crop_bbox,
            "frame_timestamp": torch.tensor([frame_i.frame_timestamp]).reshape((-1, 1)),
        }
        return img_, mask_, metadata

    def get_masks(self, list_items: list[int]) -> np.ndarray:
        """
        We refined the original masks from CoP3D and provide a download link to add them to external_data folder.
        We strongly recommend using the refined masks instead of the original ones from Cop3D.

        Return:
            - mask_list [N,H,W,1] NPFLOAT32 [0,1] (fg probability)
        """
        # Try to get the refined masks
        path_refined_masks = os.path.join(
            Keys().source_refined_masks, self.sequence_index
        )

        if not os.path.isdir(path_refined_masks):
            print(
                f"WARNING: refined masks for CoP3D sequence {self.sequence_index} not found in {path_refined_masks}."
            )
            print("This will affect result quality.")
            # Get classical masks
            mask_list = [
                np.transpose(
                    self.__getitem__(i)[1].numpy().astype(np.float32), (0, 2, 3, 1)
                )
                for i in list_items
            ]
        else:
            # Get refined masks
            list_frames = [
                os.path.join(
                    path_refined_masks,
                    "frame{:06d}.png".format(self.sequence_frame_ids[i]),
                )
                for i in list_items
            ]
            mask_list = []
            for path_i in list_frames:
                X = np.expand_dims(
                    np.array(Image.open(path_i)).astype(np.uint8), 0
                )  # Add C dimension
                if self.cop3d_resizing:
                    X_resized = (
                        resize_image(X, 800, 800, mode="bilinear")[0]
                        .unsqueeze(0)
                        .numpy()
                    )
                    mask_list.append(
                        np.transpose(X_resized.astype(np.float32), (0, 2, 3, 1))
                    )
                else:
                    X = np.expand_dims(X, 0)
                    mask_list.append(np.transpose(X.astype(np.float32), (0, 2, 3, 1)))

        mask_list = array_list_to_stack(mask_list)
        return mask_list

    @torch.no_grad()
    def get_dino_feature(self, list_items: list[int]) -> np.ndarray:
        """
        Return:
            - dino_feature_list [N,H,W,3] NPFLOAT32 [0,1]
        """

        def extract_feature(model, x, return_h_w=False):
            """Extract one x feature everytime. return the output tokens from the `n` last blocks"""
            out = model.get_intermediate_layers(x, n=1)[0]  # torch.Size([1, 3601, 384])
            out = out[
                :, 1:, :
            ]  # we discard the [CLS] token, torch.Size([1, 3600, 384])

            h, w = int(x.shape[-2] / model.patch_embed.patch_size), int(
                x.shape[-1] / model.patch_embed.patch_size
            )
            dim = out.shape[-1]

            out = out[0].reshape(h, w, dim)  # 60 60 384
            out = out.reshape(-1, dim)  # 3600, 384
            if return_h_w:
                return out, h, w
            return out

        device = "cuda"
        patch_size = 8
        image_size = (480, 480)

        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        dino.eval().to(device)

        dino_feature_list = []
        for i in list_items:
            img = Image.fromarray(
                (self.__getitem__(i)[0][0] * 255.0)
                .permute(1, 2, 0)
                .to(dtype=torch.uint8)
                .numpy()
            )  # 800 * 800
            img = transform(img)

            # make the image divisible by the patch size
            w, h = (
                img.shape[1] - img.shape[1] % patch_size,
                img.shape[2] - img.shape[2] % patch_size,
            )
            img = img[:, :w, :h].unsqueeze(0)
            img = img.to(device)

            out = extract_feature(dino, img).detach().cpu().unsqueeze(0).numpy()

            out = (out - out.min()) / (out.max() - out.min())

            dino_feature_list.append(out)

        del dino
        dino_feature_list = array_list_to_stack(dino_feature_list)

        return dino_feature_list

    def get_tracking_points(
        self, list_items: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            - tracking points N, point number, 2
            - visibles N, point number
        """
        image_size = 800
        data_path = os.path.join(
            self.cop3d_root_path,
            self.category,
            self.sequence_index,
            "tracking_points.pt",
        )
        tracking_points, visibles = torch.load(data_path, map_location="cpu")
        tracking_points = tracking_points[list_items]
        visibles = visibles[list_items]

        tracking_points[..., 0] = tracking_points[..., 0] / image_size
        tracking_points[..., 1] = tracking_points[..., 1] / image_size

        return tracking_points, visibles

    def get_part_keypoints(self, list_items: list[int], is_train=True) -> torch.Tensor:
        """
        Return:
            - part_masks_list [N,H,W] NPINT,
                idx2txt = {
                    11: 'Quadruped Head',
                    12: 'Quadruped Body',
                    13: 'Quadruped Foot',
                    14: 'Quadruped Tail'
                }

            smal vertices:
                head 1324
                tail 113
                front_left_leg 557
                front_right_leg 557
                back_left_leg 454
                back_right_leg 454
                torso 430
        """
        with open("config/smal_revised.json", "r") as f:
            smal_vert = json.load(f)

        smal_vert["leg"] = (
            smal_vert["front_left_leg"]
            + smal_vert["front_right_leg"]
            + smal_vert["back_left_leg"]
            + smal_vert["back_right_leg"]
        )

        base_path = os.path.join(
            self.cop3d_root_path, self.category, self.sequence_index, "part_masks"
        )

        if is_train:
            N = 200
        else:
            N = 50

        image_size = 800

        part_keypoints_list = []
        smal_keypoints_list = []
        last_part_keypoints, last_smal_keypoints = None, None
        for i in tqdm(list_items, desc="Prepare Part Keypoints"):
            part_mask_path = os.path.join(
                base_path, "frame{:06d}.pt".format(self.sequence_frame_ids[i])
            )
            part_mask = torch.load(part_mask_path)
            part_mask, _, _ = resize_image(
                part_mask.unsqueeze(0), image_size, image_size
            )
            part_mask = part_mask[0]

            part_keypoints, smal_keypoints = get_keypoints(
                part_mask, smal_vert, N, is_train=is_train
            )

            if part_keypoints is None:
                part_keypoints, smal_keypoints = (
                    last_part_keypoints,
                    last_smal_keypoints,
                )
            else:
                last_part_keypoints, last_smal_keypoints = (
                    part_keypoints,
                    smal_keypoints,
                )

            part_keypoints_list.append(part_keypoints)
            smal_keypoints_list.append(smal_keypoints)

        keypoints_yx = torch.tensor(part_keypoints_list, dtype=torch.float32)
        keypoints_xy = torch.flip(keypoints_yx, dims=[-1])

        keypoints_xy[..., 0] = keypoints_xy[..., 0] / image_size
        keypoints_xy[..., 1] = keypoints_xy[..., 1] / image_size

        keypoints_vert_id = torch.tensor(smal_keypoints_list)
        return keypoints_xy, keypoints_vert_id

    def get_part_masks(self, list_items: list[int]) -> torch.Tensor:
        """
        Return:
            - part_masks_list [N,H,W] NPINT,
                idx2txt = {
                    11: 'Quadruped Head',
                    12: 'Quadruped Body',
                    13: 'Quadruped Foot',
                    14: 'Quadruped Tail'
                }
        """
        base_path = os.path.join(
            self.cop3d_root_path, self.category, self.sequence_index, "part_masks"
        )
        image_size = 256

        part_masks = []
        for i in tqdm(list_items, desc="Load Part Masks"):
            part_mask_path = os.path.join(
                base_path, "frame{:06d}.pt".format(self.sequence_frame_ids[i])
            )
            part_mask = torch.load(part_mask_path)
            part_mask, _, _ = resize_image(
                part_mask.unsqueeze(0), image_size, image_size
            )
            part_mask = part_mask[0]
            part_masks.append(part_mask)

        part_masks = torch.stack(part_masks, dim=0)
        return part_masks

    def get_imgs_rgb(self, list_items: list[int]) -> np.ndarray:
        """
        Return:
            - img_list [N,H,W,3] NPFLOAT32 [0,1]
        """
        img_list = [
            np.transpose(
                self.__getitem__(i)[0].numpy().astype(np.float32), (0, 2, 3, 1)
            )
            for i in list_items
        ]
        img_list = array_list_to_stack(img_list)
        return img_list

    def get_cameras(self, list_items: list[int]) -> PerspectiveCameras:
        return join_cameras_as_batch(
            [self.__getitem__(i)[2]["camera"] for i in list_items]
        )

    def get_sparse_keypoints(
        self, list_items: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            - sparse_keypoints (N, N_KPTS, 2) NPFLOAT32
            - scores (N, N_KPTS, 1) NPFLOAT32
        """
        sparse_keypoints_path = os.path.join(
            Keys().source_refined_keypoints, f"{self.sequence_index}_keypoints.pk"
        )
        assert os.path.isfile(sparse_keypoints_path), (
            f"No keypoints data found for sequence {self.sequence_index} in: {sparse_keypoints_path},",
            'if no sparse keypoints available, please set "exp.l_optim_sparse_kp=0" in config/config.yaml',
            "This will affect result quality.",
        )

        with open(sparse_keypoints_path, "rb") as f:
            X = pk.load(f)
            sparse_keypoints, scores = X[..., :2], X[..., 2:]

        return sparse_keypoints, scores

    def get_init_shape(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            - init_betas NPFLOAT32
            - init_betas_limbs NPFLOAT32
        """
        init_shape_path = os.path.join(
            Keys().source_init_shape, f"{self.sequence_index}_init_pose.pk"
        )

        with open(init_shape_path, "rb") as f:
            X = pk.load(f)
            init_betas, init_betas_limbs = X["betas"], X["betas_limbs"]

        return init_betas, init_betas_limbs

    def get_crop_bbox_list(self, list_items=None):
        if list_items is None:
            crop_bbox_list = [
                self.__getitem__(i)[2]["crop_bbox"] for i in range(len(self))
            ]
        else:
            crop_bbox_list = [self.__getitem__(i)[2]["crop_bbox"] for i in list_items]
        return crop_bbox_list
