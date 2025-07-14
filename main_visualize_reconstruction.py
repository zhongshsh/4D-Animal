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

import os
import argparse
from data.input_cop import get_input_cop_from_archive
from model.inferencer import Inferencer
import rendering.visualization as viz
import rendering.visualization_renderer as vizrend
from scene_optim.callbacks import CallbackEval


def visualize_reconstruction(archive_path: str, frame_limit: int, device: str = "cuda"):

    # Load inputs
    ic = get_input_cop_from_archive(archive_path, frame_limit, device=device)

    image_size = ic.image_size
    images = ic.images.to(device)
    masks = ic.masks.to(device)
    X_ind = ic.X_ind  # frame id
    X_ts = ic.X_ts  # 0->1, norm of frame id
    X_ind_test = ic.X_ind_test
    X_ts_test = ic.X_ts_test
    smal = ic.smal
    renderer = ic.renderer
    texture = ic.texture.to(device)
    cameras = ic.cameras.to(device)
    cse_embedding = ic.cse_embedding.to(device)
    dino_feature = ic.dino_feature.to(device)

    # Load model (pose, texture)
    inferencer = Inferencer(archive_path, use_archived_code=True)
    pose_model = inferencer.load_pose_model().to(device)
    texture_model = inferencer.load_texture_model().to(device)

    # -- STATS
    callback_eval = CallbackEval(
        images, masks, cameras, smal, cse_embedding, image_size, device
    )

    final_eval, final_eval_str = callback_eval.call(
        pose_model, X_ind_test, X_ts_test, dino_feature[X_ind_test], texture_model
    )
    print("Results: {}".format(final_eval_str))

    with open(os.path.join(archive_path, "metric_visual.txt"), "w") as f:
        f.write(final_eval_str)

    with open("metric_result.txt", "a") as f:
        f.write(f"{final_eval_str}\t{archive_path}\n")

    # -- RENDERING
    viz.make_video_list(
        vizrend.global_visualization(
            images,
            masks,
            pose_model,
            X_ind,
            X_ts,
            dino_feature,
            smal,
            texture_model,
            cse_embedding,
            renderer,
            cameras,
            texture=texture,
        ),
        os.path.join(archive_path, "rendered_optimized.mp4"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize reconstruction results")
    parser.add_argument(
        "--archive_path", type=str, help="Archive of the reconstruction"
    )
    parser.add_argument(
        "--frame_limit", type=int, default=200, help="Archive of the reconstruction"
    )

    args = parser.parse_args()
    print(f"Archive path: {args.archive_path}")

    visualize_reconstruction(args.archive_path, args.frame_limit)
