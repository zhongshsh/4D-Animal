---
license: apache-2.0
language:
- en
tags:
- animal
size_categories:
- n<1K
---
# External Data for 4D-Animal

External data from the paper: **4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos**. 

| [**ArXiv**](xxx) | [**Code**](xxx) |

## Data Tree

```sh
.
├── cse                  # DensePose model weights and configuration files
│   ├── Base-DensePose-RCNN-FPN.yaml
│   ├── densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml
│   └── model_final_8c9d99.pkl
├── lbos                 # Laplace-Beltrami Operator results
│   ├── lbo_cse.pk
│   ├── lbo_cse_to_smal.pk
│   └── lbo_smal.pk
├── refined_masks        # Refined segmentation masks
├── preprocessing
│   ├── cse_embeddings   # Continuous Surface Embedding (CSE) data
│   └── pnp_ransac       # Refined camera parameters via PnP + RANSAC
├── smal                 # SMAL model data
│   ├── my_smpl_39dogsnorm_newv3_dog.pkl
│   └── symmetry_inds.json
├── sparse_keypoints     # Sparse keypoints predicted using BITE
├── textures             # RGB texture for CSE and SMAL
│   ├── texture_cse.pk
│   └── texture_smal.pk
└── cop3d_data           # Subset of COP3D dataset
```

## DensePose Model Weights and Configuration Files

Pretrained DensePose models and configuration files from [Detectron2](https://github.com/facebookresearch/detectron2):

- [`densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml`](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k.yaml)
- [`Base-DensePose-RCNN-FPN.yaml`](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/DensePose/configs/cse/Base-DensePose-RCNN-FPN.yaml)
- [`model_final_8c9d99.pkl`](https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_soft_animals_I0_finetune_i2m_16k/270727461/model_final_8c9d99.pkl)

## Laplace-Beltrami Operators

For details on Laplace-Beltrami operators used in this project, refer to Section 3.1 of [Continuous Surface Embeddings (CSE) paper](https://arxiv.org/pdf/2011.12438).

## SMAL Model

The SMAL model files are taken from the project [BITE: Beyond Priors for Improved 3D Dog Pose Estimation](https://github.com/runa91/bite_release):

- [`my_smpl_39dogsnorm_newv3_dog.pkl`](https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj/download?path=%2Fdata%2Fsmal_data%2Fnew_dog_models&files=my_smpl_39dogsnorm_newv3_dog.pkl&downloadStartSecret=21p5mlf8old)
- [`symmetry_inds.json`](https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj/download?path=%2Fdata%2Fsmal_data&files=symmetry_inds.json&downloadStartSecret=ecjw1bt2rbv)

## Sparse Keypoints

Sparse keypoints are generated using the keypoint prediction module provided by [BITE](https://github.com/runa91/bite_release).

## COP3D Dataset

This project is built using the [COP3D dataset](https://github.com/facebookresearch/cop3d). We provide a subset of 50 videos used in our experiments. To access the full dataset, follow the [official download instructions](https://github.com/facebookresearch/cop3d#download).

Each video directory contains:

- `images`: Original video frames from COP3D  
- `masks`: Corresponding foreground masks  
- `part_masks`: Part segmentation masks (head, tail, body, legs) generated using [PartGLEE](https://github.com/ProvenceStar/PartGLEE)  
- `tracking_points.pt`: Tracking data generated using [BootsTAP](https://bootstap.github.io/)

## License

This is **not** a new dataset. All data used in this project are derived from existing public sources, such as [COP3D](https://github.com/facebookresearch/cop3d), [BITE](https://github.com/runa91/bite_release), and others.

We fully adhere to the terms of use and licenses of all referenced datasets. If you believe that any part of the data presented here violates applicable rights or policies, please contact us. We are committed to addressing any such concerns promptly and appropriately.


<!-- ## Citation

```
@article{xxx,
  title={4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos},
  author={xxx},
  journal={ArXiv},
  year={2025},
  volume={abs/xxx},
}
``` -->
