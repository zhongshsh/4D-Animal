# 4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos

| [**ArXiv**](xxx) | [**Code**](https://github.com/zhongshsh/4D-Animal) |

This repository contains the official implementation of the paper:  
**4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos**

**Authors:**  
[Shanshan Zhong](https://github.com/zhongshsh/4D-Animal), [Jiawei Peng](https://scholar.google.com/citations?user=4jdUy5AAAAAJ), [Zehan Zheng](https://scholar.google.com/citations?user=Pig6X6MAAAAJ), [Zhongzhan Huang](https://scholar.google.com/citations?user=R-b68CEAAAAJ), [Wufei Ma](https://scholar.google.com/citations?user=mYkvHdIAAAAJ), [Guofeng Zhang](https://scholar.google.com/citations?user=vl0mzhEAAAAJ), [Qihao Liu](https://scholar.google.com/citations?user=WFl3hH0AAAAJ), [Alan Yuille](https://scholar.google.com/citations?user=FJ-huxgAAAAJ), [Jieneng Chen](https://scholar.google.com/citations?user=yLYj88sAAAAJ)


## News
- **2025-07-15**: Initial code release 🎉


## Prepraration

Clone the repository:
```sh
git clone https://github.com/zhongshsh/4D-Animal
cd 4D-Animal
```

Create and activate the conda environment:
```sh
conda create -n animal python=3.10 -y
conda activate animal
```

Install required dependencies:
```sh
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

pip install cogapp triton plotly
git clone https://github.com/facebookresearch/lightplane
cd lightplane
pip install -e .

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html

pip install pandas sqlalchemy plotly hydra-core tensorboard lpips opencv-python imageio[ffmpeg] pyyaml Pillow natsort
```

Download the [external_data.tar.gz](https://huggingface.co/datasets/zhongshsh/4D-Animal/blob/main/external_data.tar.gz) and unzip the file.

```sh
wget --header "Authorization: Bearer your_hf_token" https://huggingface.co/datasets/zhongshsh/4D-Animal/resolve/main/external_data.tar.gz 
tar -xzvf external_data.tar.gz
```
💡 Replace your_hf_token with your HuggingFace token from https://huggingface.co/settings/tokens.

## Optimize a CoP3D scene

To optimize a CoP3D scene (e.g., `1030_23106_17099`) and save results in the `experiments` directory:

```sh
python main_optimize_scene.py 'exp.sequence_index="1030_23106_17099"' 'exp.experiment_folder="experiments"'
```

Hyperparameters for reconstruction can be modified in `config/config.yaml`.

## Visualize Reconstruction

To visualize reconstruction of a trained model:

```sh
python main_visualize_reconstruction.py --archive_path experiments/1030_23106_17099
```

## Acknowledgement

We would like to express our sincere gratitude to the authors of [Animal Avatar](https://arxiv.org/pdf/2403.17103) for their well-structured and inspiring codebase, which served as a valuable reference for our implementation.

We also thank the developers of the following projects [DINO](https://github.com/facebookresearch/dino), [CSE](https://arxiv.org/pdf/2011.12438), [PartGLEE](https://github.com/ProvenceStar/PartGLEE), and [BootsTAP](https://bootstap.github.io/) for contributing such impressive models to our community. 

<!-- ## Citation

If you find our work useful, please consider citing:
```
@article{xxx,
  title={4D-Animal: Freely Reconstructing Animatable 3D Animals from Videos},
  author={xxx},
  journal={ArXiv},
  year={2025},
  volume={abs/xxx},
}
``` -->
