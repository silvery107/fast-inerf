# Fast iNeRF

## TODO
- [ ] Reconstruction loss: only do loss on key points (eg. only on object, coarsen by area)
- [ ] Mask R-CNN / PoseCNN to segment, then only do loss on the segmented part 
- [ ] Use PoseCNN to estimate initial guess of camera pose 

- [x] Main file [pose_estimation.ipynb](pose_estimation.ipynb)

### [Project Page](https://yenchenlin.me/inerf/) | [Video](https://www.youtube.com/watch?v=eQuCZaQN0tI&feature=emb_logo) | [Paper](https://arxiv.org/pdf/2012.05877.pdf)

<img src="https://user-images.githubusercontent.com/7057863/161620132-2ce16dca-53f6-413d-97ab-fe6086f1661c.gif" height=200>

PyTorch implementation of iNeRF, an RGB-only method that inverts neural radiance fields (NeRFs) for 6DoF pose estimation.

## Overview

This preliminary codebase currently only shows how to apply iNeRF with pixelNeRF. However, iNeRF can work with the original NeRF as well.

## Environment setup

To start, install `pytorch` and `torchvision` according to your own GPU version, and then create the environment using conda:
- Yulun: tested with `pytorch==1.11` and `torchvision==0.12`
- If you see `ParseException: Expected '}', found '=' (at char 759), (line:34, col:18)` error, check [here](https://github.com/sxyu/pixel-nerf/issues/61)
```sh
conda env create -f environment.yml
conda activate pixelnerf
pip install mediapy
pip install jupyter
```



Please make sure you have up-to-date NVIDIA drivers supporting CUDA 10.2 at least.

## Quick start

1. Download all pixelNeRF's pretrained weight files from [here](https://drive.google.com/file/d/1UO_rL201guN6euoWkCOn-XpqR2e8o6ju/view?usp=sharing).
Create and extract it to `./checkpoints/` folder, so that `./checkpoints/srn_car/pixel_nerf_latest` exists.

1. Open `pose_estimation.ipynb` and run through it. You can preview the results [here](https://github.com/yenchenlin/iNeRF-public/blob/master/pixel-nerf/pose_estimation.ipynb). In the following, we show the overlay of images rendered with our predicted poses and the target image.

<img src="https://user-images.githubusercontent.com/7057863/161636178-c4f36310-eb62-44fc-abad-7d90b0637de6.gif" width=128>


# BibTeX

```
@inproceedings{yen2020inerf,
  title={{iNeRF}: Inverting Neural Radiance Fields for Pose Estimation},
  author={Lin Yen-Chen and Pete Florence and Jonathan T. Barron and Alberto Rodriguez and Phillip Isola and Tsung-Yi Lin},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems ({IROS})},
  year={2021}
}
```

# Acknowledgements

This implementation is based on Alex Yu's [pixel-nerf](https://github.com/sxyu/pixel-nerf).
