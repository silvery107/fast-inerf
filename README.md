# Fast iNeRF

## TODO

- [ ] Do sampling only on pixels within an object, using masks of each object class from PoseCNN
- [ ] Set a initial guess of camera pose, using the estimated object pose from PoseCNN
- [ ] Train a NeRF model on PROPS Datatset, using [this script](https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py) to transfer image sequences to a NeRF datatset.

#### Test Scripts
- `python tests/test_posecnn.py` for training and evaluating PoseCNN
- `python tests/test_inerf.py --config configs/lego.txt` for optimizing iNeRF

### [Project Page](https://yenchenlin.me/inerf/) | [Video](https://www.youtube.com/watch?v=eQuCZaQN0tI&feature=emb_logo) | [Paper](https://arxiv.org/pdf/2012.05877.pdf)

<img src="https://user-images.githubusercontent.com/7057863/161620132-2ce16dca-53f6-413d-97ab-fe6086f1661c.gif" height=200>

PyTorch implementation of iNeRF, an RGB-only method that inverts neural radiance fields (NeRFs) for 6DoF pose estimation.


## Installation

To start, install `pytorch` and `torchvision` according to your own GPU version, and then create the environment using conda:
- Yulun: tested with `pytorch==1.11` and `torchvision==0.12`
- Sibo: tested with `pytorch==1.13` and `torchvision==0.14`
- If you see `ParseException: Expected '}', found '=' (at char 759), (line:34, col:18)` error, check [here](https://github.com/sxyu/pixel-nerf/issues/61)
```sh
git clone git@github.com:silvery107/fast-iNeRF.git
cd fast-iNeRF
conda env create -f environment.yml
conda activate inerf
```
Download pretrained NeRF and PoseCNN models [here](https://drive.google.com/drive/folders/1WdyWak9-75OHoA7rJ2Frxghq6LSe3q71?usp=share_link) and place them in `<checkpoints>` folder.

Download `PROPS-Pose-Dataset` [here](https://drive.google.com/file/d/15rhwXhzHGKtBcxJAYMWJG7gN7BLLhyAq/view) and extract it to `<data>` folder.

## Quick Start
To run the algorithm on _Lego_ object
```
python tests/test_inerf.py --config configs/lego.txt
```
If you want to store gif video of optimization process, set ```overlay = True```

All other parameters such as _batch size_, _sampling strategy_, _initial camera error_ you can adjust in corresponding config [files](https://github.com/silvery107/fast-iNeRF/tree/main/configs).

To run the algorithm on the llff dataset, just download the "nerf_llff_data" folder from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put the downloaded folder in the "data" folder.

All NeRF models were trained using this code [https://github.com/yenchenlin/nerf-pytorch/](https://github.com/yenchenlin/nerf-pytorch/)
```
├── data 
│   ├── nerf_llff_data   
│   ├── nerf_synthetic  
```

## Different Sampling Strategies 

![](https://user-images.githubusercontent.com/63703454/122686222-51e1e300-d210-11eb-8f4c-be25f078ffa9.gif)
![](https://user-images.githubusercontent.com/63703454/122686229-58705a80-d210-11eb-9c0f-d6c2208b5457.gif)
![](https://user-images.githubusercontent.com/63703454/122686235-5ad2b480-d210-11eb-87ec-d645ae07b8d7.gif)

Left - **random**, in the middle - **interest points**, right - **interest regions**. 
Interest regions sampling strategy provides faster convergence and doesnt stick in a local minimum like interest points. 

<!-- ## Citation

```
@inproceedings{yen2020inerf,
  title={{iNeRF}: Inverting Neural Radiance Fields for Pose Estimation},
  author={Lin Yen-Chen and Pete Florence and Jonathan T. Barron and Alberto Rodriguez and Phillip Isola and Tsung-Yi Lin},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems ({IROS})},
  year={2021}
}
``` -->