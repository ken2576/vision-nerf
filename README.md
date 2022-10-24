# Vision Transformer for NeRF-Based View Synthesis from a Single Input Image

[![arXiv](https://img.shields.io/badge/arXiv-2207.05736-b31b1b.svg)](https://arxiv.org/abs/2207.05736)

![car_gif](./gifs/000687.gif)

![chair_gif](./gifs/000931.gif)

Official PyTorch Implementation of paper "Vision Transformer for NeRF-Based View Synthesis from a Single Input Image", WACV 2023.

[Kai-En Lin](https://cseweb.ucsd.edu/~k2lin/)<sup>1*</sup>
[Lin Yen-Chen](https://yenchenlin.me/)<sup>2</sup>
[Wei-Sheng Lai](https://www.wslai.net/)<sup>3</sup>
[Tsung-Yi Lin](https://tsungyilin.info/)<sup>4**</sup>
[Yi-Chang Shih](https://www.yichangshih.com/)<sup>3</sup>
[Ravi Ramamoorthi](https://cseweb.ucsd.edu/~ravir/)<sup>1</sup>

<sup>1</sup>University of California, San Diego, <sup>2</sup>Massachusetts Institute of Technology, <sup>3</sup>Google, <sup>4</sup>NVIDIA

<sup>*</sup> Work done while interning at Google,
<sup>**</sup> Work done while at Google.


[Project Page](https://cseweb.ucsd.edu/~viscomp/projects/VisionNeRF/) | [Paper](https://arxiv.org/abs/2207.05736) | [Pretrained models](https://drive.google.com/drive/folders/1OAcwNPxBwaE8aY-0xrHreyP-EWmQYaYJ?usp=sharing)

## Requirements

### Install required packages

Make sure you have up-to-date NVIDIA drivers supporting CUDA 11.1 (10.2 could work but need to change `cudatoolkit` package accordingly)

Run

```
conda env create -f environment.yml
conda activate visionnerf
```


## Usage

1. Clone the repository ```git clone https://github.com/ken2576/vision-nerf.git``` and download dataset from [PixelNeRF](https://github.com/sxyu/pixel-nerf#getting-the-data).

2. Download [pretrained model weights](https://drive.google.com/drive/folders/1OAcwNPxBwaE8aY-0xrHreyP-EWmQYaYJ?usp=sharing).

    Here is a list of the model weights:

    * `nmr_500000.pth`: Our pretrained weights for the category-agnostic experiment.
    * `srn_cars_500000.pth`: Our pretrained weights for the category-specific experiment on ShapeNet Cars.
    * `srn_chairs_500000.pth`: Our pretrained weights for the category-specific experiment on ShapeNet Chairs.


3. Install requirements ```conda env create -f environment.yml```.

4. Setup configurations in ```configs```.

5. (Optional) Run training script with ```python train.py --config [config_path]```.

6. Run inference script with our [pretrained models](https://drive.google.com/drive/folders/1OAcwNPxBwaE8aY-0xrHreyP-EWmQYaYJ?usp=sharing):
```
python eval.py --config [path to config file] # For ShapeNet Cars/Chairs
python eval_nmr.py --config [path to config file] # For NMR
python gen_real.py --config [path to config file] # For real car data
```

### Prepare real data

Our pretrained model works with real car images.
You can prepare the data using [the same process as PixelNeRF](https://github.com/sxyu/pixel-nerf#real-car-images).

Then, run `gen_real.py` similar to the above example.

## Acknowledgement

This code is based on [DPT](https://github.com/isl-org/DPT), [IBRNet](https://github.com/googleinterns/IBRNet) and [PixelNeRF](https://github.com/sxyu/pixel-nerf).

## Citation
```
@inproceedings {lin2023visionnerf,
    booktitle = {WACV},
    title = {Vision Transformer for NeRF-Based View Synthesis from a Single Input Image},
    author = {Lin, Kai-En and Yen-Chen, Lin and Lai, Wei-Sheng and Lin, Tsung-Yi and Shih, Yi-Chang and Ramamoorthi, Ravi},
    year = {2023},
}
```