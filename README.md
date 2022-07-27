# [ICCV 2021] GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning

<h4 align="center"> Haipeng Li$^1$, Kunming Luo$^1$, Shuaicheng Liu$^{2,1}$</h4>
<h4 align="center"> 1. Megvii Research, 2. University of Electronic Science and Technology of China</h4>

This is the official implementation of our ICCV2021 paper [GyroFlow](https://openaccess.thecvf.com/content/ICCV2021/html/Li_GyroFlow_Gyroscope-Guided_Unsupervised_Optical_Flow_Learning_ICCV_2021_paper.html). We also provide a PyTorch version, check at [GyroFlow-PyTorch](https://github.com/lhaippp/GyroFlow-PyTorch)


## Abstract
Existing optical flow methods are erroneous in challenging scenes, such as fog, rain, and night because the basic optical flow assumptions such as brightness and gradient constancy are broken. To address this problem, we present an unsupervised learning approach that fuses gyroscope into optical flow learning. Specifically, we first convert gyroscope readings into motion fields named gyro field. Second, we design a self-guided fusion module to fuse the background motion extracted from the gyro field with the optical flow and guide the network to focus on motion details. To the best of our knowledge, this is the first deep learning-based framework that fuses gyroscope data and image content for optical flow learning. To validate our method, we propose a new dataset that covers regular and challenging scenes. Experiments show that our method outperforms the state-of-art methods in both regular and challenging scenes.

## Presentation video
[[Youtube](https://www.youtube.com/watch?v=6gh40PyWdHM)][[Bilibili](https://www.bilibili.com/video/BV1Tr4y127kd/)].

## Dependencies

* MegEngine==1.6.0
* Other requirements please refer to`requirements.txt`.

## Data Preparation

### GOF-Train

2021.11.15: We release the GOF_Train V1 that contains 2000 samples.

2022.06.22: We release the PyTorch Version, welcome to have a try.

The download link is [GoogleDrive](https://drive.google.com/file/d/1eG9W-AlKrQ_fsxT4As6wzGaewCksYxnK/view?usp=sharing) or [CDN](https://data.megengine.org.cn/research/gyroflow/GOF_Train.zip). Put the data into `./dataset/GOF_Train`, and the contents of directories are as follows:

```
./dataset/GOF_Train
├── sample_0
│   ├── img1.png
│   ├── img2.png
│   ├── gyro_homo.npy
├── sample_1
│   ├── img1.png
│   ├── img2.png
│   ├── gyro_homo.npy
.....................
├── sample_1999
│   ├── img1.png
│   ├── img2.png
│   ├── gyro_homo.npy

```

### GOF-Clean

For quantitative evaluation, including input frames and the corresponding gyro readings, a ground-truth optical flow is required for each pair.

The download link is [GoogleDrive](https://drive.google.com/file/d/1X9V_DT1JHJti6BeWnWnqAfR4QEzvFQoE/view?usp=sharing) or [CDN](https://data.megengine.org.cn/research/gyroflow/GOF_Clean.npy). Move the file to `./dataset/GOF_Clean.npy`.

### GOF-Final

The most difficult cases are collected in GOF-Final.

The download link is [GoogleDrive](https://drive.google.com/file/d/1n1ieGkilwWraxEN6XZUX1kA-tiTgEGlw/view?usp=sharing). Move the file to `./dataset/GOF_Final.npy`.

## Training and Evaluation

### Training

To train the model, you can just run:

```
python train.py --model_dir experiments
```

### Evaluation

Load the pretrained checkpoint and run:

```
python evaluate.py --model_dir experiments --restore_file experiments/val_model_best.pkl
```

We've updated the GOF (both trainset and testset), so the performance is a little bit different from the results reported in our paper.

MegEngine checkpoint can be download via [Google Drive].

## Citation

If you think this work is useful for your research, please kindly cite:

```
@InProceedings{Li_2021_ICCV,
    author    = {Li, Haipeng and Luo, Kunming and Liu, Shuaicheng},
    title     = {GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12869-12878}
}
```

## Acknowledgments

In this project we use (parts of) the official implementations of the following works:

* [ARFlow](https://github.com/lliuz/ARFlow)
* [UpFlow](https://github.com/coolbeam/UPFlow_pytorch)
* [RAFT](https://github.com/princeton-vl/RAFT)
* [DeepOIS](https://github.com/lhaippp/DeepOIS)

We thank the respective authors for open sourcing their methods.
