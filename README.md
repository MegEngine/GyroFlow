# [ICCV 2021] GyroFlow: Gyroscope-Guided Unsupervised Optical Flow Learning

This is the official implementation of our ICCV2021 paper [GyroFlow](https://openaccess.thecvf.com/content/ICCV2021/html/Li_GyroFlow_Gyroscope-Guided_Unsupervised_Optical_Flow_Learning_ICCV_2021_paper.html).

Our presentation video: [[Youtube](https://www.youtube.com/watch?v=6gh40PyWdHM)][[Bilibili](https://www.bilibili.com/video/BV1Tr4y127kd/)].

## Our Poster

![image](gyroflow_poster.jpg)

## Dependencies

* MegEngine==1.6.0
* Other requirements please refer to`requirements.txt`.

## Data Preparation

### GOF-Train

2021.11.15: We release the GOF_Train V1 that contains 2000 samples.

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

The download link is [GoogleDrive](https://drive.google.com/file/d/1BMuTNygl_W0xWMoHlck_h60inyu7BE1x/view?usp=sharing) or [CDN](https://data.megengine.org.cn/research/gyroflow/GOF_Final.npy). Move the file to `./dataset/GOF_Final.npy`.

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

MegEngine checkpoint can be download via [Google Drive](https://drive.google.com/file/d/18lK8MnOoE94MDoCjFtikvtdtd786DWLn/view?usp=sharing) or [CDN](https://data.megengine.org.cn/research/gyroflow/val_model_best.pkl).

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
