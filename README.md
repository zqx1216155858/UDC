# UDC
---
>Under-display camera (UDC) image restoration is a crucial aspect supporting the full-screen smartphones. 
However, the absence of a dedicated facial UDC dataset, given its role as the front-facing camera, limits this work to scene restoration only.
As collecting aligned facial UDC images is almost impossible, We propose a generative model named degradation learning generative network (DLGNet),  designed to progressively learn multi-scale complex degradations, simulating the degradation process of UDC images. 
Next, we combine the Flickr-Faces-HQ dataset and employ a pixel-level discriminator along with supervised training to simulate UDC degradation, resulting in the generation of the facial UDC dataset.  
Furthermore, we designed an multi-resolution progressive transformer (MRPFormer) for facial UDC image restoration, employing a multi-resolution progressive learning approach to  hierarchically reconstruct global facial information.  On the UDC benchmark. Our approach outperforms previous models by 5.17 dB on the P-OLED track and exceeds by 0.93 dB on the T-OLED track.
---
## Contents
1. [Architecture](#Architecture)
2. [Environment](#Environment)
3. [Performance](#Performance)
4. [Dataset](#FFHQ_UDC)

## Architecture
![DLGNet](https://github.com/zqx1216155858/UDC/assets/112853772/002f50db-e6a0-4ff3-883c-f1a5153eab0f)


## Environment

* Python3(Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
* NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
* Python packages
```python
git clone https://github.com/zqx1216155858/UDC.git
cd UDC
conda create -n UDC python=3.8
conda activate UDC
conda install pytorch=1.9.0+cu111  -c pytorch
pip install -r requirements.txt
```

## Performance
*DLGNet

| Train Datasets | Weight |  PSNR | SSIM |
| :-----: | :-----: | :------: | :------: |
| [UDC Taining Dataset(POLED)](https://drive.google.com/file/d/1zB1xoxKBghTTq0CKU1VghBoAoQc5YlHk/view) | [DLGNet_poled](https://drive.google.com/drive/folders/1gyZQ9Rjokv0YhtqyctkSyGzoNVzWpSuq) |  35.50 | 0.970|
| [UDC Taining Dataset(TOLED)](https://drive.google.com/file/d/1zB1xoxKBghTTq0CKU1VghBoAoQc5YlHk/view) | [DLGNet_toled](https://drive.google.com/drive/folders/1gyZQ9Rjokv0YhtqyctkSyGzoNVzWpSuq) |  42.80 | 0.986|


## FFHQ_UDC
*[FFHQ_UDC](https://drive.google.com/drive/folders/1gyZQ9Rjokv0YhtqyctkSyGzoNVzWpSuq)

