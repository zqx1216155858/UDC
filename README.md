# UDC
---

## 目录




1. [Performance](#性能情况)
2. [Environment](#所需环境)
3. [Download](#文件下载)
4. [How2train](#训练步骤)
5. [How2predict](#预测步骤)
7. [Reference](#Reference)



## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 |  | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VisDrone2019-DET-train | yolov7-tiny-best.pt | VisDrone2019-DET-val |  | 36.8 | 54.4|

## 所需环境
软件环境：该储存库是在pytorch 1.9.0+cu111中构建的

1. 克隆我们的储存库

```python
git clone https://github.com/zqx1216155858/yolov7-tiny.git
cd yolov7-tiny
```

2. 制作conda环境

```python
conda create -n yolov7-tiny python=3.8
conda activate yolov7-tiny
```

3. 安装依赖库

```python
conda install pytorch=1.9.0+cu111  -c pytorch
pip install -r requirements.txt
```

硬件环境：NVIDIA  A100

## 文件下载
**数据集**

https://github.com/VisDrone/VisDrone-Dataset

Task 1: Object Detection in Images

## 训练步骤


**1. 数据集的准备**  
  训练前将标签文件放在datasets文件夹下的VisDrone文件夹下的labels中。   
  训练前将图片文件放在datasets文件夹下的VisDrone文件夹下的images中。   

**2. 开始网络训练**  

```python
python train.py --workers 8 --device 0 --batch-size 32 --data data/visdrone.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```

## 预测步骤
1.**测试**

```python
python test.py --data data/visdrone.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7-tiny-best.pt --name yolov7_val
```

**2.推理**

​	视频上

```python
python detect.py --weights yolov7-tiny-best.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

​	图像上

```python
python detect.py --weights yolov7-tiny-best.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

## Reference
https://github.com/zqx1216155858/yolov7-tiny.git
