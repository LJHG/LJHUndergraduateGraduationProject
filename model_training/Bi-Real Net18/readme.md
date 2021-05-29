# 基于Bi-Real NEt18的图像分类算法

## 1. 简介

该项目使用Bi-Real Net18实现了图像分类，并在CIFAR100数据集上完成了训练，经过训练，Bi-Real Net18能够在数据集上达到62.85%的准确度，仅略差于MobileNet的64.64%。训练得到的结果在result文件夹中。

本项目的模型搭建以及模型训练主要使用了tensorflow以及larq来进行实现。

项目的运行需要python3中安装了以下的库：

1. tensorflow2
2. larq
3. larq_compute_engine
4. larq_zoo
5. numpy
6. matlpotlib

## 2. 项目结构

这一部分介绍各个文件编写的代码的相关内容，代码内容主要分为模型搭建，模型训练以及模型转换。

### 2.1 模型搭建

模型搭建部分文件包括birealnet.py, birealnet_cifar.py, bireal_full_cifar.py, mobilenet_v1.py。

1. birealnet.py 文件是Bi-Real Net18最原始的Bi-Real Net18的模型搭建的代码，即在ImageNet数据集上实现的版本。
2. birealnet_cifar.py 文件是本文对Bi-Real Net18针对CIFAR100数据集的图片经过修改后的版本，使其能够更好地提取特征。
3. bireal_full_cifar.py 文件是birealnet_cifar中搭建模型的一个全精度版本的实现，即不使用二值量化搭建的结果，用于比较模型压缩的效果。
4. mobilenet_v1.py 文件是MobileNet模型的一种实现，用于和Bi-Real Net18在精度上进行比较。

### 2.2 模型训练

该项目在train.py文件中描述了模型训练的相关细节。

### 2.3 模型转换

该项目使用converter.py文件来实现模型从.h5格式到.tflite格式的转换。



## 3. How to run

该项目通过直接调用train.py文件来进行模型训练。

```shell
python train.py
```

其中，具体的项诸如模型选择，超参数的设置需要在文件中进行具体的设置。