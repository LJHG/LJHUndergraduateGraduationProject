# 基于二值权重LSTM的图像分类算法

## 1. 简介

该项目对LSTM的权重进行了二值化，并在MNIST数据集上进行了验证，目的是为了探究二值化对于LSTM的影响。训练的结果在result文件夹中。

本项目的模型搭建以及模型训练主要使用了pytorch来进行实现。

项目的运行需要python3中安装了以下的库：

1. torch
6. matlpotlib

## 2. 项目结构

这一部分介绍各个文件编写的代码的相关内容。

1. LSTMCell.py。该文件是对于LSTM单元的基于pytorch的一种实现。
2. BinaryLSTMCell.py。该文件基于LSTMCell.py对LSTM的权重进行了二值化。
3. MultilayerLSTM.py。该文件对LSTM的单元进行了连接和封装，并支持多层的LSTM堆叠，以此来提供给外部进行调用。
4. train.py。该文件描述了模型训练相关的内容。
5. utils.py。该文件对相关工具的代码进行了编写，其中包括对矩阵求L1范数的实现。



## 3. How to run

该项目通过直接调用train.py文件来进行模型训练。

```shell
python train.py
```

其中，具体的项诸如模型选择，超参数的设置需要在文件中进行具体的设置。