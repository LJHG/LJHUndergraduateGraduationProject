# 二值网络的推理部署

## 1. 简介

此处展示的是使用C++编写相关来进行神经网络推理的流程，在这里使用了LCE(Larq Compute Engine)来实现推理。由于需要将模型部署到Android设备上，此处编写的代码使用到了JNI(Java Native Interface)。



## 2. How to run

1. 编写相关的C++代码以及相关的build文件，具体可以见LCE/jni_lce/lce.cc以及LCE/jni_lce/BUILD

2. 在LCE根目录下使用bazel对该项目进行编译。如下：

   ```shell
   bazel build  --config=android_arm64 //jni_lce:liblce.so
   ```

3. 编译后会生成一个LCE/bazel-bin文件夹。LCE/bazel-bin/jni_lce文件夹下找到liblce.so动态链接库文件，得到这个文件后，就可以将该文件添加到Android Studio中的工程项目中，并使用相关java进行调用了。

