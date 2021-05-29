import tensorflow as tf
import larq as lq
import numpy as np
import matplotlib.pyplot as plt
import larq_zoo
from birealnet import getBiRealNet
from birealnet_cifar import getBiRealNetCifar
from bireal_full_cifar import getBiRealNetFullCifar
from mobilenet_v1 import getMobileNet
#实时数据增强功能 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.random.set_seed(777)#可复现


# 数据集读取和切分
num_classes = 100

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

#数据提取
#z-score标准化
mean = np.mean(train_images, axis=(0, 1, 2, 3))#四个维度 批数 像素x像素 通道数
std = np.std(train_images, axis=(0, 1, 2, 3))

train_images = (train_images - mean) / (std + 1e-7)#trick 加小数点 避免出现整数 
test_images = (test_images - mean) / (std + 1e-7) 

# one-hot独热映射
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# 数据增强
datagen = ImageDataGenerator(
    featurewise_center=False,  # 布尔值。将输入数据的均值设置为 0，逐特征进行。
    samplewise_center=False,  # 布尔值。将每个样本的均值设置为 0。
    featurewise_std_normalization=False,  # 布尔值。将输入除以数据标准差，逐特征进行。
    samplewise_std_normalization=False,  # 布尔值。将每个输入除以其标准差。
    zca_whitening=False,  # 布尔值。是否应用 ZCA 白化。
    #zca_epsilon  ZCA 白化的 epsilon 值，默认为 1e-6。
    rotation_range=15,  # 整数。随机旋转的度数范围 (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # 布尔值。随机水平翻转。
    vertical_flip=False)  # 布尔值。随机垂直翻转。

datagen.fit(train_images)


epochs = 100
bestModel = None
bestAcc=0.0
bestLr = 0.0
lr_x = 0.01 #学习率
lr_decay = 1e-6 #学习衰减
lr_drop = 20 #衰减倍数

def lr_scheduler(epoch):
    return lr_x * (0.5 ** (epoch // lr_drop))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# model = getBiRealNet()
model = getBiRealNetCifar()
# model = getBiRealNetFullCifar()
# model = getMobileNet()
lq.models.summary(model)
print("lr is....:"+str(lr_x))
# training 
model.compile(
    tf.keras.optimizers.Adam(learning_rate=lr_x,decay=1e-6), # , decay=0.0001
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trained_model = model.fit(
    datagen.flow(train_images,train_labels,batch_size=64),
    # train_images, 
    # train_labels,
    # batch_size=64, 
    callbacks=[reduce_lr],
    epochs=epochs,
    validation_data=(test_images, test_labels),
    shuffle=True
)

# output
plt.plot(trained_model.history['accuracy'])
plt.plot(trained_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("acc_"+str(lr_x)+".png")
plt.clf()

print(np.max(trained_model.history['accuracy']))
print(np.max(trained_model.history['val_accuracy']))


plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss_"+str(lr_x)+".png")
plt.clf()

print(np.min(trained_model.history['loss']))
print(np.min(trained_model.history['val_loss']))

if(np.max(trained_model.history['val_accuracy']) > bestAcc):
    bestModel = model
    bestAcc = np.max(trained_model.history['val_accuracy'])
    bestLr = lr_x

bestModel.save("mobilenet.h5")
print("best acc is...: "+str(bestAcc))