import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,BatchNormalization
import larq as lq
import larq_compute_engine as lce
import tensorflow as tf


m = tf.keras.models.load_model("mobilev1.h5")
lq.models.summary(m)
with open("mobilenetV1.tflite", "wb") as flatbuffer_file:
    flatbuffer_bytes = lce.convert_keras_model(m)
    flatbuffer_file.write(flatbuffer_bytes)