import datetime

import tensorflow as tf
import numpy as np
import csv
# import matplotlib.pyplot as plt
import os
import glob
import cv2
import NetworkHelper
import DataManager
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# HyperPerameters
BATCH_SIZE = 1
# (width, Heigth, #ofChannels)
IMG_SHAPE = (320, 320, 3)

# Load the data into tf
train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.GetMarcinDataset()


print("**** starting *****")
print(("showning from training"))
DataManager.showOneRandomImg(train_Img, train_Lab)
print("showing from val")
DataManager.showOneRandomImg(validation_Img, validation_Lab)

# print("imgArray Shape: " + str(npImgTrainArray.shape))
# print("imgArray type: " + str(npImgTrainArray.dtype))
# print("LabelArray shape: "+ str(npLabelTrainArray.shape))
#
# if not issubclass(npImgTrainArray.dtype.type, np.float):
#    raise TypeError('float type expected')

# Defining the model:
# https://github.com/yinguobing/cnn-facial-landmark/blob/master/model.py
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=IMG_SHAPE))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=6))

model.summary()

# Keep only a single checkpoint, the best over test accuracy.
modelName = "CNNv2_logcosh"
filepath = "checkpoints/chpt_CNNv2_RGB-{epoch:04d}-loss-{val_loss:.2f}-Metric-{val_mean_absolute_error:.1f}.h5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_mean_absolute_error',
                             verbose=1,
                             mode='auto',
                             period=1)

csv_fileName = "logs/CSV_log_RGB_{}.csv".format(modelName)
logger = tf.keras.callbacks.CSVLogger(
    csv_fileName, separator=',', append=False
)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])

model.fit(x=np.array(train_Img), y=np.array(train_Lab), epochs=100, validation_data=(np.array(validation_Img), np.array(validation_Lab)),
          callbacks=[checkpoint, logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))

loss, acc = model.evaluate(x=np.array(test_Img), y=np.array(test_Lab))
print("Model performance:\n loss: {} \n Accuracy: {}".format(loss, acc))

test_Img_0 = test_Img[0]
test_Img_0 = np.expand_dims(test_Img_0, axis=0)

Predictions = model.predict(test_Img_0)
DataManager.showOneImg(test_Img[0], Predictions[0])





