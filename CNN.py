import datetime

import tensorflow as tf
import numpy as np
import csv
# import matplotlib.pyplot as plt
import os
import glob
import cv2
import DataManager
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# HyperPerameters
BATCH_SIZE = 1
IMG_HEIGHT = int(480/2)
IMG_WIDTH = int(640/2)
IMG_Channels = 3

# Load the data into tf
images = []
labels = []

one_image = 0
one_label = 0


images, labels = DataManager.getColorImages()

print(images[1].shape)
#showOneImg(one_image, one_label)
npImgArray = np.array(images)
npLabelArray = np.array(labels)

print("imgArray Shape: " + str(npImgArray.shape))
print("imgArray type: " + str(npImgArray.dtype))
print("LabelArray shape: "+ str(npLabelArray.shape))

if not issubclass(npImgArray.dtype.type, np.float):
    raise TypeError('float type expected')

 # Defining the model:
 # https://github.com/yinguobing/cnn-facial-landmark/blob/master/model.py
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_Channels)))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=1024, activation='relu'))
model.add(tf.keras.layers.Dense(units=6))

model.summary()

# Keep only a single checkpoint, the best over test accuracy.
modelName = "yinguobingWideDens"
filepath = "checkpoints/checkpoint_yinguobingWideDens_RGB-{epoch:04d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='auto',
                             period=50)


csv_fileName = "logs/CSV_log_RGB_{}.csv".format(modelName)
logger = tf.keras.callbacks.CSVLogger(
    csv_fileName, separator=',', append=False
)
 

model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["accuracy"])

model.fit(npImgArray, npLabelArray, epochs=1000, validation_split=0.2, callbacks=[checkpoint, logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))



