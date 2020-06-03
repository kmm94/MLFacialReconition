import datetime

import tensorflow as tf
import numpy as np
import csv
# import matplotlib.pyplot as plt
import os
import glob
import cv2
from tensorflow_core.python.keras.layers import Conv2D, MaxPool2D, Dropout, UpSampling2D, Concatenate

import DataManager
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# (width, Heigth, #ofChannels)
IMG_SHAPE = (320, 320, 3)

# Load the data into tf
train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.GetMarcinDataset()

print("**** startinmg *****")
print(("showning from training"))
DataManager.showOneRandomImg(train_Img, train_Lab)
print("showing from val")
DataManager.showOneRandomImg(validation_Img, validation_Lab)

# Defining the model:
# inspiration Unet

inputLayer = tf.keras.layers.Input(IMG_SHAPE)
conv1 = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(inputLayer)
conv1a = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(conv1)
maxPool1 = MaxPool2D(pool_size=2, strides=2)(conv1a)

dconv2 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(maxPool1)
dconv2a = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(dconv2)
maxpo2 = MaxPool2D(pool_size=2, strides=2)(dconv2a)

dconv3 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(maxpo2)
dconv3a = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(dconv3)
maxPo3 = MaxPool2D(pool_size=2, strides=2)(dconv3a)

dconv4 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(maxPo3)
dconv4 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(dconv4)

upsam1 = UpSampling2D((2, 2))(dconv4)
uconv1 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(upsam1)
conca1 = Concatenate(axis=3)([dconv3a, uconv1])
uconv2 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conca1)
uconv2 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(uconv2)

upsam2 = UpSampling2D((2, 2))(uconv2)
uconv3 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(upsam2)
conca2 = Concatenate(axis=3)([dconv2a, uconv3])
uconv4 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conca2)
uconv3 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(upsam2)

upsam3 = UpSampling2D((2, 2))(uconv3)
uconv4a = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(upsam3)
conca3 = Concatenate(axis=3)([conv1a, uconv4a])
uconv5 = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(conca3)

finalConv = Conv2D(filters=2, kernel_size=2, padding="same", activation="relu")(uconv5)
flatting = tf.keras.layers.Flatten()(finalConv)
dense1 = tf.keras.layers.Dense(units=128, activation='relu')(flatting) #changed from 256
output = tf.keras.layers.Dense(units=6)(dense1)

model = tf.keras.models.Model(inputLayer, output)

model.summary()

# Keep only a single checkpoint, the best over test accuracy.
modelName = "SmallEncDec_OurImgs128"
filepath = "checkpoints/ckpt_SmallEncDec_OurImgs128-{epoch:04d}-{val_mean_absolute_error:.2f}.h5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_mean_absolute_error',
                             verbose=1,
                             save_best_only=True,
                             mode='auto',
                             period=1)

csv_fileName = "logs/log_{}.csv".format(modelName)
logger = tf.keras.callbacks.CSVLogger(
    csv_fileName, separator=',', append=False
)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])

model.fit(x=np.array(train_Img), y=np.array(train_Lab), epochs=300, batch_size=1, validation_data=(np.array(validation_Img), np.array(validation_Lab)),
          callbacks=[checkpoint, logger])

model.save("./savedModels/{}.h5".format(modelName))

loss, acc = model.evaluate(x=test_Img, y=test_Lab)
print("Model performance:\n loss: {} \n Accuracy: {}".format(loss, acc))



