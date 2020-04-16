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
images, labels = DataManager.GetImgsRotatedAndFliped([90,180])
totalImg = len(images)

train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.SplitDataSet(images, labels)


print("**** startinmg *****")
print("TotalImgs: {} TrainSet size: {} validationSet size: {} testSet size: {}".format(totalImg, len(train_Img), len(validation_Img), len(test_Img)))
print(("showning from training"))
DataManager.showOneRandomImg(train_Img, train_Lab)
print("showing from val")
DataManager.showOneRandomImg(validation_Img, validation_Lab)

 # Defining the model:
 # inspiration Unet

inputLayer = tf.keras.layers.Input(IMG_SHAPE)
dconv1 = Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(inputLayer)
maxPool1 = MaxPool2D(pool_size=2, strides=2)(dconv1)
dconv2 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(maxPool1)
dconv2a = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(dconv2)
maxpo2 = MaxPool2D(pool_size=2, strides=2)(dconv2a)
dconv3 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(maxpo2)
dconv3a = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(dconv3)
dropo1 = Dropout(0.5)(dconv3a)
maxPo3 = MaxPool2D(pool_size=2, strides=2)(dropo1)
dconv4 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(maxPo3)
dconv4 = Conv2D(filters=256, kernel_size=3, padding="same", activation="relu")(dconv4)
dropo2 = Dropout(0.5)(dconv4)
upsam1 = UpSampling2D((2,2))(dropo2)
uconv1 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(upsam1)
conca1 = Concatenate(axis=3)([dconv3a, uconv1])
uconv2 = Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")(conca1)
upsam2 = UpSampling2D((2,2))(uconv2)
uconv3 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(upsam2)
conca2 = Concatenate(axis=3)([dconv2a, uconv3])
uconv4 = Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(conca2)
flatting = tf.keras.layers.Flatten()(uconv4)
dense1 = tf.keras.layers.Dense(units=64, activation='relu')(flatting)
output = tf.keras.layers.Dense(units=6)(dense1)

model = tf.keras.models.Model(inputLayer,  output)

model.summary()

# Keep only a single checkpoint, the best over test accuracy.
modelName = "EncoderDecoder"
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
 

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(x=train_Img, y=train_Lab, epochs=500, batch_size=1, validation_data=(validation_Img, validation_Lab), callbacks=[checkpoint, logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))

loss, acc = model.evaluate(x=test_Img, y=test_Lab)
print("Model performance:\n loss: {} \n Accuracy: {}".format(loss, acc))



