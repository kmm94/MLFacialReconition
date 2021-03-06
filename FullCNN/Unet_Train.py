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

import Unet

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# HyperPerameters
BATCH_SIZE = 1
# (width, Heigth, #ofChannels)
IMG_SHAPE = (320, 320, 3)

# Load the data into tf
images = []
labels = []

one_image = 0
one_label = 0

train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.GetMarcinDataset()


print("**** startinmg *****")
print("TrainSet size: {} \nValidationSet size: {} \nTestSet size: {}".format(len(train_Img), len(validation_Img), len(test_Img)))
print(("showning from training"))
DataManager.showOneRandomImg(train_Img, train_Lab)
print("showing from val")
DataManager.showOneRandomImg(validation_Img, validation_Lab)

model = Unet.get_unet(IMG_SHAPE)

model.summary()

# Keep only a single checkpoint, the best over test accuracy.
modelName = "Unet"
filepath = "checkpoints/checkpoint_Unet_RGB-{epoch:04d}-{val_loss:.2f}.h5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='val_mean_absolute_error',
                             verbose=1,
                             save_best_only=True,
                             mode='auto',
                             save_weights_only=True,
                             period=1)

csv_fileName = "logs/CSV_log_RGB_{}.csv".format(modelName)
logger = tf.keras.callbacks.CSVLogger(
    csv_fileName, separator=',', append=False
)

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])

model.fit(x=np.array(train_Img), y=np.array(train_Lab), batch_size=1, epochs=50, validation_data=(np.array(validation_Img), np.array(validation_Lab)),
          callbacks=[checkpoint, logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))

loss, acc = model.evaluate(x=np.array(test_Img), y=np.array(test_Lab))
print("Model performance:\n loss: {} \n Accuracy: {}".format(loss, acc))

test_Img_0 = test_Img[0]
test_Img_0 = np.expand_dims(test_Img_0, axis=0)

Predictions = model.predict(test_Img_0)
DataManager.showOneImg(test_Img[0], Predictions[0])





