import os
import zipfile
from _lsprof import profiler_entry
import DataManager

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, CSVLogger

# (width, Heigth)
IMG_SHAPE = (320, 320, 3)

base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

base_model.summary()

#frezing the base model
base_model.trainable = False

#define custom last layers of the network
#seeing the output of the last layer:
print("Output of the orignal model: ", str(base_model.output))
global_avg_layer = tf.keras.layers.Flatten()(base_model.output)

#the new output layer
print("The new output layer: ", str(global_avg_layer))

#oure trainable layer
dense1 = tf.keras.layers.Dense(units=2048, activation="relu")(global_avg_layer)
dense2 = tf.keras.layers.Dense(units=1024, activation="relu")(dense1)
prediction_layer = tf.keras.layers.Dense(units=6)(dense2)

#combining the network to oure layer
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

model.summary()

#compiling the model
model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['accuracy'])


#Data agumentation
images, labels = DataManager.GetImgsRotatedAndFliped()
totalImg = len(images)
train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.SplitDataSet(images, labels)

print("TotalImgs: {} TrainSet size: {} validationSet size: {} testSet size: {}".format(totalImg, len(train_Img), len(validation_Img), len(test_Img)))
print(("showning from training"))
DataManager.showOneRandomImg(train_Img, train_Lab)
print("showing from val")
DataManager.showOneRandomImg(validation_Img, validation_Lab)

modelName = "InceptionV3"

filepath = "checkpoints/checkpoint_InceptionV3_RGB-{epoch:04d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='auto',
                            period=5)


csv_fileName = "logs/CSV_log_RGB_{}.csv".format(modelName)
logger = CSVLogger(
    csv_fileName, separator=',', append=False
)

#training
model.fit(train_Img, train_Lab, batch_size=32 , epochs=500, validation_data=(validation_Img, validation_Lab), callbacks=[checkpoint, logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))

print(evaluating)
loss,acc = model.evaluate(x= test_Img, y=test_Lab)
print("Model performance:\n loss: {} \n Accuracy: {}".format(loss,acc))
