import os
import zipfile
from _lsprof import profiler_entry
import DataManager

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, CSVLogger

# (width, Heigth)
IMG_SHAPE = (320, 320, 3)

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

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
dense1 = tf.keras.layers.Dense(units=1024, activation="relu")(global_avg_layer)
dense4 = tf.keras.layers.Dense(units=512, activation="relu")(dense1)
dense5 = tf.keras.layers.Dense(units=512, activation="relu")(dense4)
prediction_layer = tf.keras.layers.Dense(units=6)(dense5)

#combining the network to oure layer
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

model.summary()

#compiling the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mean_absolute_error'])


#Data agumentation
train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.GetMarcinDataset()

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
model.fit(np.array(train_Img), np.array(train_Lab), batch_size=64 , epochs=100, validation_data=(np.array(validation_Img), np.array(validation_Lab)), callbacks=[logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))

print("evaluating")
loss,acc = model.evaluate(x= np.array(test_Img), y=np.array(test_Lab))
print("Model performance:\n loss: {} \n Accuracy: {}".format(loss,acc))
