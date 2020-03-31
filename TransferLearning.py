import os
import zipfile
from _lsprof import profiler_entry
import DataManager

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, CSVLogger


IMG_SHAPE = (320, 320, 3)

base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

base_model.summary()

#frezing the base model
base_model.trainable = False

#define custom last layers of the network
#seeing the output of the last layer:
print("Output of the orignal model: ", str(base_model.output))
global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

#the new output layer
print("The new output layer: ", str(global_avg_layer))

#oure trainable layer
prediction_layer = tf.keras.layers.Dense(units=6)(global_avg_layer)

#combining the network to oure layer
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

model.summary()

#compiling the model
model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['accuracy'])


#Data agumentation
images, labels = DataManager.getColorImagesAsRect()
npImgArray = np.array(images)
npLabelArray = np.array(labels)

modelName = "InceptionV3"

filepath = "checkpoints/checkpoint_InceptionV3_RGB-{epoch:04d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='auto',
                            period=50)


csv_fileName = "logs/CSV_log_RGB_{}.csv".format(modelName)
logger = CSVLogger(
    csv_fileName, separator=',', append=False
)

#training
model.fit(npImgArray, npLabelArray, epochs=1000, validation_split=0.2, callbacks=[checkpoint, logger])

model.save("./savedModels/RGB_{}.h5".format(modelName))

