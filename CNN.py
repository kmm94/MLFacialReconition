import datetime

import tensorflow as tf
# import IPython.display as display
from PIL import Image
import numpy as np
import csv
# import matplotlib.pyplot as plt
import os
import glob
import cv2
from tensorflow_core.python.keras.callbacks import ModelCheckpoint

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

# HyperPerameters
BATCH_SIZE = 1
IMG_HEIGHT = int(480/2)
IMG_WIDTH = int(640/2)
IMG_Channels = 1

# Load the data into tf
images = []
labels = []
path_to_image = "FinalDataSet/Images/*.BMT"
path_to_labels = "FinalDataSet/FinalDataPoints.csv"


def getImageName(image_path):
    img_parts = image_path.split(os.path.sep)
    return img_parts[-1]


index = 0
one_image = 0
one_label = 0

def size(img, labels, scale):
    # times of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    labels_resized = []
    for i in labels:
        labels_resized.append(i * scale)
    return img_resized, labels_resized


def reSizeImgAndLabels(img, labels):
    if (img.shape == (120, 160,IMG_Channels)):
        return size(img, labels, 2)
    else:
        return size(img, labels, 1)

def mirrorIMG(image, label):
    fliped_img = cv2.flip(image, 1)
    fliped_labels = []
    #right eye
    fliped_labels.append(int(IMG_HEIGHT - label[0]))
    fliped_labels.append(int(IMG_WIDTH - label[1]))
    #left eye
    fliped_labels.append(int(IMG_HEIGHT - label[2]))
    fliped_labels.append(int(IMG_WIDTH - label[3]))
    #nose
    fliped_labels.append(int(IMG_HEIGHT - label[4]))
    fliped_labels.append(int(IMG_WIDTH - label[5]))
    return fliped_img, fliped_labels

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' %
          (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

print("Image Augementation started")
totalFiles = len(glob.glob(path_to_image))
counter =0
for image_path in glob.glob(path_to_image):
    counter += 1
    print("[{}/{}]".format(totalFiles, counter))
    image_name = getImageName(image_path)
    labels_csv = open(path_to_labels, "r")
    for row in csv.reader(labels_csv, delimiter=","):
        if (row[0] == image_name):
            img_raw_labels = [int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))),
                         int(round(float(row[4]))), int(round(float(row[5]))), int(round(float(row[6])))]
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=2)
            imgResape, labelResape = reSizeImgAndLabels(img, img_raw_labels)
            #one_image, one_label = mirrorIMG(imgResape, labelResape)
            one_image = imgResape
            one_label = labelResape
            imgResape = np.expand_dims(imgResape, axis=2)
            imgNormalization = imgResape / 255.0
            labels.append(labelResape)
            images.append(imgNormalization)


# show one image with spots
def showOneImg(image, label):
    fillTheCircle = -1
    color = (0, 0, 0)
    radius = 2
    coordianates1 = (label[0], label[1])
    img = cv2.circle(image, coordianates1, radius,
                     color, fillTheCircle)

    coordianates2 = (label[2], label[3])
    img1 = cv2.circle(img, coordianates2, radius,
                      color, fillTheCircle)

    coordianates3 = (label[4], label[5])
    img2 = cv2.circle(img1, coordianates3, radius,
                      color, fillTheCircle)
    print("Image dimensions: {}".format(image.shape))
    print("Image labels: {}".format(label))
    cv2.imshow("TestImage", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#showOneImg(one_image, one_label)
npImgArray = np.array(images)
npLabelArray = np.array(labels)

print(npImgArray.shape)
print(npImgArray.dtype)
print(npLabelArray.shape)

if not issubclass(npImgArray.dtype.type, np.float):
    raise TypeError('float type expected')

 # Defining the model:
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_Channels)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.Dense(units=64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(units=6))

model.summary()

# Keep only a single checkpoint, the best over test accuracy.
filepath = "checkpoints/checkpoint-{epoch:04d}.h5"
checkpoint = ModelCheckpoint(filepath,
                            monitor='accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='auto')


csv_fileName = "logs/CSV_log.csv"
logger = tf.keras.callbacks.CSVLogger(
    csv_fileName, separator=',', append=False
)

model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["accuracy"])

model.fit(npImgArray, npLabelArray, epochs=10, batch_size=5, validation_split=0.2, callbacks=[checkpoint, logger])

model.save("./savedModels/myModel3.h5")

def test_model(model):
    list_to_validation = "./FinalDataSet/FlirOneTestBilleder/*.jpg"
    for i in range(1, 10):
        img_resized, labels_resized = size(npImgArray[i], npLabelArray[i], 1)
        test_img_input = np.reshape(img_resized, (1,240,320,1))
        prediction = model.predict(test_img_input)  # shape = [batch_size, values]
        print(prediction[0])
        showOneImg(img_resized, prediction[0])

test_model(model)


