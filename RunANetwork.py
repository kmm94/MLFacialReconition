

import tensorflow as tf
import numpy as np
import cv2
import glob

from tensorflow_core.python.keras.saving import load_model

IMG_Channels = 1


def up_size(img, labels, scale):
    # times of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    labels_resized = []
    for i in labels:
        labels_resized.append(i * scale)
    return img_resized, labels_resized


def down_size(img, labels):
    # times of original size
    width = int(img.shape[1] / 3.375)
    height = int(img.shape[0] / 6)
    dim = (width, height)
    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    labels_resized = []
    index = 0;
    for i in labels:
        if (index % 2 == 0):
            labels_resized.append(int(i / 3.375))
        else:
            labels_resized.append(int(i / 6))
        index +=1
    return img_resized, labels_resized


def reSizeImgAndLabels(img, labels):
    if (img.shape == (120, 160,IMG_Channels)):
        return up_size(img, labels, 2)
    elif img.shape == (240, 320, IMG_Channels):
        return up_size(img, labels, 1)
    else:
        return down_size(img, labels)

def showOneImg(image, label):
    fillTheCircle = -1
    radius = 2
    coordianates1 = (label[0], label[1])
    img = cv2.circle(image, coordianates1, radius,
                     (0,0,255), fillTheCircle)

    coordianates2 = (label[2], label[3])
    img1 = cv2.circle(img, coordianates2, radius,
                      (0,255,0), fillTheCircle)

    coordianates3 = (label[4], label[5])
    img2 = cv2.circle(img1, coordianates3, radius,
                      (0,0,0), fillTheCircle)
    cv2.imshow("TestImage", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_model():
    path_to_images = "./FinalDataSet/FlirOneTestBilleder/*.jpg"
    path_to_model = "savedModels/BAWresearchA.h5"
    list_of_filePaths = []
    model = load_model(path_to_model)
    for file_path in glob.glob(path_to_images):
        list_of_filePaths.append(file_path)

    for i in range(1, len(list_of_filePaths)):
        img = cv2.imread(list_of_filePaths[i])
        print("Proping imgfile: " + list_of_filePaths[i] + "IMG orignial shape: " + str(img.shape))
        #To gray scale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #To propper with and height
        width = 320
        height = 240
        dim = (width, height)
        img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

        #Normalize img
        img = img_resized/255.0

        #Put the img into the correct shape, Img needs shape (1,240,320,1)
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)

        #input into the network
        prediction = model.predict(img)  # shape = [batch_size, values]
        print("The Predictions: \n Højre øje: ({}, {}) \n Venstre Øje: ({}, {}) \n  Næse: ({}, {})".format(int(prediction[0][0]), int(prediction[0][1]), int(prediction[0][2]),
                                                                                                                                  int(prediction[0][3]), int(prediction[0][4]), int(prediction[0][5])))

        print("The Predictions on original size Img: \n Højre øje: ({}, {}) \n Venstre Øje: ({}, {}) \n  Næse: ({}, {})".format(int(prediction[0][0])*6, int(prediction[0][1]*3.375), int(prediction[0][2])*6,
                                                                                                                                  int(prediction[0][3]*3.375), int(prediction[0][4])*6, int(prediction[0][5]*3.375)))
        showOneImg(img_resized, prediction[0])

test_model()