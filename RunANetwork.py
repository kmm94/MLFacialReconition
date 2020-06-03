

import tensorflow as tf
import numpy as np
import cv2
import glob
import  DataManager

IMG_Channels = 3

def make_square(img, min_size=320):
    color = [0, 0, 0] # 'cause black!
    imgShape = img.shape[:2]
    addBoaderToWidth = min_size - imgShape[1]
    addBoaderToHeigth = min_size - imgShape[0]

    if(addBoaderToWidth <= 0):
        left, right = 0,0
    if (addBoaderToHeigth <= 0):
        top, bottom,= 0,0   
    if(addBoaderToWidth > 0):
        left, right = int(addBoaderToWidth/2), int(addBoaderToWidth/2)
    if (addBoaderToHeigth > 0):
        top, bottom,= int(addBoaderToHeigth/2), int(addBoaderToHeigth/2)
   
    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_with_border

def up_size(img, scale):
    # times of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return img_resized


def down_size(img):
    # times of original size
    scallingWidth = img.shape[1]/240
    scallingHeigth = img.shape[0]/320

    width = int(img.shape[1] / scallingWidth)
    height = int(img.shape[0] / scallingHeigth)
    dim = (width, height)
    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    return img_resized


def reSizeImg(img, IMG_Channels):
    if (img.shape == (120, 160, IMG_Channels)):
        return up_size(img, 2)
    elif img.shape == (240, 320, IMG_Channels):
        return up_size(img, 1)
    else:
        return down_size(img)

def testImgAsRect():
    modelFilePath = './savedModels/RGB_CNNv2_logcosh.h5'
    path_to_image = "./FinalDataSet/uSet/*.jpg"
    resizedImgs = []
    counter =0
    totalFiles = len(glob.glob(path_to_image))
    model = tf.keras.models.load_model(modelFilePath)
    for img in glob.glob(path_to_image):
        counter += 1
        print("[{}/{}]".format(counter,totalFiles))
        img = cv2.imread(img)
        img_resized = reSizeImg(img, 3)
        img_squa = make_square(img_resized)
        img_norm = img_squa/255.0


        #Put the img into the correct shape, Img needs shape (1,240,320,3)
        img_rdy = np.expand_dims(img_norm, axis=0)

        Predictions = model.predict(img_rdy)
        DataManager.showOneImg(img_squa, Predictions[0])
        print("End of test")


testImgAsRect()
#test_model()