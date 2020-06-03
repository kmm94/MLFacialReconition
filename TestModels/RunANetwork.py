

import tensorflow as tf
import numpy as np
import cv2
import glob

IMG_Channels = 3


def make_square(img, min_size=320):
    color = [0, 0, 0]  # 'cause black!
    imgShape = img.shape[:2]
    addBoaderToWidth = min_size - imgShape[1]
    addBoaderToHeigth = min_size - imgShape[0]

    if(addBoaderToWidth <= 0):
        left, right = 0, 0
    if (addBoaderToHeigth <= 0):
        top, bottom, = 0, 0
    if(addBoaderToWidth > 0):
        left, right = int(addBoaderToWidth/2), int(addBoaderToWidth/2)
    if (addBoaderToHeigth > 0):
        top, bottom, = int(addBoaderToHeigth/2), int(addBoaderToHeigth/2)

    img_with_border = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
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


def down_sizeMarcin(img):
    # times of original size
    scallingWidth = img.shape[1]/320
    scallingHeigth = img.shape[0]/240

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
    elif img.shape == (768, 1024, IMG_Channels):
        return down_sizeMarcin(img)
    else:
        return down_size(img)


def showOneImg(image, label=None):
    inputImg = image.copy()
    fillTheCircle = -1
    radius = 2
    if label is not None:
        coordianates1 = (int(label[0]), int(label[1]))
        img = cv2.circle(inputImg, coordianates1, radius,
                         (0, 0, 255), fillTheCircle)

        coordianates2 = (int(label[2]), int(label[3]))
        img1 = cv2.circle(img, coordianates2, radius,
                          (0, 255, 0), fillTheCircle)

        coordianates3 = (int(label[4]), int(label[5]))
        img2 = cv2.circle(img1, coordianates3, radius,
                          (255, 0, 0), fillTheCircle)
        print("Image labels: {}".format(label))
    else:
        img2 = image
    print("Image dimensions: {}".format(image.shape))
    cv2.imshow("TestImage", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def testImgAsRect(modelFilePath):
    path_to_image = "./TestModels/TestImages/*.jpg"
    resizedImgs = []
    counter = 0
    totalFiles = len(glob.glob(path_to_image))
    model = tf.keras.models.load_model(modelFilePath)
    for img in glob.glob(path_to_image):
        counter += 1
        print("[{}/{}]".format(counter, totalFiles))
        img = cv2.imread(img)
        img_resized = reSizeImg(img, 3)
        img_squa = make_square(img_resized)
        img_norm = img_squa/255.0

        # Put the img into the correct shape, Img needs shape (1,240,320,3)
        img_rdy = np.expand_dims(img_norm, axis=0)

        Predictions = model.predict(img_rdy)
        showOneImg(img_squa, Predictions[0])
    print("End of test")


testImgAsRect(
    'TestModels\TestModel\CNNv25_Marcin_DropOut04.h5')
# test_model()
