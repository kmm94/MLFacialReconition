import numpy as np
import tensorflow as tf
import cv2
import glob
import os
import csv


# HyperPerameters
BATCH_SIZE = 1
IMG_HEIGHT = int(480/2)
IMG_WIDTH = int(640/2)

# Load the data into tf
images = []
labels = []
path_to_image = "FinalDataSet/Images/*.*"
path_to_labels = "FinalDataSet/FinalDataPoints.csv"


def getImageName(image_path):
    img_parts = image_path.split(os.path.sep)
    return img_parts[-1]

index = 0
one_image = 0
one_label = 0

# show one image with spots
def showOneImg(image, label=None):
    #TODO: Color Left, right and nose and check dataset that left rigth and nose are set correctly
    fillTheCircle = -1
    color = (0, 0, 0)
    radius = 2
    if label is not None:
        coordianates1 = (label[0], label[1])
        img = cv2.circle(image, coordianates1, radius,
                        color, fillTheCircle)

        coordianates2 = (label[2], label[3])
        img1 = cv2.circle(img, coordianates2, radius,
                        color, fillTheCircle)

        coordianates3 = (label[4], label[5])
        img2 = cv2.circle(img1, coordianates3, radius,
                        color, fillTheCircle)
        print("Image labels: {}".format(label))
    else:
        img2 = image
    print("Image dimensions: {}".format(image.shape))
    cv2.imshow("TestImage", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    index = 0
    for i in labels:
        if (index % 2 == 0):
            labels_resized.append(int(i / 3.375))
        else:
            labels_resized.append(int(i / 6))
        index +=1
    return img_resized, labels_resized


def reSizeImgAndLabels(img, labels, IMG_Channels):
    if (img.shape == (120, 160, IMG_Channels)):
        return up_size(img, labels, 2)
    elif img.shape == (240, 320, IMG_Channels):
        return up_size(img, labels, 1)
    else:
        return down_size(img, labels)

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

def make_square(img, min_size=320):
    
    color = [0, 0, 0] # 'cause purple!

    # border widths; I set them all to 150

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


def getGrayImages():
    print("Image Augementation started")
    print("Resize, toGray and normalization")

    IMG_Channels = 1

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
                imgResape, labelResape = reSizeImgAndLabels(img, img_raw_labels, IMG_Channels)
                #one_image, one_label = mirrorIMG(imgResape, labelResape)
                one_image = imgResape
                one_label = labelResape
                imgResape = np.expand_dims(imgResape, axis=2)
                imgNormalization = imgResape / 255.0
                labels.append(labelResape)
                images.append(imgNormalization)
    return images, labels

def getColorImages():
    print("Image Augementation started")
    print("Resize, toGray and normalization")

    IMG_Channels = 3

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
                imgResape, labelResape = reSizeImgAndLabels(img, img_raw_labels, IMG_Channels)
                one_image = imgResape
                one_label = labelResape
                imgNormalization = imgResape / 255.0
                labels.append(labelResape)
                images.append(imgNormalization)
    showOneImg(images[500])
    return images, labels

def getColorImagesAsRect():
    print("Image Augementation started")
    print("Resize, toGray and normalization")

    IMG_Channels = 3

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
                imgResape, labelResape = reSizeImgAndLabels(img, img_raw_labels, IMG_Channels)
                one_image = imgResape
                one_label = labelResape
                imgNormalization = imgResape / 255.0

                #Add black padding to Img to make the image square
                imgNormalization = make_square(imgNormalization, min_size=320)

                labels.append(labelResape)
                images.append(imgNormalization)
    showOneImg(images[500])
    
    return images, labels

