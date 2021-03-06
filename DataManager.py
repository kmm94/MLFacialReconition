import numpy as np
import cv2
import glob
import os
import csv
import random
import math

# HyperPerameters
BATCH_SIZE = 1
IMG_HEIGHT = int(480 / 2)
IMG_WIDTH = int(640 / 2)

def getImageName(image_path):
    img_parts = image_path.split(os.path.sep)
    return img_parts[-1]


# show one image with spots
def showOneImg(image, label=None):
    inputImg = image.copy()
    fillTheCircle = -1
    radius = 2
    if label is not None:
        coordianates1 = (label[0], label[1])
        img = cv2.circle(inputImg, coordianates1, radius,
                         (0, 0, 255), fillTheCircle)

        coordianates2 = (label[2], label[3])
        img1 = cv2.circle(img, coordianates2, radius,
                          (0, 255, 0), fillTheCircle)

        coordianates3 = (label[4], label[5])
        img2 = cv2.circle(img1, coordianates3, radius,
                          (0, 0, 0), fillTheCircle)
        print("Image labels: {}".format(label))
    else:
        img2 = image
    print("Image dimensions: {}".format(image.shape))
    cv2.imshow("TestImage", img2)
    cv2.waitKey(10000)
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
    scallingWidth = img.shape[1] / 240 
    scallingHeigth = img.shape[0] / 320 

    width = int(img.shape[1] / scallingWidth)
    height = int(img.shape[0] / scallingHeigth)
    dim = (width, height)

    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    labels_resized = []
    index = 0
    for i in labels:
        if (index % 2 == 0):
            labels_resized.append(int(i / scallingWidth))
        else:
            labels_resized.append(int(i / scallingHeigth))
        index += 1
    return img_resized, labels_resized

def down_sizeMarcin(img, labels):
    # times of original size
    scallingWidth = img.shape[1] / 320 #changed for marcin
    scallingHeigth = img.shape[0] / 240 #changed for marcin

    width = int(img.shape[1] / scallingWidth)
    height = int(img.shape[0] / scallingHeigth)
    dim = (width, height)

    # resize image
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    labels_resized = []
    index = 0
    for i in labels:
        if (index % 2 == 0):
            labels_resized.append(int(i / scallingWidth))
        else:
            labels_resized.append(int(i / scallingHeigth))
        index += 1
    return img_resized, labels_resized

def reSizeImgAndLabels(img, labels, IMG_Channels):
    if (img.shape == (120, 160, IMG_Channels)):
        return up_size(img, labels, 2)
    elif img.shape == (240, 320, IMG_Channels):
        return up_size(img, labels, 1)
    elif img.shape == (768,1024, IMG_Channels):
        return down_sizeMarcin(img, labels)
    else:
        return down_size(img, labels)


def mirrorImgHorizon(image, label):
    # img.shape returns  (heigth(Y), width(X), channels)
    fliped_img = cv2.flip(image, 1)
    imgWidth = fliped_img.shape[1]
    fliped_labels = []
    # right eye
    fliped_labels.append(mirrorXCoordinate(imgWidth, label[0]))
    fliped_labels.append(label[1])
    # left eye
    fliped_labels.append(mirrorXCoordinate(imgWidth, label[2]))
    fliped_labels.append(label[3])
    # nose
    fliped_labels.append(mirrorXCoordinate(imgWidth, label[4]))
    fliped_labels.append(label[5])
    return fliped_img, fliped_labels


def mirrorXCoordinate(imgWidth, x):
    halfimageSide = imgWidth / 2
    distancToMidt = halfimageSide - x
    if (distancToMidt == 0):
        return x
    elif (distancToMidt > 0):
        return int(halfimageSide + distancToMidt)
    else:
        return int(halfimageSide - abs(distancToMidt))


def rotateAndScale(img, scaleFactor=0.5, degreesCCW=30):
    (oldY, oldX) = img.shape[0], img.shape[1]  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                scale=scaleFactor)  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor

    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    return rotatedImg


def make_square(img, labels, min_size=320):
    color = [0, 0, 0]  # 'cause black!
    imgShape = img.shape[:2]
    addBoaderToWidth = min_size - imgShape[1]
    addBoaderToHeigth = min_size - imgShape[0]

    if (addBoaderToWidth <= 0):
        left, right = 0, 0
    if (addBoaderToHeigth <= 0):
        top, bottom, = 0, 0
    if (addBoaderToWidth > 0):
        left, right = int(addBoaderToWidth / 2), int(addBoaderToWidth / 2)
    if (addBoaderToHeigth > 0):
        top, bottom, = int(addBoaderToHeigth / 2), int(addBoaderToHeigth / 2)

    labels_resized = []
    index = 0
    for i in labels:
        if (index % 2 == 0):
            labels_resized.append(int(i + (addBoaderToWidth / 2)))
        else:
            labels_resized.append(int(i + (addBoaderToHeigth / 2)))
        index += 1

    img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_with_border, labels_resized


def getImgAsRect(path_to_image, path_to_labels):
    print("Image Augementation started")
    print("Resize")
    images = [] 
    labels = []

    IMG_Channels = 3

    totalFiles = len(glob.glob(path_to_image))
    counter = 0
    for image_path in glob.glob(path_to_image):
        counter += 1
        image_name = getImageName(image_path)
        labels_csv = open(path_to_labels, "r")
        for row in csv.reader(labels_csv, delimiter=","):
            if (row[0] == image_name):
                img_raw_labels = [int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))),
                                  int(round(float(row[4]))), int(round(float(row[5]))), int(round(float(row[6])))]
                img = cv2.imread(image_path)

                imgResape, labelResape = reSizeImgAndLabels(img, img_raw_labels, IMG_Channels)

                # Add black padding to Img to make the image square
                imgNormalization, labelResized = make_square(imgResape, labelResape, min_size=320)

                labels.append(labelResized)
                images.append(imgNormalization)
    print("Done... #", totalFiles)
    randomIndex = random.randint(0, len(images) - 1)
    print('showing image # ', randomIndex)
    showOneImg(images[randomIndex], labels[randomIndex])

    return images, labels


def distToPoint(x, y, centerpoint):
    return math.sqrt(math.pow(x - centerpoint[0], 2) + math.pow(y - centerpoint[1], 2))


def rotateROI(x, y, centerPoint, newCenterPoint, degrees):
    dist = distToPoint(x, y, centerPoint)

    deltaX = x - centerPoint[0]
    deltaY = y - centerPoint[1]
    if (y > centerPoint[1]):
        deltaY = -abs(deltaY)
    elif (y < centerPoint[1]):
        deltaY = abs(deltaY)

    if (x > centerPoint[0]):
        deltaX = abs(deltaX)
    elif (x < centerPoint[0]):
        deltaX = -abs(deltaX)

    delta_new_x = deltaX*math.cos(math.radians(degrees))-deltaY*math.sin(math.radians(degrees))
    delta_new_y = deltaX*math.sin(math.radians(degrees))+deltaY*math.cos(math.radians(degrees))

    if (delta_new_x < 0):
        newX = newCenterPoint[0] - abs(delta_new_x)
    elif (delta_new_x > 0):
        newX = newCenterPoint[0] + delta_new_x
    else:
        newX = newCenterPoint[0]

    if (delta_new_y < 0):
        newY = newCenterPoint[1] + abs(delta_new_y)
    elif (delta_new_y > 0):
        newY = newCenterPoint[1] - delta_new_y
    else:
        newY = newCenterPoint[1]

    return int(newX), int(newY)


def rotateImg(img, labels, degrees):
    rotated_labels = []
    degrees_To_Rotate = degrees

    # Getting dimensions
    imgOrgHeigth = img.shape[0]  # Y
    imgOrgWidth = img.shape[1]  # X
    centerPoint = (int(imgOrgWidth / 2), int(imgOrgHeigth / 2))  # Format (x, y)

    rotated_img = rotateAndScale(img, 1,
                                 degrees_To_Rotate)  # rotated img is allready a squrer so they sould just be down scaled
    rotImgHeigth = rotated_img.shape[0]
    rotImgWidth = rotated_img.shape[1]
    newCenterPoint = (int(rotImgWidth / 2), int(rotImgHeigth / 2))  # Format (x, y)
    index=0
    for coordinate in labels:
        if index % 2 == 0:
            x, y = rotateROI(labels[index], labels[index+1], centerPoint, newCenterPoint, degrees_To_Rotate)
            rotated_labels.append(x)
            rotated_labels.append(y)
        index += 1
    return rotated_img, rotated_labels


def rotateImgs(images, labels, degrees):
    print("Image Augementation started")
    print("Rotatetion")
    images_copy = images.copy()
    lables_copy = labels.copy()

    IMG_Channels = 3

    totalFiles = len(images)
    counter = 0
    rotated_Images = []
    rotated_Lables= []
    for img in images_copy:
        imgResape, labelResape = reSizeImgAndLabels(img, lables_copy[counter], IMG_Channels)
        img_rotated, labels_rotated = rotateImg(imgResape, labelResape, degrees)
        img_squre, labels_squre = make_square(img_rotated, labels_rotated, min_size=320)
        rotated_Images.append(img_squre)
        rotated_Lables.append(labels_squre)
        counter += 1
    randomIndex = random.randint(0, len(rotated_Images) - 1)
    print("Done... #", totalFiles)
    print('showing image # ', randomIndex)
    showOneImg(rotated_Images[randomIndex], rotated_Lables[randomIndex])
    return rotated_Images, rotated_Lables

def getImgsRaw(path_to_image, path_to_labels):
    totalFiles = len(glob.glob(path_to_image))
    images = [] 
    labels = []
    counter = 0
    print("Getting imgs")
    for image_path in glob.glob(path_to_image):
        counter += 1
        image_name = getImageName(image_path)
        labels_csv = open(path_to_labels, "r")
        for row in csv.reader(labels_csv, delimiter=","):
            if (row[0] == image_name):
                img_raw_labels = [int(round(float(row[1]))), int(round(float(row[2]))), int(round(float(row[3]))),
                                  int(round(float(row[4]))), int(round(float(row[5]))), int(round(float(row[6])))]
                img = cv2.imread(image_path)
                labels.append(img_raw_labels)
                images.append(img)
    print("Done... got #", totalFiles, " images")
    return images, labels


def showOneRandomImg(images, labels):
    randomIndex = random.randint(0, len(images) - 1)
    print('showing image # ', randomIndex)
    showOneImg(images[randomIndex], labels[randomIndex])

def makeImgsSquare(imgs, labs):
    print("making img square")
    imgs_rect = []
    labs_rect = []
    counter = 0
    for img in imgs:
        img_squre, labels_squre = make_square(img, labs[counter])
        imgs_rect.append(img_squre)
        labs_rect.append(labels_squre)
        counter += 1
    print("done...")
    return imgs_rect, labs_rect

def GetImgsRotatedAndFliped(_rotations):
    path_to_image = "FinalDataSet/Images/*.*"
    path_to_labels = "FinalDataSet/FinalDataPoints.csv"
    images, labels = getImgsRaw(path_to_image, path_to_labels)
    print("Raw images loaded: ", len(images))
    counter = 0
    img_resized = []
    labels_resized = []
    for img in images:
        img, lab = reSizeImgAndLabels(img, labels[counter], 3)
        img_resized.append(img)
        labels_resized.append(lab)
        counter +=1

    images=None
    labels = None
    print("number of reSizedImgs: ", len(img_resized))
    rotations = _rotations
    rotated_imgs = []
    rotated_labs = []
    for degrees in rotations:
        img_to_beRot = img_resized.copy()
        lab_to_beRot = labels_resized.copy()
        rot_img, rot_lab = rotateImgs(img_to_beRot, lab_to_beRot, degrees)
        print("adding imgs that have been rotated {} Imgs#: {}".format(degrees, len(rot_img)))
        rotated_imgs.extend(rot_img)
        rotated_labs.extend(rot_lab)
    img_to_beRot = None
    lab_to_beRot = None
    print("done...")
    imgs_rect, labs_rect = makeImgsSquare(rotated_imgs, rotated_labs)

    org_imgSQ, org_labSQ = makeImgsSquare(img_resized, labels_resized)
    imgs_rect.extend(org_imgSQ)
    labs_rect.extend(org_labSQ)
    showOneRandomImg(imgs_rect,labs_rect)

    print("normalizing {} images".format(len(imgs_rect)))
    img_normalizied = []
    for img in imgs_rect:
        img_normalizied.append(img/255.0)
    print("Done...")

    # print("flipping Imgs...")
    # img_copy = img_normalizied.copy()
    # labs_copy = labs_rect.copy()
    # temp_imgs = []
    # temp_labs = []
    # index = 0
    # for img in img_copy:
    #     img_F, labs_F = mirrorImgHorizon(img, labs_copy[index])
    #     temp_imgs.append(img_F)
    #     temp_labs.append(labs_F)
    #     index +=1
    # img_normalizied.extend(temp_imgs)
    # labs_rect.extend(temp_labs)
    # print("done")

    return img_normalizied, labs_rect

def resizeImages(imgs, labs):
    print("Resizing imgs")
    counter = 0
    img_resized = []
    labels_resized = []
    for img in imgs:
        img, lab = reSizeImgAndLabels(img, labs[counter], 3)
        img_resized.append(img)
        labels_resized.append(lab)
        counter +=1
    print("number of reSizedImgs: ", len(img_resized))
    return img_resized, labels_resized

def rotateImages(imgs, labs, rotations):
    print("Rotating Images")
    rotations = rotations
    rotated_imgs = []
    rotated_labs = []
    for degrees in rotations:
        img_to_beRot = imgs.copy()
        lab_to_beRot = labs.copy()
        rot_img, rot_lab = rotateImgs(img_to_beRot, lab_to_beRot, degrees)
        print("adding imgs that have been rotated {} Imgs#: {}".format(degrees, len(rot_img)))
        rotated_imgs.extend(rot_img)
        rotated_labs.extend(rot_lab)
    img_to_beRot = None
    lab_to_beRot = None
    rotated_imgs.extend(imgs)
    rotated_labs.extend(labs)
    print("done...")
    return rotated_imgs, rotated_labs

def normalizeImgs(imgs):
    print("normalizing {} images".format(len(imgs)))
    img_normalizied = []
    for img in imgs:
        img_normalizied.append(img/255.0)
    print("Done...")
    return img_normalizied

def shuffleDataset(img, lab):
    print("shuffling list")
    c = list(zip(img, lab))
    random.shuffle(c)
    images, labels = zip(*c)
    print("shuffling Done...")
    return images, labels

def SplitDataSet(_images, _labels):
    train_Img = []
    train_Lab = []
    test_Img = []
    test_Lab = []
    validation_Img = []
    validation_Lab = []
    training_Split = 0.8
    test_Split = 0.05
    val_Split = 0.15

    if ((training_Split + test_Split + val_Split) != 1):
        raise AssertionError("splits should add up to to 1")

    images, labels = shuffleDataset(_images, _labels)
    showOneRandomImg(images, labels)

    number_of_samples = len(images)
    train_spit = number_of_samples*training_Split
    test_Split = train_spit+ number_of_samples*test_Split

    train_Img, test_Img, validation_Img = np.split(images, [int(train_spit), int(test_Split)])
    train_Lab, test_Lab, validation_Lab = np.split(labels, [int(train_spit), int(test_Split)])

    images=None
    labels=None

    return train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab

def GetMarcinDataset():
    path_to_MarcinIMG = "ThermalFaceDatabase/*.png"
    path_to_Marcinlabels = "marcin_coordinates.csv"
    path_to_image = "FinalDataSet/Images/*.*"
    path_to_labels = "FinalDataSet/FinalDataPoints.csv"

    #images, labels = getImgsRaw(path_to_MarcinIMG, path_to_Marcinlabels)
    images, labels = getImgsRaw(path_to_image, path_to_labels)
    
    #images.extend(ourImg)
    #labels.extend(ourLabs)

    train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = SplitDataSet(images, labels)
    images=None
    labels = None
    print("Train")
    train_Img, train_Lab = resizeImages(train_Img, train_Lab)

    print("Validation")
    validation_Img, validation_Lab = resizeImages(validation_Img, validation_Lab)

    print("Test")
    test_Img, test_Lab = resizeImages(test_Img, test_Lab)

    rotated_train_imgs, rotated_train_labs = rotateImages(train_Img, train_Lab, [90, 180, 270])

    rotated_validation_imgs, rotated_validation_lab = rotateImages(validation_Img, validation_Lab, [90,180, 270])

    validation_Img, validation_Lab = makeImgsSquare(rotated_validation_imgs, rotated_validation_lab)
    test_Img, test_Lab = makeImgsSquare(test_Img, test_Lab)
    train_Img, train_Lab = makeImgsSquare(rotated_train_imgs, rotated_train_labs)

    print("showing a training img #", len(train_Img))
    showOneRandomImg(train_Img, train_Lab)

    print("showing a validation img #", len(validation_Img))
    showOneRandomImg(validation_Img, validation_Lab)
    
    print("showing a test img #", len(test_Img))
    showOneRandomImg(test_Img, test_Lab)

    train_Img = normalizeImgs(train_Img)
    validation_Img = normalizeImgs(validation_Img)
    test_Img = normalizeImgs(test_Img)

    train_Img, train_Lab = shuffleDataset(train_Img, train_Lab)
    validation_Img, validation_Lab = shuffleDataset(validation_Img, validation_Lab)
    test_Img, test_Lab = shuffleDataset(test_Img, test_Lab)
    print("******************* Dataset Created *******************")
    print("TrainImgs: {} TrainLabs: {} \n ValidationImgs: {} ValidationLabs: {} \n TestImg: {} TestLab: {} \n \n".format(
        len(train_Img),
        len(train_Lab),
        len(validation_Img),
        len(validation_Lab),
        len(test_Img),
        len(test_Lab)))

    return train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab

def GetMarcinDatasetAndOurRotated():
    path_to_MarcinIMG = "ThermalFaceDatabase/*.png"
    path_to_Marcinlabels = "marcin_coordinates.csv"
    path_to_image = "FinalDataSet/Images/*.*"
    path_to_labels = "FinalDataSet/FinalDataPoints.csv"

    images_Marcin, labels_Marcin = getImgsRaw(path_to_MarcinIMG, path_to_Marcinlabels)
    images, labels = getImgsRaw(path_to_image, path_to_labels)

    train_Img_Marcin, train_Lab_Marcin, validation_Img_Marcin, validation_Lab_Marcin, test_Img_Marcin, test_Lab_Marcin = SplitDataSet(images_Marcin, labels_Marcin)
    
    train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = SplitDataSet(images, labels)

    images=None
    labels = None
    print("Train")
    train_Img_Marcin, train_Lab_Marcin = resizeImages(train_Img_Marcin, train_Lab_Marcin)
    train_Img, train_Lab = resizeImages(train_Img, train_Lab)

    print("Validation")
    validation_Img, validation_Lab = resizeImages(validation_Img, validation_Lab)
    validation_Img_Marcin, validation_Lab_Marcin = resizeImages(validation_Img_Marcin, validation_Lab_Marcin)

    print("Test")
    test_Img, test_Lab = resizeImages(test_Img, test_Lab)
    test_Img_Marcin, test_Lab_Marcin = resizeImages(test_Img_Marcin, test_Lab_Marcin)

    rotated_train_imgs, rotated_train_labs = rotateImages(train_Img, train_Lab, [])

    rotated_validation_imgs, rotated_validation_lab = rotateImages(validation_Img, validation_Lab, [])

    print("Training: \nour {} + Marcin {} in total: {}".format(len(rotated_train_imgs), len(train_Img_Marcin), (len(rotated_train_imgs)+len(train_Img_Marcin))))
    rotated_train_imgs.extend(train_Img_Marcin)
    rotated_train_labs.extend(train_Lab_Marcin)

    print("Validation: \nour {} + Marcin {} in total: {}".format(len(rotated_validation_imgs), len(validation_Img_Marcin), (len(rotated_validation_imgs)+len(validation_Img_Marcin))))
    rotated_validation_imgs.extend(validation_Img_Marcin)
    rotated_validation_lab.extend(validation_Lab_Marcin)

    print("Test: \nour {} + Marcin {} in total: {}".format(len(test_Img), len(test_Img_Marcin), (len(test_Img)+len(test_Img_Marcin))))
    test_Img.extend(test_Img_Marcin)
    test_Lab.extend(test_Lab_Marcin)

    validation_Img, validation_Lab = makeImgsSquare(rotated_validation_imgs, rotated_validation_lab)
    test_Img, test_Lab = makeImgsSquare(test_Img, test_Lab)
    train_Img, train_Lab = makeImgsSquare(rotated_train_imgs, rotated_train_labs)

    print("showing a training img #", len(train_Img))
    showOneRandomImg(train_Img, train_Lab)

    print("showing a validation img #", len(validation_Img))
    showOneRandomImg(validation_Img, validation_Lab)
    
    print("showing a test img #", len(test_Img))
    showOneRandomImg(test_Img, test_Lab)

    train_Img = normalizeImgs(train_Img)
    validation_Img = normalizeImgs(validation_Img)
    test_Img = normalizeImgs(test_Img)

    train_Img, train_Lab = shuffleDataset(train_Img, train_Lab)
    validation_Img, validation_Lab = shuffleDataset(validation_Img, validation_Lab)
    test_Img, test_Lab = shuffleDataset(test_Img, test_Lab)

    print("******************* Dataset Created *******************")
    print("TrainImgs: {} TrainLabs: {} \n ValidationImgs: {} ValidationLabs: {} \n TestImg: {} TestLab: {} \n \n".format(
        len(train_Img),
        len(train_Lab),
        len(validation_Img),
        len(validation_Lab),
        len(test_Img),
        len(test_Lab)))

    return train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab
#https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order to shuffle bout list
#img, labs = GetImgsRotatedAndFliped()
#showOneRandomImg(img, labs)


