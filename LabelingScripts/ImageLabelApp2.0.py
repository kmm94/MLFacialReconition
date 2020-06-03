import msvcrt

import numpy as np
import cv2
import csv
import glob

scale = 3  # times of original size
mouseCoordinates = []
imagePathname = 'data/*.BMT'


def Click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        print('mouse: ', mouseX / scale, ' ', mouseY / scale)
        mouseCoordinates.append((mouseX / scale, mouseY / scale))


def showImg(filename, titel, listOfPoints):
    img = cv2.imread(filename)
    cv2.namedWindow(titel)
    cv2.setMouseCallback(titel, Click)

    # Resize images:
    # times of original size
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # paint with points
    if listOfPoints is not None:
        for point in listOfPoints:
            fillTheCircle = -1
            color = (0, 0, 255)
            radius = 5
            coordianates = (int(point[0] * scale), int(point[1] * scale))
            resized = cv2.circle(resized, coordianates, radius, color, fillTheCircle)

    cv2.imshow(titel, resized)


points_csv = 'FacialPoints.csv'
with open(points_csv, 'a+') as file:
    print('writing headers to ' + points_csv)
    writer = csv.writer(file, delimiter=',')
    writer.writerow(["Filename", "left eye X", "left eye Y", "Right eye X", "Right eye Y", "Nose X", "Nose Y"])

print("looking for images the folder {} ".format(imagePathname))
print("Found {} images".format(len(glob.glob(imagePathname))))
counter = 0
for filename in glob.glob(imagePathname):
    counter += 1
    imageName = filename
    landmarks = "Choose your points {}".format(filename)
    finalPoints = "This is your points"

    print("[{}/{}] {}".format(len(glob.glob(imagePathname)), counter, filename ))

    showImg(filename, landmarks, None)
    cv2.waitKey(0)
    cv2.destroyWindow(landmarks)

    showImg(filename, finalPoints, mouseCoordinates)
    cv2.waitKey(0)
    cv2.destroyWindow(finalPoints)

    # write to csv
    points_csv = 'FacialPoints.csv'
    with open(points_csv, 'a+') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(
            [filename, mouseCoordinates[0][0], mouseCoordinates[0][1], mouseCoordinates[1][0], mouseCoordinates[1][1],
             mouseCoordinates[2][0], mouseCoordinates[2][1]])
        print("The point have been succesfully saved")
    mouseCoordinates = []

