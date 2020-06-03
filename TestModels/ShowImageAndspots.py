import cv2
import csv
import glob
import os

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
                          (255, 0, 0), fillTheCircle)
        img2 = img1
        print("Image labels: {}".format(label))
    else:
        img2 = image
    print("Image dimensions: {}".format(image.shape))
    cv2.imshow("TestImage", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def getImageName(image_path):
    img_parts = image_path.split(os.path.sep)
    return img_parts[-1]


def getImgsRaw():
    path_to_image = "./TestModels/TestImages/*.jpg"
    path_to_labels = "./TestModels/Spots.csv"
    totalFiles = len(glob.glob(path_to_image))
    counter = 0
    images = []
    labels = []
    print("Getting imgs")
    for image_path in glob.glob(path_to_image):
        counter += 1
        image_name = getImageName(image_path)
        print(image_name)
        labels_csv = open(path_to_labels, "r")
        for row in csv.reader(labels_csv, delimiter=","):
            filename = row[3]
            if (filename == image_name):
                img_raw_labels = [int(round(float(row[4]))), int(round(float(row[5]))),
                                  int(round(float(row[6]))), int(round(float(row[7])))]
                img = cv2.imread(image_path)
                labels.append(img_raw_labels)
                images.append(img)
    print("Done...")
    return images, labels


images, labels = getImgsRaw()
i = 0
print("total: ", len(images))
for img in images:
    showOneImg(img, labels[i])
    i += 1
