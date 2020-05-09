import os
import cv2
import shutil
import glob
import csv
import DataManager


images, labels = DataManager.GetImgsRotatedAndFliped([90, 180, 270])
train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.SplitDataSet(images, labels)


counter = 0
for img in train_Img:
    print("showing img#: ", counter)
    DataManager.showOneImg(img, train_Lab[counter])
    counter += 1
