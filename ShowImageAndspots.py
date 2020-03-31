import cv2
import csv
import glob
from pynput import keyboard
import DataManager



images, labels = DataManager.getColorImagesAsRect()
i = 0
for img in images:
   DataManager.showOneImg(img, labels[i])
   i += 1
