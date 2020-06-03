import cv2
import csv
import glob
import DataManager


train_Img, train_Lab, validation_Img, validation_Lab, test_Img, test_Lab = DataManager.getMarcinDataset()
i = 0
print("Total Img: ", len(train_Img))
for img in train_Img:
   DataManager.showOneImg(img, train_Lab[i])
   print("# ", i)
   i += 1
