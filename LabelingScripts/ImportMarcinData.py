import os
import cv2
from pynput import keyboard
import json
import csv


class LoadMarcinDataset(object):

    def __init__(self):
        self.currentFilename = None
        self.total = None
        self.counter = 0
        self.root = os.getcwd()

    def CreateCsvFile(self):
        def on_press(key):
            if key.char == "n":
                return False

        with open('marcin_coordinates.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "leftEyeX", "leftEyeY", "rightEyeX", "rightEyeY", "noseX", "noseY"])

        entries = os.listdir(self.root + "/ThermalFaceDatabase")
        self.total = len(entries)

        for filename in entries:
            if ".png" in filename:
                self.currentFilename = filename
                currentFilenameWithoutExtension = self.currentFilename[:-4]
                jsonFilepath = "ThermalFaceDatabase/" + currentFilenameWithoutExtension + ".ljson"

                if not (os.path.exists(jsonFilepath)):
                    print("Couldn't find " + jsonFilepath)
                    continue

                with open(jsonFilepath) as json_file:
                    data = json.load(json_file)
                    list = data['landmarks']["points"]
                    leftEyeX = int(list[39][1])
                    leftEyeY = int(list[39][0])
                    rightEyeX = int(list[42][1])
                    rightEyeY = int(list[42][0])
                    noseX = int(list[30][1])
                    noseY = int(list[30][0])

                    with open('marcin_coordinates.csv', 'a', newline='\n') as file:
                        writer = csv.writer(file)
                        writer.writerow([self.currentFilename, leftEyeX, leftEyeY, rightEyeX, rightEyeY, noseX, noseY])

                # *************************************************************************
                # *** Udkommentér følgende linjer hvis du vil se præcisionen af labels ***
                # *************************************************************************
                # img = cv2.imread("ThermalFaceDatabase/" + self.currentFilename)
                # cv2.circle(img, (leftEyeX, leftEyeY), 5, (0, 0, 255), -1)
                # cv2.circle(img, (rightEyeX, rightEyeY), 5, (0, 0, 255), -1)
                # cv2.circle(img, (noseX, noseY), 5, (0, 0, 255), -1)
                # cv2.namedWindow(self.currentFilename)
                # cv2.imshow(self.currentFilename, img)
                # listener = keyboard.Listener(on_press=on_press)
                # listener.start()
                # cv2.waitKey(0)
                # cv2.destroyWindow(filename)
                # *************************************************************************
                # *** Udkommentér følgende linjer hvis du vil se præcisionen af labels ***
                # *************************************************************************

                # self.counter = self.counter + 1
                # print(currentFilenameWithoutExtension)
                # print(str(self.counter) + "/" + str(self.total))


loadMarcinDataset = LoadMarcinDataset()
loadMarcinDataset.CreateCsvFile()
