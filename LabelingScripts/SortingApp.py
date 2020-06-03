import os
import cv2
import shutil
from pynput import keyboard


class SortImages(object):

    def __init__(self):
        self.currentFilename = None
        self.total = None
        self.counter = 0
        self.root = os.getcwd()

    def SortImages(self):
        def on_press(key):
            if key.char == "n":
                print("[" + str(self.counter) + "/" + str(self.total) + "]" + " Frasorteret: " + self.currentFilename)
                shutil.copy(self.root + "/Billeder/" + self.currentFilename, self.root +"/Frasorteret/" + self.currentFilename)
                return False
            elif key.char == "y":
                print("[" + str(self.counter) + "/" + str(self.total) + "]" + " Inkluderet: " + self.currentFilename)
                shutil.copy(self.root + "/Billeder/" + self.currentFilename, self.root +"/Inkluderet/" + self.currentFilename)
                return False

        entries = os.listdir(self.root + "/Billeder")
        self.total = len(entries)

        for filename in entries:
            if filename == "desktop.ini":
                continue
            self.counter += 1
            self.currentFilename = filename
            img = cv2.imread("Billeder/" + self.currentFilename)
            cv2.namedWindow(self.currentFilename)
            cv2.imshow(self.currentFilename, img)

            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            cv2.waitKey(0)
            cv2.destroyWindow(filename)


sort = SortImages()
sort.SortImages()
