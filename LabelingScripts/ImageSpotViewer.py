import os
import cv2
import shutil
import glob
import csv
from pynput import keyboard


class SortImages(object):

    def __init__(self):
        self.currentFilename = None
        self.total = None
        self.counter = 0
        self.root = os.getcwd()
        self.exit = False

    def printProgressBar(self, iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                         (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' %
              (prefix, bar, percent, suffix), end=printEnd)
        # Print New Line on Complete
        if iteration == total:
            print()

    def SortImages(self):
        def on_press(key):
            if key.char == "f":
                shutil.copy(self.currentFilename,
                            "./Fejl/" + self.currentFilename)
                return False
            elif key.char == 'y':
                return False

        imagePathname = './Images/*.*'

        print("total img: " + str(self.total))
        ImageCounter = 0
        for filename in glob.glob(imagePathname):

            ImageCounter += 1
            #self.printProgressBar(ImageCounter, self.total)
            self.currentFilename = filename
            final_csv = './FinalDataPoints.csv'
            with open(final_csv, 'r') as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    if("./Images\{}".format(row[0]) == filename):
                        Oimg = cv2.imread(filename)
                        fillTheCircle = -1
                        color = (0, 0, 0)
                        radius = 5
                        coordianates1 = (int(row[1].split('.')[0]),
                                         int(row[2].split('.')[0]))
                        img = cv2.circle(Oimg, coordianates1, radius,
                                         (0, 0, 255), fillTheCircle)

                        coordianates2 = (int(row[3].split('.')[0]),
                                         int(row[4].split('.')[0]))
                        img1 = cv2.circle(img, coordianates2, radius,
                                          (0, 255, 0), fillTheCircle)

                        coordianates3 = (int(row[5].split('.')[0]),
                                         int(row[6].split('.')[0]))
                        img2 = cv2.circle(img1, coordianates3, radius,
                                          (0, 0, 0), fillTheCircle)
                        print('Image {} Coordinates: {}, {}, {},'.format(filename,
                                                                         coordianates1, coordianates2, coordianates3))
                        cv2.imshow(filename, img2)
                        listener = keyboard.Listener(on_press=on_press)
                        listener.start()
                        cv2.waitKey(0)
                        cv2.destroyWindow(filename)


sort = SortImages()
sort.SortImages()
