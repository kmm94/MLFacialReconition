import cv2
import csv
import glob
from pynput import keyboard

final_csv = './SpotsOfInterest.csv'
imagePathname = './*.BMT'


def on_press(key):
    if key.char == "f":
        shutil.copy()
        return False
    elif key.char == "y":
        print("[" + str(self.counter) + "/" + str(self.total) + "]" +
              " Inkluderet: " + self.currentFilename)
        shutil.copy("/Billeder/" + self.currentFilename,
                    "/Inkluderet/" + self.currentFilename)
        return False


ImageCounter = 0
for filename in glob.glob(imagePathname):
    ImageCounter += 1
    print(filename)
    with open(final_csv, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if(".\{}".format(row[0]) == filename):
                Oimg = cv2.imread(filename)
                fillTheCircle = -1
                color = (0, 0, 255)
                radius = 2
                coordianates1 = (int(row[1].split('.')[0]),
                                 int(row[2].split('.')[0]))
                img = cv2.circle(Oimg, coordianates1, radius,
                                 color, fillTheCircle)

                coordianates2 = (int(row[3].split('.')[0]),
                                 int(row[4].split('.')[0]))
                img1 = cv2.circle(img, coordianates2, radius,
                                  color, fillTheCircle)

                coordianates3 = (int(row[5].split('.')[0]),
                                 int(row[6].split('.')[0]))
                img2 = cv2.circle(img1, coordianates3, radius,
                                  color, fillTheCircle)
                print('Coordinates: {}, {}, {},'.format(
                    coordianates1, coordianates2, coordianates3))
                cv2.imshow(filename, img2)
                listener = keyboard.Listener(on_press=on_press)
                listener.start()
                cv2.waitKey(0)
                cv2.destroyWindow(filename)

print("Number of images: {}".format(ImageCounter))
