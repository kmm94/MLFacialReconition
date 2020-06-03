import glob
import os

pathToImages = "./*"
numberOfChangedFileNames = 0
for filename in glob.glob(pathToImages):
    ext = filename.split('.')
    if(len(ext) == 2):
        numberOfChangedFileNames += 1
        newFileName = "{}.jpg".format(filename)
        print("changeing name on: {}".format(filename))
        os.rename(filename, newFileName)
print("Changed file names on {} number of files.".format(numberOfChangedFileNames))
