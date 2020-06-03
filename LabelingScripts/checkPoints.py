import csv
import glob

points_csv = 'FinalDataPoints.csv'
numberOfNonMatchs = 0
rowNumber = 0
numberOfMactchs = 0
with open(points_csv, 'r') as file:
    for row in csv.reader(file, delimiter=','):
        try:
            rowNumber += 1
            isMatch = False
            for filename in glob.glob("*.BMT"):
                if(row[0] == filename):
                    numberOfMactchs += 1
                    isMatch = True
                    continue
            if(isMatch == False):
                numberOfNonMatchs += 1
                print("match not found in row# {}: {} and {}".format(
                    rowNumber, row[0], filename))
        except IndexError as e:
            print(
                "matches: {} non-matches: {}".format(numberOfMactchs, numberOfNonMatchs))
