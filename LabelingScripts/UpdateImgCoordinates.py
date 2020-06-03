import csv

numberOfUpdated = 0
numberOfCopyedOld = 0
points_csv_old = 'SpotsOfInterest.csv'
with open(points_csv_old, 'r') as file:
    for rowold in csv.reader(file, delimiter=','):
        points_csv_updated = 'FacialPoints.csv'
        updatedpointsfile = open(points_csv_updated, 'r')
        updatedcsv = csv.reader(updatedpointsfile, delimiter=',')
        shouldContinue = False
        ConcatinatedCSV = open("FinalDataPoints.csv", 'a+')
        for row in updatedcsv:
            if(rowold[0] == row[0]):
                print('updating points on file: ' + rowold[0])
                numberOfUpdated += 1
                writer = csv.writer(ConcatinatedCSV, delimiter=',')
                writer.writerow(
                    [row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                shouldContinue = True
                continue
        if(shouldContinue):
            continue
        else:
            numberOfCopyedOld += 1
            print('A update has not been found points on file: ' + rowold[0])
            writer = csv.writer(ConcatinatedCSV, delimiter=',')
            writer.writerow([rowold[0], rowold[1], rowold[2],
                             rowold[3], rowold[4], rowold[5], rowold[6]])
print('Updated: {}'.format(numberOfUpdated))
print('Copied: {}'.format(numberOfCopyedOld))
print('Total writings: {}'.format(numberOfCopyedOld+numberOfUpdated))
