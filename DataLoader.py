"""
File: DataLoader.py
Description: Reads data from the UTIAS dataset and provides classes to interact with data
Authors: Seth Isaacson
"""
import pandas as pd
import argparse
import os
import copy

class Map:
    def __init__(self, data):
        self.data = data
        self.landmarkDict = self.data.set_index('Subject').to_dict('index')
        print(self.landmarkDict)

    def __getitem__(self, key):
        return self.landmarkDict[subjectID]

    def getLandmarkLocation(self, subjectID):
        return self.landmarkDict[subjectID]

class Robot:
    # All arguments are corresponding pandas dataframes
    def __init__(self, groundtruth, measurements, odometry, barcodes):
        self.groundTruth = groundtruth
        self.measurements = measurements
        self.odometry = odometry
        self.barcodes = barcodes

        # List of data in order. Serves as a backup to dataQueue
        self.dataList = []

        # List of data we will pop from.
        self.dataQueue = []

        self.buildDict()

    # Get next data. May include one or more of ground truth, measurement, and barcode
    def getNext(self):
        return self.dataQueue.pop(0)

    def empty(self):
        return len(self.dataQueue == 0)

    def reset(self):
        self.dataQueue = copy.deepcopy(self.dataList)

    def buildDict(self):
        self.dataQueue = []
        self.dataDict = {}
        barcodeDict = {}

        for row in self.barcodes.itertuples():
            barcodeDict[row.Barcode] = row.Subject

        for row in self.odometry.itertuples():
            time = row.Time
            self.dataDict[time] = {"Time": time, "GroundTruth": None, "Measurements": [], "Odometry":(row.Velocity, row.AngularVelocity)}

        for row in self.groundTruth.itertuples():
            time = row.Time
            x = row.X
            y = row.Y
            heading = row.Heading
            if time not in self.dataDict:
                self.dataDict[time] = {"Time": time, "GroundTruth": (x,y,heading), "Measurements": [], "Odometry": None}
            else:
                self.dataDict[time]["GroundTruth"] = (x,y,heading)

        for row in self.measurements.itertuples():
            if row.Barcode not in barcodeDict:
                print("Unrecogonized Barcode: {}. Skipping".format(row.Barcode))
            else:
                subject = barcodeDict[row.Barcode]

            time = row.Time
            if time not in self.dataDict:
                self.dataDict[time] = {"Time": time, "GroundTruth": None, "Measurements": [(subject, row.Range, row.Bearing)], "Odometry": None}
            else:
                self.dataDict[time]["Measurements"].append((subject, row.Range, row.Bearing))

        for t in sorted(self.dataDict.keys()):
            dict = self.dataDict[t]
            self.dataList.append(dict)

        self.reset()

class Data:
    def __init__(self, directory):
        self.directory = directory
        self.loadAllData()


    def createDfFromFile(self, fname, headers):
        i = 0
        data = []
        with open(fname) as file:
            for line in file:
                if i < 4:
                    i += 1
                    continue
                i += 1
                data.append(tuple(line.split()))


        df = pd.DataFrame(data, columns = headers)
        return df

    def loadAllData(self):
        files = os.scandir(self.directory)
        self.numRobots = int((len(list(files))-2)/3)

        self.robotGroundTruth = [None for _ in range(self.numRobots)]
        self.robotMeasurements = [None for _ in range(self.numRobots)]
        self.robotOdometry = [None for _ in range(self.numRobots)]

        i = 0

        for file in os.scandir(self.directory):
            headers = None
            if(file.path.startswith(self.directory + "Barcodes")):
                headers = ["Subject", "Barcode"]
                self.barcodes = self.createDfFromFile(file.path, headers)
            elif(file.path.startswith(self.directory + "Landmark")):
                headers = ["Subject", "X", "Y", "XStd", "YStd"]
                self.landmarks = self.createDfFromFile(file.path, headers)
            elif(file.path.endswith("Groundtruth.dat")):
                headers = ["Time", "X", "Y", "Heading"]
                robotIndex = int(file.path[len(self.directory)+5])-1
                self.robotGroundTruth[robotIndex] = self.createDfFromFile(file.path, headers)
            elif(file.path.endswith("Odometry.dat")):
                headers = ["Time", "Velocity", "AngularVelocity"]
                robotIndex = int(file.path[len(self.directory)+5])-1
                self.robotOdometry[robotIndex] = self.createDfFromFile(file.path, headers)
            elif(file.path.endswith("Measurement.dat")):
                headers = ["Time", "Barcode", "Range", "Bearing"]
                robotIndex = int(file.path[len(self.directory)+5])-1
                self.robotMeasurements[robotIndex] = self.createDfFromFile(file.path, headers)
            else:
                raise Exception("File not recognized {}".format(file))
            i += 1

        self.map = Map(self.landmarks)
        self.robots = []
        for i in range(self.numRobots):
            groundTruth = self.robotGroundTruth[i]
            measurements = self.robotMeasurements[i]
            odometry = self.robotOdometry[i]
            self.robots.append(Robot(groundTruth, measurements, odometry, self.barcodes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mavenlink Optimization Problem')
    parser.add_argument('directory', type=str, help='Directory with input files')

    args = parser.parse_args()
    data = Data(args.directory)

    entry = data.robots[0].getNext()
