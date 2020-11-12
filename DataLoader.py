"""
File: DataLoader.py
Description: Reads data from the UTIAS dataset and provides classes to interact with data
Authors: Seth Isaacson
"""
import pandas as pd
import argparse
import os
import copy
import numpy as np
import pickle

class Map:
    def __init__(self, data):
        self.data = data
        self.landmarkDict = self.data.set_index('Subject').to_dict('index')

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
        return len(self.dataQueue) == 0

    def size(self):
        return len(self.dataQueue)

    def reset(self):
        self.dataQueue = copy.deepcopy(self.dataList)

    def buildDict(self):
        self.dataQueue = []
        self.dataDict = {}
        barcodeDict = {}
        self.groundTruthPosition = []

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
            self.groundTruthPosition.append((time,x,y,heading))

        for row in self.measurements.itertuples():
            subject = None
            barcode = row.Barcode
            if barcode not in barcodeDict:
                print("Unrecogonized Barcode: {}. Skipping".format(barcode))
                continue
            else:
                subject = barcodeDict[barcode]

            time = float(row.Time)
            if time not in self.dataDict:
                self.dataDict[time] = {"Time": time, "GroundTruth": None, \
                        "Measurements": [(subject, row.Range, row.Bearing)], \
                        "Odometry": None}
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
        print("=== Data Loaded ===")


    def createDfFromFile(self, fname, headers):
        # i = 0
        # data = []
        # with open(fname) as file:
        #     for line in file:
        #         if i < 4:
        #             i += 1
        #             continue
        #         i += 1
        #         data.append(tuple(line.split()))
        #
        # df = pd.DataFrame(data, columns = headers)
        return pd.read_table(fname, names=headers, skiprows=4)

    def loadAllData(self):
        files = os.scandir(self.directory)
        self.numRobots = int((len(list(files))-2)/3)

        self.robotGroundTruth = [None for _ in range(self.numRobots)]
        self.robotMeasurements = [None for _ in range(self.numRobots)]
        self.robotOdometry = [None for _ in range(self.numRobots)]

        i = 0

        if self.directory[-1] != '/':
            self.directory += "/"

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
    pickle.dump(data, open("Jar/dataset1.pkl", "wb"))
    #
    # robotData = data.robots[0]
    #
    # dict = robotData.dataDict
    #
    # range = []
    # bearing = []
    # while not robotData.empty():
    #     data = robotData.getNext()
    #     measurements = data["Measurements"]
    #     t = data["Time"]
    #     if measurements == [] or t < 1248273512.011 or t > 1248273518.329:
    #         continue
    #     for m in measurements:
    #         if m[0] == 11:
    #             range.append(m[1])
    #             bearing.append(m[2])
    # print(range)
    # print(bearing)
    # print("Range Var: ", np.var(range))
    # print("Bearing Var: ", np.var(bearing))
