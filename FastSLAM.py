from EKF import EKF
from Particle import Particle
from DataLoader import *
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy

class State:
    def __init__(self):
        self.robotState = None
        self.landmarks = {}

class FastSLAM:
    def __init__(self, pkl, n):
        self.data = pickle.load(open(pkl, "rb"))
        self.n = n
        self.xRange = (-5, 5)
        self.yRange = (-5, 5)

        self.particles = []

        self.createParticles(self.n)

        self.stateLogs = []
        self.robotStates = []
        self.landmarks = []

        self.timeSeries = []

    def createParticles(self, n):
        for i in range(n):
            # x = np.random.rand()*10-5
            # y = np.random.rand()*10-5
            # theta = (np.random.rand()*2*np.pi)-np.pi
            x = 3.5732324
            y = -3.3328387
            theta = 2.3408
            p = Particle(self.n, [x,y,theta], self.data, i)
            self.particles.append(p)

    def runFastSLAM(self):
        robot1Data = self.data.robots[0]

        dt = 0
        prevOdomTime = robot1Data.odometry[0][0]

        while not robot1Data.empty():
            keyFrame = robot1Data.getNext()
            t = keyFrame[1][0]

            if keyFrame[0] == 'odometry':
                odometry = keyFrame[1]
                dt = t - prevOdomTime
                # print("===== Particle states =====")
                for p in self.particles:
                    p.propagateMotion(odometry, dt)
                    # print(p.robotState)
                prevOdomTime = t
            else:
                measurement = keyFrame[1]
                subject = measurement[1]
                if subject > 5:
                    for p in self.particles:
                        p.correct(measurement)

                    weights = np.array([p.weight for p in self.particles]).flatten().astype("float64")
                    weightSum = sum(weights)
                    if weightSum != 0:
                        for i in range(len(weights)):
                            weights[i] /= weightSum
                    else:
                        print("Weights were zero!!")
                        exit()
                        weights = [1/self.n for _ in range(self.n)]

                    particleIndices = np.random.choice(list(range(self.n)), self.n, replace=True, p=weights)
                    print(particleIndices)
                    self.particles = [self.particles[i] for i in particleIndices]
                    print([p.id for p in self.particles])

            self.timeSeries.append(t)

            self.stateLogs.append(copy.deepcopy(self.getStateMaxWeight()))

        plt.plot(self.timeSeries)
        plt.show()

    def runSlowSLAM(self):
        delay(1000000)
        self.runFastSLAM()
        delay(1000000)

    def getStateMaxWeight(self):
        maxWeightIndex = np.argmax([p.weight for p in self.particles])
        bestParticle = self.particles[maxWeightIndex]

        state = State()
        state.robotState = bestParticle.robotState

        for landmark in bestParticle.landmarkEKFs:
            position = bestParticle.landmarkEKFs[landmark].stateEstimate
            state.landmarks[landmark] = position

        return state

    def getStateAvg(self):
        state = State()

        xPos = [p.robotState[0] for p in self.particles]
        yPos = [p.robotState[1] for p in self.particles]
        thPos = [p.robotState[2] for p in self.particles]

        # TODO: Hack because all EKFS have ground truth right now
        for landmark in self.particles[0].landmarkEKFs:
            position = self.particles[0].landmarkEKFs[landmark].stateEstimate
            state.landmarks[landmark] = position

        state.robotState = [np.average(xPos), np.average(yPos), np.average(thPos)]
        return state



if __name__ == '__main__':
    slam = FastSLAM("Jar/dataset1.pkl", 25)
    slam.runFastSLAM()

    stateEstimates = slam.stateLogs

    xData = [s.robotState[0] for s in stateEstimates]
    yData = [s.robotState[1] for s in stateEstimates]
    thetaData = [s.robotState[2] for s in stateEstimates]


    groundTruth = slam.data.robots[0].groundTruthPosition


    def wrapToPi(th):
        th = np.fmod(th, 2*np.pi)
        if th >= np.pi:
            th -= 2*np.pi

        if th <= -np.pi:
            th += 2*np.pi
        return th

    timeTruth = [g[0] for g in groundTruth]
    xTruth = [g[1] for g in groundTruth]
    yTruth = [g[2] for g in groundTruth]
    theTruth = [wrapToPi(g[3]) for g in groundTruth]

    fig, ax = plt.subplots()

    # Plotting Yaw

    minTime = min(slam.timeSeries[0], timeTruth[0])
    plt.plot(np.array(slam.timeSeries)-minTime, thetaData, label="Estimated Angle")
    plt.plot(np.array(timeTruth)-minTime, theTruth, label="Ground Truth Angle")
    plt.title("Theta Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")

    plt.figure()
    plt.plot(xData, yData, label="Estimated Path")
    plt.plot(xTruth, yTruth, label="True Path")

    landmarks = stateEstimates[-1].landmarks
    xLandmarks = []
    yLandmarks = []

    for l in landmarks.values():
        xLandmarks.append(l[0])
        yLandmarks.append(l[1])

    xLandmarksTrue = []
    yLandmarksTrue = []

    for l in slam.data.map.landmarkDict.values():
        xLandmarksTrue.append(l["X"])
        yLandmarksTrue.append(l["Y"])

    plt.scatter(xLandmarks, yLandmarks, label="Estimated Landmarks", color='g')
    plt.scatter(xLandmarksTrue, yLandmarksTrue, label="Ground Truth Landmarks", color='r')
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("SLAM Results")
    plt.legend()
    plt.show()
