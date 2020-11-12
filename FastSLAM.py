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

    def createParticles(self, n):
        for _ in range(n):
            # x = np.random.rand()*10-5
            # y = np.random.rand()*10-5
            # theta = (np.random.rand()*2*np.pi)-np.pi
            x = 3.5732324
            y = -3.3328387
            theta = 2.3408
            p = Particle(self.n, [x,y,theta])
            self.particles.append(p)

    def runFastSLAM(self):
        robot1Data = self.data.robots[0]

        prevTime = 0

        while not robot1Data.empty():
            keyFrame = robot1Data.getNext()
            t = keyFrame["Time"]
            odometry = keyFrame["Odometry"]
            measurements = keyFrame["Measurements"]

            dt = t - prevTime
            prevTime = t

            # Move population
            for p in self.particles:
                if odometry is not None:
                    p.propagateMotion(odometry, dt)
                if measurements != []:
                    for m in measurements:
                        p.correct(m)

            # Resample
            weights = np.array([p.weight for p in self.particles]).flatten().astype("float64")
            weightSum = sum(weights)
            if weightSum != 0:
                for i in range(len(weights)):
                    old = weights[i]
                    weights[i] /= weightSum
            else:
                weights = [1/self.n for _ in range(self.n)]
            particleIndices = np.random.choice(list(range(self.n)), self.n, replace=True, p=weights)
            self.particles = [self.particles[i] for i in particleIndices]

            self.stateLogs.append(copy.deepcopy(self.getStateMaxWeight()))


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



if __name__ == '__main__':
    slam = FastSLAM("Jar/dataset1.pkl", 20)
    slam.runFastSLAM()

    stateEstimates = slam.stateLogs

    xData = [s.robotState[0] for s in stateEstimates]
    yData = [s.robotState[1] for s in stateEstimates]
    thetaData = [s.robotState[2] for s in stateEstimates]

    groundTruth = slam.data.robots[0].groundTruthPosition

    timeTruth = [g[0] for g in groundTruth]
    xTruth = [g[1] for g in groundTruth]
    yTruth = [g[2] for g in groundTruth]
    theTruth = [g[3] for g in groundTruth]

    fig, ax = plt.subplots()

    # Plotting Yaw
    timeData = sorted(slam.data.robots[0].dataDict.keys())
    plt.plot(timeData, thetaData)
    plt.plot(timeTruth, theTruth)
    plt.figure()

    plt.plot(xData, yData, label="Estimated Path")
    plt.plot(xTruth, yTruth, label="True Path")
    plt.legend()
    plt.show()

    # def init():
    #     plt.ylim(-14, 4)
    #     plt.xlim(-4,14)
    #     ax.plot([0,10,10,0, 0], [0, 0, -10, -10, 0], label= "Expected Path")
    #     plt.xlabel("X Coordinate (m)")
    #     plt.ylabel("Y Coordinate (m)")
    # # return ln,
    #
    # def update(frame):
    #     P = state_estimates[frame]
    #     ln.set_xdata([p[0] for p in P])
    #     ln.set_ydata([p[1] for p in P])
    #     return fig,
    #
    # animate = animation.FuncAnimation(fig, update, frames=range(len(state_estimates)),\
    #  init_func=init, interval=10)
    # plt.legend()
    # animate.save('./animation.gif',writer='imagemagick', fps=10)
