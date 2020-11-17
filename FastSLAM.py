from EKF import EKF
from Particle import Particle
from DataLoader import *
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.animation as animation
from tqdm import tqdm

ROBOT_ID = 0

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

        self.stateLogs = []
        self.robotStates = []
        self.landmarks = []

        self.timeSeries = []

        self.stateEstimates = []

        self.createParticles(self.n)

        # once every this many time steps, record the entire state (SLOW!)
        self.estimateSnapshotInterval = 75
        self.snapshotCounter = 0

        self.thetaSigma = .2

    def wrapToPi(self, th):
        th = np.fmod(th, 2*np.pi)
        if th >= np.pi:
            th -= 2*np.pi

        if th <= -np.pi:
            th += 2*np.pi
        return th

    def createParticles(self, n):
        for i in range(n):
            # x = np.random.rand()*10-5
            # y = np.random.rand()*10-5
            # theta = (np.random.rand()*2*np.pi)-np.pi
            groundTruthStart = self.data.robots[ROBOT_ID].groundTruthPosition[0]
            x = groundTruthStart[1]
            y = groundTruthStart[2]
            theta = groundTruthStart[3]
            p = Particle(self.n, [x,y,theta], self.data.map, i)
            self.particles.append(p.copy())
        # self.stateEstimates.append(self.particles)


    def gauss(self, x, mu, std):
        a = 1/(std*np.sqrt(2*np.pi))
        b = -0.5/(std**2)
        g = a*np.exp(b*(x-mu)**2)
        return g


    def runFastSLAM(self):
        robot1Data = self.data.robots[ROBOT_ID]

        dt = 0
        prevOdomTime = robot1Data.odometry[0][0]

        # count = 0
        size = copy.deepcopy(robot1Data.size())
        print(size)
        for i in tqdm(range(size)):
            # print(count, robot1Data.size())
            # count += 1

            keyFrame = robot1Data.getNext()
            t = keyFrame[1][0]

            if keyFrame[0] == 'odometry':
                odometry = keyFrame[1]
                dt = t - prevOdomTime
                prevOdomTime = t

                # print("===== Particle states =====")
                for p in self.particles:

                    thetaMeas = self.wrapToPi(robot1Data.getCompass(t))

                    p.propagateMotion(odometry, thetaMeas, dt)


                    # Local correction on compass measurement
                    # p.thetaWeight = self.gauss(thetaMeas, self.wrapToPi(p.robotState[2]), self.thetaSigma)

                # Resample based on theta weights
                # weights = [p.thetaWeight for p in self.particles]
                # weightSum = sum(weights)
                # if weightSum == 0:
                #     print("Weights 0 in theta")
                #     weights = [1/self.n for _ in range(self.n)]
                # else:
                #     weights = [w/weightSum for w in weights]
                # particleIndices = np.random.choice(list(range(self.n)), self.n, replace=True, p=weights)
                # # print(particleIndices)
                # self.particles = [self.particles[i].copy() for i in particleIndices]

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
                        # exit()
                        weights = [1/self.n for _ in range(self.n)]

                    particleIndices = np.random.choice(list(range(self.n)), self.n, replace=True, p=weights)
                    # print(particleIndices)
                    self.particles = [self.particles[i].copy() for i in particleIndices]
                    # print([p.id for p in self.particles])
            self.stateLogs.append(copy.deepcopy(self.getStateAvg()))

            self.timeSeries.append(t)

            if self.snapshotCounter == self.estimateSnapshotInterval:
                self.snapshotCounter = 0
                self.stateEstimates.append(copy.deepcopy(self.particles))
            self.snapshotCounter += 1

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


        for landmark in self.particles[0].landmarkEKFs:
            x = np.average([p.landmarkEKFs[landmark].stateEstimate[0] for p in self.particles] )
            y = np.average([p.landmarkEKFs[landmark].stateEstimate[1] for p in self.particles] )
            state.landmarks[landmark] = (x,y)
        state.robotState = [np.average(xPos), np.average(yPos), np.average(thPos)]
        return state



if __name__ == '__main__':
    slam = FastSLAM("Jar/dataset1.pkl", 150)
    slam.runFastSLAM()

    stateEstimates = slam.stateLogs

    xData = [s.robotState[0] for s in stateEstimates]
    yData = [s.robotState[1] for s in stateEstimates]
    thetaData = [s.robotState[2] for s in stateEstimates]


    groundTruth = slam.data.robots[ROBOT_ID].groundTruthPosition


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
    plt.plot(np.array(slam.timeSeries)-minTime, np.array(thetaData), label="Estimated Angle")
    plt.plot(np.array(timeTruth)-minTime, theTruth, label="Ground Truth Angle")



    odometryTime = [s[0] for s in slam.data.robots[ROBOT_ID].odometry]
    odometryTheta = [slam.data.robots[ROBOT_ID].getCompass(t) for t in odometryTime]

    plt.plot(np.array(odometryTime)-minTime, odometryTheta, label="INterp")

    plt.title("Theta Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")
    plt.legend()

    plt.figure()
    plt.plot(xData, yData, label="Estimated Path")
    plt.plot(xTruth, yTruth, label="True Path")

    landmarks = stateEstimates[-1].landmarks
    xLandmarks = []
    yLandmarks = []

    for l in landmarks:
        xLandmarks.append(landmarks[l][0])
        yLandmarks.append(landmarks[l][1])

    xLandmarksTrue = []
    yLandmarksTrue = []

    for l in slam.data.map.landmarkDict:
        # if l == 16:
        lm = slam.data.map.landmarkDict[l]
        xLandmarksTrue.append(lm["X"])
        yLandmarksTrue.append(lm["Y"])

    plt.scatter(xLandmarks, yLandmarks, label="Estimated Landmarks", color='g')
    plt.scatter(xLandmarksTrue, yLandmarksTrue, label="Ground Truth Landmarks", color='r')
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("SLAM Results")
    plt.legend()
    plt.show()

    """
    Animation
    """
    fig, ax = plt.subplots()

    ln, = plt.plot([], [], '.')


    plt.plot(xData, yData, label="Estimated Path")

    animationEstimates = []



    def init():
        plt.scatter(xLandmarksTrue, yLandmarksTrue, label="Ground Truth Landmarks", color='r')
        plt.scatter(xLandmarks, yLandmarks, label="Estimated Landmarks", color='g')
        plt.plot(xData, yData, label="Estimated Path")
        plt.plot(xTruth, yTruth, label="True Path")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.legend()
    # return ln,
    # print(len(slam.stateEstimates))
    def update(frame):
        particles = slam.stateEstimates[frame]
        ln.set_xdata([p.robotState[0] for p in particles])
        ln.set_ydata([p.robotState[1] for p in particles])
        return fig,

    animate = animation.FuncAnimation(fig, update, frames=range(len(slam.stateEstimates)),\
     init_func=init, interval=100)
    plt.legend()
    # animate.save('./smallSpread.gif',writer='imagemagick', fps=10)

    plt.show()
