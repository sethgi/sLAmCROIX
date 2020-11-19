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
import scipy.interpolate as interp

NUM_STEPS = 15000
NUM_PARTS = 1
ROBOT_ID = 2
class State:
    def __init__(self):
        self.robotState = None
        self.landmarks = {}

class FastSLAM:
    def __init__(self, pkl, n, id):
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


        # once every this many time steps, record the entire state (SLOW!)
        self.estimateSnapshotInterval = 200
        self.snapshotCounter = 0

        self.robotId = id

        self.createParticles(self.n)

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
            groundTruthStart = self.data.robots[self.robotId].groundTruthPosition[0]
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
        robot1Data = self.data.robots[self.robotId]

        dt = 0
        prevOdomTime = robot1Data.odometry[0][0]

        # count = 0
        size = copy.deepcopy(robot1Data.size())
        print(size)
        for i in tqdm(range(NUM_STEPS)):
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

            # else:
            #     measurement = keyFrame[1]
            #     subject = measurement[1]
            #     if subject > 5:
            #         for p in self.particles:
            #             p.correct(measurement)
            #
            #         weights = np.array([p.weight for p in self.particles]).flatten().astype("float64")
            #         weightSum = sum(weights)
            #         if weightSum != 0:
            #             for i in range(len(weights)):
            #                 weights[i] /= weightSum
            #         else:
            #             print("Weights were zero!!")
            #             # exit()
            #             weights = [1/self.n for _ in range(self.n)]
            #
            #         particleIndices = np.random.choice(list(range(self.n)), self.n, replace=True, p=weights)
            #         # print(particleIndices)
            #         self.particles = [self.particles[i].copy() for i in particleIndices]
            #         # print([p.id for p in self.particles])
            self.stateLogs.append(copy.deepcopy(self.getStateAvg()))

            self.timeSeries.append(t)

            if self.snapshotCounter == self.estimateSnapshotInterval:
                self.snapshotCounter = 0
                particleSet = [p.copy() for p in self.particles]
                self.stateEstimates.append(particleSet)
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


def rms(truth, estimate):
    if len(truth) != len(estimate):
        print("RMS is weird when arrays aren't same length")

    errors = [truth[i] - estimate[i] for i in range(len(truth))]
    return np.sqrt(sum(e**2 for e in errors)/len(errors))

def euclid(x, xt, y, yt):
    return np.sqrt((x-xt)**2 + (y-yt)**2)

def euclidRMS(xTruth, xEstimate, yTruth, yEstimate):
    errors = [euclid(xEstimate[i], xTruth[i], yEstimate[i], yTruth[i]) for i in range(len(xTruth))]
    return np.sqrt(sum(e**2 for e in errors)/len(errors))


if __name__ == '__main__':
    slam = FastSLAM("Jar/dataset1.pkl", NUM_PARTS, ROBOT_ID)
    slam.runFastSLAM()
    # pickle.dump(slam, open('slam.pkl', 'wb'))

    # data = pickle.load(open("Jar/dataset1.pkl", 'rb'))
    #
    # slam = pickle.load(open('slam.pkl', "rb"))


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



    odometryTime = [s[0] for s in slam.data.robots[slam.robotId].odometry]
    odometryTheta = [slam.data.robots[slam.robotId].getCompass(t) for t in odometryTime]

    plt.title("Theta Tracking")
    plt.xlabel("Time (s)")
    plt.ylabel("Heading (rad)")
    plt.legend()

    theTruthInterp = [slam.data.robots[0].getCompass(t) for t in slam.timeSeries]
    thetaRMS = rms(theTruthInterp, thetaData)
    thRmsMessage = "Theta RMS (rad): {}".format(thetaRMS)
    plt.annotate(thRmsMessage, xy=(0.05, 0.95), xycoords="axes fraction")

    plt.figure()

    plt.plot(np.array(slam.timeSeries)-minTime, xData, label="X Estimate")
    plt.plot(np.array(timeTruth) - minTime, xTruth, label="X Truth")
    plt.xlabel("Time (s)")
    plt.ylabel("X (m)")

    xTruthInterp = [slam.data.robots[0].getXTruth(t) for t in slam.timeSeries]
    xRMS = rms(xTruthInterp, xData)
    # plt.plot(np.array(slam.timeSeries)-minTime, xTruthInterp, label="Interp Truth")
    xRmsMessage = "X RMS (m): {}".format(xRMS)
    plt.annotate(xRmsMessage, xy=(0.05, 0.95), xycoords="axes fraction")
    plt.legend()
    plt.figure()

    yTruthInterp = [slam.data.robots[0].getYTruth(t) for t in slam.timeSeries]

    plt.plot(np.array(slam.timeSeries)-minTime, yData, label="Y Estimate")
    plt.plot(np.array(timeTruth) - minTime, yTruth, label="Y Truth")
    # plt.plot(np.array(slam.timeSeries)-minTime, yTruthInterp, label="Interp Truth")

    plt.xlabel("Time (s)")
    plt.ylabel("Y (m)")

    yRMS = rms(yTruthInterp, yData)
    yRmsMessage = "Y RMS (m): {}".format(yRMS)
    plt.annotate(yRmsMessage, xy=(0.05, 0.95), xycoords="axes fraction")
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


    pathRMS = euclidRMS(xTruthInterp, xData, yTruthInterp, yData)
    pathRmsMessage = "Path RMS (m): {}".format(pathRMS)
    plt.annotate(pathRmsMessage, xy=(0.05, 0.95), xycoords="axes fraction")


    plt.scatter(xLandmarks, yLandmarks, label="Estimated Landmarks", color='g')
    plt.scatter(xLandmarksTrue, yLandmarksTrue, label="Ground Truth Landmarks", color='r')
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.title("SLAM Results")
    plt.legend()
    # plt.show()

    """
    Animation
    """
    fig, ax = plt.subplots()

    ln, = plt.plot([], [], '.')
    lnLandmarks = plt.scatter([], [], label="Estimated Landmarks")#, marker='.',color="green", markersize=5)


    plt.plot(xData, yData, label="Estimated Path")

    animationEstimates = []



    def init():
        plt.scatter(xLandmarksTrue, yLandmarksTrue, label="Ground Truth Landmarks", color='r')
        plt.plot(xData, yData, label="Estimated Path")
        plt.plot(xTruth, yTruth, label="True Path")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.legend()
    # return ln,
    # print(len(slam.stateEstimates))
    def update(frame):
        f = frame*3
        particles = slam.stateEstimates[f]
        ln.set_xdata([p.robotState[0] for p in particles])
        ln.set_ydata([p.robotState[1] for p in particles])

        landmarks = slam.stateLogs[f*slam.estimateSnapshotInterval].landmarks
        xLandmarks = []
        yLandmarks = []

        for l in landmarks:
            xLandmarks.append(landmarks[l][0])
            yLandmarks.append(landmarks[l][1])

        lnLandmarks.set_offsets(np.c_[xLandmarks, yLandmarks])
        return fig,

    animate = animation.FuncAnimation(fig, update, frames=range(len(slam.stateEstimates)//3),\
     init_func=init, interval=100)
    plt.legend()

    # animate.save('./muchBetter.gif',writer='imagemagick', fps=10)

    plt.show()
