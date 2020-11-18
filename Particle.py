import numpy as np
from EKF import EKF
import copy
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, n, state=[0,0,0], map = None, id = 0):

        self.robotState = state#x,y,theta
        self.robotState[2] = self.wrapToPi(self.robotState[2])

        self.weight = 1/n

        # Subject number : EKF
        self.landmarkEKFs = {}

        # .01, .01
        self.velocitySigma = 0.05
        self.angleSigma = 0.003
        self.slipSigma = 0.075

        self.X_IDX = 0
        self.Y_IDX = 1
        self.THETA_IDX = 2

        self.n = n
        self.map = map

        self.id = id

        self.thetaWeight = 1


    def copy(self):
        robotState = copy.deepcopy(self.robotState)
        weight = self.weight
        landmarkEKFs = copy.deepcopy(self.landmarkEKFs)
        n = self.n
        map = self.map

        p = Particle(n, robotState, map, self.id)
        p.landmarkEKFs = landmarkEKFs

        return p

    # Control =
    def propagateMotion(self, control, thetaMeas, dt):
        velocity = control[1] + skewnorm.rvs(-.16, 0, self.velocitySigma)
        velocity = max(0, velocity)

        slipVel = np.random.normal(0, self.slipSigma)

        thetaMeas += np.random.normal(0, self.angleSigma)
        self.robotState[self.THETA_IDX] = self.wrapToPi(thetaMeas)

        xVel = velocity*np.cos(self.robotState[self.THETA_IDX])
        yVel = velocity*np.sin(self.robotState[self.THETA_IDX])

        self.robotState[self.X_IDX] += xVel*dt - slipVel*dt*np.sin(self.robotState[self.THETA_IDX])
        self.robotState[self.Y_IDX] += yVel*dt + slipVel*dt*np.cos(self.robotState[self.THETA_IDX])


    # Measurement = [time, subject, range, bearing]
    def correct(self, measurement):
        subject = measurement[1]
        range = measurement[2]
        bearing = measurement[3]

        if subject < 5:
            raise Exception("Invalid Subject")
        truth = self.map[subject]
        truth = [truth["X"], truth["Y"]]

        if subject not in self.landmarkEKFs:
            newEKF = EKF(copy.deepcopy(self.robotState), range, bearing)
            self.landmarkEKFs[subject] = newEKF
            self.weight = 1/self.n
        else:
            self.weight = self.landmarkEKFs[subject].correct(range,bearing,copy.deepcopy(self.robotState), truth)
        return self.weight

    def wrapToPi(self, th):
        th = np.fmod(th, 2*np.pi)
        if th >= np.pi:
            th -= 2*np.pi

        if th <= -np.pi:
            th += 2*np.pi
        return th

if __name__ == '__main__':
    p = Particle(1, [0,0,0])
    particles = []
    for _ in range(500):
        pNew = copy.deepcopy(p)
        particles.append(pNew)

    onePartX = []
    onePartY = []

    for _ in range(5 ):
        for i in range(500):
            particles[i].propagateMotion([0, 0.075, 0.25], np.radians(15), 0.01)
        onePartX.append(particles[0].robotState[0])
        onePartY.append(particles[0].robotState[1])

    positions = [a.robotState for a in particles]
    xPos = [a[0] for a in positions]
    yPos = [a[1] for a in positions]
    thPos = [a[2] for a in positions]

    plt.scatter(p.robotState[0], p.robotState[1], label="original")
    plt.scatter(xPos, yPos, label="Moved")
    plt.plot(onePartX, onePartY)
    plt.legend()
    plt.show()
