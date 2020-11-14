import numpy as np
from EKF import EKF
import copy
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, n, state=[0,0,0], data = None):

        self.robotState = state#x,y,theta
        self.robotState[2] = self.wrapToPi(self.robotState[2])

        self.weight = 1

        # Subject number : EKF
        self.landmarkEKFs = {}

        # .01, .01
        self.velocitySigma = 0.01
        self.angleSigma = 0.01

        self.X_IDX = 0
        self.Y_IDX = 1
        self.THETA_IDX = 2

        self.n = n
        self.data = data

    # Control =
    def propagateMotion(self, control, dt):
        velocity = control[1] + np.random.normal(0, self.velocitySigma)
        angularVelocity = control[2] + np.random.normal(0, self.angleSigma)

        xVel = velocity*np.cos(self.robotState[self.THETA_IDX])
        yVel = velocity*np.sin(self.robotState[self.THETA_IDX])

        self.robotState[self.X_IDX] += xVel*dt
        self.robotState[self.Y_IDX] += yVel*dt

        dTheta = angularVelocity*dt
        self.robotState[self.THETA_IDX] += angularVelocity*dt
        self.robotState[self.THETA_IDX] = self.wrapToPi(self.robotState[self.THETA_IDX])

    # Measurement = [subject, range, bearing]
    def correct(self, measurement):
        subject = measurement[1]
        range = measurement[2]
        bearing = measurement[3]

        if subject <= 5:
            raise Exception("Invalid Subject")
        truth = self.data.map[subject]
        truth = [truth["X"], truth["Y"]]

        if subject not in self.landmarkEKFs:
            newEKF = EKF(self.robotState, range, bearing)
            self.landmarkEKFs[subject] = newEKF
            self.weight = 1/self.n
        else:
            self.weight = self.landmarkEKFs[subject].correct(range,bearing,self.robotState, truth)
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

    for _ in range(50):
        for i in range(500):
            particles[i].propagateMotion([0.075, 0.25], 0.01)
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
