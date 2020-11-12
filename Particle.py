import numpy as np
from EKF import EKF

class Particle:
    def __init__(self, n, state=[0,0,0]):

        self.robotState = state#x,y,theta
        self.robotState[2] = self.wrapToPi(self.robotState[2])
        self.weight = 1

        # Subject number : EKF
        self.landmarkEKFs = {}

        # .01, .01
        self.velocitySigma = 0.05
        self.angleSigma = 0.05

        self.X_IDX = 0
        self.Y_IDX = 1
        self.THETA_IDX = 2

        self.n = n

    # Control = {Time: , }
    def propagateMotion(self, control, dt):
        velocity = control[0] + np.random.normal(0, self.velocitySigma)
        angularVelocity = control[1] + np.random.normal(0, self.angleSigma)

        # print(angle, angularVelocity)

        self.robotState[self.X_IDX] += velocity*np.cos(self.robotState[self.THETA_IDX])*dt
        self.robotState[self.Y_IDX] += velocity*np.sin(self.robotState[self.THETA_IDX])*dt
        self.robotState[self.THETA_IDX] += angularVelocity*dt
        self.robotState[self.THETA_IDX] = self.wrapToPi(self.robotState[self.THETA_IDX])


    # Measurement = [subject, range, bearing]
    def correct(self, measurement):
        subject = measurement[0]
        range = measurement[1]
        bearing = measurement[2]

        if subject not in self.landmarkEKFs:
            newEKF = EKF(self.robotState, range, bearing)
            self.landmarkEKFs[subject] = newEKF
            self.weight = 1/self.n
        else:
            self.weight = self.landmarkEKFs[subject].correct(range,bearing,self.robotState)


    def wrapToPi(self, th):
        th = np.fmod(th, 2*np.pi)
        if th >= np.pi:
            th -= 2*np.pi

        if th <= -np.pi:
            th += 2*np.pi
        return th
