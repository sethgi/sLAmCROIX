"""
File: EKF.py
Author: Seth Isaacson
Description: A general, abstract implementation of an n-dof EKF, with support
for arbitrarily many prediction steps per correction step (and vice-versa).
"""

import numpy as np

class EKF:

    def __init__(self, robotState, range, bearing):


        self.n = 2

        # Hard-coded covariance matrices
        # Found by taking samples from robot 1 measuring landmark 11
        #   at times 1248273512.011 to 1248273518.329 (odometry was 0)
        # [.01, .001]
        self.sigmaZ = np.diag([0.1, 0.1]) #range, bearing

        robotX = robotState[0]
        robotY = robotState[1]
        robotTheta = robotState[2]


        x = robotX + range*np.cos(self.wrapToPi(robotTheta) + self.wrapToPi(bearing))
        y = robotY + range*np.sin(self.wrapToPi(robotTheta) + self.wrapToPi(bearing))

        # Running estimate
        self.stateEstimate = np.reshape(np.array([x,y]), (2,1))
        self.stateCovariance = np.ones((self.n, self.n))


        self.stateEstimateLogs = []
        self.stateCovarianceLogs = []

    # The most important function
    def wrapToPi(self, th):
        th = np.fmod(th, 2*np.pi)
        if th >= np.pi:
            th -= 2*np.pi

        if th <= -np.pi:
            th += 2*np.pi
        return th

    def setMeasurementCovariance(self, cov):
        self.sigmaZ = cov

    # Give predicted measurement
    def measurementModel(self, robotState):
        robotX = robotState[0]
        robotY = robotState[1]
        robotTheta = self.wrapToPi(robotState[2])

        range = np.sqrt((robotX - self.stateEstimate[0])**2 + \
                        (robotY - self.stateEstimate[1])**2)

        bearing = np.arctan2(self.stateEstimate[1] - robotY, self.stateEstimate[0]-robotX)-robotTheta
        bearing = self.wrapToPi(bearing)

        return np.array([range, bearing])

    # Returns jacobian of measurement wrt previous state
    def computeMeasurementJacobian(self, range, bearing, robotState):
        robotX = robotState[0]
        robotY = robotState[1]

        xDist = self.stateEstimate[0] - robotX
        yDist = self.stateEstimate[1] - robotY

        denominator = xDist**2 + yDist**2

        jacobian = np.array([[xDist/np.sqrt(denominator), yDist/np.sqrt(denominator)],\
                            [-yDist/denominator, xDist/denominator ]]).reshape((2,2))
        return jacobian

    # Correction step for EKF
    def correct(self, range, bearing, robotState, truth=None):

        predictedMeasurement = self.measurementModel(robotState)
        zHat = np.reshape(predictedMeasurement, (2,1))

        H = self.computeMeasurementJacobian(range, bearing, robotState)

        Q = H @ self.stateCovariance @ H.T + self.sigmaZ

        if np.linalg.matrix_rank(Q) != 2:
            raise Exception("Singular whoopsies")
        Qinv = np.linalg.inv(Q)

        # print("Q: ", Q)

        K = self.stateCovariance @ H.T @ Qinv

        zt = np.reshape(np.array([range, bearing]), (2,1))

        diff = zt - zHat

        # https://stackoverflow.com/questions/7570808/how-do-i-calculate-the-difference-of-two-angle-measures/30887154
        # angleDist = abs(zt[1] - zHat[1])%2*np.pi
        # if angleDist > np.pi:
        #     angleDist = 2*np.pi - angleDist
        # sign = 1 if (zt[1] - zHat[1] >= 0 and zt[1] - zHat[1] <= np.pi) or \
        #        (zt[1] - zHat[1] <=-np.pi and zt[1]- zHat[1]>= -2*np.pi)  \
        #        else  -1
        # angleDist *= sign

        diff[1] = self.wrapToPi(diff[1])
        # print("expected: ", zHat, "Actual: ", zt)
        self.stateEstimate = self.stateEstimate + K@(diff)
        self.stateCovariance = (np.identity(2) - K @ H) @ self.stateCovariance

        self.stateEstimate = truth

        weight = np.linalg.det(2*np.pi*Q)**-.5 * \
                    np.exp(-.5*(diff).T @ Qinv @ (diff))

        # print("Actual: ", zt.reshape((1,2)), " Expected: ", zHat.reshape((1,2)))
        # print(np.exp(-.5*(zt-zHat).T @ Qinv @ (zt-zHat)))

        # Note: Notation on z is inconsistent on p. 450
        # print("weight ", weight)
        return weight
