import numpy as np
import math

class EKF:


    def __init__(self, robotState, range, bearing):


        self.n = 2

        # Hard-coded covariance matrices
        # Found by taking samples from robot 1 measuring landmark 11
        #   at times 1248273512.011 to 1248273518.329 (odometry was 0)
        # [.01, .001]
        self.sigmaZ = np.diag([.075, .025]) #range, bearing

        robotX = robotState[0]
        robotY = robotState[1]
        robotTheta = robotState[2]


        x = robotX + range*np.cos(self.wrapToPi(robotTheta) + self.wrapToPi(bearing))
        y = robotY + range*np.sin(self.wrapToPi(robotTheta) + self.wrapToPi(bearing))

        # Running estimate
        self.stateEstimate = np.reshape(np.array([x,y]), (2,1))
        self.stateCovariance = np.ones((self.n, self.n))*.01


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

    def gauss(self, x, mu, std, scale):
        a = scale * 1/(std*math.sqrt(2*math.pi))
        b = -0.5/(std**2)
        g = a*math.exp(b*(x-mu)**2)
        return g

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

        diff[1] = self.wrapToPi(diff[1])

        weightRange = self.gauss(zt[0], zHat[0], .075, 1)
        weightBearing = self.gauss(zt[1], zHat[1], .025, 1)
        weight = weightRange * weightBearing

        self.stateEstimate = self.stateEstimate + K@(diff)
        self.stateCovariance = (np.identity(2) - K @ H) @ self.stateCovariance

        # self.stateEstimate = truth

        # weight = np.linalg.det(2*np.pi*Q)**-.5 * \
        #             np.exp(-.5*(diff).T @ Qinv @ (diff))

        return weight
