from EKF import EKF
from Particle import Particle
from DataLoader import *
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import scipy.interpolate as interp
from scipy.stats import skewnorm


def wrapToPi(th):
    th = np.fmod(th, 2*np.pi)
    if th >= np.pi:
        th -= 2*np.pi

    if th <= -np.pi:
        th += 2*np.pi
    return th

def propagateMotion(state, control, thetaMeas, dt):
        state_new = [0,0,0]
        velocity = control[1] + skewnorm.rvs(-0.5, 0, 0.04) # -0.16, 0, 0.04
        velocity = max(0, velocity)

        slipVel = np.random.normal(0, 0.07)

        thetaMeas += np.random.normal(0, 0.0125)
        state_new[2] = wrapToPi(thetaMeas)

        xVel = velocity*np.cos(state_new[2])
        yVel = velocity*np.sin(state_new[2])

        state_new[0] = state[0] + xVel*dt - slipVel*dt*np.sin(state[2])
        state_new[1] = state[1] + yVel*dt + slipVel*dt*np.cos(state[2])

        return state_new

def propagateMotionNOSLIP(state, control, thetaMeas, dt):
        state_new = [0,0,0]
        velocity = control[1] # + skewnorm.rvs(-0.5, 0, 0.04) # -0.16, 0, 0.04
        velocity = max(0, velocity)

        slipVel = np.random.normal(0, 0.07)

        thetaMeas += np.random.normal(0, 3) #(0, 0.0125)
        state_new[2] = wrapToPi(thetaMeas)

        xVel = velocity*np.cos(state_new[2])
        yVel = velocity*np.sin(state_new[2])

        state_new[0] = state[0] + xVel*dt # - slipVel*dt*np.sin(state[2])
        state_new[1] = state[1] + yVel*dt # + slipVel*dt*np.cos(state[2])

        return state_new

def propagateRandom(state, control, dt):
        state_new = [0,0,0]
        d_th = wrapToPi(np.random.normal(0, 0.013))
        d_d = np.random.normal(0.0009,0.0005)

        state_new[2] = state[2] + d_th
        state_new[0] = state[0] + d_d*np.cos(state_new[2])
        state_new[1] = state[1] + d_d*np.sin(state_new[2])
        return state_new

def deltaD(state, state_new):
    return np.sqrt((state_new[0]-state[0])**2 + (state_new[1]-state[1])**2)

def deltaTH(state, state_new):
    return (state_new[2] - state[2])

def clean(data, min, max):
    cleandata = []
    for d in data:
        if (d<max):
            cleandata.append(d)
    return cleandata

if __name__ == '__main__':
    robotdata = pickle.load(open("Jar/robotdata.pkl", "rb"))

    # pickle.dump(robotdata, open("Jar/robotdata.pkl", "wb"))


    # velocities = [odom[1] for odom in robotdata.odometry]
    # omegas = [odom[2] for odom in robotdata.odometry]
    t = robotdata.odometry[0][0]

    mm0_d = []
    mm0_th = []
    mm0_x = []
    mm0_y = []
    
    mm1_d = []
    mm1_th = []
    mm1_x = []
    mm1_y = []

    mm2_d = []
    mm2_th = []
    mm2_x = []
    mm2_y = []

    mm3_d = []
    mm3_th = []
    mm3_x = []
    mm3_y = []
    

    for odom in robotdata.odometry[1:]:
        dt = t - odom[0]
        t = odom[0] # time
        v = odom[1] # velocity
        w = odom[2] # omega
        state = [robotdata.getXTruth(t), robotdata.getYTruth(t), robotdata.getCompass(t)]

        if (1): # (0.06<v<0.08 and -0.1<w<0.1):
            control = [t,v,w]
            thetaMeas = robotdata.getCompass(t)
            nextstate_propagated = propagateMotion(state, control, robotdata.getCompass(t+dt), dt)
            nextstate_random = propagateRandom(state, control, dt)
            nextstate_propagatedNOSLIP = propagateMotionNOSLIP(state, control, robotdata.getCompass(t+dt), dt)
            nextstate_truth = [robotdata.getXTruth(t+dt), robotdata.getYTruth(t+dt), robotdata.getCompass(t+dt)]
            
            mm0_d.append(deltaD(state, nextstate_truth))
            mm0_th.append(deltaTH(state, nextstate_truth))
            mm0_x.append(nextstate_truth[0]-state[0])
            mm0_y.append(nextstate_truth[1]-state[1])

            mm1_d.append(deltaD(state, nextstate_propagated))
            mm1_th.append(deltaTH(state, nextstate_propagated))
            mm1_x.append(nextstate_propagated[0]-state[0])
            mm1_y.append(nextstate_propagated[1]-state[1])

            mm2_d.append(deltaD(state, nextstate_random))
            mm2_th.append(deltaTH(state, nextstate_random))
            mm2_x.append(nextstate_random[0]-state[0])
            mm2_y.append(nextstate_random[1]-state[1])

            mm3_d.append(deltaD(state, nextstate_propagatedNOSLIP))
            mm3_th.append(deltaTH(state, nextstate_propagatedNOSLIP))
            mm3_x.append(nextstate_propagatedNOSLIP[0]-state[0])
            mm3_y.append(nextstate_propagatedNOSLIP[1]-state[1])

    


    plt.hist(velocities)
    plt.hist(omegas)
    
    # print(max(mm0_d))

    # mm0_d_plotting = clean(mm0_d, -0.002, 0.005)
    # mm1_d_plotting = clean(mm1_d, -0.002, 0.005)
    # # mm2_d_plotting = clean(mm2_d, -0.002, 0.005)
    # mm3_d_plotting = clean(mm3_d, -0.002, 0.005)
    # bins = np.linspace(-0.002, 0.005, 30)

    # mm0_x_plotting = clean(mm0_x, -0.005, 0.005)
    # mm1_x_plotting = clean(mm1_x, -0.005, 0.005)
    # # mm2_x_plotting = clean(mm2_x, -0.005, 0.005)
    # mm3_x_plotting = clean(mm3_x, -0.005, 0.005)

    # mm0_y_plotting = clean(mm0_y, -0.005, 0.005)
    # mm1_y_plotting = clean(mm1_y, -0.005, 0.005)
    # # mm2_y_plotting = clean(mm2_y, -0.005, 0.005)
    # mm3_y_plotting = clean(mm3_y, -0.005, 0.005)

    # mm0_th_plotting = clean(mm0_th, -0.05, 0.05)
    # mm1_th_plotting = clean(mm1_th, -0.05, 0.05)
    # # mm2_th_plotting = clean(mm2_th, -0.05, 0.05)
    # mm3_th_plotting = clean(mm3_th, -0.05, 0.05)
    # bins_th = np.linspace(-0.05, 0.05, 25)

    # bins_x = np.linspace(-0.005, 0.005, 25)
    
    # plt.hist(mm0_d_plotting, bins, label='Truth', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm1_d_plotting, bins, label='Motion Model', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm3_d_plotting, bins, label='Motion Model - NO SLIP', histtype='step', stacked=True, fill=False, linewidth=2)
    # # plt.hist(mm2_d_plotting, bins, label='Random Model', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.xlabel('Meters')
    # plt.legend()
    # plt.figure()

    # plt.hist(mm0_x_plotting, bins_x, label='Truth', histtype='step', stacked=True, fill=False, linewidth=3)
    # # plt.hist(mm2_x_plotting, bins_x, label='Random', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm3_x_plotting, bins_x, label='Motion - No Slip', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm1_x_plotting, bins_x, label='Motion', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.title('Change in X')
    # plt.xlabel('Meters')
    # plt.legend()
    # plt.figure()

    # plt.hist(mm0_y_plotting, bins_x, label='Truth', histtype='step', stacked=True, fill=False, linewidth=2)
    # # plt.hist(mm2_y_plotting, bins_x, label='Random', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm3_y_plotting, bins_x, label='Motion - No Slip', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm1_y_plotting, bins_x, label='Motion', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.title('Change in Y')
    # plt.xlabel('Meters')
    # plt.legend()
    # plt.figure()

    # plt.hist(mm0_th_plotting, bins_th, label='Truth', histtype='step', stacked=True, fill=False, linewidth=2)
    # # plt.hist(mm2_th_plotting, bins_th, label='Random', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm3_th_plotting, bins_th, label='Motion - No Slip', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.hist(mm1_th_plotting, bins_th, label='Motion', histtype='step', stacked=True, fill=False, linewidth=2)
    # plt.title('Change in Theta')
    # plt.xlabel('Radians')

    # plt.legend()
    plt.show()


    # robotdata.getCompass(t)