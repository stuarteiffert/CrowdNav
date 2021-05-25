#!/usr/bin/env python3
"""
Uses a Kalman Filter to model the motion of each observed agent
Can be used for constant velocity prediction if no observations supplied
"""

import numpy as np
from pykalman import KalmanFilter

class KFMulti():
    """
    Tracks multiple objects (without association) using pykalman
    
    """
    def __init__(self, measurements):
        #shared KF (same dynamics used by all objects, we update tr and obs)
        self.num_tracks = len(measurements)
        self.observation_matrix = [[1, 0, 0, 0],
                                   [0, 0, 1, 0]]
        self.transition_matrix = [[1, 1, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 1],
                                  [0, 0, 0, 1]]
        self.xs = []   #state estimates
        self.Ps = []  #cov matrices
        self.kf = KalmanFilter(transition_matrices = self.transition_matrix,observation_matrices = self.observation_matrix)
        self.createKFs(measurements)
        
    def createKFs(self,meas_all):
        #create a KF object for each tracked object, sharing tr and obs, but using individual states
        for meas in meas_all:
            init_state =[meas[0][0],0,meas[0][1],0]  #in form [x,xv,y,yv]
            kf = KalmanFilter(transition_matrices = self.transition_matrix,
                  observation_matrices = self.observation_matrix,
                  initial_state_mean = init_state)
            kf = kf.em(np.array(meas), n_iter=5)
            x, P = kf.smooth(np.array(meas))
            self.xs.append(x[-1, :]) #save the last state estimate
            self.Ps.append(P[-1, :]) #save last cov matrix estimate
            
    def update(self,obs_all=None):
        #update each state and cov estimate using latest obs (or None for Constant Vel)
        #if obs_all is set, length must be same as objects we have init on
        #return just the [x,y] for each tracked object (ie no vel estimate)
        if obs_all is not None:
            if len(obs_all) != self.num_tracks:
                ##Incorrect length of observations
                return None            
        
        for i in range(self.num_tracks):
            obs = None
            if obs_all is not None:
                obs = obs_all[i]
            (x_new, P_new) = self.kf.filter_update(filtered_state_mean = self.xs[i],
                                               filtered_state_covariance = self.Ps[i],
                                               observation = obs)
            self.xs[i] = x_new
            self.Ps[i] = P_new
            
        new_pred = []
        for obj in self.xs:
            new_pred.append([obj[0],obj[2]])
        return new_pred

def create_kf_input(history_len,num_agents,X,R):
    #initialise a KF for each agent based on last 5 observations, which we use to predict future path for CV approach
    history_len = 4 #don't need entire observation history for a constant vel model
    agent_pos = [[] for i in range(num_agents)]
    for timestep in X[-(history_len-1):]:
        for i, agent in enumerate(timestep):
            #ensure no missed observations
            #if agent[0] != 0 and agent[1] != 0:
            agent_pos[i].append(agent[0:2])
    for i, agent in enumerate(R[0]):
        #if agent[0] != 0 and agent[1] != 0:
        agent_pos[i].append(agent[0:2])
    return agent_pos
