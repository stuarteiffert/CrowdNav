#!/usr/bin/env python
"""

Implements a MPC approach to dynamic path planning using a Generative RNN and MCTS
 

Last edits:
19/07/19 - Stuart Eiffert - created
02/08/19 - Stuart Eiffert - updated for use within simulated testing environmnet
"""

from __future__ import print_function
import sys
import numpy as np
import time
import threading
import copy
from crowd_nav.policy.predictive_planner.generative_model import ResponseRNN
import crowd_nav.policy.predictive_planner.mcts_par as mcts_par
from crowd_nav.policy.predictive_planner.kalman_model import KFMulti, create_kf_input

class Planner(object):

    def __init__(self, model_path, lookahead, stdev=None, use_cv=False, use_sef2=False):

        # Flags
        self.det_updated = False
        self.obs_current = None
        self.num_agents = None
        self.robot_actions =None
        self.obs_history = []
        self.robot_history = []
        self.lookahead=lookahead
        self.model_path = model_path
        self.using_cv = use_cv #constant velocity model
        self.limit_disturbance=use_sef2

        ###############################
        #Build and Load Predictive Model
        if not self.using_cv:
            print('Loading model...')
            self.model = ResponseRNN(self.model_path, stdev=stdev)
            print('done')
            self.NUM_PARS = 40 #how many parallel streams to run in predictive model
        else:
            self.model = None # don't init KFMulti() until we have observations
            self.NUM_PARS = 1 #no parallel in cv
        self.cv_preds = None
        self.OBSTACLES=np.zeros(shape=(80,80)) #discretised 1m grid around position (0,0) = OBSTACLES[10,10]
   
        ################################
        #Planner Variables:
        self.NUM_DIMS = 2
        self.OBS_WINDOW = 12 #observation history per plan (needs to be same as self.seq_length_in in model)
        self.GOAL = [0,0,1] #placeholder
        self.HORIZON = 20 # how many steps into the future we search
        self.TIME_THRESH = 0.3

        #Define Action Space:  (done in mcts_par currently)
        #self.YAW_RATES = [-20,-5,0,5,20]   
        #self.ACCS = [-0.15,-0.05,0, 0.05,0.15]
        #if self.NUM_PARS > (len(self.ACCS)*len(self.YAW_RATES))**2:
        #    print('Error! Not currently capable of using more than num_actions^2 parallel ')

        #Define FPS of observations / prediction
        self.FPS=4 #what Hz of observations/predictions to use (ORCA only)


    def planPath(self,X, R, offset):
        if not self.using_cv:
            enc_state = self.model.encode_observed(copy.deepcopy(X))
        else:
            enc_state = None
            kf_history = 4 ##how long into past KF considers
            agent_pos = create_kf_input(kf_history,self.num_agents,X,R) 
            tracks = KFMulti(agent_pos) 
            self.cv_preds = [tracks.update() for i in range(self.HORIZON*2)]
        root_node = mcts_par.createRootNode(copy.deepcopy(X),
                                            copy.deepcopy(R),
                                            enc_state,
                                            self.HORIZON,
                                            self.num_agents,
                                            use_cv=self.using_cv)

        if self.limit_disturbance:
            self.GOAL[2] = 2 #sets cost function to SEF2
        else:
            self.GOAL[2] = 1 #sets cost function to SEF1
        planner_goal = [self.GOAL[0] - offset[0], self.GOAL[1] - offset[1], self.GOAL[2]]
        #t1 = time.time()
        updated_root = mcts_par.searchUCT_par(root_node, 
                                              self.num_agents, 
                                              self.NUM_PARS,
                                              self.TIME_THRESH,
                                              planner_goal,
                                              self.model,
                                              cv_preds=self.cv_preds,
                                              obs_map=self.OBSTACLES,
                                              return_tree=False)
        #t2 = time.time()
        #print('ACT', t2-t1)
        path = []
        actions = []
        moves = []
        next_best_node = mcts_par.bestChildPar(updated_root, True)

        #print('next_best_node', next_best_node.state.robo_pos)
        #print('updated_root', updated_root.state.robo_pos, offset)
        last_pos = updated_root.state.robo_pos
        use_holonomic_yaws = False #for swagbot testing
        use_crowdnav_actionspace = True
        for iter in range(self.HORIZON):
            # add node position and actions to lists
            # convert path back to world frame (add offset)
            path.append([next_best_node.state.robo_pos[0] + offset[0], next_best_node.state.robo_pos[1] + offset[1]])
            #actions.append([next_best_node.state.ACCS[next_best_node.state.past_moves[-1][1]],next_best_node.state.YAW_RATES[next_best_node.state.past_moves[-1][0]]])
            actions.append(next_best_node.state.past_moves[-1])
            #Below checks done due to changes made when changes to swagbot
            if use_crowdnav_actionspace:
                new_pos = next_best_node.state.robo_pos
                new_move = [new_pos[0] - last_pos[0], new_pos[1] - last_pos[1]]
                moves.append(new_move)
                last_pos = new_pos

            else:
                if use_holonomic_yaws:
                    speed = next_best_node.state.VELS[next_best_node.state.past_moves[-1][1]]
                    yaw = next_best_node.state.YAWS[next_best_node.state.past_moves[-1][0]]
                    dX = -speed*np.sin(np.deg2rad(yaw)) #negative(anti?) clockwise from north
                    dY = speed*np.cos(np.deg2rad(yaw))
                    moves.append([dX,dY])
                else:
                    moves.append([next_best_node.state.GRID_MOVES[next_best_node.state.past_moves[-1][0]],next_best_node.state.GRID_MOVES[next_best_node.state.past_moves[-1][1]]])
            next_step = mcts_par.bestChildPar(next_best_node, True)
            next_best_node = next_step
            if next_best_node == None:
                break
        #print('movementmct2', moves )
        #print('actions', actions)
        #print('path', path)
        return path, moves


    def convertInput(self, input_obs, num_pars):
        """
        expect input_obs to be a list (len (num_agents+1)) of lists, of len (dT+1),  where dT is the observation window (ie dT = encoder_input_size +1)
        robot list (first element in outer list) can leave off last element, as unused
        first dimension is robot positions, at T+1
        all other dims are agent positions, at T 
        all positions are in world coords
        First element of robot is used as offset for all positions
        First element of other agents is not used.
        Last element of robot is not used (should be zeros, as we dont yet know our future position)
        #Assumes robot input has already been offset by self.lookahead steps
        """
        len_enc = len(input_obs[0])-2 #length X. first element dropped, last element becomes R.
        len_pred = 1 #length R. for use in single step inference (changed in training)

        init_offset = input_obs[0][0] #robot position at t=-dT (this observation is not used for other agents, only required for offset relative to robo)
        #print("robo last pos = {}".format(input_obs[0][-2]))
        #num_pars = 1
        #create placeholders for parallelisation. We only use first block (length num_agents) initially.
        X = [ [ [0,0,0,0] for y in range(self.num_agents*num_pars) ] for x in range(len_enc)]  #encoder inputs
        R = [ [ [0,0,0,0] for y in range(self.num_agents*num_pars) ] for x in range(len_pred)] #decoder inputs (for use in mcts)

        #populate model inputs:        
        for t in range(len_enc):
            # X -> [x_t, y_t, rx_t, ry_t] for all t<current
            # we dont add last element of input_obs, this becomes R instead.
            for i in range(self.num_agents):
                X[t][i][0] = input_obs[i+1][t+1][0] - init_offset[0] # agent position
                X[t][i][1] = input_obs[i+1][t+1][1] - init_offset[1]        
                X[t][i][2] = input_obs[0][t+1][0] - init_offset[0]#robot position appended to each agent position
                X[t][i][3] = input_obs[0][t+1][1] - init_offset[1]

        for t in range(len_pred):
            # R -> [x_t, y_t, 0, 0] for each agent, then 0,0,0,0 as padding for parallelisation (contains current observed positions of each agent)
            #Note: not currently set up for t>1
            for i in range(self.num_agents):
                R[t][i][0] = input_obs[i+1][-1][0] - init_offset[0] #first input to decoder is last observed position
                R[t][i][1] = input_obs[i+1][-1][1] - init_offset[1]   

        return X, R, init_offset

    def reset(self):
        self.num_agents = None
        self.robot_actions =None
        self.obs_history = []
        self.robot_history = []


    def getAction(self, robo_pos, goal, obs):
        """
        expects observations of type provided by CrowdNav: input_obs = [ob * number_observations]
         where each ob contains a list of nearby agents
         expects complete observability (ie no missed detections between frames) as not implemented with object id

        print('num-agents',len(obs))
        first element of robot_history is only used as offset
        we leave off last element as it is unused anyway in convertInput (it is fed with zeros in ROS version)
        first element of obs_history unused
        last element is used without a known robot position, as input to decoder        
        During initialisation, we don't know history, so we extrapolate using each agent pref velocity and goal direction of each agent and robot
        """
        swagbot_goal_thresh = 1.5 #if within 1.5m return goal as path
        vec2goal = [goal[0] - robo_pos[0], goal[1] - robo_pos[1]]
        dist2goal = np.linalg.norm(vec2goal)
        if dist2goal < swagbot_goal_thresh:
            return vec2goal


        if len(self.obs_history) == 0:
            #Lookahead must be 1! (see paper, cite tbd)
            self.GOAL = [goal[0], goal[1], 1] #last element decides cost function to use
            assert self.lookahead ==1
            self.num_agents = len(obs)
            #we fill with expected last positions based on known velocity of each agent. For robot, we assume it has been still.
            extrap_hist_robot = [ [robo_pos[0],robo_pos[1]] for i in range(self.OBS_WINDOW+1)  ]
            self.robot_history = extrap_hist_robot

            extrap_hist_agents = [ [ [0,0] for step in range(self.OBS_WINDOW+2)] for i in range(self.num_agents) ]
            for i, agent in enumerate(obs):
                agent_pos = [agent.px,agent.py]
                extrap_hist_agents[i][-1]=agent_pos
                speed_pref = agent.v_pref / self.FPS*0.25 #in m/timestep, ORCA agents start slower it seems
                v_goal = np.array([agent.gx, agent.gy])-np.array(agent_pos) 
                v_history = - speed_pref * v_goal /  np.linalg.norm(v_goal) #vector in direction away from goal, at agent preferred speed
                for step in range(self.OBS_WINDOW+1):
                    pos = agent_pos+v_history*(step+1)
                    extrap_hist_agents[i][-(step+2)] = [pos[0],pos[1]]
            self.obs_history = extrap_hist_agents

        else:
            for i, agent in enumerate(obs):
                self.obs_history[i].append([agent.px,agent.py])
                self.obs_history[i].pop(0)
            self.robot_history.append([robo_pos[0],robo_pos[1]])
            self.robot_history.pop(0)


        full_obs = copy.deepcopy(self.obs_history)
        full_obs.insert(0,copy.deepcopy(self.robot_history))
        full_obs[0].append([0,0]) #placeholder for robot action at time=t+1

        X, R, offset = self.convertInput(full_obs, self.NUM_PARS)

        try:
            next_path, next_actions = self.planPath(X,R,offset)
            #print('sending', next_actions[0])
            return next_actions[0]
        except:
            print('No action returned. Error in planner:')
            return (0,0)
    def cleanup(self):
        self.det_updated = False
        if not self.using_cv:
            self.model.sess.close()
        #delete all created objects here

