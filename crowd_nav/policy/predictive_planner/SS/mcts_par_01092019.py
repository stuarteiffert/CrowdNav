#!/usr/bin/env python
"""
MCTS planner
Uses either a predictive model, or a set of future predictions (from constant velocity models)
Predicts the optimal path for a robot, given the goal of either:
    1. herding all agernts in a direction
    2. reaching a waypoint

#################
Last edits:
19/07/19 - Stuart Eiffert - created
09/08/19 - Stuart Eiffert - edited for use with constant velocity predictions



##############
Usage:
1. create root node 
    -   based on observations (X), 
    -   action placeholder (R) containing last obs at R[0], 
    -   encoded state from model (None if using cv). 
    -   Lookahead is limit to future timesteps to search.
    -   Number of agents must be known prior (from observations)
2. Find best path using UCT method. 
    -   If using cv, set num_pars (number parallel MCTS searches to 1).
    -   GOAL takes form [x, y, type:(0=herding, 1=waypoint)], eg planner_goal = [10,10, 1]

3. Output Best found path from final tree

###############
#EXAMPLE:

import mcts_par

lookahead = 25
use_cv =False
preds=None
planner_goal = [10,10, 1] #[x, y, type (0=herding, 1=waypoint)]
time_thresh = 1
num_pars=parallel_predictions

if use_cv:
    preds = [tracks.update() for i in range(lookahead)]
    model = None
    enc_state = None
else:
    MODEL_PATH = "/home/stuart/acfr/code/sequence_prediction/models/"+dataset_type+"/LSTM_2_16_timestep3_prob"
    model = ResponseRNN(MODEL_PATH)
    enc_state = model.encode_observed(copy.deepcopy(X))

root_node = mcts_par.createRootNode(copy.deepcopy(X),
                                    copy.deepcopy(R),
                                    enc_state,
                                    lookahead,
                                    num_agents,
                                    use_cv=use_cv)

updated_root, TREE = mcts_par.searchUCT_par(root_node, 
                                          num_agents, 
                                          num_pars,
                                          time_thresh,
                                          planner_goal,
                                          model,
                                          cv_preds=preds,
                                          obs_map=OBSTACLES,
                                          return_tree=True)

final_node = mcts_par.bestChildPar(updated_root, True)
#Best found path:
print('\nBest Path:')
print(final_node.state.ACCS[final_node.state.past_moves[0][1]], 
      final_node.state.YAW_RATES[final_node.state.past_moves[0][0]], final_node.state.robo_pos)
for iter in range(lookahead):
    next_step = mcts_par.bestChildPar(final_node, True)
    final_node = next_step
    if final_node == None:
        break
    print(final_node.state.ACCS[final_node.state.past_moves[-1][1]], 
          final_node.state.YAW_RATES[final_node.state.past_moves[-1][0]], final_node.state.robo_pos)


"""


import random
import math
import hashlib
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Ellipse

#global placeholder for parallel simulation
enc_state_template = None
TREE = []
VIS = False
OBSTACLES=np.zeros(shape=(100,100))

class State():
    #Summary of the enviro state for a node, including position of agents, associated rewards, and previsou actions
    #YAW_RATES = [-5,0,5]
    #ACCS = [-0.05,0,0.05]

    YAW_RATES = [-20,-5,0,5,20]
    ACCS = [-0.2,0,0.2]

    #stdev=4.9 HANDLED IN PRED MODEL!
    #ACCS = [-0.040, 0, 0.040]
    MAX_SPEED = 1.0*0.25 #4 fps

    def __init__(self, robo_vel, robo_pos, steps_left, moves=[], agent_pos=None, enc_state=None, past_pos=[]):
        self.robo_vel = robo_vel #[last_speed,  last_yaw] of robot, for determining next position from action
        self.robo_pos = robo_pos #[x,y] position of node (ie the robot's 'action' in the predictive model)
        self.enc_state = enc_state # encoded state after prediction using robo_vel (known for root prior)
        self.agent_pos = agent_pos #positions of uncontrolled agents, for use in eval function (known for root prior)
        self.agent_history = past_pos #positions of uncontrolled agents, for use in eval function (known for root prior)
        #self.agent_accs = agent_pos #positions of uncontrolled agents, for use in eval function (known for root prior)
        self.uncertainty = []
        self.reward=0
        self.in_obstacle = False
        self.steps_left=steps_left
        self.past_moves=moves
        self.holonomic = False #use a grid based action space, rather than accel and yaw
        self.grid_actions = False

        #if robo_vel[0] >= self.MAX_SPEED:
        #    print('fast', robo_vel, self.past_moves)
        #    #limit action space
        #    self.ACCS = [0]      
        if self.holonomic:
            if self.grid_actions:
                self.GRID_MOVES = [-self.MAX_SPEED, -self.MAX_SPEED*0.5, 0, self.MAX_SPEED*0.5, self.MAX_SPEED]
                self.ACTION_SPACE = [[dx,dy] for dx in range(len(self.GRID_MOVES)) for dy in range(len(self.GRID_MOVES))]

            else:
                #use yaw and velocities
                self.YAWS = np.linspace(0,360,16,endpoint=False).tolist() #discretise into 16 yaw actions
                self.VELS = [self.MAX_SPEED*0.2,self.MAX_SPEED*0.5,self.MAX_SPEED]
                self.ACTION_SPACE = [[y,v] for y in range(len(self.YAWS)) for v in range(len(self.VELS))] #only need one action with zero velocity
        else:
            self.ACTION_SPACE = [[y,a] for y in range(len(self.YAW_RATES)) for a in range(len(self.ACCS))] 
        self.num_moves=len(self.ACTION_SPACE)

    def update_reward(self, GOAL):
        #tbd: define evaluation function
        #     function should be normalised about 0
        #   
        #tester eval function used below gives a higher reward for herding agents up
        #[0] index in loop required as only a single step of decoder is used (can output list of steps)
        #
        #Note: Calculating squared distance between two points thousands of times is pretty slow. TBD
        r = 0
        num_agents = len(self.agent_pos)
        rad_hum = 0.3
        rad_robot = 0.3
        method =1        
        #method = GOAL[2]
        # reward methods: [herding, obstacle_avoid, ???] maybe include 'check agent's not 'impacted' '?
        obstacle_thresh = 1.5 #avoidance threshold (metres)

        if method ==0:
            #herding
            # reward based on movement of agents in the 'goal vector' direction
            x_goal = GOAL[0]
            y_goal = GOAL[1]
            for i, agent in enumerate(self.agent_pos):
                x_pos = agent[0]            
                y_pos = agent[1]
                if self.uncertainty is not None:
                    r += (x_goal*x_pos + y_goal*y_pos) / (1+self.uncertainty[i]) #reward is dependent on uncertainty of agent prediction
                else:
                    r += (x_goal*x_pos + y_goal*y_pos) 
                # avoid hitting any agents
                sqr_dist_agent = (self.robo_pos[0] - x_pos)**2 + (self.robo_pos[1] - y_pos)**2
                #update thresh to be proportional to uncertainty
                #update repulsion to not be step function
                if sqr_dist_agent**0.5 < obstacle_thresh:
                    r -= 0.5*2*(1 / sqr_dist_agent**0.5 - 1 / obstacle_thresh) # 
                    #r -= obstacle_thresh*10 # stepwise function
                #r += (x_goal*x_pos + y_goal*y_pos) / (1+self.uncertainty[i]) #reward is dependent on uncertainty of agent prediction
            r /= num_agents
            
        elif method ==1:
            #obstacle-avoidance
            # reward based on distance to goal and 'not hitting' anything (use map size 10x10)
            # works similar to a potential field approach, updated each state, using just distance from robo pos to goal and obstacles
            #reward based on squared euclidean dist to each obstacle and goal,  
            # approach based on http://portal.ku.edu.tr/~cbasdogan/Courses/Robotics/projects/algorithm_poten_field.pdf 
            #x_goal = -15
            #y_goal = -5
            x_goal = GOAL[0]
            y_goal = GOAL[1]
            sqr_dist_goal = (self.robo_pos[0] - x_goal)**2 + (self.robo_pos[1] - y_goal)**2
            r = -(sqr_dist_goal)**0.5 #quadratic possibly better,  ie no sqrt (too slow though?)
            r = -(sqr_dist_goal)
            
            for i, agent in enumerate(self.agent_pos):
                # for each agent, we add a negative reward based on distance 
                x_pos = agent[0]            
                y_pos = agent[1]
                sq_dist_agent = (self.robo_pos[0] - x_pos)**2 + (self.robo_pos[1] - y_pos)**2
                #update thresh to be proportional to uncertainty
                #update repulsion to not be step function
                dist_agent = sq_dist_agent**0.5
                sig = 1 #self.uncertainty[i] #size of obstacle 'potential field' is proportional to uncertainty
                uncertainty_thresh = obstacle_thresh * sig
                if dist_agent < uncertainty_thresh:

                    #version 1:
                    # obstacle_thresh = 1.5
                    # #c = 10*math.sqrt(2)
                    #r -=  (uncertainty_thresh - dist_agent ) * sig * 10  #linear potential field increase

                    #version2:
                    # obstacle_thresh = 1.5
                    # #c = 10*math.sqrt(2)
                    #r -=  sig * 100 / dist_agent  #cost shown in paper
                    #??r -= uncertainty_thresh*10 # stepwise function

                    #version3:
                    # obstacle_thresh = 1.5
                    # #c = 10*math.sqrt(2)
                    #r -= 100*(1 / dist_agent - 1 / obstacle_thresh) # 

                    #version4:
                    # obstacle_thresh = 1.5
                    # #c = 10*math.sqrt(2)
                    #r -= 10*(1 / dist_agent - 1 / obstacle_thresh) # 

                    #version5:
                    # obstacle_thresh = 1.5
                    # #c = 2*math.sqrt(2)
                    r -= 10*(1 / dist_agent - 1 / obstacle_thresh) # AND change c constant to 2*math.sqrt(2)

                    #version6                                                              :
                    # obstacle_thresh = 1.5 + rad_hum + rad_robot
                    # #c = 10*math.sqrt(2)
                    # r= quadratic + linear 
                    r -=  (obstacle_thresh - dist_agent ) * 100 #linear potential field increase

                    #V1 collides too much: success rate: 0.84, collision rate: 0.14, nav time: 17.18, total reward: 0.1230, avg comp time: 0.5345, avg path len: 15.97
                    #V2 timeouts too much: success rate: 0.68, collision rate: 0.00, nav time: 20.26, total reward: 0.0839, avg comp time: 0.5253, avg path len: 21.30                   
                    #.: either need to lower the 100 in V2, or raise the 10 in
                    #V3 TEST  has success rate: 0.78, collision rate: 0.00, nav time: 20.16, total reward: 0.0977, avg comp time: 0.5266, avg path len: 20.81
                    #V4 TEST  has success rate: 0.77, collision rate: 0.22, nav time: 17.88, total reward: 0.0944, avg comp time: 0.5177, avg path len: 16.15
                    #V5 TEST  has success rate: 0.72, collision rate: 0.26, nav time: 17.88, total reward: 0.0811, avg comp time: 0.5108, avg path len: 15.83

                    #Extra Notes:   - Speed of MCTS-CV grows with more agents. whoops!
                    #               - PF is much better than reported. Unsure why. Maybe really unlucky sampling before?
                


        elif method ==2:
            #limit disturbance
            x_goal = GOAL[0]
            y_goal = GOAL[1]
            sqr_dist_goal = (self.robo_pos[0] - x_goal)**2 + (self.robo_pos[1] - y_goal)**2
            #r = -(sqr_dist_goal)**0.5 #quadratic possibly better,  ie no sqrt (too slow though?)
            r = -(sqr_dist_goal)            
            for i, agent in enumerate(self.agent_pos):
                # for each agent, big negative if within collision distance, and small negative if impacting acceleration
                x_pos = agent[0]            
                y_pos = agent[1]
                sqr_dist_agent = (self.robo_pos[0] - x_pos)**2 + (self.robo_pos[1] - y_pos)**2
                dist_agent = sqr_dist_agent**0.5
                if dist_agent < obstacle_thresh:
                    #r -= 0.5*2*(1 / sqr_dist_agent**0.5 - 1 / obstacle_thresh) # 
                    r -=  (obstacle_thresh - dist_agent ) * 100 #linear potential field increase
                #determine agent's change in velocity between last node and this:
                if dist_agent < obstacle_thresh*2:
                    v_past =  [self.agent_history[-1][i][0] - self.agent_history[-2][i][0], self.agent_history[-1][i][1] - self.agent_history[-2][i][1]]
                    v_now =  [self.agent_pos[i][0] - self.agent_history[-1][i][0], self.agent_pos[i][1] - self.agent_history[-1][i][1]]
                    #print('VELS', v_past, v_now)
                    acc = np.linalg.norm([v_past[0] - v_now[0], v_past[1] - v_now[1]])
                    #print('ACC', acc)
                    if acc > 0.2:
                        #print('ACC',acc, r, acc**2 * 100)
                        #print('ACC',acc)
                        r -=  acc * 1000 #linear potential field increase
           
        self.reward = r        

        return r
    
    def update_pos(self, new_speed, new_yaw):
        #updates x,y of position based on new linear acc and change in yaw, for child node state
        old_x = self.robo_pos[0]
        old_y = self.robo_pos[1]
        dX = -new_speed*np.sin(np.deg2rad(new_yaw)) #negative(anti?) clockwise from north
        dY = new_speed*np.cos(np.deg2rad(new_yaw))
        new_pos = [old_x + dX, old_y + dY]
        return new_pos

    def next_state(self):
        #UNUSED?
        #randomly choose a move from action space
        nextmove=random.choice([x for x  in self.ACTION_SPACE])
        #return the predicted next state of the environment based on the planned moved.
        predicted_env = self.predict(nextmove)
        next_state=State(predicted_env, self.steps_left-1, self.past_moves+[nextmove])
        return next_state
    
    def new_child(self, action):
        if self.holonomic:
            if self.grid_actions:
                new_pos = [self.robo_pos[0]+self.GRID_MOVES[action[0]], self.robo_pos[1]+self.GRID_MOVES[action[1]]] #update to reflect how we actually move in the gym environment...
                new_vel = [ np.linalg.norm(action), 0] #we dont use yaw in grid holonomic
            else:
                new_yaw = self.YAWS[action[0]]
                new_speed = self.VELS[action[1]]
                dX = -new_speed*np.sin(np.deg2rad(new_yaw)) #negative(anti?) clockwise from north
                dY = new_speed*np.cos(np.deg2rad(new_yaw))
                new_pos = [self.robo_pos[0]+dX, self.robo_pos[1]+dY] #update to reflect how we actually move in the gym environment...
                new_vel = [new_speed, new_yaw] #velocity of child node
        else:
            new_speed = self.robo_vel[0] + self.ACCS[action[1]] #whoops, put yaw and acc in wrong order TODO
            if new_speed > self.MAX_SPEED:
                print('SPEED', new_speed)
                new_speed = self.MAX_SPEED 
            elif new_speed < -self.MAX_SPEED:
                print('SPEED', new_speed)
                new_speed = -self.MAX_SPEED

            new_yaw = self.robo_vel[1] + self.YAW_RATES[action[0]]
            if new_yaw > 180:
                new_yaw -= 360
            elif new_yaw < -180:
                new_yaw += 360
            new_vel = [new_speed, new_yaw] #velocity of child node
            new_pos = self.update_pos(new_speed, new_yaw) #position of child node
        #create child State. We don't yet know the predicted agent positions or encoding
        next_state=State(new_vel, new_pos, self.steps_left-1, self.past_moves+[action], past_pos =self.agent_history+[self.agent_pos] ) 
        return next_state
    
    def predict(self,action):
        #UNUSED?
        pred_state = copy.deepcopy(self.env_state)  #whatta ya mean I have to deepcopy? jeez
        return pred_state


    def terminal(self):
        if self.in_obstacle:
            return True
        if self.steps_left == 0:
            return True
        return False

    def __hash__(self):
        return int(hashlib.md5(str(self.past_moves).encode('utf-8')).hexdigest(),16)
    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False
    def __repr__(self):
        s="Value: %d; Moves: %s"%(self.reward,self.past_moves)
        return s

class Node():
    def __init__(self, state, parent=None):
        self.visits=0
        self.value=0.0    
        self.state=state
        self.children=[]
        self.untried_actions=state.ACTION_SPACE #specified here for speed in node Expansion
        self.parent=parent    
    def add_child(self,child_state):
        child=Node(child_state,self)
        self.children.append(child)
        self.untried_actions.remove(child_state.past_moves[-1])
    def update(self,reward):
        self.value+=reward
        self.visits+=1
    def fully_expanded(self):
        if len(self.children)==self.state.num_moves:
            return True
        return False
    def __repr__(self):
        s="Node; children: %d; visits: %d; value: %f"%(len(self.children),self.visits,self.value)
        return s


class NodePar():
    #allows parallelisation of treePolicy
    def __init__(self, state, parent=None):
        self.visits=0
        self.value=0.0 
        self.virtual_value=0.0
        self.state=state
        self.children=[]
        self.virtual_children=[]
        self.untried_actions=state.ACTION_SPACE #specified here for speed in node Expansion
        self.parent=parent
        global VIS
        if VIS:
            global TREE
            TREE.append(copy.copy(self)) #for visulaisation
    def add_child(self,child_state):
        child=NodePar(child_state,self)
        self.children.append(child)
        self.untried_actions.remove(child_state.past_moves[-1])
    def add_virtual_child(self,child_state):
        child=NodePar(child_state,self)
        self.virtual_children.append(child)
        self.untried_actions.remove(child_state.past_moves[-1])
    def update(self,reward):
        self.value+=reward
        self.visits+=1
        self.children += self.virtual_children
        self.virtual_children = []
        self.virtual_value=0.0
    def fully_expanded(self):
        if (len(self.children) + len(self.virtual_children)) == self.state.num_moves:
            return True
        return False
    def __repr__(self):
        s="Action: "+str(self.state.past_moves) + " ; children: %d; visits: %d; value: %f"%(len(self.children),
                                                                                            self.visits,
                                                                                            self.value)
        return s

def checkObstacleMap(pos):
        #if pos is in an obstacle, return false 
        global OBSTACLES
        obstacle_map = OBSTACLES #global variable,  bool array
        x_pos = int(pos[0] + obstacle_map.shape[0]*0.5) #offset 0,0 to be centre of grid
        y_pos = int(pos[1] + obstacle_map.shape[1]*0.5)
        if x_pos > obstacle_map.shape[0]-1:
            x_pos = obstacle_map.shape[0] -1
        elif x_pos < 0:
            x_pos = 0
            
        if y_pos > obstacle_map.shape[1]-1:
            y_pos = obstacle_map.shape[1] -1
        elif y_pos < 0:
            y_pos = 0
        return obstacle_map[x_pos,y_pos] # return bool value of grid cell

    
def searchUCT(budget, root, num_agents, goal):
    #budget: number of iterations to run until return reult
    #root: starting 'node' to build the tree from
    for iter in range(int(budget)):
        front = None
        front = treePolicy(root) #select and expand nodes
        reward = simulatePolicy(front, num_agents, goal, predictive_model) #simulate 
        backUp(front,reward)
    return root

def searchUCT_par(root, num_agents, pars, time_thresh, goal, predictive_model, cv_preds=None,obs_map=None,return_tree=False):
    #root: starting 'node' to build the tree from
    #pars: number of parallel predictions to make
    #time_thresh: cut off when we return the root node as is
    #cv_preds: constant velocity predictions for all agent future positions
    #return_tree: for visualisation purposes    

    #do one iteration before loop as we know all new nodes will have root as parent,
    # and we should only explore at most num_actions of nodes
    if obs_map is not None:
        global OBSTACLES
        OBSTACLES = obs_map
    global VIS
    VIS = return_tree
    time_start = time.time()
    fronts = None
    fronts = treePolicyPar(root, pars, is_root=True) #select and expand nodes, returning pars num nodes
    simulatePolicyPar(fronts, num_agents, pars, goal, predictive_model, cv_preds=cv_preds, is_root=True) #simulate and backup 
    
    while (time.time() - time_start) < time_thresh:
        fronts = None
        fronts = treePolicyPar(root, pars) #select and expand nodes, returning pars num nodes
        simulatePolicyPar(fronts, num_agents, pars, goal, predictive_model, cv_preds=cv_preds) #simulate and backup  
        time_now = time.time()

    if VIS:
        global TREE
        return root, TREE
    return root

def treePolicy(node):
    while node.state.terminal() == False:
        if len(node.children)==0: #if node has no children, we expand
            return expand(node)
        elif random.uniform(0,1)<.5: #sometimes randomly SELECT even with unexplored actions
            node=bestChild(node)
        else:
            if node.fully_expanded()==False:    
                return expand(node)
            else:
                node=bestChild(node)
    return node

def treePolicyPar(root_node, num_par, is_root=False):
    #find the N best nodes (N=num_par) to expand
    #node starts as root, should we make a copy of it here?,  no we aren't altering it, just children 
    #TODO: add virtual_loss to stop all parallel streams converging
    nodes = []
    #first 'if is_root' required to not get stuck when action_space is less than num_pars, and so can't expand root_node anymore
    if is_root:
        #first iteration, just expand root as much as we want
        untried_actions = copy.copy(root_node.untried_actions) #should be entire ACTION_SPACE here
        # copied as we mutate it as we iterate through
        for iter, action in enumerate(untried_actions):
            if len(nodes) < num_par:
                new_state=root_node.state.new_child(action) #does not alter root_node.state 
                root_node.add_virtual_child(new_state)
                nodes.append(root_node.virtual_children[-1])
            else:
                #print('More options in Root than Pars')
                return nodes
    else:    
        while len(nodes) < num_par:
            #until we have as many nodes as we want, keep selecting
            node = root_node #once an expanded virtual child added to nodes, we go back to top
            node_added = False
            while node.state.terminal() == False:
                #travel down tree until we expand, or hit terminal node
                if len(node.children)==0: #if node has no children, we expand
                    if node.fully_expanded()==False: #we may have already filled node with virtual_children
                        nodes.append(expandPar(node)) 
                        node_added = True 
                    break  # we break even if no node added, as we are at a dead end      
                # if current node has children, or is full of virtual_children                                                      
                elif random.uniform(0,1)<.5:
                    node=bestChildPar(node) #this should already be a node, ie no problem selecting multiple times till it 'fills up'
                else:
                    if node.fully_expanded()==False: #needs to include both existing children and 'virtual children'
                        nodes.append(expandPar(node))
                        node_added = True
                        break
                    else:
                        node=bestChildPar(node) #this should already be a node, ie no problem selecting multiple times till it 'fills up'
            
            #this would add a node that has already been expanded and backuped to nodes,  not wanted!
            #if not node_added: 
            #    nodes.append(node)
    return nodes

def bestChild(node, explore=True):
    bestscore=-1000
    scalar = 0.1*math.sqrt(2) # c constant in UCB1
    if not explore:
        scalar = 0
    bestchildren=[]
    for c in node.children:
        exploit=c.value/c.visits  #avg value (right term in UCB1)
        explore=math.sqrt(math.log(node.visits)/float(c.visits)) #(left term in UCB1)   
        score=exploit+scalar*explore
        if score==bestscore:
            bestchildren.append(c)
        if score>bestscore:
            bestchildren=[c]
            bestscore=score
    if len(bestchildren)==0:
        print("No best child found, probably fatal!")
    return random.choice(bestchildren)

def bestChildPar(node, ret_final=False):
    #same as non parallel function
    bestscore=-1000
    scalar = 10*math.sqrt(2) # c constant in UCB1
    virtual_loss_multiplier = 1.0 #dependent on size of reward of each state!

    bestchildren=[]
    for c in node.children: #do not consider virtual_children,  as we dont know their value or enc_state
        if not ret_final:
            exploit=(c.value + c.virtual_value*virtual_loss_multiplier)/c.visits  #avg value (right term in UCB1)
            explore=math.sqrt(math.log(node.visits)/float(c.visits)) #(left term in UCB1)   
            score=exploit+scalar*explore
        else:
            score = c.visits
        if score==bestscore:
            bestchildren.append(c)
        if score>bestscore:
            bestchildren=[c]
            bestscore=score
    if len(bestchildren)==0:
        if ret_final:
            return None
        print("No best child found, probably fatal!")
    node = random.choice(bestchildren)
    node.virtual_value -= 1
    return node

def simulatePolicy(node, num_agents, goal, predictive_model):
    #performs a single prediction step using the nodes action and determines reward
    enc_state = node.parent.state.enc_state
    R = node.state.robo_pos #position of robot 
    R_all  = [[0,0,R[0],R[1]] for i in range(num_agents)] #prediction is done per agent, so we need an input for each
    output_step, dec_state =  predictive_model.decode_step(R_all, enc_state)
    node.state.enc_state = dec_state
    node.state.agent_pos = output_step
    reward = node.state.update_reward(goal)  
    return reward

def simulatePolicyPar(nodes, num_agents, num_pars, goal, predictive_model, cv_preds=None, is_root=False):
    #performs a single prediction step using the nodes action and determines reward
    # for each observed agent, we get:
    #    output =  [mu_x, mu_y, sigma_x, sigma_y, rho]
    #    enc_state = [[l1h,l1c],[l2h,l2c]]  (for 2 LSTM layers)
    global enc_state_template
    enc_state = enc_state_template # a deepcopy of enc_state to save recreating each time, we just overwrite

    R_all = []
    if is_root:
        #all nodes share same parent, so let's reuse enc_state and agent_pos (last obs) for all
        enc_state_root = nodes[0].parent.state.enc_state 
        agent_pos_root = nodes[0].parent.state.agent_pos 
        for i, node in enumerate(nodes):
            if cv_preds is None:
                enc_idx = i*num_agents #position to start overwriting for parallel node
                enc_state[0][0][0][enc_idx:(enc_idx+num_agents)] = enc_state_root[0][0] #layer 1, c
                enc_state[0][0][1][enc_idx:(enc_idx+num_agents)] = enc_state_root[0][1] #layer 1, h
                enc_state[0][1][0][enc_idx:(enc_idx+num_agents)] = enc_state_root[1][0] #layer 2, c
                enc_state[0][1][1][enc_idx:(enc_idx+num_agents)] = enc_state_root[1][1] #layer 2, h
                
            #update robot actions for input
            R = node.state.robo_pos #position of robot in future node (t=T+1)
            R_all += [[agent_pos_root[j][0],agent_pos_root[j][1],R[0],R[1]] for j in range(num_agents)]  #updated for X_T=[x_T,r_T+1]    
            #R_all should now be of length num_agents*num_nodes
            # TODO: we may need to expand R_all with dummy data if length num_nodes < num_pars!
        if len(nodes) < num_pars: #can be the case when num_pars is larger than available actions from the root node
            for iter in range(num_pars - len(nodes)):
                R_all += [[0,0,0,0] for j in range(num_agents)] #fill actions with dummy data
    else:
        #get encoded state from each nodes parent:

        for i, node in enumerate(nodes):
            if cv_preds is None:
                #overwrite enc_state, for each agent, rnn layer, h and c (for lstm),  and hidden unit
                node_enc = node.parent.state.enc_state
                enc_idx = i*num_agents #position to start overwriting for parallel node
                enc_state[0][0][0][enc_idx:(enc_idx+num_agents)] = node_enc[0][0] #layer 1, c
                enc_state[0][0][1][enc_idx:(enc_idx+num_agents)] = node_enc[0][1] #layer 1, h
                enc_state[0][1][0][enc_idx:(enc_idx+num_agents)] = node_enc[1][0] #layer 2, c
                enc_state[0][1][1][enc_idx:(enc_idx+num_agents)] = node_enc[1][1] #layer 2, h

            #update robot actions for input
            R = node.state.robo_pos #position of robot 
            R_all += [[0,0,R[0],R[1]] for j in range(num_agents)]
    #perform prediction across all agents and parallel nodes
    if cv_preds is None:
        outputs, dec_states =  predictive_model.decode_step(R_all, enc_state)
    #now we set the enc_states for each node.state
    for i, node in enumerate(nodes):
        dec_idx = i*num_agents #position to start overwriting for parallel nod

        if cv_preds is not None:
            output_step = cv_preds[len(node.state.past_moves)]
            uncertainty = None
            dec_state = None
        else:
            #get the nodes encoded state and output state
            l1h = dec_states[0][0][dec_idx:(dec_idx+num_agents)] #layer 1, c
            l1c = dec_states[0][1][dec_idx:(dec_idx+num_agents)] #layer 1, h
            l2h = dec_states[1][0][dec_idx:(dec_idx+num_agents)] #layer 2, c
            l2c = dec_states[1][1][dec_idx:(dec_idx+num_agents)] #layer 2, h
            dec_state = [[l1h,l1c],[l2h,l2c]]
            output_step = outputs[0][dec_idx:(dec_idx+num_agents)] #zero index required for single step use of pred model
        
            #get the nodes state uncertainty
            # unctertainty per agent is the determinant of the covariant matrix of the 2D Gaussian: 
            # where C = [[sigma_x**2, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y**2]]
            #     det(C) = (sigma_x*sigma_y)**2*(1-rho**2)
            # Other option is trace of C:
            #     tr(C) = (sigma_x*sigma_y)**2
            uncertainty = [ ((np.exp(agent[2])*np.exp(agent[3]))**2*(1-np.tanh(agent[4])**2))**0.5 for agent in output_step] # det(C)
            #uncertainty = [ (np.exp(agent[2])*np.exp(agent[3])) for agent in output_step] # trace(C)
        
        #update the nodes state:
        node.state.enc_state = dec_state
        node.state.agent_pos = output_step
        node.state.uncertainty = uncertainty
        #print(np.mean(uncertainty), node.state.steps_left)
        
        #check if action is valid (ie are we in an 'obstacle'?)
        invalid_move = False #we don't yet use obstacles
        #invalid_move = checkObstacleMap(node.state.robo_pos, OBSTACLE_MAP)
        if invalid_move:
            reward = -500
            node.state.in_obstacle=True
            node.state.reward=reward
        else:
            reward = node.state.update_reward(goal)     
        backUp(node, reward)   
    return reward

def backUp(node,reward):
    while node!=None:
        node.update(reward)
        node=node.parent
    return

def expand(node):
    untried_actions = node.untried_actions
    new_action = random.choice(untried_actions)
    new_state=node.state.new_child(new_action)
    node.add_child(new_state)
    return node.children[-1]

def expandPar(node):
    untried_actions = node.untried_actions #accounts for virtual_children
    new_action = random.choice(untried_actions)
    new_state=node.state.new_child(new_action) #does not alter node
    node.add_virtual_child(new_state)
    return node.virtual_children[-1]

def get_yaw(v):
    #anticlockwise from north. 180 is positive
    v1 = np.array(v)
    v0 = np.array([0,1])
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def createRootNode(X_root,R_root, X_encoded, horizon, num_agents, use_cv=False):
    #Define Root Node from observations X and R
    # X holds all past agent positions and current robot action (at time of observation t)
    # R holds current agent positions (at time t) and placeholder for next robot action
    # 1. get last positions of all non-controlled agents
    global enc_state_template 

    last_agent_pos = [[x[0],x[1]] for x in R_root[0][0:num_agents]] #updated for X_T=[x_T,r_T+1]    
    prior_agent_pos = [[x[0],x[1]] for x in X_root[-1][0:num_agents]] #updated for X_T=[x_T,r_T+1]
    #get last speed and yaw of controlled agent (robot) (robot input same for all agents, so use 0th agent as 2nd index)
    last_robot_pos = [X_root[-1][0][2], X_root[-1][0][3]]
    last_robot_vel = [X_root[-1][0][2] - X_root[-2][0][2], X_root[-1][0][3] - X_root[-2][0][3]]
    last_robot_speed = np.linalg.norm(last_robot_vel)
    last_robot_yaw = get_yaw(last_robot_vel)
    
    # create root node to start action search from:
    root_pos = last_robot_pos
    root_speed = last_robot_speed  #last speed of robot
    root_yaw = last_robot_yaw    #last yaw of robot
    root_vel = [root_speed, root_yaw] #last known speed and yaw
    root_agent_pos = last_agent_pos  #last positions of all agents (not robot)
    enc_state_copy = copy.deepcopy(X_encoded)  #last encoded state
    enc_state_template = copy.deepcopy(X_encoded) # for use in each simulation step
    if use_cv:
        #if using constant velocity, we expect X_encoded to also be None
        root_enc_state = None
    else:
        # reformat root_enc_state to be just state = [[l1h,l1c],[l2h,l2c]] for num_agents
        root_l1h = enc_state_copy[0][0][0][0:num_agents] #layer 1, c
        root_l1c = enc_state_copy[0][0][1][0:num_agents] #layer 1, h
        root_l2h = enc_state_copy[0][1][0][0:num_agents] #layer 2, c
        root_l2c = enc_state_copy[0][1][1][0:num_agents] #layer 2, h
        root_enc_state = [[root_l1h, root_l1c],[root_l2h, root_l2c]]
    
    # Create State object for root node 
    root_state = State(root_vel, root_pos, horizon, [], root_agent_pos,  root_enc_state, past_pos=[prior_agent_pos])
    #print('root vel: speed = {:.3f} m/s, yaw = {:.3f} deg'.format(root_vel[0], root_vel[1]))
    #print('root initial pos: x = {:.2f}m, y = {:.2f}m '.format(root_pos[0], root_pos[1]))
    # Create Root Node (using parallel object)
    root = NodePar(root_state)    
    
    return root

def visMCTS(nodes_obs, ids, tree, root, obst=None, isDist=False):
    len_obs = len(nodes_obs)
    plt.figure(num=None, figsize=(12, 12), dpi=100, facecolor='w', edgecolor='k')
    axes = plt.gca()
    axes.set_xlim([-20,12])
    axes.set_ylim([-12,12])
    cols = [[1,0,0],[0,1,0],[0,0,1],[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1],[0,0.5,1],[0.5,0,1],[1,0,0.5],[1,0.5,0],[0,1,0.5],[0.5,1,0]]

    #0. draw obstacle map if available
    if obst is not None:
        #draw obstacle grid array
        width = obst.shape[0]*0.5
        height = obst.shape[1]*0.5
        for i, row in enumerate(obst):
            for j, col in enumerate(row):
                if col:
                    #draw obstacle:
                    axes.add_patch(Rectangle((i-width,j-height),1,1,linewidth=1,edgecolor=[0,0,0],facecolor=[0.2,0.2,0.2]))
    #1. Plot Observations
    #   past robot movements

    moved_x = [step[0][2] for step in nodes_obs]
    moved_y = [step[0][3] for step in nodes_obs]
    plt.plot(moved_x, moved_y, color=[0,0,0], linestyle='solid', linewidth=1, marker='o', markersize=6)
    plt.plot(moved_x[-1], moved_y[-1], color=[0,0,0], linestyle='solid', linewidth=1, marker='o', markersize=12)
    #   past agent movements
    #   ids is a list of track_ids that we want to visualise
    for idx in ids:
        obs_x = [step[idx][0] for step in nodes_obs if step[idx][0] !=0 ]
        obs_y = [step[idx][1] for step in nodes_obs if step[idx][1] !=0 ]
        plt.plot(obs_x, obs_y, color=[0.6,0.6,0.6], linestyle='solid', linewidth=1, marker='o',markersize=4)
        #plt.plot(obs_x[-1], obs_y[-1], color=[0.6,0.6,0.6], linestyle='solid', linewidth=1, marker='o',markersize=8)

    #2. Plot explored Tree Actions
    #   tree_root is the parent of all child nodes etc,  we want to plot the position of each node
    depths = []
    widths = []
    for node in tree:
        if node.parent !=None:
            start = node.parent.state.robo_pos
            end = node.state.robo_pos
            depths.append(len(node.state.past_moves))
            widths.append(len(node.children))
            plt.plot([start[0],end[0]], [start[1],end[1]], color=[0,0,1], linestyle='solid', linewidth=0.5, marker='o',markersize=1,alpha=0.1)

    #   plot best chosen path
    current_node = root
    best_nodes = [current_node]
    for iter in range(20):
        #only bother checking 20 timesteps, we won't plot beyond that even if found
        current_node = bestChildPar(current_node, True)
        if current_node == None:
            #print('Length of Path = ', iter)
            break
        best_nodes.append(current_node)
        start = current_node.parent.state.robo_pos
        end = current_node.state.robo_pos
        plt.plot([start[0],end[0]], [start[1],end[1]], color=[1,0.1,0.1], linestyle='solid', linewidth=2, marker='o',markersize=2)
    #print(len(best_nodes))
    #3. Plot responses to best path
    agent_preds = [ [] for i in nodes_obs[0] ] #get all agent positions for each planned step
    for step in best_nodes:
        pos_all = step.state.agent_pos
        for i in range(len(agent_preds)):
            agent_preds[i].append(pos_all[i])
    # if isDist,  then agent_preds will now include the bivariate distribution for each predicted state
    for i, agent in enumerate(agent_preds):
        xs = [x[0] for x in agent]
        ys = [x[1] for x in agent]
        #Root node was last observed position, lets join to observed track
        x_link = [nodes_obs[-1][i][0], xs[0]]
        y_link = [nodes_obs[-1][i][1], ys[0]]
        plt.plot(x_link, y_link, color=[0.6,0.6,0.6], linestyle='solid', linewidth=1, marker='o',markersize=4)
        plt.plot(xs[0],ys[0], color=[0.6,0.6,0.6], linestyle='solid', linewidth=1, marker='o',markersize=8)
        if not any(x==0 for x in xs):
            #missed observations are 0's, so we assume this path is invalid
            plt.plot(xs,ys, color=[1,0.1,0.1], linestyle='solid', linewidth=2, marker='o',markersize=2)
        if isDist:
            sigma_xs = [np.exp(x[2]) for x in agent[1:]] #root node has no uncertainty
            sigma_ys = [np.exp(x[3]) for x in agent[1:]]
            corrs = [np.tanh(x[4]) for x in agent[1:]]
            sigma_xs.insert(0,sigma_xs[0]) # add in placeholder for root node (uncertainty same as 1st pred)
            sigma_ys.insert(0,sigma_ys[0])
            corrs.insert(0,corrs[0])
            
            for num_stds in range(3, 0,-1):
                for j in range(len(sigma_xs)):
                    cov = np.array([[sigma_xs[j]**2,corrs[j]*sigma_xs[j]*sigma_ys[j]],[corrs[j]*sigma_xs[j]*sigma_ys[j],sigma_ys[j]**2]])
                    lambda_, v = np.linalg.eig(cov)
                    lambda_ = np.sqrt(lambda_)
                    dist_cols = [[1,0,0],[1,0.5,0],[1,1,0.2]] 
                    alpha = 0.5 /num_stds
                    ell = Ellipse(xy=(xs[j],ys[j]),
                                  width=lambda_[0]*num_stds, #*0.5, 
                                  height=lambda_[1]*num_stds, #*0.5,
                                  angle=np.rad2deg(np.arccos(v[0, 0])),
                                  alpha=alpha)
                    ell.set_edgecolor(dist_cols[num_stds-1])
                    ell.set_facecolor(dist_cols[num_stds-1])
                    ell.set_linewidth(0.2)
                    axes.add_artist(ell) 
        #plt.plot(xs,ys, color=[1,0,0], linestyle='solid', linewidth=1, marker='o',markersize=2)

        

        
    plt.show()
    plt.gcf().clear()
    plt.clf()
    
    return depths, widths
