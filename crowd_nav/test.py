#!/usr/bin/env python
"""
Testing of various path planning approaches in a simulated ORCA environment, as per Eiffert et al. 2020 (https://arxiv.org/abs/2001.11597)
This script allows testing of SARL, LM-SARL, MCTS-GRNN, MCTS-CV and PF planners.

##############################################################################
Usage:

python test.py --policy='mctscv' 
python test.py --policy='mctsrnn' --gpu --pred_model_dir=$saved_rnn_model_dir
python test.py --policy='sarl' --gpu --model_dir=$saved_rl_model_dir
python test.py --policy='pf' 

For SARL with ORCA state transition: saved_rl_model_dir=data/sarl_orca
For SARL with CV state transition: saved_rl_model_dir=data/sarl_cv
For LM-SARL with ORCA state transition: saved_rl_model_dir=data/sarl_lm_orca
Edit testing configuration via file pointed to by --env_config

Options:
--output_dir: Save results of each episode to given directory
--save_fig:  Save resultant trajectories as png (requires output_dir set)
--interactive: Allow interaction with the orca environmnet in a GUI.
--comparison:  Compare behaviour in set scenarios
"""
import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym
import copy
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from tkinter import *

class App:
    def __init__(self, master, env, pattern):
        self.master = master
        self.env = env
        frame = Frame(self.master)
        self.master.bind('<Left>', self.leftKey)
        self.master.bind('<Right>', self.rightKey)
        self.master.bind('<Up>', self.upKey)
        self.master.bind('<Down>', self.downKey)
        #self.master.bind('<space>', self.actionSelect)
        frame.pack()
        self.quitButton = Button(frame, text="QUIT", fg="red", cnf={},
            command=quit)
        self.quitButton.grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.starttButton = Button(frame, text="Start",
            command=self.start)
        self.starttButton.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.resetButton = Button(frame, text="Reset", command=self.reset)
        self.resetButton.grid(row=3, column=0, sticky='w', padx=5, pady=5)

        self.state = [] #contains positions etc of all agents, Robot is always pos 0
        self.last_state = [] #contains positions etc of all agents, Robot is always pos 0
        self.actions = [0,1,2,3]
        self.icons = []
        self.score = 0
        self.horizon = 10
        self.started = False
        self.agent_pattern = pattern #0: approaching, 1: circle


        self.board_width = 400
        self.board_height = 400
        robot_speed = 4
        self.scale = 20

        self.ob = self.env.reset('int', comparison=self.agent_pattern)
        self.done = False

        self.canvas = Canvas(self.master, bg="blue", height=self.board_height, width=self.board_width)
        self.canvas.pack()
        

    def initUI(self,state):
        #state[0] = robot, state[1] = all other agents
        robo_pos = state[0]
        x1 = int(robo_pos[0]*self.scale-5)+self.board_width*0.5
        x2 = int(robo_pos[0]*self.scale+5)+self.board_width*0.5
        y1 = int(robo_pos[1]*self.scale-5)+self.board_height*0.5
        y2 = int(robo_pos[1]*self.scale+5)+self.board_height*0.5
        col = "green"
        icon = self.canvas.create_oval(x1,y1,x2,y2, fill=col)
        self.icons.append(icon)

        for idx, agent in enumerate(state[1]):
            if idx==0:
                col = "black"
            else:
                col = "red"
            pos_now = agent
            x1 = int(pos_now[0]*self.scale-5)+self.board_width*0.5
            x2 = int(pos_now[0]*self.scale+5)+self.board_width*0.5
            y1 = int(pos_now[1]*self.scale-5)+self.board_height*0.5
            y2 = int(pos_now[1]*self.scale+5)+self.board_height*0.5
            icon = self.canvas.create_oval(x1,y1,x2,y2, fill=col)
            self.icons.append(icon)

    def leftKey(self,event):
        user_action = ActionXY(-1,0)
        self.updateState(user_action)

    def rightKey(self,event):
        user_action = ActionXY(1,0)
        self.updateState(user_action)

    def upKey(self,event):
        user_action = ActionXY(0,-1)
        self.updateState(user_action)
        

    def downKey(self,event):
        user_action = ActionXY(0,1)
        self.updateState(user_action)

    def updateState(self, action):
        self.last_state = [ self.state[0], [self.state[1][i] for i in range(len(self.state[1]))] ]  #copy.deepcopy(self.state)
        self.ob, _, done, info = self.env.step(action)
        self.state = [ self.env.robot.get_position(), [self.ob[i].position for i in range(len(self.ob)) ] ]
        self.done = done

        #update robo pos
        robo_move = [self.state[0][0] - self.last_state[0][0], self.state[0][1] - self.last_state[0][1] ]
        self.canvas.move(self.icons[0],robo_move[0]*self.scale,robo_move[1]*self.scale)

        for i in range(len(self.state[1])):
            move = [self.state[1][i][0] - self.last_state[1][i][0], self.state[1][i][1] - self.last_state[1][i][1] ]
            self.canvas.move(self.icons[i+1],move[0]*self.scale,move[1]*self.scale)


    def getScore(self,state):
        score = 0 #update
        for agent in state[1:]:
            dist = np.linalg.norm(agent.pos - goal)
            score += dist
        return int(score)

    def start(self):

        val = np.random.randint(10)+1        
        self.init_env(val)

    def reset(self):
        self.started = False
        self.state = []
        for icon in self.icons:
            self.canvas.delete(icon)
        self.icons = []
        self.score = 0
        self.start()

    def init_env(self,num):
        self.started = True
        print("Starting Game.")
        self.goals = []
        self.poss = []
        self.agents=[]       

        self.ob = self.env.reset('int', comparison=self.agent_pattern)
        self.done = False
        self.state = [ self.env.robot.get_position(), [self.ob[i].position for i in range(len(self.ob)) ] ]

        self.initUI(self.state)

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--test_policy', type=str, default='mctsrnn')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_fig', default=False, action='store_true')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--pred_model_dir', type=str, default=None)
    parser.add_argument('--use_mcts_sef2', default=False, action='store_true')
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--mixed', default=True, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--interactive', default=False, action='store_true')
    parser.add_argument('--compare_scenarios', default=False, action='store_true')
    parser.add_argument('--pattern', type=int, default=0)
    args = parser.parse_args()

    #Need to create a robot policy within the simulated environmnet, even though we do not use it for testing when MCTS or PF is used
    if args.policy == 'mctsrnn':
        args.policy = 'orca'
        args.test_policy = 'mctsrnn'
    elif args.policy == 'mctscv':
        args.policy = 'orca'
        args.test_policy = 'mctscv'
    elif args.policy == 'pf':
        args.policy = 'orca'
        args.test_policy = 'pf'
    elif args.policy == 'sarl':
        args.test_policy = 'sarl'
    elif args.policy == 'orca':
        args.test_policy = 'orca'
    else:
        sys.exit("--policy must be one of 'mctsrnn', 'mctscv', 'pf', 'sarl', 'orca'")

    if args.model_dir is not None:
        #env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = args.model_dir + '/policy.config'
        env_config_file = args.env_config
        #policy_config_file = args.policy_config
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        #policy_config_file = args.env_config
        policy_config_file = args.policy_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)

    policy.configure(policy_config)

    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))
    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    if args.mixed:
        env.test_sim = 'mixed'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)
    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0.2
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.interactive:
        #for testing how orca responds to action inputs
        print('phase', 'interactive')
        root = Tk()
        app = App(root, env, args.pattern)
        root.mainloop()

    elif args.visualize:
        print('phase', args.phase)
        ob = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        while not done:
            action = robot.act(ob)
            
            ob, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
        if args.traj:
            env.render('traj', args.video_file)
        else:
            env.render('video', args.video_file)

        logging.info('It takes %.2f seconds to finish. Final status is %s', env.global_time, info)
        if robot.visible and info == 'reach goal':
            human_times = env.get_human_times()
            logging.info('Average time for humans to reach goal: %.2f', sum(human_times) / len(human_times))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], 
                                args.phase, 
                                print_failure=True, 
                                test_policy=args.test_policy, 
                                output_dir=args.output_dir, 
                                save_fig=args.save_fig, 
                                model_dir=args.pred_model_dir, 
                                use_mcts_sef2=args.use_mcts_sef2,
                                comparisons=args.compare_scenarios)


if __name__ == '__main__':
    main()
