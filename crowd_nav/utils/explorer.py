#!/usr/bin/env python
"""
Used for training and testing of SARL.
Adapted for use with testing of MCTS-GRNN approach, and compariuson to MCTS-CV and PotentialField (PF) baselines

"""

import logging
import copy
from crowd_sim.envs.utils.info import *
import numpy as np
import time
from crowd_sim.envs.utils.action import ActionXY
import crowd_nav.policy.potential_field as potential_field
import crowd_nav.policy.predictive_planner.mcts_planner as predictive_planner
import torch
import csv

class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        training = False #tbd elsewhere
        if training:
            import torch

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    # @profile
    #run test episodes: define set number of agents in set positions!
    #default behaviour is to run SARL policy
    def run_k_episodes(self, k, 
                            phase, 
                            update_memory=False, 
                            imitation_learning=False, 
                            imitate_real_data=False, 
                            episode=None,
                            print_failure=False, 
                            randomise=True, 
                            comparisons=False, 
                            test_policy='sarl', 
                            output_dir=None, 
                            save_fig=False, 
                            model_dir=None, 
                            use_mcts_sef2=False):
        
        print('Using test policy:', test_policy)


        if output_dir is not None:
            output_csv_file = output_dir + 'test_data_'+ test_policy +'.csv'
            output_fields = ['episode','num_agents','result','near_misses','closest_dist', 'path_length', 'path_time', '%_disturbed', '%_half_disturbed', '%_quart_disturbed']
            with open(output_csv_file,'w') as fd:
                writer = csv.writer(fd)
                writer.writerow(output_fields)

        if comparisons:
            randomise = False #comparing behaviour in specific test cases

        if self.robot.policy is not None:
            self.robot.policy.set_phase(phase)       


        if test_policy == 'mctsrnn' or test_policy == 'mctscv':
            #init tf sess
            lookahead = 1  
            if test_policy == 'mctsrnn':
                use_cv=False
                print('Using Model:', model_dir)
                if model_dir == None:
                    raise SystemExit('Model directly must be supplied to use mctsrnn')  
            else:
                #test_policy == 'mctscv'
                use_cv=True            
                print('Using constant vel model')

            rnn_planner = predictive_planner.Planner(model_dir, lookahead, stdev=[4.9,4.9], use_cv=use_cv, use_sef2=use_mcts_sef2)

        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        comp_times = []
        path_lens = []
        accs=[]
        for i in range(k):      
            near_misses_iter = 0
            closest_dist_iter = 5.0
            path_time_iter = self.env.time_limit
            comp_times_iter = []
            accs_iter = []
            if randomise:
                np.random.seed() #numpy random doesnt work properly in torch with RNG states and forked processes https://github.com/pytorch/pytorch/issues/5059
                num_agents = np.random.randint(2,12)
                self.env.human_num = num_agents
            if comparisons:
                if i == 0:
                    print('Approaching Comparison...')
                elif i == 1:
                    print('Circle Comparison...')
                elif i == 2:
                    print('Crossing Comparison...')
                else:
                    print('Comparisons Complete')
                    break
                #explicitly set what comparison to do
                num_agents = 10
                self.env.human_num = num_agents
            if test_policy == 'mctsrnn' or test_policy == 'mctscv':
                #assumes that we haven't just appeared in a room of people with no knowledge of their history, so needs goal and v_pref to assume history is linear away from goal

                if comparisons:
                    ob = self.env.reset('comp',use_predictive_model=True,comparison=i)
                else:
                    ob = self.env.reset(phase,use_predictive_model=True)
                rnn_planner.reset() #removes all history of tracks
            else:
                if comparisons:
                    ob = self.env.reset('comp',comparison=i)
                else:
                    ob = self.env.reset(phase)
            done = False
            states = []
            actions = []
            rewards = []
            steps=0
            robo_pos_last = [0,0]

            if imitation_learning and imitate_real_data:
                #we don't imitate ORCA, we instead imitate real world observed interactions
                states.append(None) #what are these? observed states?
                actions = None
                info = None
                actions.append(None)
                rewards.append(None)
            else:
                #generate using ORCA
                while not done:
                    steps +=1

                    start = time.time()
                    if test_policy == 'pf':
                        #we use potential field reactive approach instead of RL                    
                        move = potential_field.getAction((self.robot.px,self.robot.py),(self.robot.gx,self.robot.gy),ob, self.env.circle_radius*3)
                        #print('MOVE-------', move)
                        action = ActionXY(move[0],move[1])
                    elif test_policy == 'mctsrnn' or test_policy == 'mctscv':
                        #we use potential field reactive approach instead of RL                    
                        move = rnn_planner.getAction((self.robot.px,self.robot.py),(self.robot.gx,self.robot.gy),ob)
                        #print('move', move)
                        l= np.linalg.norm(move)
                        #l = 0.35 #testing: (2*0.25^2)^0.5
                        if l != 0:
                            unit_move = np.array(move) / l
                        else:
                            unit_move = [0,0]
                        #print('MOVE-------', unit_move)
                        action = ActionXY(unit_move[0],unit_move[1])
                    else:
                        
                        action = self.robot.act(ob)
                        #print(np.linalg.norm([action.vx,action.vy]))
                        #print(action)

                    #print('Time:' , time.time()-start)
                    comp_times_iter.append(time.time()-start)
                    ob, reward, done, info = self.env.step(action)
                    if steps >200:
                        done = True
                        info = 'Test'
                    states.append(self.robot.policy.last_state)
                    actions.append(action)

                    rewards.append(reward)
                    if phase in ['test']:
                        #determine % steps that change in velocity of agents nearby robot exceeded threshold
                        dist_thresh = 5.0 #do not consider agent accelerations beyond 2m away

                        robo_pos = np.array(self.env.states[-1][0].position)
                        #print('d2g', np.linalg.norm([robo_pos[0]-self.robot.gx,robo_pos[1]-self.robot.gy]))
                        #print('actual move', np.linalg.norm([robo_pos[0]-robo_pos_last[0],robo_pos[1]-robo_pos_last[1]]))
                        robo_pos_last = robo_pos 
                        if len(self.env.states) > 2:

                            #can't determine acceleration with 2 points
                            for agent_num, agent in enumerate(self.env.states[-1][1]):
                                dist = np.linalg.norm(np.array(agent.position)-robo_pos)
                                if dist < dist_thresh:      
                                    #print(dist, agent.position, robo_pos)
                                    #change in vel of agent after latest robot action: 
                                    acc = np.linalg.norm(np.array(agent.velocity) - np.array(self.env.states[-2][1][agent_num].velocity))
                                    accs_iter.append(acc)

                    if isinstance(info, Danger):
                        print('DIST:', info.min_dist)
                        too_close += 1
                        near_misses_iter +=1
                        min_dist.append(info.min_dist)
                        if info.min_dist < closest_dist_iter:
                            closest_dist_iter = info.min_dist


            comp_times += comp_times_iter
            accs += accs_iter
            result = None
            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
                result = 'Success'
                path_time_iter = self.env.global_time
            elif isinstance(info, Collision):
                #print('COLL:', info.min_dist)
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
                result = 'Collision'
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
                result = 'Timeout'
            elif info == 'Test':
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
                result = 'Timeout'
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            if phase in ['test']:
                pts = np.array([state[0].position for state in self.env.states])
                lengths = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)) # Length between corners
                total_length = np.sum(lengths)
                path_lens.append(total_length)
                acc_thresh = 1.0*0.25 # m/s^2 * s/steps #based on  T. Korhonen (normal time to max_v is ~1s) and v_pref set in config of 1.0
                acc_exceed_ep = 0
                acc_exceed_half_ep = 0
                acc_exceed_quart_ep = 0
                if len(accs_iter) >0:
                    acc_exceed_ep = len([acc for acc in accs_iter if acc > acc_thresh]) / len(accs_iter)   
                    acc_exceed_half_ep = len([acc for acc in accs_iter if acc > acc_thresh*0.5]) / len(accs_iter)  
                    acc_exceed_quart_ep = len([acc for acc in accs_iter if acc > acc_thresh*0.25]) / len(accs_iter)  

                if output_dir is not None:
                    #fname = str(i)+'_traj.mp4'
                    #self.env.render(mode='video', output_file=fname)
                    #if save_fig:

                    fname = output_dir + test_policy + '_' + str(i)+'_traj.png'
                    self.env.render(mode='traj', output_file=fname)
                    #save data to csv
                    output_data = [i+1,num_agents,result,near_misses_iter,closest_dist_iter, total_length, path_time_iter, acc_exceed_ep, acc_exceed_half_ep, acc_exceed_quart_ep]

                    with open(output_csv_file,'a') as fd:
                        writer = csv.writer(fd)
                        writer.writerow(output_data)

            logging.info('Test {} of {}. Result: {}, Time Avg: {}'.format(i+1,k, info, np.mean(np.array(comp_times_iter))))


        if not comparisons:
            if test_policy == 'mctsrnn' or test_policy == 'mctscv':
                rnn_planner.cleanup()
            success_rate = success / k
            collision_rate = collision / k
            assert success + collision + timeout == k
            avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

            extra_info = '' if episode is None else 'in episode {} '.format(episode)

            if phase in ['test']:
                acc_thresh = 1.0*0.25 # m/s^2 * s/steps #based on  T. Korhonen (normal time to max_v is ~1s) and v_pref set in config of 1.0
                acc_exceed = 0
                acc_exceed_half = 0
                acc_exceed_quart = 0
                if len(accs) >0:
                    acc_exceed = len([acc for acc in accs if acc > acc_thresh]) / len(accs)   
                    acc_exceed_half = len([acc for acc in accs if acc > acc_thresh*0.5]) / len(accs)  
                    acc_exceed_quart = len([acc for acc in accs if acc > acc_thresh*0.25]) / len(accs)  
                logging.info('Agent Accelerations: Threshold {:.2f} @ dist {:.2f}m (full, half, quart):  {:.2f}, {:.2f}, {:.2f},'.
                             format(acc_thresh, 2, acc_exceed, acc_exceed_half, acc_exceed_quart))

            if phase in ['test','val']:
                logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}, avg comp time: {:.4f}, avg path len: {:.2f} '.
                             format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                                    average(cumulative_rewards), np.mean(np.array(comp_times)), np.mean(np.array(path_lens))))



                total_time = sum(success_times + collision_times + timeout_times) * self.robot.time_step
                logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                             too_close / total_time, average(min_dist))

            if print_failure:
                logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
                logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
