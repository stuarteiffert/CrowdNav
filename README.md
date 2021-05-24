# CrowdNav
This repository contains the codes for our ICRA 2018 paper. For more details, please refer to the paper
[Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning](https://arxiv.org/abs/1809.08835).


Updated by Stuart Eiffert:
    Testing other methods in same simulated environment:
        MCTS-RNN
        PF
        MCTS-CV
    Instructructions:
        1. Train an RNN prediction model
        2. Train an RL policy
        3. Test navigation via: cd crowd_nav & python3 test.py --test_policy='mctsrnn' --output_dir='/data/outputs/images' --save_fig --mixed --gpu

        Set test variables in crowd_nav/config/env.configs

        Use SEF2 in MCTS via use_sef2=True, set via --use_mcts_sef2 in test.py (Note: this will not work for MCTS-CV which does not predict any response) 

Testing:

python3 test.py --policy='sarl' --test_policy='sarl' --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/comp_test/' --save_fig --gpu --model_dir=data/output_02082019_cv


Comparison:
    test_policy='mctscv'
    python test.py --test_policy=$test_policy --compare_scenarios --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/comp_test/' --save_fig --gpu --pred_model_dir='/home/stuart/acfr/code/sequence_prediction/models/ORCA/LSTM_2_64_timestep3_prob_1' 

python3 test.py --policy='sarl' --test_policy='sarl' --compare_scenarios --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/comp_test/' --save_fig --gpu --model_dir=data/output_02082019_cv


python3 test.py --test_policy='pf' --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/images/' --save_fig --mixed --gpu
python3 test.py --test_policy='sarl' --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/images/' --save_fig --mixed --gpu
python3 test.py --test_policy='mctscv' --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/images/' --save_fig --mixed --gpu
python3 test.py --test_policy='mctsrnn' --output_dir='/home/stuart/code/CrowdNav/crowd_nav/data/outputs/images/' --pred_model_dir='/home/stuart/acfr/code/sequence_prediction/models/ORCA/LSTM_2_64_timestep3_prob_1' --save_fig --mixed --gpu

## References

Makes use of ORCA, SARL


## Citation
If using this repo, please consider citing:

@INPROCEEDINGS{Eiffert2020,
  author={S. {Eiffert} and H. {Kong} and N. {Pirmarzdashti} and S. {Sukkarieh}},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Path Planning in Dynamic Environments using Generative RNNs and Monte Carlo Tree Search}, 
  year={2020},
  pages={10263-10269},
  doi={10.1109/ICRA40945.2020.9196631}}


## Abstract
Mobility in an effective and socially-compliant manner is an essential yet challenging task for robots operating in crowded spaces.
Recent works have shown the power of deep reinforcement learning techniques to learn socially cooperative policies.
However, their cooperation ability deteriorates as the crowd grows since they typically relax the problem as a one-way Human-Robot interaction problem.
In this work, we want to go beyond first-order Human-Robot interaction and more explicitly model Crowd-Robot Interaction (CRI).
We propose to (i) rethink pairwise interactions with a self-attention mechanism, and
(ii) jointly model Human-Robot as well as Human-Human interactions in the deep reinforcement learning framework.
Our model captures the Human-Human interactions occurring in dense crowds that indirectly affects the robot's anticipation capability.
Our proposed attentive pooling mechanism learns the collective importance of neighboring humans with respect to their future states.
Various experiments demonstrate that our model can anticipate human dynamics and navigate in crowds with time efficiency,
outperforming state-of-the-art methods.


## Method Overview
<img src="https://i.imgur.com/YOPHXD1.png" width="1000" />

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy sarl
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy sarl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy sarl --model_dir data/output --phase test --visualize --test_case 0
```
5. Plot training curve.
```
python utils/plot.py data/output/output.log
```


## Simulation Videos
CADRL             | LSTM-RL
:-------------------------:|:-------------------------:
<img src="https://i.imgur.com/vrWsxPM.gif" width="400" />|<img src="https://i.imgur.com/6gjT0nG.gif" width="400" />
SARL             |  OM-SARL
<img src="https://i.imgur.com/rUtAGVP.gif" width="400" />|<img src="https://i.imgur.com/UXhcvZL.gif" width="400" />


## Learning Curve
Learning curve comparison between different methods in an invisible setting.

<img src="https://i.imgur.com/l5UC3qa.png" width="600" />

## Citation
If you find the codes or paper useful for your research, please cite our paper:
```
@misc{1809.08835,
Author = {Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
Title = {Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
Year = {2018},
Eprint = {arXiv:1809.08835},
}
```
