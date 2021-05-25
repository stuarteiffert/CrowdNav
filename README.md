# MCTS-GRNN
This repository supplements the ICRA 2020 paper [Path Planning in Dynamic Environments using Generative RNNs and Monte Carlo Tree Search](https://arxiv.org/abs/2001.11597).

Included within this repository is the code required to reproduce simulated planning results, and the resultant data from these simulations.
Note that in order to reproduce the MCTS-GRNN SEF1 and SEF2 results shown in Table 1 of the paper a trained RNN model is required. This model has not been provided in this repository.

****
## Updates

* [2021.05.23] SRLSTM (ref) compared to the generative RNN model used within this work in terms of ability to model the response of an agent to a robot's planned action (Ref TBD). SRLSTM found to allow significantly better modelling of response and will be integrated in future work.

* [2021.05.18] Errors in the original potential field (PF) planner implementation fixed. Results are now significantly better, updated in data directory.

* [2021.05.17] Per episode data reproduced and saved in \data directory, as original paper only saved summary data. Note that this data differs slightly from the original reported results, suggesting that significantly more than 500 episodes should be used in future for testing.

* [2021.05.16] Reported average computation time per method reported in Table 1 found to not have included time required to convert data into required form for MCTS versions. This is approxiately 30ms for MCTS-GRNN and 120 ms for MCTS-CV, due to added need for creating kalman filter objects per tracked agent. Note that this time has not been optimised for this work.

* [2021.05.16] Note: SARL was used in the original paper rather than LM-SARL as original noted. New training of LM-SARL in 2021 has not resulted in a stable planner.


****
## Results



Updated results for Table 1 of paper, based on per episode data, improved PF, and use of ORCA vs constant velocity in SARL.

					Disturbance		
Method	| Success %	| Coll % | Avg Len (m) | Avg Comp (s) |	Disturbance > 1 m/s^2 | > 0.5 m/s^2 | > 0.25 m/s^2
:-----------:|:--------:|:------:|:-------:|:-------:|:-------:|:-------:|:-------:
MCTS-RNN SEF1 | 98.20%	 | 0.40% | 	17.68 | 	0.3* | 	1.52% | 	6.81% | 	13.49%
MCTS-RNN SEF2 | 91.80% | 0.00%	 | 18.3975 | 0.3* | 	1.08% | 	5.44% | 	10.75%
MCTS-CV | 97.00%	 | 0.20%	 | 16.3845	0.3*	1.61%	 | 7.16%	 | 14.31%
SARL | 95.40%	 | 0.00% | 	17.01	 | 0.251	 | 1.55%	 | 6.91%	 | 12.59%
SARL-CV | 97.00% | 	0.00% | 	17.01	 | 0.15	 | 1.71%	 | 7.12%	 | 12.74%
PF | 91.40%	 | 0.80%	 | 19.63	 | 0.01	 | 1.85%	 | 7.46%	 | 13.50%

****
## Usage

### Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip. See setup.py for list of packages being installed
```
pip install -e .
```

### Testing

Default testing uses config from /crowd_nav/config/env.config. 
Default is 500 mixed test cases.
Policy should be one of ['mctsrnn', 'mctscv', 'pf', 'sarl', 'orca']
If using 'sarl', provide model_dir of trained RL policy. See /models. Additionally, ensure that the environmnet config in the policy config file matches the test config.

Example usage:

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


### Training

SARL trained as per (ref):
```
python train.py --policy sarl
```

Generative RNN trained as per ... in tensorflow 1.10.1

## References

Makes use of ORCA, SARL.
See SARL repo for more info on simulated environmnet


## Citation
If using this repo, please cite:

```
@INPROCEEDINGS{Eiffert2020,
  author={S. {Eiffert} and H. {Kong} and N. {Pirmarzdashti} and S. {Sukkarieh}},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Path Planning in Dynamic Environments using Generative RNNs and Monte Carlo Tree Search}, 
  year={2020},
  pages={10263-10269},
  doi={10.1109/ICRA40945.2020.9196631}}
```

and also consider citing the work that this repo builds upon:

```
@misc{1809.08835,
Author = {Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
Title = {Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
Year = {2018},
Eprint = {arXiv:1809.08835},
}
```
