# Welcome to the 2022 NeurIPS Driving SMARTS Competition
Thank you for your interest in the 2022 NeurIPS Driving SMARTS competition. Please take a look at the [competition site](https://codalab.lisn.upsaclay.fr/competitions/6618).

## Registration
In order to be elligible for any rewards, either 

+ Fill out the following form: [Registration Form](https://docs.google.com/forms/d/1bIXTQL420q-cB65j1df2vhbh-79NMlm0M2G0uQtwvds)

+ Or, send a response to smarts4ad@gmail.com including the following information:
    ```text
    Public group name [required]:      
    Group members [required]:      
    Declared affiliations (orgs or any relation to organisers) [required]:      
    Primary contact email [required]:
    ```

## Competition Tracks
Validation Stage: This stage is to validate that your submission will work without errors during Track1 and Track2 evaluation.

There are two competition tracks.
+ Track 1: The participants may use any method to develop their solutions.
+ Track 2: The participants are only allowed to train their methods on the offline datasets.

## Prizes
Top participants in each track will receive the following prizes:

* Gold US$6000
* Silver US$4000
* Bronze US$2000

Additional prizes:

* US$1000 for the most innovative approach out of top-6 entries in both tracks
* US$1000 given to one of the valid submissions not in top-3 positions in either track

Winners in each track will receive cash prizes and will get a chance to present their innovative solutions during a virtual ceremony.

## First Steps
Code and instructions related to the competition may be found in the [competition directory](./competition/) where it is recommended you read the READMEs of each section.
- [Track 1 training](./competition/track1/train/README.md)
- [Track 1 submission](./competition/track1/submission/README.md)
- [Track 2](./competition/track2/README.md)
- [Evaluation](./competition/evaluation/README.md)

## Starting Kits
Starting code may be found for each track in the following locations.
- [Track 1](./competition/track1/)
- [Track 2](./competition/track2/)

## Submission
Deliverables may be submitted to the following site: https://codalab.lisn.upsaclay.fr/competitions/6618#participate-submit_results

Track1 and Track2 deliverables can only be submitted a limited number of times. Therefore, the Validation Stage may be used to ensure your model does not encounter errors during Track1 and Track2 submission.

#

# SMARTS
[![SMARTS CI Base Tests Linux](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-base-tests-linux.yml?query=branch%3Amaster) 
[![SMARTS CI Format](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml/badge.svg?branch=master)](https://github.com/huawei-noah/SMARTS/actions/workflows/ci-format.yml?query=branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/smarts/badge/?version=latest)](https://smarts.readthedocs.io/en/latest/?badge=latest)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SMARTS (Scalable Multi-Agent Reinforcement Learning Training School) is a simulation platform for multi-agent reinforcement learning (RL) and research on autonomous driving. Its focus is on realistic and diverse interactions. It is part of the [XingTian](https://github.com/huawei-noah/xingtian/) suite of RL platforms from Huawei Noah's Ark Lab.

Check out the paper at [SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving](https://arxiv.org/abs/2010.09776).

![](docs/_static/smarts_envision.gif)

# Documentation
:rotating_light: :bell: Read the docs :notebook_with_decorative_cover: at [smarts.readthedocs.io](https://smarts.readthedocs.io/) . :bell: :rotating_light:

# Setup and Quickstart
1. [Set up SMARTS](https://smarts.readthedocs.io/en/latest/setup.html).
2. [Run a simple experiment](https://smarts.readthedocs.io/en/latest/quickstart.html).

# Examples 
### Egoless
Simulate a SMARTS environment without any ego agents, but with only background traffic.
1. [Egoless](examples/egoless.py) example.

### Control Theory
Several agent control policies and agent [action types](smarts/core/controllers/__init__.py) are demonstrated.

1. Chase Via Points
   + script: [control/chase_via_points.py](examples/control/chase_via_points.py)
   + Multi agent
   + ActionSpaceType: LaneWithContinuousSpeed
1. Trajectory Tracking
   + script: [control/trajectory_tracking.py](examples/control/trajectory_tracking.py)
   + ActionSpaceType: Trajectory
1. OpEn Adaptive Control
   + script: [control/ego_open_agent.py](examples/control/ego_open_agent.py)
   + ActionSpaceType: MPC
1. Laner
   + script: [control/laner.py](examples/control/laner.py)
   + Multi agent
   + ActionSpaceType: Lane
1. Parallel environments
   + script: [control/parallel_environment.py](examples/control/parallel_environment.py)
   + Multiple SMARTS environments in parallel
   + ActionSpaceType: LaneWithContinuousSpeed

### RL Model
1. [Intersection](examples/rl/intersection) using PPO from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3).
    <img src="examples/rl/intersection/docs/_static/intersection.gif" height="450" width="600"/>
1. [Racing](examples/rl/racing) using world model based RL.
    <img src="examples/rl/racing/docs/_static/racing.gif" height="350" width="600"/>

### RL Environment
1. [ULTRA](https://github.com/smarts-project/smarts-project.rl/blob/master/ultra) provides a gym-based environment built upon SMARTS to tackle intersection navigation, specifically the unprotected left turn.

# Issues, Bugs, Feature Requests 
1. First, read how to communicate issues, report bugs, and request features [here](./docs/resources/contributing.rst#communication).
1. Next, raise them using appropriate tags at [https://github.com/huawei-noah/SMARTS/issues](https://github.com/huawei-noah/SMARTS/issues).

# Cite this work
If you use SMARTS in your research, please cite the [paper](https://arxiv.org/abs/2010.09776). In BibTeX format:

```bibtex
@misc{SMARTS,
    title={SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving},
    author={Ming Zhou and Jun Luo and Julian Villella and Yaodong Yang and David Rusu and Jiayu Miao and Weinan Zhang and Montgomery Alban and Iman Fadakar and Zheng Chen and Aurora Chongxi Huang and Ying Wen and Kimia Hassanzadeh and Daniel Graves and Dong Chen and Zhengbang Zhu and Nhat Nguyen and Mohamed Elsayed and Kun Shao and Sanjeevan Ahilan and Baokuan Zhang and Jiannan Wu and Zhengang Fu and Kasra Rezaee and Peyman Yadmellat and Mohsen Rohani and Nicolas Perez Nieves and Yihan Ni and Seyedershad Banijamali and Alexander Cowen Rivers and Zheng Tian and Daniel Palenicek and Haitham bou Ammar and Hongbo Zhang and Wulong Liu and Jianye Hao and Jun Wang},
    url={https://arxiv.org/abs/2010.09776},
    primaryClass={cs.MA},
    booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
    year={2020},
    month={11}
}
```
