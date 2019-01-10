# Project 2: Continuous Control

## Introduction

The goal of this project is to train an agent to move a double-jointed robot hand to a goal location and keep it there. The agent receives a reward of +0.1 for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the goal location for as many time steps as possible.

The observation space consists of 33 variables corresponding to the position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to the torque applicable to the two joints. Each entry in the action vector is a number between -1 and +1.

This project addresses an environment where 20 identical agents are trained simultaneously. The agents must get an average score of +30 over 100 consecutive episodes to solve the environment.


## Setting up the project

0. Clone this repository.

1. Download the environment from one of the links below. Select the environment that matches your operating system:

    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip);
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip);
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip);
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip).
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the repository folder and unzip (or decompress) the file. 


## Training and testing the agents

### Training the agents

Run the `agent_train.py` script as follows:

```bash
$ python agent_train.py --episodes 1000 --actor weights_actor.pth --critic weights_critic.pth --number 20 --plot plot.png
```

The script accepts the following parameters:
- **`--episodes`** (default: 1000): the maximum number of episodes;
- **`--actor`**: name of the file to store the weights of the actor network;
- **`--critic`**: name of the file to store the weights of the critic network;
- **`--number`** (default: 20): the number of agents;
- **`--plot`** (default: plot.png): name of the file to store the plot of the obtained scores.

### Testing the agents

Run the `agent_test.py` script as follows:

```bash
$ python agent_test.py --actor weights_actor.pth --critic weights_critic.pth --number 20
```

The script accepts the following parameters:
- **`--actor`**: name of the file to load the weights of the actor network;
- **`--critic`**: name of the file to load the weights of the critic network;
- **`--number`** (default: 20): the number of agents.
