"""
    MOPSI Project
RL and autonomous vehicles

Authors : Even Matencio - Charles.A Gourio
Date : 15/02:2021
"""


# Standard library
import os
import sys
from matplotlib import pyplot as plt
import pygame
import gym
from tqdm import tqdm
import numpy as np

# Local source
import highway_env

# 3rd party packages
from datetime import datetime
import imageio

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#=====================================================================================
#================== CONFIGURATION AND GLOBAL VARIABLES ===============================
#=====================================================================================

env = gym.make('mopsi-env-v0')

# Configuration
env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 3
env.config["controlled_vehicles"] = 1
env.config["duration"] = 500

env.config["screen_width"] = 1000
env.config["screen_height"] = 1000

# Choose to train or not the agent
TRAIN = False

obs = env.reset()
done = True

#=====================================================================================
#============================= MAIN PROGRAM ==========================================
#=====================================================================================

if TRAIN:
    model = PPO("MlpPolicy", env,1e-5,verbose=1, device='cuda')
    model.learn(total_timesteps=100000)
    model.save("PPO_mopsi_highway")
    del model

try :
    model = PPO.load("PPO_mopsi_highway")
except :
    print("File not found, try to train the agent before launching again")
    sys.exit()

for i in tqdm(range(env.config["duration"])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()



