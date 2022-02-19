"""
    MOPSI Project
RL and autonomous vehicles

Authors : Even Matencio - Charles.A Gourio
Date : 15/02:2021
"""


# Standard library
import os
from matplotlib import pyplot as plt
import pygame
import gym
from tqdm import tqdm

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
#=====================================================================================
#=====================================================================================

TRAIN = True

env = gym.make('mopsi-env-v0')

obs = env.reset()

if TRAIN:
    model = PPO("MlpPolicy", env,1e-5,verbose=1, device='cuda')
    model.learn(total_timesteps=10000)
    model.save("PPO_mopsi_highway")
    del model # remove to demonstrate saving and loading

model = PPO.load("PPO_mopsi_highway")

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()



