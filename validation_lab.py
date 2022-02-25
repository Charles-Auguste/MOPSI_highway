"""
    MOPSI Project
RL and autonomous vehicles

Authors : Even Matencio - Charles.A Gourio
Date : 15/02:2021
"""


# Standard library
import os
import sys
from datetime import datetime

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
import numpy as np
from typing import Callable

# Local source
import highway_env

# 3rd party packages
from mopsi_callback import MopsiCallback_single_core


#=====================================================================================
#============================= CONFIGURATION =========================================
#=====================================================================================

env_id = 'mopsi-env-v0'

load_from = "model_after_10000"



#=====================================================================================
#============================= MAIN PROGRAM ==========================================
#=====================================================================================

if __name__ == "__main__":

    env = gym.make(env_id)

    # Configuration
    env.config["number_of_lane"] = 1
    env.config["other_vehicles"] = 0
    env.config["controlled_vehicles"] = 1
    env.config["duration"] = 1000
    env.config["screen_width"] = 1000
    env.config["screen_height"] = 1000

    try:
        model = PPO.load(load_from)
    except:
        print("File not found, try to train the agent before launching again")
        sys.exit()

    obs = env.reset()
    for i in tqdm(range(env.config["duration"])):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

