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
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

# Local source
import highway_env

# 3rd party packages
from datetime import datetime
import imageio
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

#=====================================================================================
#================== CONFIGURATION AND GLOBAL VARIABLES ===============================
#=====================================================================================

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


#=====================================================================================
#============================= MAIN PROGRAM ==========================================
#=====================================================================================

if __name__ == "__main__":
    env_id = "mopsi-env-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Choose to train or not the agent
    TRAIN = True
    done = True



    if TRAIN:
        model = PPO("MlpPolicy", env,1e-1,verbose=1)
        model.learn(total_timesteps=1e5)
        model.save("PPO_mopsi_highway")
        del model

    try :
        model = PPO.load("PPO_mopsi_highway")
    except :
        print("File not found, try to train the agent before launching again")
        sys.exit()

    obs = env.reset()
    for i in tqdm(range(env.config["duration"])):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

