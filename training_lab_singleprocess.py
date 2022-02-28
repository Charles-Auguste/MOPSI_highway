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
#================== CONFIGURATION AND GLOBAL VARIABLES ===============================
#=====================================================================================

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


# Configuration

nb_iteration = 1000000 # Number of time steps for learning

learning_rate = 1e-3

debug_info = 0  # 0 for nothing, 1 for minimum, 2 for max

saving_rate = 10000 # Interval between each saves (optimal between 10000 and 100000)

comment = "__test7__" # add a comment to the name of the simulation

load_from = "model/model__2022-02-26___20_32_45PPO_mopsi_highway___test5__"


# Initialisation
time = str(datetime.now().date()) + "___" + str(
        datetime.now().hour) + "_" + str(datetime.now().minute) + "_" + str(datetime.now().second)
saving_path = "model/model__" + time

env_id = 'mopsi-env-v0'


#=====================================================================================
#============================= MAIN PROGRAM ==========================================
#=====================================================================================

if __name__ == "__main__":

    # Create the vectorized environment
    env = gym.make(env_id)

    if load_from != "":
        try :
            model = PPO.load(load_from)
            model.set_env(env)
        except :
            print("Error : file not found")
            sys.exit()
    else :
        model = PPO("MlpPolicy", env, learning_rate, verbose=debug_info, tensorboard_log="ppo_mopsi_tensorboard/")
    M_callback = MopsiCallback_single_core(nb_step=saving_rate, log_dir=saving_path, env = env)
    model.learn(total_timesteps = nb_iteration, tb_log_name="mopsi_run_"+time+comment, callback = M_callback)
    model.save(saving_path +"PPO_mopsi_highway_"+comment)
    del model

