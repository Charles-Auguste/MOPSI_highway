"""
    MOPSI Project
RL and autonomous vehicles

Authors : Even Matencio - Charles.A Gourio
Date : 15/02:2021
"""


# Standard library
import os
import sys
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from tqdm import tqdm
import numpy as np

# Local source
import highway_env
from rl_callback import SaveCallback

# 3rd party packages


#=====================================================================================
#================== CONFIGURATION AND GLOBAL VARIABLES ===============================
#=====================================================================================


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

    # Choose to train or not the agent
    TRAIN = True
    done = True

    saving_path = "model/"

    env_id = 'mopsi-env-v0'
    if TRAIN:
        num_cpu = 4  # Number of processes to use
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    else:
        env = gym.make(env_id)




    if TRAIN:
        model = PPO("MlpPolicy", env,1e-3,verbose=1,tensorboard_log="ppo_mopsi_tensorboard/")
        callback = SaveCallback(nb_step=100000, log_dir=saving_path+"inter/")
        model.learn(total_timesteps=200000,tb_log_name="mopsi_run")
        model.save(saving_path +"PPO_mopsi_highway")
        del model

    try :
        model = PPO.load("model/PPO_mopsi_highway")
    except :
        print("File not found, try to train the agent before launching again")
        sys.exit()

    obs = env.reset()
    for i in tqdm(range(env.config["duration"])):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

