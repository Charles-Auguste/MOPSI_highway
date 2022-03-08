"""
    MOPSI Project
RL and autonomous vehicles

Authors : Even Matencio - Charles.A Gourio
Date : 15/02:2021
"""

# Standard library
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pygame
import gym
from tqdm import tqdm

# Local source
import highway_env

# 3rd party packages
from datetime import datetime
import imageio


#=====================================================================================
#============================== FUNCTIONS ============================================
#=====================================================================================

def show_var_infos(vari,title="swow_var_info", dirpath = None):
    title_file = title
    fig,ax = plt.subplots()
    ax.plot(vari)
    ax.set_title("variance evolution from t=0 to t=T")
    ax.set_xlabel("Time (nb it)")
    ax.set_ylabel("variance (m/s)")
    if dirpath != None:
        fig.savefig(dirpath+"/"+ title_file +".png")
    plt.show()

env = gym.make('mopsi-env-v0')


#=====================================================================================
#================== CONFIGURATION AND GLOBAL VARIABLES ===============================
#=====================================================================================

# Configuration
env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 0
env.config["controlled_vehicles"] = 1
env.config["duration"] = 100
env.config["circle_radius"] = 200

env.config["screen_width"] = 1000
env.config["screen_height"] = 1000

env.reset("sim")

#=====================================================================================
#============================ MAIN PROGRAM ===========================================
#=====================================================================================

if __name__ == "__main__":

    # Main Loop
    plt.ion()
    fig, ax_lst = plt.subplots()
    for i in range(env.config["duration"]):
        # Action
        obs, reward, done, info = env.step([0,0])
        road = np.array(obs[3])
        presence = np.array(obs[0])
        ax_lst.imshow(presence+road)
        fig.canvas.draw()
        fig.canvas.flush_events()



