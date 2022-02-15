"""
    MOPSI Project
RL and autonomous vehicles

Authors : Even Matencio - Charles.A Gourio
Date : 15/02:2021
"""


# Standard library
from matplotlib import pyplot as plt
import pygame
import gym
from tqdm import tqdm

# Local source
import highway_env

# 3rd party packages
from datetime import datetime


#=====================================================================================
#=====================================================================================
#=====================================================================================

# Functions

def show_var_infos(vari,title="swow_var_info"):
    title_file = title
    fig,ax = plt.subplots()
    ax.plot(vari)
    ax.set_title("variance evolution from t=0 to t=T")
    fig.savefig("results/"+ title_file +".png")
    plt.show()

env = gym.make('mopsi-env-v0')

# Configuration

env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 40
env.config["controlled_vehicles"] = 1
env.config["duration"] = 1000


env.config["screen_width"] = 1000
env.config["screen_height"] = 1000

env.reset()

# Main program

if __name__ == "__main__":

    hist = []
    nb_vehicles = env.config["other_vehicles"] + env.config["controlled_vehicles"]
    duration = env.config["duration"]

    for i in tqdm(range(env.config["duration"])):

        # Action
        obs, reward, done, info = env.step([0,0])

        # Rendu graphique ( à commenter pour accelerer le code )
        #env.render()

        # Construction des histogrammes
        hist.append(env.var_speed())
        last_speed = env.get_speeds()

    time = str(datetime.now().date()) + "___" + str(
    datetime.now().hour) + "_" + str(datetime.now().minute) + "_" + str(datetime.now().second)
    show_var_infos(hist[10:], "traffic_" + str(nb_vehicles) + "_vehicles" + "__" + str(duration) + "it__" + time)


