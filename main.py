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
env.config["other_vehicles"] = 30
env.config["controlled_vehicles"] = 1
env.config["duration"] = 1500

env.config["screen_width"] = 1000
env.config["screen_height"] = 1000

# Saving results and datas
SAVE_SIMULATION = True

# Put "sim" bellow for an IDM simulation
env.reset("sim")

#=====================================================================================
#============================ MAIN PROGRAM ===========================================
#=====================================================================================

if __name__ == "__main__":

    # Initialisation
    hist = []
    done = False
    nb_vehicles = env.config["other_vehicles"] + env.config["controlled_vehicles"]
    real_nb_vehicles = len(env.road.vehicles)
    print(real_nb_vehicles)
    duration = env.config["duration"]

    if SAVE_SIMULATION and duration < 100:
        raise SystemError("Simulation must have at least 100it")

    # Initialisation du dossier résultats
    if SAVE_SIMULATION:
        time = str(datetime.now().date()) + "___" + str(
        datetime.now().hour) + "_" + str(datetime.now().minute) + "_" + str(datetime.now().second)
        result_folder_path = "results/Simulation__"+ time
        os.mkdir(result_folder_path)

        time_gif = duration//2
        duration_gif = 50
        filenames = []


    # Main Loop
    for i in tqdm(range(duration)):

        # Building renders for the beginning and the end of the simulation
        if i == 1:
            name_picture = result_folder_path + "/" + "render_begin" + str(i) + ".png"
            plt.imsave(name_picture, env.render(mode="rgb_array"))

        if i == duration - 1:
            name_picture = result_folder_path + "/" + "render_end" + str(i) + ".png"
            plt.imsave(name_picture, env.render(mode="rgb_array"))

        # Gif
        if SAVE_SIMULATION and i >= time_gif and i <= time_gif + duration_gif:
            name_picture = result_folder_path + "/" + "render" + str(i) + ".png"
            filenames.append(name_picture)
            plt.imsave(name_picture, env.render(mode="rgb_array"))

        # Failure
        if done and i!=duration - 1:
            name_picture = result_folder_path + "/" + "render_fail" + str(i) + ".png"
            plt.imsave(name_picture, env.render(mode="rgb_array"))
            sys.exit()

        # Action
        obs, reward, done, info = env.step([0,0])

        # Histogram
        hist.append(env.var_speed())


    # Create gif

    if SAVE_SIMULATION:
        with imageio.get_writer(result_folder_path + "/results.gif", mode = "I") as writer :
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        for filename in set(filenames):
            os.remove(filename)

        show_var_infos(hist[10:], "traffic_" + str(real_nb_vehicles) + "_vehicles" + "__" + str(duration) + "it__",
                   dirpath=result_folder_path)

    else :
        show_var_infos(hist[10:], "traffic_" + str(real_nb_vehicles) + "_vehicles" + "__" + str(duration) + "it__",
                   dirpath= None)
