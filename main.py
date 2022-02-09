import gym
import highway_env
from matplotlib import pyplot as plt
import time as t
import pygame



def show_var_infos(vari,title="swow_var_info"):
    title_file = title
    fig,ax = plt.subplots()
    ax.plot(vari)
    ax.set_title("variance evolution from t=0 to t=T")
    fig.savefig("results/"+ title_file +".png")
    plt.show()

env = gym.make('mopsi-env-v0')


env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 35
env.config["controlled_vehicles"] = 1
env.reset()

done = False

hist = []
while not done:
    obs, reward, done, info = env.step([0,0])
    env.render()
    hist.append(env.var_speed())
    last_speed = env.get_speeds()

show_var_infos(hist[10:],"traffic_36_vehicles")


