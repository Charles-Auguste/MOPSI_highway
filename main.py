import gym
import highway_env
from matplotlib import pyplot as plt
import time as t
import pygame

env = gym.make('mopsi-env-v0')

# ATTENTION
############

# Pas mettre plus de 20 voitures
# Se corrige tres facilement mais autre petit bug a corriger avant


env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 6
env.config["controlled_vehicles"] = 1
env.reset()

done = False

while not done:
    obs, reward, done, info = env.step([0,0])
    print("reward : ",reward)
    env.render()

plt.imshow(env.render(mode="rgb_array"))

