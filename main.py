import gym
import highway_env
from matplotlib import pyplot as plt
import time as t
import pygame

# env = gym.make('racetrack-v0')
env = gym.make('mopsi-env-v0')

env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 1
env.config["controlled_vehicles"] = 1
env.reset()

done = False

while not done:
    obs, reward, done, info = env.step([0,0])
    print("reward : ",reward)
    env.render()
    print(env.get_speeds())

plt.imshow(env.render(mode="rgb_array"))

