import gym
import highway_env
from matplotlib import pyplot as plt
import time as t

env = gym.make('mopsi-env-v0')

# ATTENTION
############

# Pas mettre plus de 20 voitures
# Se corrige tres facilement mais autre petit bug a corriger avant


env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 20
env.config["controlled_vehicles"] = 1
env.reset()



for _ in range(1000):
    obs, reward, done, info = env.step([0,0])
    print("reward : ",reward)
    env.render()

plt.imshow(env.render(mode="rgb_array"))

