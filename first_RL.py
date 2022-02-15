import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('mopsi-env-v0')


env.config["number_of_lane"] = 1
env.config["other_vehicles"] = 29
env.config["controlled_vehicles"] = 1
obs = env.reset()

done = False

for i in range(100):
    obs, reward, done, info = env.step([0,0])
    env.render()

