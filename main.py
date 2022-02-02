import gym
import highway_env
from matplotlib import pyplot as plt
import time as t

env = gym.make('mopsi-env-v0')

for i in range (1,7):
    env.config["number_of_lane"] = i
    env.reset()
    env.render()
    t.sleep(2)




#for _ in range(150):
    #action = env.action_type.actions_all["IDLE"]
    #obs, reward, done, info = env.step(action)
    #env.render()

#plt.imshow(env.render(mode="rgb_array"))

