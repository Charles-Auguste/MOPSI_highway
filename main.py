import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make('mopsi-env-v0')
env.reset()

while True:
    env.render()




# for _ in range(150):
#     action = env.action_type.actions_all["IDLE"]
#     obs, reward, done, info = env.step(action)
#     env.render()

# plt.imshow(env.render(mode="rgb_array"))
