import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
img = env.render(mode='rgb_array')
print(img)
