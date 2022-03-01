import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')
render = lambda : plt.imshow(env.render(mode='rgb_array'))
env.reset()
render()
