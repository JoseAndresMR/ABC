import gym
env = gym.make('CartPole-v0')
env.reset()
img = env.render(mode='rgb_array')
print(img)
