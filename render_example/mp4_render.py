import gym
from gym import wrappers
env_to_wrap = gym.make('CartPole-v0')
env = wrappers.Monitor(env=env_to_wrap,
                       directory='renders',
                       force=True)
observation = env.reset()

env.close()
env_to_wrap.close()
