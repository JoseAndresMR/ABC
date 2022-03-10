import gym
import os
import shutil

shutil.rmtree('renders',
              ignore_errors=True)
environment = 'Pendulum-v1'
env = gym.make(environment)
os.makedirs(os.path.join('renders',
                         environment),
            exist_ok=True)
env = gym.wrappers.Monitor(env=env,
                           directory=os.path.join('renders',
                                                  environment),
                           force=True)
observation = env.reset()
for t in range(1000):
    # env.render()
    # print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # env.render()
    # env.enabled = True
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break

env.close()
