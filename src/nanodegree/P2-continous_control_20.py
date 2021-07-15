import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

from unityagents import UnityEnvironment
import numpy as np

# select this option to load version 1 (with a single agent) of the environment
#env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')

# select this option to load version 2 (with 20 agents) of the environment
env = UnityEnvironment(file_name='/home/jamillan4i/Libraries/ABC/bin/environments/Reacher_20/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# create a new agent
agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size, random_seed=2)

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

def ddpg(n_episodes=1000, max_t=3000, print_every=25):
    scores_deque = deque(maxlen=print_every)
    scores = []
    beta = 0.1  # factor the random noise gets multiplied with
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get the current states
        #print("states shape", states.shape)
        e_scores = np.zeros(20)  # the scores of an episode for each of the 20 reachers
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states, noise_factor=beta)# let the agent select actions
            #print("actions shape", actions.shape)
            env_info = env.step(actions)[brain_name]      # execute the selected actions and save the new information about the environment
            rewards = env_info.rewards                    # get the rewards
            next_states = env_info.vector_observations    # get the resulting states
            dones = env_info.local_done                   # check whether episodes have finished
            agent.step(states, actions, rewards, next_states, dones)  # pass the information to the agent
            states = next_states
            e_scores += rewards
            if np.any(dones):
                break 
        avg_score = np.mean(e_scores)  # the average score of the agents
        scores_deque.append(avg_score)
        scores.append(avg_score)
        beta = max(0.995 * beta, 0.01)  # reduce the noise a bit while training
        print('\rEpisode {:d}\tscore: {:.2f}\taverage score over the last 10 episodes: {:.2f}'.format(i_episode, scores_deque[-1], np.mean(list(scores_deque)[-10:])), end="")
        if i_episode > 100 and np.mean(scores_deque) > 30:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            #plt.plot(np.arange(1, len(scores)+1), scores)
            #plt.ylabel('Score')
            #plt.xlabel('Episode #')
            #plt.show()
            
    return scores

scores = ddpg()
env.close()  # close the environment as it is no longer needed

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()