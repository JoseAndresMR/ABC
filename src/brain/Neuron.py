from logging import log
import numpy as np
import os, copy
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from brain.rl_agents.DdpgAgent import DdpgAgent
from brain.rl_agents.DQNAgent import DQNAgent

class Neuron(object):

    def __init__(self, neuron_type, config, log_path, k_dim, v_dim, environment_signal_size = None):

        self.state, self.next_state, self.action, self.reward = None, None, None, None
        self.neuron_type = neuron_type
        self.log_path = log_path
        self.step = 0
        self.config = config
        self.k_dim, self. v_dim, self.environment_signal_size = k_dim, v_dim, environment_signal_size
        self.attended = []
        self.scores_deque = deque(maxlen=1000)
        self.scores = []
        self.tensorboard_writer = SummaryWriter(os.path.join(log_path,"neurons", "{}".format(self.config["ID"])))
        self.log_every = 5000
        self.no_reward_penalty = 0

        self.buildRlAgent()

    def buildRlAgent(self):
        if self.neuron_type == "sensory-motor":
            self.state_size = self.environment_signal_size[0]
            self.action_size = self.environment_signal_size[1]
            self.key = None

        elif self.neuron_type == "sensory":
            self.state_size = self.environment_signal_size
            self.action_size = self.k_dim + self.v_dim
            self.key = np.random.rand(1,self.k_dim)
            
        elif self.neuron_type == "intern":
            self.state_size = self.v_dim
            self.action_size = self.k_dim*2 + self.v_dim
            self.query = np.random.rand(1,self.k_dim)
            self.key, self.output_value = None, None

        elif self.neuron_type == "motor":
            self.state_size = self.v_dim
            self.action_size = self.k_dim + self.environment_signal_size
            self.query = np.random.rand(1,self.k_dim)

        print("Neuron: Building a {} neuron with {} state size and {} action size".format(self.neuron_type, self.state_size, self.action_size))

        self.config["agent"]["ID"] = self.config["ID"]
        if self.config["agent"]["type"] == "DDPG":
            self.rl_agent = DdpgAgent(self.config["agent"], self.log_path, self.state_size, self.action_size, random_seed = 2)
        if self.config["agent"]["type"] == "DQN":
            self.rl_agent = DQNAgent(self.config["agent"], self.log_path, self.state_size, self.action_size, random_seed = 2)

    def setNextInputValue(self, state):
        self.step += 1
        if type(self.state) != type(np.array(1)):
            self.state = copy.deepcopy(state)
        else:
            self.next_state = copy.deepcopy(state)

    def setReward(self, reward):
        if reward == []:
            reward = self.no_reward_penalty
        self.reward = reward
        self.scores_deque.append(reward)
        self.scores.append(reward)
        if self.step % self.log_every == 0:
            self.tensorboard_writer.add_scalar('avg_score',
                                    np.mean(self.scores_deque),
                                    self.step)

    def forward(self):
        self.action = self.rl_agent.act(self.state)
        self.decomposeAction()
        return self.action

    def backprop(self):
        if type(self.next_state) != type(np.array(1)):
            done = True
        else:
            done = False
        self.rl_agent.step(self.state, self.action, [self.reward], self.next_state, [done])
        self.state = self.next_state
        self.next_state = None

    def decomposeAction(self):
        if self.neuron_type == "sensory-motor":
            self.output_value = self.action
        elif self.neuron_type == "sensory":
            self.key = self.action[:,:self.k_dim]
            self.output_value = self.action[:,self.k_dim:]
        elif self.neuron_type == "intern":
            self.query = self.action[:,:self.k_dim]
            self.key = self.action[:,self.k_dim:self.k_dim*2]
            self.output_value = self.action[:,self.k_dim*2:]
        elif self.neuron_type == "motor":
            self.query = self.action[:,:self.k_dim]
            self.output_value = self.action[:,self.k_dim:]

    def makeRewardPlot(self):
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        plt.plot(np.arange(1, len(self.scores)+1), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Step #')
        plt.title("Neuron: Score")
        plt.show()
