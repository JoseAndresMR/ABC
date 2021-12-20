from logging import log
import numpy as np
import os, copy
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from brain.rl_agents.DdpgAgent import DdpgAgent
from brain.rl_agents.DQNAgent import DQNAgent

class Neuron(object):
    """ Wrapping of the RL agent that defines each neuron. Interface to the Brain architecture. """
    
    def __init__(self, neuron_type, config, log_path, k_dim, v_dim, environment_signal_size = None):
        """ Definition and initilialisation of the structure of the nueron.
        
        Args:
            neuron_type (string): Wether sensory, intern, motor or sensory-motor.
            config (dict): Definition of the RL agent at the core of the nueron.
            log_path (string): Path on disk to store gathered information about the experience
            k_dim (int): Keys and queries length
            v_dim (int): Value length
            environment_signal_size (): Input or output dim, or both depending on type.
        """

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
        """Define state, action, key, query and value sizes depeding on neuron type.
        Create the object of the RL agent. """
        
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
        """ Receive next state from the Brain.
        
        Args:
            state (np.array): Observations taken by the agent on the Environment. """

        self.step += 1
        if not isinstance(self.state, np.array):
            self.state = copy.deepcopy(state)
        else:
            self.next_state = copy.deepcopy(state)

    def setReward(self, reward):
        """ Receive reward from the Brain
        
        Args:
            reward (int): Reward taken by the agent from the Environment. """

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
        """ Make the RL agent to apply its policy given the current state and generate an action.

        Results:
            action (np.array): Selected action. """
        self.action = self.rl_agent.act(self.state)
        self.decomposeAction()
        return self.action

    def backprop(self):
        """ Apply the learning step and actualize state. """

        if not isinstance(self.next_state, np.array):
            done = True
        else:
            done = False
        self.rl_agent.step(self.state, self.action, [self.reward], self.next_state, [done])
        self.state = self.next_state
        self.next_state = None

    def decomposeAction(self):
        """ Split the outcome of the RL agent into the pieces of the attention mechanism. """

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