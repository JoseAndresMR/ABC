from logging import log
import numpy as np
import os
import copy
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from .rl_agent.ddpg_agent import DDPGAgent
from .rl_agent.dqn_agent import DQNAgent


class Neuron(object):
    """ Wrapping of the RL agent that defines each neuron. Interface to the Brain architecture. """

    neuron_types = ["sensory-motor",
                    "sensory",
                    "intern",
                    "motor"]

    def __init__(self, neuron_type: str,
                 config: dict,
                 log_path: str,
                 k_dim: int,
                 v_dim: int,
                 environment_signal_size=None):
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
        self.tensorboard_writer = SummaryWriter(os.path.join(
            log_path, "neurons", "{}".format(self.config["ID"])))
        self.log_every = 5000
        self.no_reward_penalty = 0

        self.build_rl_agent()

    def build_rl_agent(self):
        """Define state, action, key, query and value sizes depeding on neuron type.
        Create the object of the RL agent. """

        if self.neuron_type == "sensory-motor":
            self.state_size = self.environment_signal_size[0]
            self.action_size = self.environment_signal_size[1]
            self.key = None

        elif self.neuron_type == "sensory":
            self.state_size = self.environment_signal_size
            self.action_size = self.k_dim + self.v_dim
            self.key = np.random.rand(1, self.k_dim)

        elif self.neuron_type == "intern":
            self.state_size = self.v_dim
            self.action_size = self.k_dim*2 + self.v_dim
            self.query = np.random.rand(1, self.k_dim)
            self.key, self.output_value = None, None

        elif self.neuron_type == "motor":
            self.state_size = self.v_dim
            self.action_size = self.k_dim + self.environment_signal_size
            self.query = np.random.rand(1, self.k_dim)

        else:
            raise ValueError('neuron_type must be:', self.neuron_types)

        print("Neuron: Building a {} neuron with {} state size and {} action size".format(
            self.neuron_type, self.state_size, self.action_size))

        self.config["agent"]["ID"] = self.config["ID"]
        if self.config["agent"]["type"] == "DDPG":
            self.rl_agent = DDPGAgent(
                self.config["agent"], self.log_path, self.state_size, self.action_size, random_seed=2)
        elif self.config["agent"]["type"] == "DQN":
            self.rl_agent = DQNAgent(
                self.config["agent"], self.log_path, self.state_size, self.action_size, random_seed=2)
        else:
            raise ValueError('neuron_type must be DDPG or DQN')

    def set_next_input_value(self, state: np.ndarray):
        """ Receive next state from the Brain.

        Args:
            state (np.array): Observations taken by the agent on the Environment. """
        # Sometimes, list appears
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        self.step += 1
        if not isinstance(self.state, np.ndarray):
            self.state = copy.deepcopy(state)
        else:
            self.next_state = copy.deepcopy(state)

    def set_reward(self, reward: int):
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
        self.decompose_action()
        return self.action

    def backprop(self):
        """ Apply the learning step and actualize state. """

        if not isinstance(self.next_state, np.ndarray):
            done = True
        else:
            done = False
        self.rl_agent.step(self.state, self.action, [
                           self.reward], self.next_state, [done])
        self.state = self.next_state
        self.next_state = None

    def decompose_action(self):
        """ Split the outcome of the RL agent into the pieces of the attention mechanism. """

        if self.neuron_type == "sensory-motor":
            self.output_value = self.action
        elif self.neuron_type == "sensory":
            self.key = self.action[:, :self.k_dim]
            self.output_value = self.action[:, self.k_dim:]
        elif self.neuron_type == "intern":
            self.query = self.action[:, :self.k_dim]
            self.key = self.action[:, self.k_dim:self.k_dim*2]
            self.output_value = self.action[:, self.k_dim*2:]
        elif self.neuron_type == "motor":
            self.query = self.action[:, :self.k_dim]
            self.output_value = self.action[:, self.k_dim:]
