from logging import log
import numpy as np
import os
import random
import copy
from collections import namedtuple, deque
from copy import deepcopy

from brainrl.brain.util.nn_model import NnModel
from brainrl.brain.util.prioritized_replay_buffer import PrioritizedReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRIORITIZED_ER_e = 5e-2 # added to td for the calculus of the probability
PRIORITIZED_ER_a = 0.4
PRIORITIZED_ER_b = 0.6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected device: ", device)


class DQNAgent(object):
    """RL agent of Deep Q-Network type."""

    def __init__(self, config, log_path, state_size, action_size, random_seed):
        """Initialize agent paramenters, models, and memory.

        Args:
            config (dict): configuration of parameters and model structure.
            log_path (string): Path on disk to store gathered information about the experience.
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            random_seed (int): Random seed.
        """
        self.config = config
        self.log_path = log_path
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.learn_step = 0  # for learning every n steps
        self.eps = copy.deepcopy(self.config["eps"][0])
        self.eps_decay = self.config["eps"][1]
        self.eps_end = self.config["eps"][2]
        sizes = {"state" : state_size, "action" : action_size}

        self.qnetwork_local = NnModel(deepcopy(config["models"]["actor"]), sizes, random_seed).to(device)
        self.qnetwork_target = NnModel(deepcopy(config["models"]["actor"]), sizes, random_seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.copy_weights(self.qnetwork_local, self.qnetwork_target)

        self.log_path = os.path.join(os.path.dirname(__file__),'..','..','..',"data/runs/experiments")
        fake_inputs = {"state" : torch.randn(1, state_size).to(device), "action" : torch.randn(1, action_size).to(device)}
        writer = SummaryWriter(os.path.join(self.log_path,"neurons", "{}".format(config["ID"]), "actor_graph"))
        writer.add_graph(self.qnetwork_local, fake_inputs)
        writer.close()

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, PRIORITIZED_ER_a)

    def copy_weights(self, source, target):
        """Copies the weights from the source to the target.
        
        Args:
            source (NnModel): model where the weights are copied from. 
            target (NnModel): model where the weights are copied to. """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Args:
            states (np.array): Current observations on the environment.
            actions (np.array): Action already selected by the agent given State.
            rewards (np.array): Reward received from the Environment when taken the action.
            next_states (np.array): Next observations on the environment.
            dones (list of bools): Wether episode has finished in this step or not. """
        # Save experience in replay memory
        td = self.td(state, action, reward, next_state, done) + PRIORITIZED_ER_e
        self.memory.add(state, action, reward, next_state, done, td)

        # Learn every UPDATE_EVERY time steps.
        self.learn_step = (self.learn_step + 1) % UPDATE_EVERY
        if self.learn_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Args:
            state (np.array): Current observation from the Environment.

        Returns:
            action (np.array): Selected action. """
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            inputs = {"state" : state}
            action_values = self.qnetwork_local(inputs).cpu().data.numpy()
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        self.eps = max(self.eps_end, self.eps_decay*self.eps)
        if random.random() > self.eps:
            return np.array([[np.argmax(action_values)]])
        else:
            return np.array([[random.choice(np.arange(self.action_size))]])

    def reset(self):
        """ Restart the seed of the noise. """
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Args:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): Discount factor.
        """
        states, actions, rewards, next_states, dones, probs = experiences

        # Double DQN
        input = {"state" : next_states}
        local_argmax_actions = self.qnetwork_local(input).detach().argmax(dim=1).unsqueeze(1)
        next_qvalues = self.qnetwork_target(input).gather(1,local_argmax_actions).detach()
        
        # Single DQN
        #next_qvalues = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        Q_targets = rewards + (gamma * next_qvalues * (1 - dones))

        input = {"state" : states}
        Q_expected = self.qnetwork_local(input).gather(1, actions)
        Q_targets = Q_expected + (Q_targets - Q_expected)*(1/(len(self.memory)*probs.unsqueeze(1)))**PRIORITIZED_ER_b 

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def td(self, state, action, reward, next_state, done):
        """ Temporal Difference error. 
        
        Args:
            states (np.array): Current observations on the environment.
            actions (np.array): Action already selected by the agent given State.
            rewards (np.array): Reward received from the Environment when taken the action.
            next_states (np.array): Next observations on the environment.
            dones (list of bools): Wether episode has finished in this step or not. """
        with torch.no_grad():
            # inputs = {"state" : torch.Tensor(state)}
            # action_values = self.qnetwork_local(inputs)
            # state = (torch.from_numpy(state)).float().to(device)
            self.qnetwork_target.eval()
            next_inputs = {"state" : (torch.from_numpy(next_state)).float().to(device)}
            # next_state = (torch.from_numpy(next_state)).float().to(device)
            q_target_next = self.qnetwork_target.forward(next_inputs).detach().max(1)
            Q_target = reward[0] + GAMMA*(q_target_next[0]) * (1-done[0])
            inputs = {"state" : (torch.from_numpy(state)).float().to(device)}
            self.qnetwork_local.eval()
            Q_expected = self.qnetwork_local.forward(inputs)#[0][action[0]]
            Q_expected = Q_expected[0]
            Q_expected = Q_expected[action[0]]
            td = float(Q_target) - float(Q_expected)

        return abs(td)
                      

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: PyTorch model (weights will be copied from).
            target_model: PyTorch model (weights will be copied to).
            tau (float): Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)