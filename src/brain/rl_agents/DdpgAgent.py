import numpy as np
import os
import random
import copy
from copy import deepcopy

from brain.utils.NnModel import NnModel
from brain.utils.OUNoise import OUNoise
from brain.utils.ReplayBuffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.995            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
LEARN_EVERY = 32         # learn every LEARN_EVERY steps
LEARN_STEPS = 16            # how often to execute the learn-function each LEARN_EVERY steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected device: ", device)


class DdpgAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, config, state_size, action_size, random_seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.learn_step = 0  # for learning every n steps
        sizes = {"state" : state_size, "action" : action_size}

        self.log_path = os.path.join(os.path.dirname(__file__),'..','..','..',"data/runs/experiments")

        # Actor Network (w/ Target Network)
        self.actor_local = NnModel(deepcopy(config["models"]["actor"]), sizes, random_seed).to(device)
        self.actor_target = NnModel(deepcopy(config["models"]["actor"]), sizes, random_seed).to(device)
        fake_inputs = {"state" : torch.randn(1, state_size).to(device), "action" : torch.randn(1, action_size).to(device)}
        writer = SummaryWriter(os.path.join(self.log_path,"neurons", "{}".format(config["ID"]), "actor_graph"))
        writer.add_graph(self.actor_local, fake_inputs)
        writer.close()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = NnModel(deepcopy(config["models"]["critic"]), sizes, random_seed).to(device)
        self.critic_target = NnModel(deepcopy(config["models"]["critic"]), sizes, random_seed).to(device)
        writer = SummaryWriter(os.path.join(self.log_path,"neurons", "{}".format(config["ID"]), "critic_graph"))
        writer.add_graph(self.critic_local, fake_inputs)
        writer.close()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        self.copy_weights(self.critic_local, self.critic_target)
        self.copy_weights(self.actor_local, self.actor_target)

        # Noise process
        self.noise = OUNoise(1 * action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)   
        

    def copy_weights(self, source, target):
        """Copies the weights from the source to the target"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experiences / rewards
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.learn_step = (self.learn_step + 1) % LEARN_EVERY
        # Learn every LEARN_EVERY steps if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self.learn_step == 0:
            for _ in range(LEARN_STEPS):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, noise_factor=1.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            inputs = {"state" : state}
            action = self.actor_local(inputs).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += noise_factor * self.noise.sample()
        # return np.clip(action, -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target({"state" : next_states})
        Q_targets_next = self.critic_target({"state" : next_states, "action" : actions_next})
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Current expected Q-values
        Q_expected = self.critic_local({"state" : states, "action" : actions})
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets[:,:1])
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local({"state" : states})
        actor_loss = -self.critic_local({"state" : states, "action" : actions_pred}).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)