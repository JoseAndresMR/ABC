import numpy as np
import os
import random
import json
from copy import deepcopy

from brainrl.brain.util.nn_model import NnModel
from brainrl.brain.util.ou_noise import OUNoise
from brainrl.brain.util.replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Selected device: ", device)

class DdpgAgent(object):
    """RL agent of Deep Deterministic Policy Gradient type."""

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
        self.metaparams = self.config["definition"]["metaparameters"]
        self.log_path = log_path
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.learn_step = 0
        sizes = {"state" : state_size, "action" : action_size}

        # Actor Network (w/ Target Network)
        self.actor_local = NnModel(deepcopy(config["models"]["actor"]), sizes, random_seed).to(device)
        self.actor_target = NnModel(deepcopy(config["models"]["actor"]), sizes, random_seed).to(device)
        fake_inputs = {"state" : torch.randn(1, state_size).to(device), "action" : torch.randn(1, action_size).to(device)}
        writer = SummaryWriter(os.path.join(self.log_path,"neurons", "{}".format(config["ID"]), "actor_graph"))
        writer.add_graph(self.actor_local, fake_inputs)
        writer.close()
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.metaparams["lr_actor"])

        # Critic Network (w/ Target Network)
        self.critic_local = NnModel(deepcopy(config["models"]["critic"]), sizes, random_seed).to(device)
        self.critic_target = NnModel(deepcopy(config["models"]["critic"]), sizes, random_seed).to(device)
        writer = SummaryWriter(os.path.join(self.log_path,"neurons", "{}".format(config["ID"]), "critic_graph"))
        writer.add_graph(self.critic_local, fake_inputs)
        writer.close()
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.metaparams["lr_critic"])

        self.copy_weights(self.critic_local, self.critic_target)
        self.copy_weights(self.actor_local, self.actor_target)

        # Noise process
        self.noise = OUNoise(1 * action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.metaparams["buffer_size"], self.metaparams["batch_size"], random_seed)   
        

    def copy_weights(self, source, target):
        """Copies the weights from the source to the target.
        
        Args:
            source (NnModel): model where the weights are copied from. 
            target (NnModel): model where the weights are copied to. """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.
        
        Args:
            states (np.array): Current observations on the environment.
            actions (np.array): Action already selected by the agent given State.
            rewards (np.array): Reward received from the Environment when taken the action.
            next_states (np.array): Next observations on the environment.
            dones (list of bools): Wether episode has finished in this step or not. """
        # Save experiences / rewards
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.learn_step = (self.learn_step + 1) % self.metaparams["learn_every"]
        # Learn every LEARN_EVERY steps if enough samples are available in memory
        if len(self.memory) > self.metaparams["batch_size"] and self.learn_step == 0:
            for _ in range(self.metaparams["learn_steps"]):
                experiences = self.memory.sample()
                self.learn(experiences, self.metaparams["gamma"])

    def act(self, state, add_noise=False, noise_factor=1.0):
        """Returns actions for given state as per current policy.
        
        Args:
            state (np.array): Current observation from the Environment.
            add_noise (bool): Wether to apply noise over the selected action or not. 
            noise_factor (float): Measure of the influence of the added noise.
            
        Returns:
            action (np.array): Selected action. """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            inputs = {"state" : state}
            action = self.actor_local(inputs).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += noise_factor * self.noise.sample()
        return action

    def reset(self):
        """ Restart the seed of the noise. """
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Args:
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples.
            gamma (float): Discount factor.
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
        self.soft_update(self.critic_local, self.critic_target, self.metaparams["tau"])
        self.soft_update(self.actor_local, self.actor_target, self.metaparams["tau"])

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Args:
            local_model: PyTorch model (weights will be copied from).
            target_model: PyTorch model (weights will be copied to).
            tau (float): Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)