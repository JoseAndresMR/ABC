from collections import namedtuple, deque
import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, PRIORITIZED_ER_a):
        """Initialize the parameters.

        Args:
            action_size (int): Dimension of the actions.
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "td"])
        self.seed = random.seed(seed)
        self.PRIORITIZED_ER_a = PRIORITIZED_ER_a
    
    def add(self, state, action, reward, next_state, done, td):
        """Add a new experience to memory.
        Args:
            states (np.array): Current observations on the environment.
            actions (np.array): Action already selected by the agent given State.
            rewards (np.array): Reward received from the Environment when taken the action.
            next_states (np.array): Next observations on the environment.
            dones (list of bools): Wether episode has finished in this step or not. """
        e = self.experience(state, action, reward, next_state, done, td)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.
        Returns:
            states (np.array): Current observations on the environment.
            actions (np.array): Action already selected by the agent given State.
            rewards (np.array): Reward received from the Environment when taken the action.
            next_states (np.array): Next observations on the environment.
            dones (list of bools): Wether episode has finished in this step or not. """
        probs = np.array([abs(e.td) for e in self.memory if e is not None], dtype=np.float)
        probs = probs**self.PRIORITIZED_ER_a / sum(probs**self.PRIORITIZED_ER_a)
        #probs = np.ones(len(self.memory))/len(self.memory)
        chosen_indexes = random.choices(range(len(self.memory)), k=self.batch_size, weights=probs)
        experiences = [self.memory[i] for i in chosen_indexes]
        probs = probs[chosen_indexes]
        experiences = random.sample(self.memory, k=self.batch_size)

        probs = torch.from_numpy(probs).float().to(device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, probs)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)