from mlagents_envs.environment import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from .environment import Environment

class UnityEpisodicEnvironment(Environment):
    """ OpenAI Unity, episodic environment. """
    
    def __init__(self, file_path, id, log_path):
        super().__init__(id, log_path)
        """ Creates the Unity environment. Gathers Unity specific information structure and presents it as a common ABC frame.

        Args:
            file_path (string): path to the binary in memory of disk.
            id (string): unique identification in the whole experience.
            log_path (string): Path on disk to store gathered information about the experience  
        """

        self.env = UnityEnvironment(file_name=file_path, no_graphics=True)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        states = env_info.vector_observations
        state_size = states.shape[1]
        state_size = brain.vector_state_space_size
        state_type = brain.vector_state_space_type
        action_size = brain.vector_action_space_size
        action_type = brain.vector_action_space_type

        print("Environment: Starting Unity Environment called {}".format(id))
        print("Environment: State type: {}. State size: {}".format(state_type, state_size))
        print("Environment: State type: {}. State size: {}".format(action_type, action_size))
        self.env_info = {"num_agents" : self.num_agents, "state_size" : state_size, "state_type" : state_type, "action_size" : action_size, "action_type" : action_type}

    def start_episodes(self, n_episodes=10000, max_t=3000, success_avg = 30, print_every=3):
        """
        Start a new stack of episodes.
        
        Args:
            n_episodes (int): max number of episodes before the stack finishes.
            max_t (int): max number of timesteps befor the stack finishes.
            success_avg (int): boundary that determines when the performance is good enough and the stack finishes.
            print_every (int): number of episodes skipped before the log is updated.
        """

        self.n_episodes, self.max_t, self.print_every, self.success_avg = n_episodes, max_t, print_every, success_avg
        self.current_episode, self.current_t = 0, 0

        self.scores_deque = deque(maxlen=self.print_every)
        self.scores = []
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.states = env_info.vector_observations  # get the current states
        self.e_scores = np.zeros(self.num_agents)  # the scores of an episode for each of the 20 reachers
        return self.states

    def step(self):
        """ Apply the chosen actions in the enviornemnt, then receive the reward and next observations.
        When the episode finishes, check if the performance is good enough and if log is required.
        
        Returns:
            rewards (list of ints): Used to measure the performance and hence learning.
            next_states (list np.arrays): Next observation of the agent.
            dones (list of bools): Wether the current episode is already finished or not.
            env_finished (list of bools): Wehter the environment is solved or max episodes reached. 
        """

        episode_finished = False
        if self.current_t < self.max_t:
            # if self.env_info["action_type"] == "discrete":
            #     self.actions = np.random.choice(np.arange(len(self.actions[0])),p=self.actions[0]) ### TODO: Fix when parallelization
            env_info = self.env.step(self.actions)[self.brain_name]      # execute the selected actions and save the new information about the environment
            rewards = env_info.rewards                    # get the rewards
            next_states = env_info.vector_observations    # get the resulting states
            dones = env_info.local_done                   # check whether episodes have finished
            self.states = next_states
            self.e_scores += rewards
            self.current_t += 1
            if np.any(dones):
                self.state = None
                episode_finished = True
        else:
            episode_finished = True
        
        if episode_finished:
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            self.states = env_info.vector_observations  # get the current states

            avg_score = np.mean(self.e_scores)  # the average score of the agents
            self.scores_deque.append(avg_score)
            self.scores.append(avg_score)
            print('\rEpisode {:d}\tscore: {:.2f}\taverage score over the last 10 episodes: {:.2f}'.format(self.current_episode, self.scores_deque[-1], np.mean(list(self.scores_deque)[-10:])), end="")
            if self.current_episode > 100 and np.mean(self.scores_deque) > self.success_avg:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(self.current_episode-100, np.mean(self.scores_deque)))
                env_finished = True
            if self.current_episode % self.print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.current_episode, np.mean(self.scores_deque)))
                self.tensorboard_writer.add_scalar('scores',
                                                self.scores_deque[-1],
                                                self.current_episode)
            self.e_scores = np.zeros(self.num_agents)  # the scores of an episode for each of the 20 reachers
            self.current_episode += 1
            self.current_t = 0
        env_finished = self.current_episode == self.n_episodes + 1
        return (rewards, next_states, dones, env_finished)

    def finish_environment(self):
        """ Close the environment to free memory and computation. """
        self.env.close()  # close the environment as it is no longer needed
