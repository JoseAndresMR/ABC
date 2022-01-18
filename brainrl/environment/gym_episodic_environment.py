import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from .environment import Environment


class GymEpisodicEnvironment(Environment):
    """ OpenAI Gym, episodic environment. """

    def __init__(self, id, name, log_path):
        super().__init__(id, log_path)
        """ Creates the Gym environment. Gathers Gym specific information structure and presents it as a common ABC frame.

        Args:
            id (string): unique identification in the whole experience.
            name (string): name of the environment for Gym.
            log_path (string): Path on disk to store gathered information about the experience  
        """

        self.env = gym.make(name)
        observations = self.env.reset()
        state_type = type(self.env.observation_space)
        if state_type == gym.spaces.box.Box:
            state_size = self.env.observation_space.shape
        elif state_type == gym.spaces.discrete.Discrete:
            state_size = self.env.observation_space.n

        action_type = type(self.env.action_space)
        if action_type == gym.spaces.box.Box:
            action_size = self.env.action_space.shape
        elif action_type == gym.spaces.discrete.Discrete:
            action_size = self.env.action_space.n

        print("Environment: Starting Gym Environment called {}".format(name))
        print("Environment: State type: {}. State size: {}".format(state_type, state_size))
        print("Environment: Action type: {}. Action size: {}".format(action_type, action_size))

        self.env_info = {"num_agents" : 1,
                         "state_type": state_type,
                         "state_size": state_size,
                         "action_size": action_size,
                         "action_type": action_type}

    def start_episodes(self, n_episodes=1000, max_t=3000, success_avg=30, print_every=50):
        """
        Start a new stack of episodes.

        Args:
            n_episodes (int): max number of episodes before the stack finishes.
            max_t (int): max number of timesteps befor the stack finishes.
            success_avg (int): boundary that determines when the performance is good enough and the stack finishes.
            print_every (int): number of episodes skipped before the log is updated.
        """

        self.n_episodes, self.max_t, self.print_every, self.success_avg = \
            n_episodes, max_t, print_every, success_avg
        self.current_episode, self.current_t = 0, 0

        self.scores_deque = deque(maxlen=self.print_every)
        self.scores = []
        self.states = np.array([self.env.reset()])
        # the scores of an episode for each of the 20 reachers
        self.e_scores = np.zeros(1)
        return self.states

    def step(self):
        """ Apply the chosen actions in the environment, then receive the reward and next observations.
        When the episode finishes, check if the performance is good enough and if log is required.

        Returns:
            rewards (list of ints): Used to measure the performance and hence learning.
            observations (list np.arrays): Next state observed by the agent.
            dones (list of bools): Wether the current episode is already finished or not.
            env_finished (list of bools): Wehter the environment is solved or max episodes reached. 
        """

        episode_finished = False
        env_finished = False
        if self.current_t < self.max_t:
            if self.env_info["action_type"] == gym.spaces.discrete.Discrete:
                self.actions = np.argmax(self.actions)
            observation, reward, done, info = self.env.step(self.actions[0])

            if self.render_flag:
                self.env.render()
            self.states = np.array([observation])
            self.e_scores += [reward]
            self.current_t += 1
            if done:
                self.state = None
                episode_finished = True
        else:
            episode_finished = True

        if episode_finished:
            self.states = np.array(self.env.reset())
            self.current_episode += 1
            self.current_t = 0

            # the average score of the agents
            avg_score = np.mean(self.e_scores)
            self.scores_deque.append(avg_score)
            self.scores.append(avg_score)
            print('\rEpisode {:d}\tscore: {:.2f}\taverage score over the last 10 episodes: {:.2f}'.format(
                self.current_episode, self.scores_deque[-1], np.mean(list(self.scores_deque)[-10:])))
            if self.current_episode > 10 and np.mean(self.scores_deque) > self.success_avg:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    self.current_episode-100, np.mean(self.scores_deque)))
                env_finished = True
            if self.current_episode % self.print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                    self.current_episode, np.mean(self.scores_deque)))
                self.tensorboard_writer.add_scalar('scores',
                                                   self.scores_deque[-1],
                                                   self.current_episode)
            self.e_scores = np.zeros(1)

        env_finished = env_finished or self.current_episode == self.n_episodes + 1
        return ([reward], np.array([observation]), [done], env_finished)

    def finish_environment(self):
        """ Close the environment to free memory and computation. """
        self.env.close()
