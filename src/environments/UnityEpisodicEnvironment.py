from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from environments.Environment import Environment

class UnityEpisodicEnvironment(Environment):
    def __init__(self, file_path, id):
        super().__init__(id)
        # self.file_path = os.path.join(os.path.dirname(__file__), "..","..","bin",file_path)
        self.env = UnityEnvironment(file_name=file_path, no_graphics=True)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents)
        action_size = brain.vector_action_space_size
        states = env_info.vector_observations
        state_size = states.shape[1]
        self.env_info = {"num_agents" : num_agents, "state_size" : state_size, "action_size" : action_size}
        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)

    def startEpisodes(self, n_episodes=1000, max_t=3000, success_avg = 30, print_every=3):
        self.n_episodes, self.max_t, self.print_every, self.success_avg = n_episodes, max_t, print_every, success_avg
        self.current_episode, self.current_t = 0, 0

        self.scores_deque = deque(maxlen=self.print_every)
        self.scores = []
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.states = env_info.vector_observations  # get the current states
        self.e_scores = np.zeros(1)  # the scores of an episode for each of the 20 reachers
        return self.states

    def step(self):
        episode_finished = False
        if self.current_t < self.max_t:
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
            self.current_episode += 1
            self.current_t = 0

            avg_score = np.mean(self.e_scores)  # the average score of the agents
            self.scores_deque.append(avg_score)
            self.scores.append(avg_score)
            print('\rEpisode {:d}\tscore: {:.2f}\taverage score over the last 10 episodes: {:.2f}'.format(self.current_episode, self.scores_deque[-1], np.mean(list(self.scores_deque)[-10:])), end="")
            if self.current_episode > 100 and np.mean(self.scores_deque) > self.success_avg:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(self.current_episode-100, np.mean(self.scores_deque)))
                env_finished = True
            if self.current_episode % self.print_every == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(self.current_episode, np.mean(self.scores_deque)))
                # plt.figure()
                plt.plot(np.arange(1, len(self.scores)+1), self.scores)
                plt.title("Ennvironment: score")
                # plt.ylabel('Score')
                # plt.xlabel('Episode #')
                # plt.draw()
                # plt.ioff()
                # plt.show()
                # self.fig.show()
                # self.f_d.set_data(np.arange(1, len(self.scores)+1), self.scores)
                self.tensorboard_writer.add_scalar('scores',
                                                self.scores_deque[-1],
                                                self.current_episode)
            self.e_scores = np.zeros(20)  # the scores of an episode for each of the 20 reachers

        env_finished = self.current_episode == self.n_episodes + 1
        return (rewards, next_states, dones, env_finished)

    def finishEnvironment(self):
        self.env.close()  # close the environment as it is no longer needed
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(self.scores)+1), self.scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
