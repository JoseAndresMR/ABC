from .attention_field import AttentionField
from .neuron import Neuron
from torch.utils.tensorboard import SummaryWriter
import copy
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import json
import os
import seaborn as sns
sns.set_theme()


class Brain(object):
    """ Creation, interconnection and management of neurons. """

    def __init__(self, config: dict, log_path: str):
        """ Initialization of neurons, attention field, and data management. 

        Args:
            config (dict): Configuration of the brain. Contents of each env:
                - neurons (dict): define how each neuron is
                    - type (dict): type of neurons to be defined
                        - neurons (list): Not for intern neurons. Definition of each neurons
                            - agent (dict): agent that defines the neuron
                                - type (string): type of RL agent
                                - additional_dim (list/int): input or output dim, or both depending on type.
                                - models (dict): neural network model in the predefined models
                                    - actor (string)
                                    - critic (string)
                        - quantity (dict): Just for intern neurons.
                            - quantity (int): number of these neurons, all equal.
                            - Same definition as above
                - attention_field (dict): 
                    - key_dim (int): keys and queries length
                    - value_dim (int): value length
                    - reward_backprop_thr (int): minimum reward to stop its backpropagation
            log_path (string): Path on disk to store gathered information about the experience
        """

        self.neurons = {"all": [], "sensory-motor": [],
                        "sensory": [], "intern": [], "motor": []}
        self.config = config
        self.log_path = os.path.join(log_path, "brain")
        self.k_dim, self.v_dim = self.config["attention_field"][
            "key_dim"], self.config["attention_field"]["value_dim"]
        self.attention_field = AttentionField(self.k_dim, self.v_dim)
        self.spawn_neurons()
        self.forward_step = 0
        self.tensorboard_writer = SummaryWriter(self.log_path)

        self.scores_deque = deque(maxlen=1000)
        self.scores = []
        self.performance = -99999
        self.start_plots()

    def spawn_neurons(self):
        """ Batch initialisation of all kinds of neurons. TODO: dynamically spawn during the development of the experience. """
        print("Brain: Spawning neurons")
        for neuron_type, type_neuron_config in self.config["neurons"].items():
            if neuron_type == "sensory-motor" or neuron_type == "sensory" or neuron_type == "motor":
                for neuron_config in type_neuron_config["neurons"]:
                    neuron_config["ID"] = len(self.neurons["all"]) + 1
                    self.spawn_one_neuron(neuron_type, neuron_config, self.k_dim,
                                          self.v_dim, neuron_config["agent"]["additional_dim"])

            elif neuron_type == "intern":
                for _ in range(type_neuron_config["quantity"]):
                    neuron_config = copy.deepcopy(type_neuron_config)
                    neuron_config.pop("quantity")
                    neuron_config["ID"] = len(self.neurons["all"]) + 1
                    self.spawn_one_neuron(
                        neuron_type, neuron_config, self.k_dim, self.v_dim)

    def spawn_one_neuron(self, neuron_type, config, k_dim, v_dim, additional_dim=None):
        """ Creation of a neuron and its inclussion in the management objects. """
        empty_neuron = {"neuron": None, "state": [], "next_state": [
        ], "action": [], "reward": [], "attended": [], "info": {"type": ""}}
        neuron = deepcopy(empty_neuron)
        neuron["neuron"] = Neuron(
            neuron_type, config, self.log_path, k_dim, v_dim, additional_dim)
        neuron["info"]["type"] = neuron_type
        self.neurons["all"].append(neuron)
        self.neurons[neuron_type].append(neuron)

    def forward(self):
        """ Composed and structured learning from prior steps and forward propagation of current step.
        The logical sequence follows sensory, intern, motor and sensory-motor agents.
        All of them follow a first learning from pior steps, later propagation of current step.
        First attention us run sensory -> intern and later (sensory, intern) -> motor. """

        if self.forward_step > 0:
            [neuron["neuron"].backprop() for neuron in self.neurons["sensory"]]
        [neuron["neuron"].forward() for neuron in self.neurons["sensory"]]
        if len(self.neurons["intern"]) > 0:
            self.run_attention_field_step(1)
            if self.forward_step > 0:
                [neuron["neuron"].backprop()
                 for neuron in self.neurons["intern"]]
            [neuron["neuron"].forward() for neuron in self.neurons["intern"]]
        if len(self.neurons["motor"]) > 0:
            self.run_attention_field_step(2)
        if self.forward_step > 0:
            [neuron["neuron"].backprop()
             for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]
        for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]:
            neuron["neuron"].forward()
            neuron["action"] = neuron["neuron"].output_value
        self.forward_step += 1
        if self.forward_step % 50 == 0:
            self.make_plots()

    def run_attention_field_step(self, stage: int):
        """ Compute attention weights in two possible stages.
        First attention us run sensory -> intern and later (sensory, intern) -> motor. """

        if stage == 1 or stage == 2:
            for neuron in self.neurons["sensory"]:
                self.attention_field.add_entries(
                    None, neuron["neuron"].key, neuron["neuron"].output_value)
        if stage == 1:
            for neuron in self.neurons["intern"]:
                self.attention_field.add_entries(neuron["neuron"].query,
                                                 neuron["neuron"].key,
                                                 neuron["neuron"].output_value)
        elif stage == 2:
            for neuron in self.neurons["intern"]:
                self.attention_field.add_entries(None,
                                                 neuron["neuron"].key,
                                                 neuron["neuron"].output_value)
        if stage == 2:
            for neuron in self.neurons["motor"]:
                self.attention_field.add_entries(
                    neuron["neuron"].query, None, None)

        values, attended = self.attention_field.run_step()
        if stage == 1:
            neurons = self.neurons["intern"]
        elif stage == 2:
            neurons = self.neurons["motor"]
        for i, value in enumerate(values):
            neurons[i]["neuron"].set_next_input_value(np.array([value]))
            neurons[i]["neuron"].attended = attended[i]

    def set_state_and_reward(self):
        """ Transports the state and reward information from the information object in this class to inside each neurons' class
        and to the overall performace storage. """
        [neuron["neuron"].set_next_input_value(
            neuron["state"]) for neuron in self.neurons["sensory-motor"] + self.neurons["sensory"]]
        [self.allocate_reward(np.array(neuron["reward"]).mean(), neuron["neuron"].attended)
         for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]
        [neuron["neuron"].set_reward(neuron["reward"])
         for neuron in self.neurons["sensory"]]
        [neuron["neuron"].set_reward(neuron["reward"])
         for neuron in self.neurons["intern"]]
        [neuron["neuron"].set_reward(np.array(neuron["reward"]).mean(
        )) for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]

        self.scores_deque.append(np.array([np.array(neuron["reward"]).mean(
        ) for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]).sum())
        self.scores.append(np.array([np.array(neuron["reward"]).mean(
        ) for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]).sum())
        self.performance = np.mean(self.scores_deque)

        for neuron in self.neurons["all"]:
            neuron["reward"] = []

    def allocate_reward(self, reward, attendeds):
        """ Split or backpropagate the reward given the attention weights each agent used to definde its state. """
        split_rewards = np.array(attendeds) * reward
        neurons = self.neurons["sensory"] + self.neurons["intern"]

        for i, split_reward in enumerate(split_rewards):
            if abs(split_reward) > self.config["attention_field"]["reward_backprop_thr"]:
                if len(neurons[i]["reward"]) == 0:
                    neurons[i]["reward"] = [split_reward]
                else:
                    neurons[i]["reward"] += split_reward
                if neurons[i]["neuron"].neuron_type != "sensory":
                    self.allocate_reward(
                        split_reward, neurons[i]["neuron"].attended)

    def get_performance(self):
        return self.performance

    def start_plots(self):
        ### Attention heatmap
        self.attention_heatmap_fig = plt.figure()
        self.attention_heatmap_ax = self.attention_heatmap_fig.add_subplot(111)
        self.attention_heatmap_ax.set_xlabel('AttendeDs: Sensory (1-{}) and Intern ({}-{})'.format(len(self.neurons["sensory"]),
                                                                        len(self.neurons["sensory"])+1,
                                                                        len(self.neurons["all"])-len(self.neurons["motor"])))
        self.attention_heatmap_ax.set_ylabel('AttendeRs: Intern ({}-{}) and Motor ({}-{})'.format(len(self.neurons["sensory"])+1,
                                                                        len(self.neurons["all"])-len(self.neurons["motor"]),
                                                                        len(self.neurons["all"])-len(self.neurons["motor"])+1,
                                                                        len(self.neurons["all"])))
        self.attention_heatmap_ax.set_title("Brain: Full Attention field")

        ### Neuron rewards
        self.neuron_reward_fig = plt.figure()
        self.neuron_reward_ax = self.neuron_reward_fig.add_subplot(111)

    def make_plots(self):
        """ Different visualizations about performance and attention. TODO: Add attention heatmap and 3D plot to tensorboard """
        plt.clf()
        # Reward scalar
        self.tensorboard_writer.add_scalar('avg_score',
                                           np.mean(self.scores_deque),
                                           self.forward_step)
        neuron_rewards_df = pd.DataFrame([neuron["reward"] for neuron in self.neurons["all"]])

        sns.heatmap( data = neuron_rewards_df, vmin=0, vmax=1.0)
        # fig, ax = plt.subplots()
        attention_heatmap_df = pd.DataFrame(data=np.array([neuron["neuron"].attended for neuron in self.neurons["intern"] + self.neurons["motor"]]),
                            index=range(len(self.neurons["sensory"])+1,len(self.neurons["all"])+1),
                            columns=range(1,len(self.neurons["all"])-len(self.neurons["motor"])+1))
        # sns.heatmap( data = pd.DataFrame(), ax = self.ax, vmin=0, vmax=1.0)
        sns.heatmap( data = attention_heatmap_df, vmin=0, vmax=1.0)
        plt.pause(0.1)
        self.attention_heatmap_fig.canvas.draw()
        # plt.savefig('Attention.png')

        # 3D Attention Field. TODO: to tensorboard
        # plt.rcParams["legend.fontsize"] = 10
        # fig = plt.figure()
        # ax = fig.gca(projection="3d")
        # ax.set_xlim3d(-0.2, 1.2)
        # ax.set_ylim3d(-0.2, 1.2)
        # ax.set_zlim3d(-0.2, 1.2)

        # for neuron in self.neurons["sensory"]:
        #     key = neuron["neuron"].key[0]
        #     x = list(zip(np.zeros(self.config["attention_field"]["key_dim"]), key))
        #     ax.plot(x[0], x[1], x[2], color = "black")
        #     ax.scatter3D(key[0], key[1], key[2], marker = "o", color = "green")

        # other_neurons = self.neurons["sensory"] + self.neurons["intern"]
        # for neuron in self.neurons["intern"]:
        #     key = neuron["neuron"].key[0]
        #     query = neuron["neuron"].query[0]
        #     x = list(zip(key, query))
        #     ax.plot(x[0], x[1], x[2], color = "black")
        #     ax.scatter3D(key[0], key[1], key[2], marker = "o", color = "green")
        #     ax.scatter3D(query[0], query[1], query[2], marker = "o", color = "red")

        #     for i, attention in enumerate(neuron["neuron"].attended):
        #         if attention > 0.3:
        #             key = other_neurons[i]["neuron"].key[0]
        #             x = list(zip(key, query))
        #             ax.plot(x[0], x[1], x[2], color = "blue")

        # for neuron in self.neurons["motor"]:
        #     query = neuron["neuron"].query[0]
        #     x = list(zip(query, np.ones(self.config["attention_field"]["key_dim"])))
        #     ax.plot(x[0], x[1], x[2], color = "black")
        #     ax.scatter3D(query[0], query[1], query[2], marker = "o", color = "red")

        #     for i, attention in enumerate(neuron["neuron"].attended):
        #         if attention > 0.3:
        #             key = other_neurons[i]["neuron"].key[0]
        #             x = list(zip(key, query))
        #             ax.plot(x[0], x[1], x[2], color = "blue")

        # ax.legend()
        # plt.title("Brain: 3D Attention Field")
        # plt.savefig('3D attention field.png')
