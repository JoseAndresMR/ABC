import numpy as np
from copy import deepcopy
import json
import os
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import copy
from torch.utils.tensorboard import SummaryWriter

from brain.Neuron import Neuron
from brain.AttentionField import AttentionField

class Brain(object):

    def __init__(self, config, log_path):

        self.neurons = {"all": [], "sensory-motor" : [], "sensory" : [], "intern" : [], "motor" : []}
        self.config = config
        self.log_path = os.path.join(log_path,"brain")
        self.k_dim, self.v_dim = self.config["attention_field"]["key_dim"], self.config["attention_field"]["value_dim"]
        self.attention_field = AttentionField(self.k_dim, self.v_dim, self.config["attention_field"]["reward_backprop_thr"])
        self.spawnNeurons()
        self.forward_step = 0
        self.tensorboard_writer = SummaryWriter(self.log_path)

        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(111)
        self.scores_deque = deque(maxlen=1000)
        self.scores = []

    def spawnNeurons(self): ### SIMPLIFY FUNCTION
        print("Brain: Spawning neurons")
        for neuron_type, type_neuron_config in self.config["neurons"].items():
            if neuron_type == "sensory-motor" or neuron_type == "sensory" or neuron_type == "motor":
                for neuron_config in type_neuron_config["neurons"]:
                    neuron_config["ID"] = len(self.neurons["all"]) + 1
                    self.spawnOneNeuron(neuron_type, neuron_config, self.k_dim, self.v_dim, neuron_config["agent"]["additional_dim"])

            elif neuron_type == "intern":
                for _ in range(type_neuron_config["quantity"]):
                    neuron_config = copy.deepcopy(type_neuron_config)
                    neuron_config.pop("quantity")
                    neuron_config["ID"] = len(self.neurons["all"]) + 1
                    self.spawnOneNeuron(neuron_type, neuron_config, self.k_dim, self.v_dim)

    def spawnOneNeuron(self, neuron_type, config, k_dim, v_dim, additional_dim = None):
        empty_neuron = {"neuron" : None, "state" : [], "next_state" : [], "action" : [], "reward" : [], "attended" : [], "info" : {"type" : ""}}
        neuron = deepcopy(empty_neuron)
        neuron["neuron"] = Neuron(neuron_type, config, self.log_path, k_dim, v_dim, additional_dim)
        neuron["info"]["type"] = neuron_type
        self.neurons["all"].append(neuron)
        self.neurons[neuron_type].append(neuron)

    def forward(self):
        # print("Brain: forward step {}".format(self.forward_step))
        if self.forward_step > 0:
            [neuron["neuron"].backprop() for neuron in self.neurons["sensory"]]
        [neuron["neuron"].forward() for neuron in self.neurons["sensory"]]
        if self.neurons["intern"]:
            self.runAttentionFieldStep(1)
            if self.forward_step > 0:
                [neuron["neuron"].backprop() for neuron in self.neurons["intern"]]
            [neuron["neuron"].forward() for neuron in self.neurons["intern"]]
        if len(self.neurons["all"]) > len(self.neurons["sensory-motor"]):
            self.runAttentionFieldStep(2)
        if self.forward_step > 0:
            [neuron["neuron"].backprop() for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]
        for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]:
            neuron["neuron"].forward()
            neuron["action"] = neuron["neuron"].output_value
        self.forward_step += 1
        if self.forward_step % 5000 == 0:
            self.makePlots()

    def runAttentionFieldStep(self, stage):
        if stage == 1 or stage == 2:
            for neuron in self.neurons["sensory"]:
                self.attention_field.addEntries(None, neuron["neuron"].key, neuron["neuron"].output_value)
        if stage == 1:
            for neuron in self.neurons["intern"]:
                self.attention_field.addEntries(neuron["neuron"].query,
                                                neuron["neuron"].key,
                                                neuron["neuron"].output_value)
        elif stage == 2:
            for neuron in self.neurons["intern"]:
                self.attention_field.addEntries(None,
                                                neuron["neuron"].key,
                                                neuron["neuron"].output_value)
        if stage == 2:
            for neuron in self.neurons["motor"]:
                self.attention_field.addEntries(neuron["neuron"].query, None, None)

        values, attended = self.attention_field.runStep()
        if stage == 1:
            neurons = self.neurons["intern"]
        elif stage == 2:
            neurons = self.neurons["motor"]
        for i, value in enumerate(values):
            neurons[i]["neuron"].setNextInputValue(np.array([value]))
            neurons[i]["neuron"].attended = attended[i]

    def setStateAndReward(self):
        [neuron["neuron"].setNextInputValue(neuron["state"]) for neuron in self.neurons["sensory-motor"] + self.neurons["sensory"]]
        [self.allocateReward(np.array(neuron["reward"]).mean(), neuron["neuron"].attended) for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["sensory"]]
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["intern"]]
        [neuron["neuron"].setReward(np.array(neuron["reward"]).mean()) for neuron in self.neurons["sensory-motor"] + self.neurons["motor"]]
        
        self.scores_deque.append(np.array([np.array(neuron["reward"]).mean() for neuron in self.neurons["motor"]]).sum())
        self.scores.append(np.array([np.array(neuron["reward"]).mean() for neuron in self.neurons["motor"]]).sum())

        for neuron in self.neurons["all"]:
            neuron["reward"] = []

    def allocateReward(self, reward, attendeds):
        split_rewards = np.array(attendeds)*reward
        neurons = self.neurons["sensory"] + self.neurons["intern"]
        
        for i, split_reward in enumerate(split_rewards):
            if abs(split_reward) > 0.01:
                if neurons[i]["reward"] == []:
                    neurons[i]["reward"] = split_reward
                else:
                    neurons[i]["reward"] += split_reward
                if neurons[i]["neuron"].neuron_type != "sensory":
                    self.allocateReward(split_reward, neurons[i]["neuron"].attended)

    def makePlots(self):
        ### Attention heatmap
        # fig, ax = plt.subplots()
        # df = pd.DataFrame(data=np.array([neuron["neuron"].attended for neuron in self.neurons["intern"] + self.neurons["motor"]]),
        #                     index=range(len(self.neurons["sensory"])+1,len(self.neurons["all"])+1),
        #                     columns=range(1,len(self.neurons["all"])-len(self.neurons["motor"])+1))
        # sns.heatmap(df, vmin=0, vmax=1.0)
        # plt.xlabel('AttendeDs: Sensory (1-{}) and Intern ({}-{})'.format(len(self.neurons["sensory"]),
        #                                                                 len(self.neurons["sensory"])+1,
        #                                                                 len(self.neurons["all"])-len(self.neurons["motor"])))
        # plt.ylabel('AttendeRs: Intern ({}-{}) and Motor ({}-{})'.format(len(self.neurons["sensory"])+1,
        #                                                                 len(self.neurons["all"])-len(self.neurons["motor"]),
        #                                                                 len(self.neurons["all"])-len(self.neurons["motor"])+1,
        #                                                                 len(self.neurons["all"])))
        # plt.title("Brain: Full Attention field")
        # plt.savefig('Attention.png')

        ### Reward
        # fig, ax = plt.subplots()
        # plt.plot(np.arange(1, len(self.scores)+1), self.scores)
        # plt.title("Brain: Reward")
        # plt.ylabel('Score')
        # plt.xlabel('Step #')
        self.tensorboard_writer.add_scalar('avg_score',
                                        np.mean(self.scores_deque),
                                        self.forward_step)
        # plt.show()

        ### 3D Attention Field
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
        