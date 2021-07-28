import numpy as np
from copy import deepcopy
import json
import os

from brain.Neuron import Neuron
from brain.AttentionField import AttentionField

class Brain(object):

    def __init__(self):

        self.neurons = {"all": [], "sensory" : [], "intern" : [], "motor" : []}
        with open(os.path.join(os.path.dirname(__file__),'config.json'), 'r') as j:
            self.config = json.load(j)
        self.k_dim, self.v_dim = self.config["attention_field"]["key_dim"], self.config["attention_field"]["value_dim"]
        self.startAttentionField()
        self.spawnNeurons()
        self.forward_step = 0

    def spawnNeurons(self): ### SIMPLIFY FUNCTION
        for neuron_type, type_neuron_config in self.config["neurons"].items():
            if neuron_type == "sensory":
                for neuron_config in type_neuron_config["neurons"]:
                    self.spawnOneNeuron(neuron_type, self.k_dim, self.v_dim, neuron_config["agent"]["additional_dim"])

            elif neuron_type == "intern":
                for _ in range(type_neuron_config["quantity"]):
                    self.spawnOneNeuron(neuron_type, self.k_dim, self.v_dim)

            elif neuron_type == "sensory" or neuron_type == "motor":
                for neuron_config in type_neuron_config["neurons"]:
                    self.spawnOneNeuron(neuron_type, self.k_dim, self.v_dim, neuron_config["agent"]["additional_dim"])

    def spawnOneNeuron(self, neuron_type, k_min, v_min, additional_dim = None):
        empty_neuron = {"neuron" : None, "state" : [], "next_state" : [], "action" : [], "reward" : [], "attended" : [], "info" : {"type" : ""}}
        neuron = deepcopy(empty_neuron)
        neuron["neuron"] = Neuron(neuron_type, k_min, v_min, additional_dim)
        neuron["info"]["type"] = neuron_type
        self.neurons["all"].append(neuron)
        self.neurons[neuron_type].append(neuron)

    def forward(self):
        # print("Brain: forward step {}".format(self.forward_step))
        if self.forward_step > 0:
            [neuron["neuron"].backprop() for neuron in self.neurons["sensory"]]
        [neuron["neuron"].forward() for neuron in self.neurons["sensory"]]
        self.runAttentionFieldStep()
        if self.forward_step > 0:
            [neuron["neuron"].backprop() for neuron in self.neurons["motor"]]
        if self.forward_step > 1:
            [neuron["neuron"].backprop() for neuron in self.neurons["intern"]]
        [neuron["neuron"].forward() for neuron in self.neurons["intern"]]
        for neuron in self.neurons["motor"]:
            neuron["neuron"].forward()
            neuron["action"] = neuron["neuron"].output_value
        self.forward_step += 1

    def runAttentionFieldStep(self):
        for neuron in self.neurons["sensory"]:
            self.attention_field.addEntries(None, neuron["neuron"].key, neuron["neuron"].output_value)
        for neuron in self.neurons["intern"]:
            self.attention_field.addEntries(neuron["neuron"].query,
                                            neuron["neuron"].key,
                                            neuron["neuron"].output_value)
        for neuron in self.neurons["motor"]:
            self.attention_field.addEntries(neuron["neuron"].query, None, None)

        values, attended = self.attention_field.runStep()
        neurons = self.neurons["intern"] + self.neurons["motor"]
        for i, value in enumerate(values):
            neurons[i]["neuron"].setNextInputValue(np.array([value]))
            neurons[i]["neuron"].attended = attended[i]

    def setStateAndReward(self):
        [neuron["neuron"].setNextInputValue(neuron["state"]) for neuron in self.neurons["sensory"]]

        [self.allocateReward(neuron["reward"], neuron["neuron"].attended) for neuron in self.neurons["motor"]]
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["sensory"]]
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["intern"]]
        [neuron["neuron"].setReward(neuron["reward"]) for neuron in self.neurons["motor"]]

    def allocateReward(self, reward, attendeds):
        split_rewards = np.array(attendeds)*np.array(reward)
        neurons = self.neurons["sensory"] + self.neurons["intern"]
        
        for i, split_reward in enumerate(split_rewards):
            if split_reward > 0.1:
                neurons[i]["reward"] =+ split_reward
                if neurons[i].neuron_type != "sensory":
                    self.allocateReward(split_reward, neurons[i]["neuron"].attended)            

    def startAttentionField(self):
        af_config = self.config["attention_field"]
        self.attention_field = AttentionField(af_config["key_dim"], af_config["value_dim"])